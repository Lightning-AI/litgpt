import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Optional

import lightning as L
import numpy as np
import torch
from lightning.fabric.strategies import DeepSpeedStrategy, XLAStrategy

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from generate.base import generate
from lit_parrot.lora import mark_only_lora_as_trainable, lora, lora_state_dict
from lit_parrot.model import Parrot, Config
from lit_parrot.tokenizer import Tokenizer
from lit_parrot.utils import lazy_load, check_valid_checkpoint_dir, step_csv_logger
from lit_parrot.speed_monitor import SpeedMonitor, measure_flops, estimate_flops
from scripts.prepare_alpaca import generate_prompt


eval_interval = 100
save_interval = 100
eval_iters = 100
log_interval = 1
devices = 1

# Hyperparameters
learning_rate = 3e-4
batch_size = 128
micro_batch_size = 4
gradient_accumulation_iters = batch_size // micro_batch_size
assert gradient_accumulation_iters > 0
max_iters = 50000  # train dataset size
weight_decay = 0.01
lora_r = 8
lora_alpha = 16
lora_dropout = 0.05
warmup_iters = 100

ds_config = {
    "train_micro_batch_size_per_gpu": micro_batch_size,
    "gradient_accumulation_steps": gradient_accumulation_iters,
    "zero_optimization": {"stage": 2},
}

hparams = {k: v for k, v in locals().items() if isinstance(v, (int, float, str)) and not k.startswith("_")}


def setup(
    data_dir: Path = Path("data/alpaca"),
    checkpoint_dir: Path = Path("checkpoints/stabilityai/stablelm-base-alpha-3b"),
    out_dir: Path = Path("out/lora/alpaca"),
    precision: Optional[str] = None,
    tpu: bool = False,
):
    if precision is None:
        precision = "32-true" if tpu else "bf16-mixed"
    strategy = (
        "auto"
        if devices <= 1
        else XLAStrategy(sync_module_states=False) if tpu else DeepSpeedStrategy(config=ds_config)
    )
    # For multi-host TPU training, the device count for Fabric is limited to the count on a single host.
    fabric_devices = "auto" if (tpu and devices > 1) else devices

    logger = step_csv_logger(out_dir.parent, out_dir.name, flush_logs_every_n_steps=log_interval)
    fabric = L.Fabric(devices=fabric_devices, strategy=strategy, precision=precision, loggers=logger)
    fabric.print(hparams)
    fabric.launch(main, data_dir, checkpoint_dir, out_dir)


def main(fabric: L.Fabric, data_dir: Path, checkpoint_dir: Path, out_dir: Path):
    check_valid_checkpoint_dir(checkpoint_dir)

    speed_monitor = SpeedMonitor(fabric, window_size=50, time_unit="seconds")

    fabric.seed_everything(1337 + fabric.global_rank)

    if fabric.global_rank == 0:
        os.makedirs(out_dir, exist_ok=True)

    train_data = torch.load(data_dir / "train.pt")
    val_data = torch.load(data_dir / "test.pt")

    config = Config.from_name(name=checkpoint_dir.name)
    checkpoint_path = checkpoint_dir / "lit_model.pth"
    fabric.print(f"Loading model {str(checkpoint_path)!r} with {config.__dict__}")
    with fabric.init_module(), lora(r=lora_r, alpha=lora_alpha, dropout=lora_dropout, enabled=True):
        model = Parrot(config)
    with lazy_load(checkpoint_path) as checkpoint:
        # strict=False because missing keys due to LoRA weights not contained in state dict
        model.load_state_dict(checkpoint, strict=False)

    mark_only_lora_as_trainable(model)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    fabric.print(f"Number of trainable parameters: {num_params}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    model, optimizer = fabric.setup(model, optimizer)

    with open(data_dir / "config.json") as data_config_path:
        max_seq_length = json.load(data_config_path).get("max_seq_length", model.config.block_size)

    train_time = time.time()
    train(fabric, model, optimizer, train_data, val_data, checkpoint_dir, out_dir, max_seq_length, speed_monitor)
    fabric.print(f"Training time: {(time.time()-train_time):.2f}s")

    # Save the final LoRA checkpoint at the end of training
    save_path = out_dir / "lit_model_lora_finetuned.pth"
    save_lora_checkpoint(fabric, model, path=save_path)


def train(
    fabric: L.Fabric,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_data: np.ndarray,
    val_data: np.ndarray,
    checkpoint_dir: Path,
    out_dir: Path,
    max_seq_length: int,
    speed_monitor: SpeedMonitor,
) -> None:
    tokenizer = Tokenizer(checkpoint_dir / "tokenizer.json", checkpoint_dir / "tokenizer_config.json")

    validate(fabric, model, val_data, tokenizer, max_seq_length)  # sanity check

    with torch.device("meta"):
        meta_model = Parrot(model.config)
        # estimated is too much of an optimistic estimate, left just for reference
        estimated_flops = estimate_flops(meta_model) * micro_batch_size
        fabric.print(f"Estimated TFLOPs: {estimated_flops * fabric.world_size / 1e12:.2f}")
        x = torch.randint(0, 1, (micro_batch_size, model.config.block_size))
        measured_flops = measure_flops(meta_model, x)
        fabric.print(f"Measured TFLOPs: {measured_flops * fabric.world_size / 1e12:.2f}")
        del meta_model, x

    step_count = 0
    total_lengths = 0
    total_t0 = time.time()

    if fabric.device.type == "xla":
        import torch_xla.core.xla_model as xm

        xm.mark_step()
    for iter_num in range(max_iters):
        if step_count <= warmup_iters:
            # linear warmup
            lr = learning_rate * step_count / warmup_iters
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

        iter_t0 = time.time()

        input_ids, targets = get_batch(fabric, train_data, max_seq_length)

        is_accumulating = (iter_num + 1) % gradient_accumulation_iters != 0
        with fabric.no_backward_sync(model, enabled=is_accumulating):
            logits = model(input_ids, max_seq_length=max_seq_length)
            loss = loss_fn(logits, targets)
            fabric.backward(loss / gradient_accumulation_iters)

        if not is_accumulating:
            optimizer.step()
            if fabric.device.type == "xla":
                xm.mark_step()
            optimizer.zero_grad()
            step_count += 1
        elif fabric.device.type == "xla":
            xm.mark_step()

        t1 = time.time()
        total_lengths += input_ids.size(1)
        speed_monitor.on_train_batch_end(
            (iter_num + 1) * micro_batch_size,
            t1 - total_t0,
            # this assumes that device FLOPs are the same and that all devices have the same batch size
            fabric.world_size,
            flops_per_batch=measured_flops,
            lengths=total_lengths,
        )
        if iter_num % log_interval == 0:
            fabric.print(
                f"iter {iter_num} step {step_count}: loss {loss.item():.4f}, iter time:"
                f" {(t1 - iter_t0) * 1000:.2f}ms{' (optimizer.step)' if not is_accumulating else ''}"
            )

        if not is_accumulating and step_count % eval_interval == 0:
            t0 = time.time()
            val_loss = validate(fabric, model, val_data, tokenizer, max_seq_length)
            t1 = time.time() - t0
            speed_monitor.eval_end(t1)
            fabric.print(f"step {iter_num}: val loss {val_loss:.4f}, val time: {t1 * 1000:.2f}ms")
            fabric.barrier()
        if not is_accumulating and step_count % save_interval == 0:
            checkpoint_path = out_dir / f"iter-{iter_num:06d}-ckpt.pth"
            fabric.print(f"Saving checkpoint to {str(checkpoint_path)!r}")
            save_lora_checkpoint(fabric, model, checkpoint_path)


@torch.no_grad()
def validate(
    fabric: L.Fabric, model: torch.nn.Module, val_data: np.ndarray, tokenizer: Tokenizer, max_seq_length: int
) -> torch.Tensor:
    fabric.print("Validating ...")
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        input_ids, targets = get_batch(fabric, val_data, max_seq_length)
        logits = model(input_ids)
        loss = loss_fn(logits, targets)
        losses[k] = loss.item()
    val_loss = losses.mean()

    # produce an example:
    instruction = "Recommend a movie for me to watch during the weekend and explain the reason."
    fabric.print(instruction)
    sample = {"instruction": instruction, "input": ""}
    prompt = generate_prompt(sample)
    encoded = tokenizer.encode(prompt, device=model.device)
    max_returned_tokens = len(encoded) + 100
    output = generate(
        model, idx=encoded, max_returned_tokens=max_returned_tokens, max_seq_length=max_returned_tokens, temperature=0.8
    )
    output = tokenizer.decode(output)
    fabric.print(output)

    model.train()
    return val_loss.item()


def loss_fn(logits, targets):
    # shift the targets such that output n predicts token n+1
    logits = logits[..., :-1, :].contiguous()
    targets = targets[..., 1:].contiguous()
    loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
    return loss


def get_batch(fabric: L.Fabric, data: np.ndarray, max_seq_length: int):
    ix = torch.randint(len(data), (micro_batch_size,))

    input_ids = [data[i]["input_ids"].type(torch.int64) for i in ix]
    labels = [data[i]["labels"].type(torch.int64) for i in ix]

    max_len = max(len(s) for s in input_ids) if fabric.device.type != "xla" else max_seq_length

    def pad_right(x, pad_id):
        # pad right based on the longest sequence
        n = max_len - len(x)
        return torch.cat((x, torch.full((n,), pad_id, dtype=x.dtype)))

    x = torch.stack([pad_right(x, pad_id=0) for x in input_ids])
    y = torch.stack([pad_right(x, pad_id=-1) for x in labels])

    if fabric.device.type in ("mps", "xla"):
        x, y = fabric.to_device((x, y))
    else:
        x, y = fabric.to_device((x.pin_memory(), y.pin_memory()))
    return x, y


def save_lora_checkpoint(fabric, model, path):
    fabric.print(f"Saving LoRA weights to {str(path)!r}")
    checkpoint = lora_state_dict(model)
    torch.save(checkpoint, path)


if __name__ == "__main__":
    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    torch.set_float32_matmul_precision("high")

    from jsonargparse.cli import CLI

    warnings.filterwarnings(
        # false positive using deepspeed: https://github.com/Lightning-AI/lightning/pull/17761#discussion_r1219705307
        "ignore",
        message="Remove `.no_backward_sync()` from your code",
    )

    CLI(setup)
