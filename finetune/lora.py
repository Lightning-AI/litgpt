"""
Instruction-tuning with LoRA on the Alpaca dataset.

Note: If you run into a CUDA error "Expected is_sm80 to be true, but got false", uncomment the line
`torch.backends.cuda.enable_flash_sdp(False)` in the script below (see https://github.com/Lightning-AI/lit-llama/issues/101).
"""
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
from lit_parrot.utils import lazy_load, check_valid_checkpoint_dir  # TODO: Use lazy_load
from scripts.prepare_alpaca import generate_prompt


instruction_tuning = True
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
max_iters = 500000 * 3 // micro_batch_size
weight_decay = 0.0
lora_r = 8
lora_alpha = 16
lora_dropout = 0.05
warmup_iters = 100

ds_config = {
    "train_micro_batch_size_per_gpu": micro_batch_size,
    "gradient_accumulation_iters": gradient_accumulation_iters,
    "zero_optimization": {"stage": 2},
}


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
    fabric = L.Fabric(devices=fabric_devices, strategy=strategy, precision=precision)
    fabric.launch(main, data_dir, checkpoint_dir, out_dir)


def main(
    fabric: L.Fabric = None,
    data_dir: Path = Path("data/alpaca"),
    checkpoint_dir: Path = Path("checkpoints/stabilityai/stablelm-base-alpha-3b"),
    out_dir: Path = Path("out/lora/alpaca"),
):
    check_valid_checkpoint_dir(checkpoint_dir)
    fabric.seed_everything(1337 + fabric.global_rank)

    if fabric.global_rank == 0:
        os.makedirs(out_dir, exist_ok=True)

    train_data, val_data = load_datasets(data_dir=data_dir)

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

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model, optimizer = fabric.setup(model, optimizer)
    train(fabric, model, optimizer, train_data, val_data, checkpoint_dir, out_dir)

    # Save the final LoRA checkpoint at the end of training
    save_path = out_dir / "lit_model-lora-finetuned.pth"
    save_lora_checkpoint(fabric, model, path=save_path)


def train(
    fabric: L.Fabric,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_data: np.ndarray,
    val_data: np.ndarray,
    checkpoint_dir: Path,
    out_dir: Path,
) -> None:
    """The training loop.

    Loosely based on the nanoGPT implementation: https://github.com/karpathy/nanoGPT.
    """
    step_count = 0

    tokenizer = Tokenizer(checkpoint_dir / "tokenizer.json", checkpoint_dir / "tokenizer_config.json")

    if fabric.device.type == "xla":
        import torch_xla.core.xla_model as xm

        xm.mark_step()
    for iter_num in range(max_iters):
        if step_count <= warmup_iters:
            # linear warmup
            lr = learning_rate * step_count / warmup_iters
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

        t0 = time.time()

        input_ids, targets = get_batch(fabric, train_data)

        with fabric.no_backward_sync(model, enabled=((iter_num + 1) % gradient_accumulation_iters != 0)):
            logits = model(input_ids)
            loss = loss_fn(logits, targets)
            fabric.backward(loss / gradient_accumulation_iters)

        if (iter_num + 1) % gradient_accumulation_iters == 0:
            optimizer.step()
            if fabric.device.type == "xla":
                xm.mark_step()
            optimizer.zero_grad()
            step_count += 1

            if step_count % eval_interval == 0:
                val_loss = validate(fabric, model, val_data, tokenizer)
                fabric.print(f"step {iter_num}: val loss {val_loss:.4f}")
                fabric.barrier()

            if step_count % save_interval == 0:
                # We are only saving the LoRA weights
                # TODO: Provide a function/script to merge the LoRA weights with pretrained weights
                save_lora_checkpoint(model, path=os.path.join(out_dir, f"iter-{iter_num:06d}-ckpt.pth"))
        else:
            if fabric.device.type == "xla":
                xm.mark_step()

        dt = time.time() - t0
        if iter_num % log_interval == 0:
            fabric.print(f"iter {iter_num}: loss {loss.item():.4f}, time: {dt*1000:.2f}ms")


@torch.no_grad()
def validate(fabric: L.Fabric, model: torch.nn.Module, val_data: np.ndarray, tokenizer: Tokenizer) -> torch.Tensor:
    fabric.print("Validating ...")
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        input_ids, targets = get_batch(fabric, val_data)
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
    output = generate(
        model,
        idx=encoded,
        max_returned_tokens=len(encoded) + 100,
        max_seq_length=model.config.block_size,
        temperature=0.8,
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


def get_batch(fabric: L.Fabric, data: list):
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


def load_datasets(data_dir: Path):
    train_data = torch.load(data_dir / "train.pt")
    val_data = torch.load(data_dir / "test.pt")
    return train_data, val_data


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
