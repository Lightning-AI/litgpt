# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import lightning as L
import torch
from lightning.fabric.loggers import CSVLogger
from lightning.fabric.plugins import BitsandbytesPrecision
from lightning.fabric.strategies import FSDPStrategy
from lightning.fabric.utilities import ThroughputMonitor

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from generate.base import generate
from lit_gpt.adapter import GPT, Block, Config, adapter_filter, mark_only_adapter_as_trainable
from lit_gpt.tokenizer import Tokenizer
from lit_gpt.utils import (
    check_valid_checkpoint_dir,
    chunked_cross_entropy,
    get_default_supported_precision,
    load_checkpoint,
    num_parameters,
)
from scripts.prepare_alpaca import generate_prompt

eval_interval = 600
save_interval = 1000
eval_iters = 100
eval_max_new_tokens = 100
log_interval = 1
devices = 1

# Hyperparameters
learning_rate = 3e-3
batch_size = 64 // devices
micro_batch_size = 4
gradient_accumulation_iters = batch_size // micro_batch_size
assert gradient_accumulation_iters > 0
max_seq_length = None  # assign value to truncate
epoch_size = 50000  # train dataset size
num_epochs = 5
max_iters = num_epochs * epoch_size // devices // micro_batch_size
max_steps = num_epochs * epoch_size // devices // batch_size
weight_decay = 0.02
warmup_steps = 2 * (epoch_size // devices // batch_size)  # 2 epochs

hparams = {k: v for k, v in locals().items() if isinstance(v, (int, float, str)) and not k.startswith("_")}


def setup(
    data_dir: Path = Path("data/alpaca"),
    checkpoint_dir: Path = Path("checkpoints/stabilityai/stablelm-base-alpha-3b"),
    out_dir: Path = Path("out/adapter/alpaca"),
    precision: Optional[str] = None,
    quantize: Optional[Literal["bnb.nf4", "bnb.nf4-dq", "bnb.fp4", "bnb.fp4-dq", "bnb.int8-training"]] = None,
) -> None:
    precision = precision or get_default_supported_precision(training=True)

    plugins = None
    if quantize is not None and quantize.startswith("bnb."):
        if "mixed" in precision:
            raise ValueError("Quantization and mixed precision is not supported.")
        dtype = {"16-true": torch.float16, "bf16-true": torch.bfloat16, "32-true": torch.float32}[precision]
        plugins = BitsandbytesPrecision(quantize[4:], dtype)
        precision = None

    if devices > 1:
        if quantize:
            raise NotImplementedError(
                "Quantization is currently not supported for multi-GPU training. Please set devices=1 when using the"
                " --quantize flag."
            )
        strategy = FSDPStrategy(
            auto_wrap_policy={Block},
            activation_checkpointing_policy={Block},
            state_dict_type="full",
            limit_all_gathers=True,
            cpu_offload=False,
        )
    else:
        strategy = "auto"

    logger = CSVLogger(out_dir.parent, out_dir.name, flush_logs_every_n_steps=log_interval)
    fabric = L.Fabric(devices=devices, strategy=strategy, precision=precision, loggers=logger, plugins=plugins)
    fabric.print(hparams)
    fabric.launch(main, data_dir, checkpoint_dir, out_dir)


def main(fabric: L.Fabric, data_dir: Path, checkpoint_dir: Path, out_dir: Path) -> None:
    check_valid_checkpoint_dir(checkpoint_dir)

    fabric.seed_everything(1337)  # same seed for every process to init model (FSDP)

    if fabric.global_rank == 0:
        os.makedirs(out_dir, exist_ok=True)

    train_data = torch.load(data_dir / "train.pt")
    val_data = torch.load(data_dir / "test.pt")

    config = Config.from_name(name=checkpoint_dir.name)
    checkpoint_path = checkpoint_dir / "lit_model.pth"
    fabric.print(f"Loading model {str(checkpoint_path)!r} with {config.__dict__}")
    with fabric.init_module(empty_init=(devices > 1)):
        model = GPT(config)
    mark_only_adapter_as_trainable(model)

    fabric.print(f"Number of trainable parameters: {num_parameters(model, requires_grad=True):,}")
    fabric.print(f"Number of non trainable parameters: {num_parameters(model, requires_grad=False):,}")

    model = fabric.setup_module(model)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if isinstance(fabric.strategy.precision, BitsandbytesPrecision):
        import bitsandbytes as bnb

        optimizer = bnb.optim.PagedAdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)
    optimizer = fabric.setup_optimizers(optimizer)
    scheduler = get_lr_scheduler(optimizer, warmup_steps=warmup_steps, max_steps=max_steps)

    # strict=False because missing keys due to Adapter weights not contained in state dict
    load_checkpoint(fabric, model, checkpoint_path, strict=False)

    fabric.seed_everything(1337 + fabric.global_rank)

    train_time = time.perf_counter()
    train(fabric, model, optimizer, scheduler, train_data, val_data, checkpoint_dir, out_dir)
    fabric.print(f"Training time: {(time.perf_counter()-train_time):.2f}s")
    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")

    # Save the final checkpoint at the end of training
    save_path = out_dir / "lit_model_adapter_finetuned.pth"
    save_adapter_checkpoint(fabric, model, save_path)


def train(
    fabric: L.Fabric,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler,
    train_data: List[Dict],
    val_data: List[Dict],
    checkpoint_dir: Path,
    out_dir: Path,
) -> None:
    tokenizer = Tokenizer(checkpoint_dir)
    longest_seq_length, longest_seq_ix = get_longest_seq_length(train_data)
    model.max_seq_length = min(longest_seq_length, max_seq_length or float("inf"))
    fabric.print(
        f"The longest sequence length in the train data is {longest_seq_length}, the model's maximum sequence length is"
        f" {model.max_seq_length} and context length is {model.config.block_size}"
    )

    validate(fabric, model, val_data, tokenizer, max_iters=2)  # sanity check

    throughput = ThroughputMonitor(fabric, window_size=50)
    step_count = 0
    total_lengths = 0
    total_t0 = time.perf_counter()

    for iter_num in range(1, max_iters + 1):
        iter_t0 = time.perf_counter()

        input_ids, targets = get_batch(fabric, train_data, longest_seq_ix if iter_num == 1 else None)

        is_accumulating = iter_num % gradient_accumulation_iters != 0
        with fabric.no_backward_sync(model, enabled=is_accumulating):
            logits = model(input_ids, lm_head_chunk_size=128)
            # shift the targets such that output n predicts token n+1
            logits[-1] = logits[-1][..., :-1, :]
            loss = chunked_cross_entropy(logits, targets[..., 1:])
            fabric.backward(loss / gradient_accumulation_iters)

        if not is_accumulating:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            step_count += 1

        total_lengths += input_ids.numel()
        if iter_num % log_interval == 0:
            loss_item = loss.item()  # expensive device-to-host synchronization
            t1 = time.perf_counter()
            throughput.update(
                time=t1 - total_t0, batches=iter_num, samples=iter_num * micro_batch_size, lengths=total_lengths
            )
            throughput.compute_and_log(step=iter_num)
            fabric.print(
                f"iter {iter_num} step {step_count}: loss {loss_item:.4f}, iter time:"
                f" {(t1 - iter_t0) * 1000:.2f}ms{' (optimizer.step)' if not is_accumulating else ''}"
            )

        if not is_accumulating and step_count % eval_interval == 0:
            t0 = time.perf_counter()
            val_loss = validate(fabric, model, val_data, tokenizer, max_iters=eval_iters)
            t1 = time.perf_counter() - t0
            fabric.print(f"step {iter_num}: val loss {val_loss.item():.4f}, val time: {t1 * 1000:.2f}ms")
            fabric.barrier()
        if not is_accumulating and step_count % save_interval == 0:
            checkpoint_path = out_dir / f"iter-{iter_num:06d}-ckpt.pth"
            save_adapter_checkpoint(fabric, model, checkpoint_path)


# the adapter "kv cache" cannot be initialized under `inference_mode`
@torch.no_grad()
def validate(fabric: L.Fabric, model: GPT, val_data: List[Dict], tokenizer: Tokenizer, max_iters: int) -> torch.Tensor:
    fabric.print("Validating ...")
    model.eval()
    losses = torch.zeros(max_iters)
    for k in range(max_iters):
        input_ids, targets = get_batch(fabric, val_data)
        logits = model(input_ids)
        losses[k] = chunked_cross_entropy(logits[..., :-1, :], targets[..., 1:], chunk_size=0)
    val_loss = losses.mean()

    # produce an example:
    instruction = "Recommend a movie for me to watch during the weekend and explain the reason."
    fabric.print(instruction)
    sample = {"instruction": instruction, "input": ""}
    prompt = generate_prompt(sample)
    encoded = tokenizer.encode(prompt, device=fabric.device)
    with fabric.init_tensor():
        # do not set `max_seq_length=max_returned_token` because memory is not a concern here
        model.set_kv_cache(batch_size=1)
    output = generate(
        model, encoded, max_returned_tokens=len(encoded) + eval_max_new_tokens, temperature=0.8, eos_id=tokenizer.eos_id
    )
    model.clear_kv_cache()
    output = tokenizer.decode(output)
    fabric.print(output)

    model.train()
    return val_loss


def get_batch(
    fabric: L.Fabric, data: List[Dict], longest_seq_ix: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(len(data), (micro_batch_size,))
    if longest_seq_ix is not None:
        # force the longest sample at the beginning so potential OOMs happen right away
        ix[0] = longest_seq_ix

    input_ids = [data[i]["input_ids"].type(torch.int64) for i in ix]
    labels = [data[i]["labels"].type(torch.int64) for i in ix]

    # this could be `longest_seq_length` to have a fixed size for all batches
    max_len = max(len(s) for s in input_ids)

    def pad_right(x, pad_id):
        # pad right based on the longest sequence
        n = max_len - len(x)
        return torch.cat((x, torch.full((n,), pad_id, dtype=x.dtype)))

    x = torch.stack([pad_right(x, pad_id=0) for x in input_ids])
    y = torch.stack([pad_right(x, pad_id=-1) for x in labels])

    # Truncate if needed
    if max_seq_length:
        x = x[:, :max_seq_length]
        y = y[:, :max_seq_length]

    if fabric.device.type == "cuda" and x.device.type == "cpu":
        x, y = fabric.to_device((x.pin_memory(), y.pin_memory()))
    else:
        x, y = fabric.to_device((x, y))
    return x, y


def get_lr_scheduler(optimizer, warmup_steps: int, max_steps: int):
    # linear warmup followed by cosine annealing
    scheduler1 = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: step / warmup_steps)
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(max_steps - warmup_steps))
    return torch.optim.lr_scheduler.SequentialLR(optimizer, [scheduler1, scheduler2], milestones=[warmup_steps])



def get_longest_seq_length(data: List[Dict]) -> Tuple[int, int]:
    # find out the minimum max_seq_length required during fine-tuning (saves memory!)
    lengths = [len(d["input_ids"]) for d in data]
    longest_seq_length = max(lengths)
    longest_seq_ix = lengths.index(longest_seq_length)
    return longest_seq_length, longest_seq_ix


def save_adapter_checkpoint(fabric: L.Fabric, model: torch.nn.Module, file_path: Path) -> None:
    fabric.print(f"Saving adapter weights to {str(file_path)!r}")
    fabric.save(file_path, {"model": model}, filter={"model": adapter_filter})


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    from jsonargparse import CLI

    CLI(setup)
