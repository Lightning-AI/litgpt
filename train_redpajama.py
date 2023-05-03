import os
import math
import glob
import time
from functools import partial
from pathlib import Path
from typing import Tuple, Optional

import lightning as L
from lightning.fabric.strategies import FSDPStrategy

import torch
from torch.utils.data import DataLoader
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

import numpy as np

from lit_llama.model import Block, LLaMA, LLaMAConfig
from lit_llama.packed_dataset import PackedDataset, CombinedDataset
from lit_llama.utils import save_model_checkpoint


out_dir = "out/training"
save_interval = 1000
eval_interval = 1000
eval_iters = 100
log_interval = 1

# compile = False

# Hyperparameters
learning_rate = 6e-4
batch_size = 125
micro_batch_size = 5
max_iters = 600000  # num_epochs * epoch_size // devices
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = True
warmup_iters = 2000
lr_decay_iters = max_iters
min_lr = 6e-5


# Data proportions from https://arxiv.org/pdf/2302.13971.pdf Table 1
data_config = [
    ("arxiv", 2.5),
    ("book", 4.5),
    ("c4", 15.0),
    ("cc", 67.0),
    ("github", 4.5),
    ("stackexchange", 2.0),
    ("wikipedia", 4.5),
]


def main(
    devices: int = 4,
    train_data_dir: Path = "data/red_pajama",
    val_data_dir: Path = "data/red_pajama_val",
) -> None:
    auto_wrap_policy = partial(
        transformer_auto_wrap_policy, transformer_layer_cls={Block}
    )
    strategy = FSDPStrategy(
        auto_wrap_policy=auto_wrap_policy, activation_checkpointing=Block
    )

    fabric = L.Fabric(
        accelerator="cuda", devices=devices, precision="bf16-mixed", strategy=strategy
    )
    fabric.launch()
    fabric.seed_everything(1337)

    if fabric.global_rank == 0:
        os.makedirs(out_dir, exist_ok=True)

    config = LLaMAConfig.from_name("7B")

    train_dataloader, val_dataloader = create_dataloaders(
        batch_size=micro_batch_size,
        block_size=config.block_size,
        train_data_dir=train_data_dir,
        val_data_dir=val_data_dir,
        seed=1338,
    )
    train_dataloader, val_dataloader = fabric.setup_dataloaders(train_dataloader, val_dataloader)

    with fabric.device:
        torch.set_default_dtype(torch.bfloat16)
        model = LLaMA(config)
        model.apply(model._init_weights)
        torch.set_default_dtype(torch.float32)

    # if compile:
    #     model = torch.compile(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(beta1, beta2),
    )

    model, optimizer = fabric.setup(model, optimizer)

    process_batch_size = batch_size // devices
    grad_accum_steps = process_batch_size // micro_batch_size

    train(fabric, model, optimizer, train_dataloader, val_dataloader, grad_accum_steps)


def train(
    fabric: L.Fabric,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_dataloader: DataLoader,
    val_dataloader: Optional[DataLoader],
    grad_accum_steps: int,
) -> None:
    """The training loop.

    Loosely based on the nanoGPT implementation: https://github.com/karpathy/nanoGPT.
    """

    step_count = 0

    step_time = 0.0
    tokens = 0
    tokens_sec = 0.0
    prev_t1 = time.time()

    for iter_num, train_data in enumerate(train_dataloader):
        t0 = time.time()

        # determine and set the learning rate for this iteration
        lr = get_lr(iter_num) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr


        input_ids = train_data[:, 0 : model.config.block_size].contiguous()
        targets = train_data[:, 1 : model.config.block_size + 1].contiguous()
        
        is_accumulating = (iter_num + 1) % grad_accum_steps != 0

        with fabric.no_backward_sync(model, enabled=is_accumulating):
            logits = model(input_ids)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
            fabric.backward(loss / grad_accum_steps)

        t1 = time.time()

        if not is_accumulating:
            fabric.clip_gradients(model, optimizer, max_norm=grad_clip)

            optimizer.step()
            optimizer.zero_grad()
            step_count += 1

            t1 = time.time()

            if step_count % eval_interval == 0:
                val_loss = validate(fabric, model, val_dataloader)
                fabric.print(f"step {iter_num}: val loss {val_loss:.4f}")
                fabric.barrier()
                fabric.log_dict(
                    {"iter": iter_num, "val_loss": val_loss, "step": step_count, "lr": lr}
                )

            if step_count % save_interval == 0:
                fabric.print(f"Saving checkpoint to {out_dir}")
                save_model_checkpoint(
                    fabric, model, os.path.join(out_dir, f"iter-{iter_num:06d}-ckpt.pth")
                )

        dt = t1 - t0

        tokens += micro_batch_size * model.config.block_size
        step_time += t1 - prev_t1
        prev_t1 = t1

        if iter_num % log_interval == 0:
            tokens_sec_str = f"{tokens / step_time:.0f}" if not is_accumulating else "-"

            fabric.log_dict(
                {"iter": iter_num, "train_loss": loss, "step": step_count, "lr": lr}
            )
            fabric.print(
                    f"iter {iter_num}: loss {loss.item():.4f}, time: {dt*1000:.2f}ms, speed: {tokens_sec_str} toks/s/device"
            )

        if not is_accumulating:
            tokens = 0
            step_time = 0.0

        if iter_num > max_iters:
            break


@torch.no_grad()
def validate(
    fabric: L.Fabric, model: torch.nn.Module, val_dataloader: DataLoader
) -> torch.Tensor:
    fabric.print("Validating ...")
    model.eval()
    losses = torch.zeros(eval_iters)
    for k, val_data in enumerate(val_dataloader):
        input_ids = val_data[:, 0 : model.config.block_size].contiguous()
        targets = val_data[:, 1 : model.config.block_size + 1].contiguous()
        logits = model(input_ids)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
        )
        losses[k] = loss.item()
    out = losses.mean()
    model.train()
    return out


def create_dataloader(
    batch_size: int,
    block_size: int,
    data_dir: str,
    shuffle: bool = True,
    seed: int = 12345,
) -> DataLoader:
    datasets = []
    for prefix, _ in data_config:
        filenames = glob.glob(os.path.join(data_dir, prefix + "*"))
        dataset = PackedDataset(
            filenames, n_chunks=10, block_size=block_size, shuffle=shuffle, seed=seed
        )
        datasets.append(dataset)

    if not datasets:
        raise RuntimeError(
            f"No data found at {data_dir}. Make sure you ran prepare_redpajama.py to create the dataset."
        )

    weights = [weight for _, weight in data_config]
    sum_weights = sum(weights)
    weights = [el / sum_weights for el in weights]

    combined_dataset = CombinedDataset(datasets=datasets, seed=seed, weights=weights)

    return DataLoader(combined_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)


def create_dataloaders(
    batch_size: int,
    block_size: int,
    train_data_dir: str = "data/red_pajama",
    val_data_dir: str = "data/red_pajama_val",
    seed: int = 12345,
) -> Tuple[DataLoader, DataLoader]:
    # Increase by one because we need the next word as well
    effective_block_size = block_size + 1
    train_dataloader = create_dataloader(
        batch_size=batch_size,
        block_size=effective_block_size,
        data_dir=train_data_dir,
        shuffle=True,
        seed=seed,
    )
    val_dataloader = (
        create_dataloader(
            batch_size=batch_size,
            block_size=effective_block_size,
            data_dir=val_data_dir,
            shuffle=False,
            seed=seed,
        )
        if val_data_dir
        else None
    )
    return train_dataloader, val_dataloader


# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


if __name__ == "__main__":
    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    torch.set_float32_matmul_precision("high")

    from jsonargparse.cli import CLI

    CLI(main)
