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
batch_size = 128
micro_batch_size = 4
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

    with fabric.device:
        torch.set_default_tensor_type(torch.HalfTensor)
        model = LLaMA(config).bfloat16()
        model.apply(model._init_weights)
        torch.set_default_tensor_type(torch.FloatTensor)

    # if compile:
    #     model = torch.compile(model)

    model = fabric.setup_module(model)

    train_dataloader, val_dataloader = create_dataloaders(
        batch_size=micro_batch_size,
        block_size=config.block_size,
        train_data_dir=train_data_dir,
        val_data_dir=val_data_dir,
        seed=1338,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(beta1, beta2),
    )
    optimizer = fabric.setup_optimizers(optimizer)

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

    for iter_num, train_data in enumerate(train_dataloader):
        # determine and set the learning rate for this iteration
        lr = get_lr(iter_num) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        t0 = time.time()

        input_ids, targets = get_batch(
            fabric, train_data, block_size=model.config.block_size
        )

        is_accumulating = (iter_num + 1) % grad_accum_steps == 0

        with fabric.no_backward_sync(model, enabled=is_accumulating):
            logits = model(input_ids)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
            fabric.backward(loss / grad_accum_steps)

        if not is_accumulating:
            fabric.clip_gradients(model, optimizer, max_norm=grad_clip)

            optimizer.step()
            optimizer.zero_grad()
            step_count += 1

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

        dt = time.time() - t0
        if iter_num % log_interval == 0:
            fabric.log_dict(
                {"iter": iter_num, "train_loss": loss, "step": step_count, "lr": lr}
            )
            fabric.print(
                f"iter {iter_num}: loss {loss.item():.4f}, time: {dt*1000:.2f}ms"
            )

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
        input_ids, targets = get_batch(
            fabric,
            val_data,
            block_size=model.config.block_size,  # type: ignore[union-attr,arg-type]
        )
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

    return DataLoader(combined_dataset, batch_size=batch_size, shuffle=False)


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


def get_batch(
    fabric: L.Fabric, data: np.ndarray, block_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    x = data[:, 0:block_size]
    y = data[:, 1 : block_size + 1]
    x, y = fabric.to_device((x.pin_memory(), y.pin_memory()))
    return x, y


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
