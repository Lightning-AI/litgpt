# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

"""
This script is adapted from TinyLlama:
https://github.com/jzhang38/TinyLlama/blob/main/pretrain/tinyllama.py
"""

import math
import os
import sys
import time
from functools import partial
from pathlib import Path
from typing import Tuple, Union

import lightning as L
import torch
import torch.nn as nn
from lightning.fabric.loggers import CSVLogger, TensorBoardLogger
from lightning.fabric.strategies import FSDPStrategy
from lightning.fabric.utilities.throughput import ThroughputMonitor, measure_flops
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader
from torchmetrics.aggregation import RunningMean

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_gpt.model import GPT, Block, CausalSelfAttention, Config, LLaMAMLP
from lit_gpt.packed_dataset import CombinedDataset
from lit_gpt.utils import CycleIterator, chunked_cross_entropy, num_parameters

# System settings
model_name = "tiny-llama-1.1b"
name = "lit-tiny-llama-1.1b"
out_dir = Path(os.getenv("LIGHTNING_ARTIFACTS_DIR", "out")) / name
logger_name = "tensorboard"
devices = torch.cuda.device_count() or 1

# Hyperparameters
global_batch_size = 512
learning_rate = 4e-4
micro_batch_size = 4
max_tokens = int(3e12)  # 3 trillion
warmup_steps = 2000
log_step_interval = 1
eval_iters = 100
save_step_interval = 1000
eval_step_interval = 1000

weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = True
min_lr = 4e-5

batch_size = global_batch_size // devices
gradient_accumulation_iters = batch_size // micro_batch_size
assert gradient_accumulation_iters > 0
warmup_iters = warmup_steps * gradient_accumulation_iters
log_iter_interval = log_step_interval * gradient_accumulation_iters


hparams = {k: v for k, v in locals().items() if isinstance(v, (int, float, str)) and not k.startswith("_")}


def setup(resume: Union[bool, Path] = False):
    logger = choose_logger(logger_name, name=name, resume=resume)

    strategy = FSDPStrategy(auto_wrap_policy={Block}, state_dict_type="full", sharding_strategy="HYBRID_SHARD")
    fabric = L.Fabric(devices=devices, strategy=strategy, precision="bf16-mixed", loggers=[logger])
    fabric.launch()

    fabric.print(hparams)
    if logger_name in ("tensorboard", "wandb"):
        fabric.logger.log_hyperparams(hparams)

    main(fabric, resume)


def main(fabric, resume):
    if fabric.global_rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    config = Config.from_name(model_name)

    train_dataloader, val_dataloader = create_dataloaders(batch_size=micro_batch_size, block_size=config.block_size)
    train_dataloader, val_dataloader = fabric.setup_dataloaders(train_dataloader, val_dataloader)

    fabric.seed_everything(3407)  # same seed for every process to init model (FSDP)

    fabric.print(f"Loading model with {config.__dict__}")
    t0 = time.perf_counter()
    with fabric.init_module(empty_init=False):
        model = GPT(config)
        model.apply(partial(init_weights, n_layer=config.n_layer, n_embd=config.n_embd))

    fabric.print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.")
    fabric.print(f"Total parameters: {num_parameters(model):,}")

    model = torch.compile(model)
    model = fabric.setup(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2), fused=True
    )
    optimizer = fabric.setup_optimizers(optimizer)

    state = {"model": model, "optimizer": optimizer, "hparams": hparams, "iter_num": 0, "step_count": 0}

    if resume is True:
        resume = max(out_dir.glob("*.pth"), key=(lambda p: int(p.name.split("-")[1])))
    if resume:
        fabric.print(f"Resuming training from {resume}")
        fabric.load(resume, state)

    train_time = time.perf_counter()
    train(fabric, state, train_dataloader, val_dataloader, resume)
    fabric.print(f"Training time: {(time.perf_counter()-train_time):.2f}s")
    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")


def train(fabric, state, train_dataloader, val_dataloader, resume):
    model = state["model"]
    optimizer = state["optimizer"]

    validate(fabric, model, val_dataloader, max_iters=2)  # sanity check
    throughput = ThroughputMonitor(fabric, window_size=5)

    with torch.device("meta"):
        meta_model = GPT(model.config)
        x = torch.randint(0, 1, (micro_batch_size, meta_model.config.block_size))
        model_fwd = lambda: meta_model(x)
        model_loss = lambda y: chunked_cross_entropy(y, x, chunk_size=0)
        measured_flops = measure_flops(meta_model, model_fwd, model_loss)
        fabric.print(f"Measured TFLOPs: {measured_flops * fabric.world_size / 1e12:.2f}")
        del meta_model, x

    max_tokens_per_device = max_tokens // fabric.world_size
    tokens_per_iter = micro_batch_size * model.config.block_size
    max_iters = max_tokens_per_device // tokens_per_iter
    initial_iter = state["iter_num"]
    train_iterator = CycleIterator(train_dataloader)

    # resume data loader state by fast-forwarding through all seen batches
    # drop this once streaming dataset supports proper resuming
    if resume:
        resume_t0 = time.perf_counter()
        for resume_iter in range(initial_iter):
            next(train_iterator)
            if resume_iter % 1000 == 0:
                fabric.print(f"Resuming dataset: {resume_iter} / {initial_iter}")
        fabric.barrier()
        fabric.print(
            f"Resuming data loader finished. Took {time.perf_counter() - resume_t0:.1f} seconds to reach iteration"
            f" {initial_iter}, epoch {train_iterator.epoch}."
        )

    running_loss = RunningMean(window=gradient_accumulation_iters, sync_on_compute=False).to(fabric.device)
    fabric.barrier()
    total_t0 = time.perf_counter()

    for train_data in train_iterator:
        if state["iter_num"] >= max_iters:
            break

        # determine and set the learning rate for this iteration
        lr = get_lr(state["iter_num"], warmup_iters, max_iters) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        state["iter_num"] += 1
        iter_t0 = time.perf_counter()

        input_ids = train_data[:, 0 : model.config.block_size].contiguous().long()
        targets = train_data[:, 1 : (model.config.block_size + 1)].contiguous().long()

        is_accumulating = state["iter_num"] % gradient_accumulation_iters != 0
        with fabric.no_backward_sync(model, enabled=is_accumulating):
            logits = model(input_ids)
            loss = chunked_cross_entropy(logits, targets)
            fabric.backward(loss / gradient_accumulation_iters)

        running_loss.update(loss.detach())

        if not is_accumulating:
            fabric.clip_gradients(model, optimizer, max_norm=grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            state["step_count"] += 1

        if state["iter_num"] % log_iter_interval == 0:
            loss = running_loss.compute().item()  # expensive device-to-host synchronization
            t1 = time.perf_counter()
            throughput.update(
                time=(t1 - total_t0),
                flops=(measured_flops * log_iter_interval),
                batches=state["iter_num"],
                samples=(state["iter_num"] * micro_batch_size),
                lengths=(state["iter_num"] * micro_batch_size * model.config.block_size),
            )
            metrics = {
                "loss": loss,
                "iter": state["iter_num"],
                "step": state["step_count"],
                "epoch": train_iterator.epoch,
                "iter_time": t1 - iter_t0,
                "remaining_time": (
                    (t1 - total_t0) / (state["iter_num"] - initial_iter) * (max_iters - state["iter_num"])
                ),
                "tokens": state["iter_num"] * micro_batch_size * model.config.block_size,
                "total_tokens": state["iter_num"] * micro_batch_size * model.config.block_size * fabric.world_size,
                "learning_rate": lr,
            }

            fabric.print(
                f"iter {metrics['iter']} | step {metrics['step']}: loss {metrics['loss']:.4f}, iter time:"
                f" {metrics['iter_time'] * 1000:.2f} ms{' (optimizer.step),' if not is_accumulating else ','}"
                f" remaining time: {metrics['remaining_time'] / 3600 / 24:.2f} days"
            )

            throughput_metrics = throughput.compute()
            metrics.update(throughput_metrics)
            fabric.log_dict(metrics, step=state["iter_num"])

        if val_dataloader is not None and not is_accumulating and state["step_count"] % eval_step_interval == 0:
            t0 = time.perf_counter()
            val_loss = validate(fabric, model, val_dataloader, max_iters=eval_iters)
            val_loss = val_loss.item()
            td = time.perf_counter() - t0

            fabric.print(f"iter {state['iter_num']}: val loss {val_loss:.4f}, val time: {td * 1000:.2f} ms")
            metrics = {"val_loss": val_loss, "val_ppl": math.exp(val_loss)}
            fabric.log_dict(metrics, step=state["iter_num"])
            fabric.barrier()

        if not is_accumulating and state["step_count"] % save_step_interval == 0:
            checkpoint_path = out_dir / f"step-{state['step_count']:08d}.pth"
            fabric.print(f"Saving checkpoint to {str(checkpoint_path)!r}")
            fabric.save(checkpoint_path, state)


@torch.no_grad()
def validate(fabric: L.Fabric, model: nn.Module, val_dataloader: DataLoader, max_iters: int) -> torch.Tensor:
    fabric.print("Validating ...")
    model.eval()

    losses = torch.zeros(max_iters, device=fabric.device)
    for k, val_data in enumerate(val_dataloader):
        if k >= max_iters:
            break
        input_ids = val_data[:, 0 : model.config.block_size].contiguous().long()
        targets = val_data[:, 1 : (model.config.block_size + 1)].contiguous().long()
        logits = model(input_ids)
        loss = chunked_cross_entropy(logits, targets)
        losses[k] = loss

    model.train()
    return losses.mean()


def create_dataloaders(batch_size: int, block_size: int) -> Tuple[DataLoader, DataLoader]:
    from lightning.data import StreamingDataset
    from lightning.data.streaming.item_loader import TokensLoader

    # Increase by one because we need the next word as well
    effective_block_size = block_size + 1

    train_datasets = [
        StreamingDataset(
            input_dir="data/slimpajama/train",
            item_loader=TokensLoader(block_size=effective_block_size),
            shuffle=True,
            drop_last=True,
        ),
        StreamingDataset(
            input_dir="data/starcoder",
            item_loader=TokensLoader(block_size=effective_block_size),
            shuffle=True,
            drop_last=True,
        ),
    ]

    # Mix SlimPajama data and Starcoder data with these proportions:
    weights = (0.693584, 0.306416)
    combined_dataset = CombinedDataset(datasets=train_datasets, seed=42, weights=weights)
    train_dataloader = DataLoader(
        combined_dataset, batch_size=batch_size, pin_memory=True, num_workers=8, drop_last=True
    )

    val_dataset = StreamingDataset(
        input_dir="data/slimpajama/val",
        item_loader=TokensLoader(block_size=effective_block_size),
        shuffle=True,
        # Consider setting to False, but we would lose some samples due to truncation when world size > 1
        drop_last=True,
    )
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=True, num_workers=8, drop_last=True)
    return train_dataloader, val_dataloader


# learning rate decay scheduler (cosine with linear warmup)
def get_lr(it: int, warmup_iters: int, max_iters: int) -> float:
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > max_iters, return min learning rate
    if it > max_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


def init_weights(module: nn.Module, n_layer: int, n_embd: int):
    # Follows GPT-NeoX: https://arxiv.org/abs/2204.06745
    if isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=math.sqrt(2.0 / 5 / n_embd))
    elif isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=math.sqrt(2.0 / 5 / n_embd))
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    for name, param in module.named_parameters():
        if name == "proj.weight" and isinstance(module, (LLaMAMLP, CausalSelfAttention)):
            nn.init.normal_(param, mean=0.0, std=(1 / math.sqrt(n_embd) / n_layer))


def choose_logger(logger_name: str, name: str, resume: Union[bool, Path], *args, **kwargs):
    if logger_name == "csv":
        return CSVLogger(root_dir=(out_dir / "logs"), name="csv", *args, **kwargs)
    if logger_name == "tensorboard":
        return TensorBoardLogger(root_dir=(out_dir / "logs"), name="tensorboard", *args, **kwargs)
    if logger_name == "wandb":
        return WandbLogger(project="tinyllama", name=name, resume=(resume is not False), *args, **kwargs)
    raise ValueError(f"`logger={logger_name}` is not a valid option.")


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    from jsonargparse import CLI
    from lightning.fabric.utilities.imports import _TORCH_GREATER_EQUAL_2_2

    if not _TORCH_GREATER_EQUAL_2_2:
        raise ImportError("The tinyllama.py training script requires PyTorch 2.2 (nightly) or higher to run.")

    CLI(setup)
