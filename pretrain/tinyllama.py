"""
This script is adapted from TinyLlama:
https://github.com/jzhang38/TinyLlama/blob/main/pretrain/tinyllama.py


TODO LIST:
- [ ] check that seed is correctly set and each rank sees a partition of the data
- [x] implement init-weights
- [ ] install torch nightly
- [ ] use fake dataset to compare batches/sec numbers
- [ ] determine global batch size
- [ ] add torch.compile
- [ ] verify script can be resumed
- [ ] resolve TODOs in script below
"""
import glob
import math
import sys
import time
from pathlib import Path
from typing import Optional, Tuple, Union
import math
import lightning as L
import torch
import torch.nn as nn
from lightning.fabric.strategies import FSDPStrategy
from lightning.data import StreamingDataset, StreamingDataLoader
from lightning.data.streaming.item_loader import TokensLoader
from torch.utils.data import DataLoader
from functools import partial

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_gpt.model import GPT, Block, Config, CausalSelfAttention, LLaMAMLP
from lit_gpt.packed_dataset import CombinedDataset, PackedDataset
from lit_gpt.speed_monitor import SpeedMonitorFabric as SpeedMonitor
from lit_gpt.speed_monitor import estimate_flops, measure_flops
from lit_gpt.utils import chunked_cross_entropy, num_parameters
from lightning.pytorch.loggers import WandbLogger
import random

model_name = "TinyLlama-1.1B-intermediate-step-480k-1T"
name = "tinyllama_1b"
out_dir = Path("out") / name

# Hyperparameters
devices = 4  # TODO: undo: 8 needed

global_batch_size = 32  # TODO: should be 512?
learning_rate = 4e-4
micro_batch_size = 1  # TODO: should be 8
max_step = 715256 * 2
warmup_steps = 2000
log_step_interval = 10
eval_iters = 100
save_step_interval = 5000
eval_step_interval = 5000

weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = True
min_lr = 4e-5

batch_size = global_batch_size // devices
gradient_accumulation_steps = batch_size // micro_batch_size
assert gradient_accumulation_steps > 0
warmup_iters = warmup_steps * gradient_accumulation_steps

max_iters = max_step * gradient_accumulation_steps
lr_decay_iters = max_iters
log_iter_interval = log_step_interval * gradient_accumulation_steps


hparams = {k: v for k, v in locals().items() if isinstance(v, (int, float, str)) and not k.startswith("_")}
logger = WandbLogger(project="tinyllama")


def setup(precision: str = "bf16-mixed", resume: Union[bool, Path] = False):
    if devices > 1:
        strategy = FSDPStrategy(
            auto_wrap_policy={Block},
            activation_checkpointing_policy=None,
            state_dict_type="full",
            limit_all_gathers=True,
            cpu_offload=False,
        )
    else:
        strategy = "auto"

    fabric = L.Fabric(devices=devices, strategy=strategy, precision=precision, loggers=[logger])
    fabric.launch()

    fabric.print(hparams)
    if fabric.global_rank == 0:
        logger.experiment.config.update(hparams)
    
    main(fabric, resume)


def main(fabric, resume):
    monitor = SpeedMonitor(fabric, window_size=5, time_unit="seconds")

    if fabric.global_rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    config = Config.from_name(model_name)

    train_dataloader, val_dataloader = create_dataloaders(
        batch_size=micro_batch_size,
        block_size=config.block_size,
        fabric=fabric,
    )
    if val_dataloader is None:
        train_dataloader = fabric.setup_dataloaders(train_dataloader)
    else:
        train_dataloader, val_dataloader = fabric.setup_dataloaders(train_dataloader, val_dataloader)

    fabric.seed_everything(3407)  # same seed for every process to init model (FSDP)

    fabric.print(f"Loading model with {config.__dict__}")
    t0 = time.perf_counter()
    with fabric.init_module(empty_init=False):
        model = GPT(config)
        model.apply(partial(init_weights, n_layer=config.n_layer))

    fabric.print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.")
    fabric.print(f"Total parameters {num_parameters(model):,}")

    # model = torch.compile(model, fullgraph=True)
    model = fabric.setup(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2))
    optimizer = fabric.setup_optimizers(optimizer)

    state = {"model": model, "optimizer": optimizer, "hparams": hparams, "iter_num": 0, "step_count": 0}

    if resume is True:
        resume = max(out_dir.glob("*.pth"), key=(lambda p: int(p.name.split("-")[1])))
    if resume :
        fabric.print(f"Resuming training from {resume}")
        fabric.load(resume, state)

    train_time = time.perf_counter()
    train(fabric, state, train_dataloader, val_dataloader, monitor, resume)
    fabric.print(f"Training time: {(time.perf_counter()-train_time):.2f}s")
    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")


def train(fabric, state, train_dataloader, val_dataloader, monitor, resume):
    model = state["model"]
    optimizer = state["optimizer"]

    # if val_dataloader is not None:
    #     validate(fabric, model, val_dataloader)  # sanity check

    with torch.device("meta"):
        meta_model = GPT(model.config)
        # "estimated" is not as precise as "measured". Estimated is optimistic but widely used in the wild.
        # When comparing MFU or FLOP numbers with other projects that use estimated FLOPs,
        # consider passing `SpeedMonitor(flops_per_batch=estimated_flops)` instead
        estimated_flops = estimate_flops(meta_model) * micro_batch_size
        fabric.print(f"Estimated TFLOPs: {estimated_flops * fabric.world_size / 1e12:.2f}")
        x = torch.randint(0, 1, (micro_batch_size, model.config.block_size))
        measured_flops = measure_flops(meta_model, x)
        fabric.print(f"Measured TFLOPs: {measured_flops * fabric.world_size / 1e12:.2f}")
        del meta_model, x

    total_lengths = 0
    total_t0 = time.perf_counter()
    initial_iter = state["iter_num"]
    curr_iter = 0

    for train_data in train_dataloader:
        # resume loader state. This is not elegant but it works. Should rewrite it in the future.
        if resume:
            if curr_iter < initial_iter:
                curr_iter += 1
                continue
            else:
                resume = False
                curr_iter = -1
                fabric.barrier()
                fabric.print("resume finished, taken {} seconds".format(time.perf_counter() - total_t0))
        if state["iter_num"] >= max_iters:
            break

        # determine and set the learning rate for this iteration
        lr = get_lr(state["iter_num"]) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        iter_t0 = time.perf_counter()

        input_ids = train_data[:, 0:model.config.block_size].contiguous().long()
        targets = train_data[:, 1:(model.config.block_size + 1)].contiguous().long()

        is_accumulating = (state["iter_num"] + 1) % gradient_accumulation_steps != 0
        with fabric.no_backward_sync(model, enabled=is_accumulating):
            logits = model(input_ids)
            loss = chunked_cross_entropy(logits, targets, chunk_size=0)
            fabric.backward(loss / gradient_accumulation_steps)

        if not is_accumulating:
            fabric.clip_gradients(model, optimizer, max_norm=grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            state["step_count"] += 1

        state["iter_num"] += 1

        total_lengths += input_ids.size(1)
        loss = loss.item()
        t1 = time.perf_counter()

        metrics = {
            "loss": loss,
            "iter": state['iter_num'],
            "step": state['step_count'],
            "iter_time": (t1 - iter_t0),
            "remaining_time": (t1 - total_t0) / (state['iter_num'] - initial_iter) * (max_iters - state['iter_num']),
        }

        fabric.print(
            f"iter {metrics['iter']} step {metrics['step']}: loss {metrics['loss']:.4f}, iter time:"
            f" {metrics['iter_time'] * 1000:.2f} ms{' (optimizer.step)' if not is_accumulating else ''}"
            f" remaining time: {metrics['remaining_time'] / 3600 / 24:.2f} days"
        )
        if state["iter_num"] % log_iter_interval == 0:
            fabric.log_dict(metrics)
 
        monitor.on_train_batch_end(
            train_elapsed=(t1 - total_t0),
            # this assumes that device FLOPs are the same and that all devices have the same batch size
            world_size=fabric.world_size,
            flops_per_batch=measured_flops,
            samples=((state["iter_num"] + 1) * micro_batch_size),
            lengths=total_lengths,
        )

        if val_dataloader is not None and not is_accumulating and state["step_count"] % eval_step_interval == 0:
            t0 = time.perf_counter()
            val_loss = validate(fabric, model, val_dataloader)
            t1 = time.perf_counter() - t0

            fabric.print(f"step {state['iter_num']}: val loss {val_loss:.4f}, val time: {t1 * 1000:.2f}ms")
            fabric.log_dict({"metric/val_loss": val_loss.item(), "total_tokens":  model.config.block_size * (state["iter_num"] + 1) * micro_batch_size * fabric.world_size},state["step_count"])
            fabric.log_dict({"metric/val_ppl": math.exp(val_loss.item()), "total_tokens":  model.config.block_size * (state["iter_num"] + 1) * micro_batch_size * fabric.world_size},state["step_count"])
            fabric.barrier()

        if not is_accumulating and state["step_count"] % save_step_interval == 0:
            checkpoint_path = out_dir / f"iter-{state['iter_num']:06d}-ckpt.pth"
            fabric.print(f"Saving checkpoint to {str(checkpoint_path)!r}")
            fabric.save(checkpoint_path, state)


@torch.no_grad()
def validate(fabric: L.Fabric, model: nn.Module, val_dataloader: DataLoader) -> torch.Tensor:
    fabric.print("Validating ...")
    model.eval()

    losses = torch.zeros(eval_iters, device=fabric.device)
    for k, val_data in enumerate(val_dataloader):
        if k >= eval_iters:
            break
        input_ids = val_data[:, 0:model.config.block_size].contiguous().long()
        targets = val_data[:, 1:(model.config.block_size + 1)].contiguous().long()
        logits = model(input_ids)
        loss = chunked_cross_entropy(logits, targets, chunk_size=0)
        losses[k] = loss.item()

    out = losses.mean()
    model.train()
    return out


def create_dataloaders(
    batch_size: int,
    block_size: int,
    fabric,
) -> Tuple[DataLoader, DataLoader]:
    
    # if True:  # TODO: undo
    #     return DataLoader(FakeDataset(), batch_size=batch_size, shuffle=False, pin_memory=True)

    # Increase by one because we need the next word as well
    effective_block_size = block_size + 1

    train_datasets = [
        # TODO: change to slimpajama/train and starcoder
        StreamingDataset(
            name="slimpajama/val", 
            version="latest", 
            item_loader=TokensLoader(block_size=effective_block_size), 
            shuffle="full",
        ),
        StreamingDataset(
            name="slimpajama/test", 
            version="latest", 
            item_loader=TokensLoader(block_size=effective_block_size), 
            shuffle="full",
        ),
    ]

    # Mix SlimPajama data and Starcoder data with these proportions:
    weights = (0.693584, 0.306416)
    weights = [w / sum(weights) for w in weights]

    combined_dataset = CombinedDataset(datasets=train_datasets, seed=42, weights=weights)
    train_dataloader = DataLoader(combined_dataset, batch_size=batch_size, pin_memory=True)
    
    val_dataset = StreamingDataset(
        name="slimpajama/val", 
        version="latest", 
        item_loader=TokensLoader(block_size=effective_block_size), 
        # Consider setting to False, but we would lose some samples due to truncation when world size > 1
        shuffle="full",
    )
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=True)
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


def init_weights(module: nn.Module, n_layer: int):
    # Follows GPT-NeoX: https://arxiv.org/abs/2204.06745
    if isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=math.sqrt(2.0 / 5 / module.weight.size(1)))
    elif isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=math.sqrt(2.0 / 5 / module.weight.size(1)))
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    for name, param in module.named_parameters():
        # TODO: Do we need SwiGLU?
        if (name == "proj.weight" and isinstance(module, LLaMAMLP)): # or (name == "w3.weight" and isinstance(module, SwiGLU)):
            nn.init.normal_(param, mean=0.0, std=(1 / math.sqrt(param.shape[-1]) / n_layer))


# TODO: remove
class FakeDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 1000
    def __getitem__(self, index):
        return torch.randint(0, 10, size=(2049,))


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    from jsonargparse import CLI

    CLI(setup)
