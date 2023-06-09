import json
import math
import sys
import time
from pathlib import Path
from types import MethodType
from typing import Tuple

import lightning as L
import numpy as np
import torch

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_parrot import Config, Parrot, Tokenizer
from lit_parrot.utils import check_valid_checkpoint_dir
from generate.base import generate
from lit_parrot.speed_monitor import SpeedMonitor, total_flops
from lightning.fabric.loggers import CSVLogger


devices = 1
precision = "16-mixed"

out_dir = Path("out") / "openwebtext"
data_dir = Path("data") / "openwebtext"
name = "train-openwebtext"
eval_interval = 200
save_interval = 400
eval_iters = 100
log_interval = 1

# Hyperparameters
learning_rate = 6e-4
batch_size = 64 / devices
micro_batch_size = 4  # FIXME
gradient_accumulation_steps = batch_size // micro_batch_size
assert gradient_accumulation_steps > 0
epoch_size = 50000  # train dataset size
# num_epochs = 5
max_iters = 600000  # num_epochs * (epoch_size // micro_batch_size) // devices
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = True
warmup_iters = 2000
lr_decay_iters = max_iters
min_lr = 6e-5
seed = 1338
fake_data = True  # FIXME

hparams = {k: v for k, v in locals().items() if isinstance(v, (int, float, str)) and not k.startswith("_")}

logger = CSVLogger(out_dir)


def merge_by(dicts, key):
    from collections import defaultdict

    out = defaultdict(dict)
    for d in dicts:
        if key in d:
            out[d[key]].update(d)
    return [v for _, v in sorted(out.items())]


def save(self) -> None:
    """Overriden to merge CSV by the step number."""
    import csv

    if not self.metrics:
        return
    metrics = merge_by(self.metrics, "step")
    keys = sorted({k for m in metrics for k in m})
    with self._fs.open(self.metrics_file_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(metrics)


logger.experiment.save = MethodType(save, logger.experiment)

speed_monitor = SpeedMonitor(logger, precision, window_size=50, time_unit="seconds")


def main(checkpoint_dir: Path = Path(f"checkpoints/EleutherAI/pythia-1b")) -> None:
    fabric = L.Fabric(devices=devices, precision=precision)
    fabric.launch()

    fabric.print(hparams)

    check_valid_checkpoint_dir(checkpoint_dir)

    with open(checkpoint_dir / "lit_config.json") as fp:
        config = Config(**json.load(fp))

    fabric.seed_everything(seed + fabric.global_rank)

    if fabric.global_rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    train_data, val_data = load_datasets(data_dir)

    fabric.print(f"Loading model with {config.__dict__}", file=sys.stderr)
    t0 = time.time()
    with fabric.init_module():
        model = Parrot(config)
    fabric.print(f"Time to instantiate model: {time.time() - t0:.02f} seconds.", file=sys.stderr)

    num_total_params = sum(p.numel() for p in model.parameters())
    fabric.print(f"Total parameters {num_total_params}")

    tokenizer = Tokenizer(checkpoint_dir / "tokenizer.json", checkpoint_dir / "tokenizer_config.json")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2))
    model, optimizer = fabric.setup(model, optimizer)

    train(fabric, model, tokenizer, optimizer, train_data, val_data)


def train(
    fabric: L.Fabric,
    model: Parrot,
    tokenizer: Tokenizer,
    optimizer: torch.optim.Optimizer,
    train_data: np.ndarray,
    val_data: np.ndarray,
) -> None:
    validate(fabric, model, tokenizer, val_data)  # sanity check

    flops = total_flops(model)
    fabric.print(
        f"TFLOPs per sequence {flops / 1e12:.2f}, total TFLOPs"
        f" {flops * micro_batch_size * fabric.world_size / 1e12:.2f}"
    )
    step_count = 0
    total_t0 = time.time()

    for iter_num in range(max_iters):
        # determine and set the learning rate for this iteration
        lr = get_lr(iter_num) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        did_step = False
        iter_t0 = time.time()

        input_ids, targets = get_batch(fabric, train_data, model.config.block_size)
        with fabric.no_backward_sync(model, enabled=((iter_num + 1) % gradient_accumulation_steps != 0)):
            logits = model(input_ids)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
            fabric.backward(loss / gradient_accumulation_steps)

        if (iter_num + 1) % gradient_accumulation_steps == 0:
            fabric.clip_gradients(model, optimizer, max_norm=grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            step_count += 1
            did_step = True

        t1 = time.time()
        speed_monitor.batch_end(
            (iter_num + 1) * micro_batch_size,
            t1 - total_t0,
            fabric.world_size,
            flops_per_batch=flops * micro_batch_size,
            max_seq_length=model.config.block_size,
        )
        if iter_num % log_interval == 0:
            fabric.print(
                f"iter {iter_num} step {step_count}: loss {loss.item():.4f}, train time:"
                f" {(t1 - iter_t0) * 1000:.2f}ms{' (optimizer.step)' if did_step else ''}"
            )

        if did_step and step_count % eval_interval == 0:
            t0 = time.time()
            val_loss = validate(fabric, model, tokenizer, val_data)
            t1 = time.time() - t0
            speed_monitor.eval_end(t1)
            fabric.print(f"step {iter_num}: val loss {val_loss:.4f}, val time: {t1 * 1000:.2f}ms")
            fabric.barrier()
        if did_step and step_count % save_interval == 0:
            checkpoint_path = out_dir / f"{name}.pth"
            fabric.print(f"Saving checkpoint to {str(checkpoint_path)!r}")
            fabric.save(checkpoint_path, {"model": model})


@torch.no_grad()
def validate(
    fabric: L.Fabric, model: torch.nn.Module, tokenizer: Tokenizer, val_data: np.ndarray, num_samples: int = 3
) -> torch.Tensor:
    fabric.print("Validating ...")
    model.eval()

    losses = torch.zeros(eval_iters, device=fabric.device)
    for k in range(eval_iters):
        input_ids, targets = get_batch(fabric, val_data, model.config.block_size)
        logits = model(input_ids)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        losses[k] = loss.item()
    out = losses.mean()

    if not fake_data:  # not really useful to see results with fake data
        input_ids, _ = get_batch(fabric, val_data, model.config.block_size)
        input_ids = input_ids[:, :num_samples]
        for ids in input_ids:
            prompt = tokenizer.decode(ids)
            y = generate(
                model,
                ids,
                max_returned_tokens=ids.size(0) + 100,
                max_seq_length=model.config.block_size,
                temperature=0.8,
                top_k=50,
            )
            fabric.print(f"Prompt: {prompt!r}: {tokenizer.decode(y)!r}")

    model.train()
    return out


def get_batch(fabric: L.Fabric, data: np.ndarray, block_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    if fake_data:
        x = torch.randint(0, 100, (micro_batch_size, block_size), device=fabric.device)
        y = torch.randint_like(x, 0, 100)
        return x, y
    ix = torch.randint(len(data) - block_size, (micro_batch_size,))
    x = torch.stack([torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64)) for i in ix])
    x, y = fabric.to_device((x.pin_memory(), y.pin_memory()))
    return x, y


def load_datasets(data_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    if fake_data:
        return None, None  # type: ignore
    train_data = np.memmap(str(data_dir / "train.bin"), dtype=np.uint16, mode="r")
    val_data = np.memmap(str(data_dir / "val.bin"), dtype=np.uint16, mode="r")
    return train_data, val_data


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
    torch.set_float32_matmul_precision("high")

    from jsonargparse import CLI

    CLI(main)
