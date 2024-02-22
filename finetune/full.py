# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
import dataclasses
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import lightning as L
import torch
from lightning.fabric.loggers import CSVLogger
from lightning.fabric.strategies import FSDPStrategy
from torchmetrics import RunningMean

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from generate.base import generate
from lit_gpt.args import EvalArgs, IOArgs, TrainArgs
from lit_gpt.model import GPT, Block, Config
from lit_gpt.tokenizer import Tokenizer
from lit_gpt.utils import (
    CLI,
    check_valid_checkpoint_dir,
    chunked_cross_entropy,
    get_default_supported_precision,
    load_checkpoint,
    num_parameters,
)
from scripts.prepare_alpaca import generate_prompt


def setup(
    precision: Optional[str] = None,
    devices: int = 1,
    resume: Union[bool, Path] = False,
    io_args: IOArgs = IOArgs(
        train_data_dir=Path("data/alpaca"),
        val_data_dir=Path("data/alpaca"),
        checkpoint_dir=Path("checkpoints/stabilityai/stablelm-base-alpha-3b"),
        out_dir=Path("out/full/alpaca"),
    ),
    train_args: TrainArgs = TrainArgs(
        save_interval=1000,
        log_interval=1,
        global_batch_size=64,
        micro_batch_size=1,
        lr_warmup_steps=100,
        epochs=5,
        epoch_size=50000,
        learning_rate=3e-3,
        max_seq_length=None,
    ),
    eval_args: EvalArgs = EvalArgs(interval=600, max_new_tokens=100, max_iters=100),
) -> None:
    print(locals())
    precision = precision or get_default_supported_precision(training=True)

    if devices > 1:
        strategy = FSDPStrategy(
            auto_wrap_policy={Block},
            activation_checkpointing_policy={Block},
            state_dict_type="full",
            limit_all_gathers=True,
            cpu_offload=False,
        )
    else:
        strategy = "auto"

    logger = CSVLogger(io_args.out_dir.parent, io_args.out_dir.name, flush_logs_every_n_steps=train_args.log_interval)
    fabric = L.Fabric(devices=devices, strategy=strategy, precision=precision, loggers=logger)
    fabric.launch(
        main, devices, resume, Config.from_name(name=io_args.checkpoint_dir.name), io_args, train_args, eval_args
    )


def main(
    fabric: L.Fabric,
    devices: int,
    resume: Union[bool, Path],
    config: Config,
    io_args: IOArgs,
    train_args: TrainArgs,
    eval_args: EvalArgs,
) -> None:
    validate_args(io_args, train_args, eval_args)

    steps_per_epoch = train_args.epoch_size // devices // train_args.batch_size(devices)
    lr_max_steps = train_args.epochs * steps_per_epoch

    check_valid_checkpoint_dir(io_args.checkpoint_dir)

    fabric.seed_everything(1337)  # same seed for every process to init model (FSDP)

    if fabric.global_rank == 0:
        os.makedirs(io_args.out_dir, exist_ok=True)

    train_data = torch.load(io_args.train_data_dir / "train.pt")
    val_data = torch.load(io_args.val_data_dir / "test.pt")

    checkpoint_path = io_args.checkpoint_dir / "lit_model.pth"
    fabric.print(f"Loading model {str(checkpoint_path)!r} with {config.__dict__}")
    with fabric.init_module(empty_init=(devices > 1)):
        model = GPT(config)

    fabric.print(f"Number of trainable parameters: {num_parameters(model, requires_grad=True):,}")

    model = fabric.setup(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_args.learning_rate,
        weight_decay=train_args.weight_decay,
        betas=(train_args.beta1, train_args.beta2),
    )
    optimizer = fabric.setup_optimizers(optimizer)
    scheduler = get_lr_scheduler(optimizer, warmup_steps=train_args.lr_warmup_steps, max_steps=lr_max_steps)
    state = {"model": model, "optimizer": optimizer, "scheduler": scheduler, "iter_num": 0, "step_count": 0}

    if resume is True:
        resume = max(io_args.out_dir.glob("*.pth"), key=(lambda p: int(p.name.split("-")[1])))
    if resume:
        fabric.print(f"Resuming training from {resume}")
        fabric.load(resume, state)
    else:
        load_checkpoint(fabric, state["model"], checkpoint_path)

    fabric.seed_everything(1337 + fabric.global_rank)

    train_time = time.perf_counter()
    train(fabric, state, train_data, val_data, devices, resume, io_args, train_args, eval_args)
    fabric.print(f"Training time: {(time.perf_counter()-train_time):.2f}s")
    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")

    # Save the final checkpoint at the end of training
    fabric.save(io_args.out_dir / "lit_model_finetuned.pth", {"model": state["model"]})


def train(
    fabric: L.Fabric,
    state: Dict,
    train_data: List[Dict],
    val_data: List[Dict],
    devices: int,
    resume: Union[bool, Path],
    io_args: IOArgs,
    train_args: TrainArgs,
    eval_args: EvalArgs,
) -> None:
    model = state["model"]
    optimizer = state["optimizer"]
    scheduler = state["scheduler"]
    tokenizer = Tokenizer(io_args.checkpoint_dir)
    longest_seq_length, longest_seq_ix = get_longest_seq_length(train_data)
    model.max_seq_length = min(longest_seq_length, train_args.max_seq_length or float("inf"))
    fabric.print(
        f"The longest sequence length in the train data is {longest_seq_length}, the model's maximum sequence length is"
        f" {model.max_seq_length} and context length is {model.config.block_size}"
    )

    validate(
        fabric, model, val_data, tokenizer, dataclasses.replace(eval_args, max_iters=2), train_args
    )  # sanity check
    initial_iter = state["iter_num"]

    # resume data loader state by fast-forwarding through all seen batches
    if resume:
        resume_t0 = time.perf_counter()
        for resume_iter in range(initial_iter):
            get_batch(fabric, train_data, None)
            if resume_iter % 1000 == 0:
                fabric.print(f"Resuming dataset: {resume_iter} / {initial_iter}")
        fabric.barrier()
        fabric.print(
            f"Resuming data loader finished. Took {time.perf_counter() - resume_t0:.1f} seconds to reach iteration"
            f" {initial_iter}."
        )

    running_loss = RunningMean(window=train_args.gradient_accumulation_iters(devices), sync_on_compute=False).to(
        fabric.device
    )
    fabric.barrier()

    for state["iter_num"] in range(state["iter_num"] + 1, train_args.max_iters(devices) + 1):
        iter_t0 = time.perf_counter()

        input_ids, targets = get_batch(
            fabric,
            train_data,
            train_args.micro_batch_size,
            train_args.max_seq_length,
            longest_seq_ix if state["iter_num"] == 1 else None,
        )

        is_accumulating = state["iter_num"] % train_args.gradient_accumulation_iters(devices) != 0
        with fabric.no_backward_sync(model, enabled=is_accumulating):
            logits = model(input_ids)
            # shift the targets such that output n predicts token n+1
            loss = chunked_cross_entropy(logits[..., :-1, :], targets[..., 1:])
            fabric.backward(loss / train_args.gradient_accumulation_iters(devices))

        running_loss.update(loss.detach())

        if not is_accumulating:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            state["step_count"] += 1

        if state["iter_num"] % train_args.log_interval == 0:
            loss = running_loss.compute().item()  # expensive device-to-host synchronization
            t1 = time.perf_counter()
            metrics = {
                "loss": loss,
                "iter": state["iter_num"],
                "step": state["step_count"],
                "iter_time": t1 - iter_t0,
                "tokens": state["iter_num"] * train_args.micro_batch_size * model.config.block_size,
                "total_tokens": (
                    state["iter_num"] * train_args.micro_batch_size * model.config.block_size * fabric.world_size
                ),
                # TODO: log learning rate
            }
            fabric.print(
                f"iter {metrics['iter']} | step {metrics['step']}: loss {metrics['loss']:.4f}, iter time:"
                f" {metrics['iter_time'] * 1000:.2f} ms{' (optimizer.step)' if not is_accumulating else ''}"
            )
            fabric.log_dict(metrics, step=state["iter_num"])

        if not is_accumulating and state["step_count"] % eval_args.interval == 0:
            t0 = time.perf_counter()
            val_loss = validate(fabric, model, val_data, tokenizer, eval_args, train_args)
            t1 = time.perf_counter() - t0
            fabric.print(f"iter {state['iter_num']}: val loss {val_loss.item():.4f}, val time: {t1 * 1000:.2f} ms")
            metrics = {"val_loss": val_loss, "val_ppl": math.exp(val_loss)}
            fabric.log_dict(metrics, step=state["iter_num"])
            fabric.barrier()
        if not is_accumulating and state["step_count"] % train_args.save_interval == 0:
            checkpoint_path = io_args.out_dir / f"step-{state['step_count']:06d}.pth"
            fabric.print(f"Saving checkpoint to {str(checkpoint_path)!r}")
            fabric.save(checkpoint_path, state)


# FSDP has issues with `inference_mode`
@torch.no_grad()
def validate(
    fabric: L.Fabric, model: GPT, val_data: List[Dict], tokenizer: Tokenizer, eval_args: EvalArgs, train_args: TrainArgs
) -> torch.Tensor:
    fabric.print("Validating ...")
    model.eval()
    losses = torch.zeros(eval_args.max_iters)
    for k in range(eval_args.max_iters):
        input_ids, targets = get_batch(fabric, val_data, train_args.micro_batch_size, train_args.max_seq_length)
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
        model,
        encoded,
        max_returned_tokens=len(encoded) + eval_args.max_new_tokens,
        temperature=0.8,
        eos_id=tokenizer.eos_id,
    )
    model.clear_kv_cache()
    output = tokenizer.decode(output)
    fabric.print(output)

    model.train()
    return val_loss


def get_batch(
    fabric: L.Fabric,
    data: List[Dict],
    micro_batch_size: int,
    max_seq_length: Optional[int],
    longest_seq_ix: Optional[int] = None,
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


def validate_args(io_args: IOArgs, train_args: TrainArgs, eval_args: EvalArgs) -> None:
    unsupported = [(train_args, ["max_tokens", "max_norm"])]
    for args, names in unsupported:
        for name in names:
            if getattr(args, name) is not None:
                raise ValueError(f"{__file__} doesn't support the {name!r} argument. This is set in {args}")
    required = [
        (io_args, ["checkpoint_dir", "train_data_dir", "val_data_dir"]),
        (train_args, ["epoch_size", "epochs"]),
        (eval_args, ["max_new_tokens"]),
    ]
    for args, names in required:
        for name in names:
            if getattr(args, name) is None:
                raise ValueError(f"{__file__} requires the {name!r} argument. This is set in {args}")


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    CLI(setup)
