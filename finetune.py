"""
Instruction-tuning with LoRA on the Alpaca dataset.
"""
import os
import time

import lightning as L
import numpy as np
import torch
from lightning.fabric.strategies import DeepSpeedStrategy

from generate import generate
from lit_llama.lora import mark_only_lora_as_trainable, with_lora
from lit_llama.model import LLaMA, LLaMAConfig
from lit_llama.tokenizer import Tokenizer
from scripts.prepare_alpaca import generate_prompt

out_dir = "out"
eval_interval = 100
eval_iters = 100
log_interval = 1

# Hyperparameters
learning_rate = 2e-5
batch_size = 32
micro_batch_size = 4
gradient_accumulation_steps = batch_size // micro_batch_size

# TODO: Limit to 3 epochs
max_iters = 100000000 #  50000 * 3 // 4 // batch_size
weight_decay = 0.0
block_size = 256

# TODO: LR scheduling
warmup_steps = 100

ds_config = {
    "gradient_accumulation_steps": gradient_accumulation_steps,
    "train_micro_batch_size_per_gpu": micro_batch_size,
    "bf16": {
        "enabled": True
    },
    "gradient_clipping": 1,
    "zero_optimization": {
        "stage": 2,
        "offload_param": {
            "device": "none"
        },
            "offload_optimizer": {
            "device": "none"
        },
        "allgather_partitions": True,
        "allgather_bucket_size": 5e8,
        "contiguous_gradients": True
    }
}


def main() -> None:
    fabric = L.Fabric(
        accelerator="cuda", 
        devices=4,
        strategy=DeepSpeedStrategy(config=ds_config)
    )
    fabric.launch()
    fabric.seed_everything(1337 + fabric.global_rank)

    if fabric.global_rank == 0:
        os.makedirs(out_dir, exist_ok=True)

    train_data, val_data = load_datasets()

    config = LLaMAConfig.from_name("7B")
    config.block_size = block_size

    with fabric.device, with_lora(r=8, alpha=32, dropout=0.1, enabled=True):
        model = LLaMA(config)

    checkpoint = torch.load("checkpoints/lit-llama/7B/state_dict.pth")
    
    # strict=False because missing keys due to lora weights not contained in checkpoint state
    model.load_state_dict(checkpoint, strict=False) 
    mark_only_lora_as_trainable(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    model, optimizer = fabric.setup(model, optimizer)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_iters, last_epoch=-1)

    train(fabric, model, optimizer, train_data, val_data)


def train(
    fabric: L.Fabric,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_data: np.ndarray,
    val_data: np.ndarray,
) -> None:
    """The training loop.

    Loosely based on the nanoGPT implementation: https://github.com/karpathy/nanoGPT.
    """
    for iter_num in range(max_iters):

        # evaluate the loss on train/val sets and write checkpoints
        if iter_num % eval_interval == 0:
            val_loss = validate(fabric, model, val_data)
            fabric.print(f"step {iter_num}: val loss {val_loss:.4f}")
            # TODO: Save with Fabric
            # print(f"Saving checkpoint to {out_dir}")
            # checkpoint = {"model": model, "optimizer": optimizer, "iter": iter, "val_loss": val_loss}
            #fabric.save(os.path.join(out_dir, f"iter-{iter_num:06d}-ckpt.pt"), checkpoint)
            fabric.barrier()
        

        t0 = time.time()

        input_ids, targets = get_batch(fabric, train_data)

        logits = model(input_ids)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        fabric.backward(loss)

        if (iter_num + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            # scheduler.step()
            optimizer.zero_grad()

        dt = time.time() - t0
        if iter_num % log_interval == 0:
            fabric.print(f"iter {iter_num}: loss {loss.item():.4f}, time: {dt*1000:.2f}ms")


def generate_response(model, instruction):
    tokenizer = Tokenizer("checkpoints/lit-llama/tokenizer.model")
    sample = {"instruction": instruction, "input": ""}
    prompt = generate_prompt(sample)
    encoded = tokenizer.encode(prompt, bos=True, eos=True)
    encoded = encoded[None, :]  # add batch dimension
    encoded = encoded.to(model.device)

    output = generate(
        model,
        idx=encoded,
        max_seq_length=block_size,
        max_new_tokens=100,
    )
    output = tokenizer.decode(output[0].cpu())
    return output.split("### Response:")[1].strip()


@torch.no_grad()
def validate(fabric: L.Fabric, model: torch.nn.Module, val_data: np.ndarray) -> torch.Tensor:
    fabric.print("Validating ...")
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        input_ids, targets = get_batch(fabric, val_data)
        logits = model(input_ids)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        losses[k] = loss.item()
    out = losses.mean()

    # produce an example:
    instruction = "Recommend a movie for me to watch during the weekend and explain the reason."
    fabric.print(instruction)
    fabric.print(generate_response(model, instruction))

    model.train()
    return out


def get_batch(fabric: L.Fabric, data: list, pad_id: int = 0):
    ix = torch.randint(len(data), (micro_batch_size,))

    def pad(x):
        # TODO: optimize this to pad to the next multiple of 8 or so?
        n = block_size - len(x)
        return torch.cat((x, torch.full((n,), pad_id, dtype=x.dtype)))

    def shift_right(x):
        return x[1:]

    x = torch.stack([pad(data[i]["input_ids"]) for i in ix]).type(torch.int64)
    y = torch.stack([pad(shift_right(data[i]["labels"])) for i in ix]).type(torch.int64)
    x, y = fabric.to_device((x.pin_memory(), y.pin_memory()))
    return x, y


def load_datasets(data_dir: str = "data/alpaca"):
    train_data = torch.load(os.path.join(data_dir, "train.pt"))
    val_data = torch.load(os.path.join(data_dir, "test.pt"))
    return train_data, val_data


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
