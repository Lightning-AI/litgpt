"""
Instruction-tuning with LLaMA-Adapter on the Alpaca dataset following the paper

LLaMA-Adapter: Efficient Fine-tuning of Language Models with Zero-init Attention
https://arxiv.org/abs/2303.16199

Note: If you run into a CUDA error "Expected is_sm80 to be true, but got false", install
the PyTorch nightly version for a fix (see https://github.com/Lightning-AI/lit-llama/issues/101).
"""
import os
import time

import lightning as L
import numpy as np
import torch

from generate import generate
from lit_llama.adapter import LLaMA, LLaMAConfig, mark_only_adapter_as_trainable, adapter_state_dict
from lit_llama.tokenizer import Tokenizer
from lit_llama.utils import EmptyInitOnDevice
from scripts.prepare_alpaca import generate_prompt


out_dir = "out/adapter/"
eval_interval = 40
save_interval = 200
eval_iters = 100
log_interval = 1

# Hyperparameters
learning_rate = 9e-3
batch_size = 64
micro_batch_size = 4
gradient_accumulation_steps = batch_size // micro_batch_size
epoch_size = 50000  # train dataset size
num_epochs = 100
max_iters = epoch_size * num_epochs // micro_batch_size  # 5 epochs
weight_decay = 0.02
block_size = 256
warmup_steps = epoch_size * 2 // micro_batch_size  # 2 epochs


def main():
    fabric = L.Fabric(accelerator="cuda", devices=1, precision="bf16-mixed")
    fabric.launch()
    fabric.seed_everything(1337 + fabric.global_rank)

    if fabric.global_rank == 0:
        os.makedirs(out_dir, exist_ok=True)

    train_data, val_data = load_datasets()

    config = LLaMAConfig()
    config.block_size = block_size

    checkpoint = torch.load("checkpoints/lit-llama/7B/state_dict.pth")

    with EmptyInitOnDevice(device=fabric.device, dtype=torch.bfloat16):
        model = LLaMA(config)
        # strict=False because missing keys due to adapter weights not containted in state dict
        model.load_state_dict(checkpoint, strict=False)
    
    mark_only_adapter_as_trainable(model)

    num_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print(f"Number of trainable parameters: {num_params}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    model, optimizer = fabric.setup(model, optimizer)
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
    step_count = 0

    for iter_num in range(max_iters):

        if step_count <= warmup_steps:
            # linear warmup
            lr = learning_rate * step_count / warmup_steps
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        t0 = time.time()

        input_ids, targets = get_batch(fabric, train_data)
        logits = model(input_ids)
        loss = loss_fn(logits, targets)
        with fabric.no_backward_sync(model, enabled=((iter_num + 1) % gradient_accumulation_steps != 0)):
            fabric.backward(loss / gradient_accumulation_steps)

        # fabric.clip_gradients(model, optimizer, clip_val=1.0)

        if (iter_num + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            step_count += 1
                
            if step_count % eval_interval == 0:
                val_loss = validate(fabric, model, val_data)
                fabric.print(f"step {iter_num}: val loss {val_loss:.4f}")
                fabric.barrier()

            if step_count % save_interval == 0:
                print(f"Saving adapter weights to {out_dir}")
                
                # only save the adapter weights for smaller checkpoint files
                checkpoint = adapter_state_dict(model)
                # TODO: Provide a function/script to merge the adapter weights with pretrained weights
                if fabric.is_global_zero:
                    torch.save(checkpoint, os.path.join(out_dir, f"iter-{iter_num:06d}-ckpt.pt"))
                fabric.barrier()

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
    return output # output.split("### Response:")[1].strip()


@torch.no_grad()
def validate(fabric: L.Fabric, model: torch.nn.Module, val_data: np.ndarray) -> torch.Tensor:
    fabric.print("Validating ...")
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        input_ids, targets = get_batch(fabric, val_data)
        logits = model(input_ids)
        loss = loss_fn(logits, targets)
        losses[k] = loss.item()
    out = losses.mean()

    # produce an example:
    instruction = "Recommend a movie for me to watch during the weekend and explain the reason."
    
    output = generate_response(model, instruction)
    fabric.print(instruction)
    fabric.print(output)

    model.train()
    return out.item()

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

    max_len = max(len(s) for s in input_ids)

    def pad_right(x, pad_id):
        # pad right based on the longest sequence
        n = max_len - len(x)
        return torch.cat((x, torch.full((n,), pad_id, dtype=x.dtype)))

    x = torch.stack([pad_right(x, pad_id=0) for x in input_ids])
    y = torch.stack([pad_right(x, pad_id=-1) for x in labels])
    x, y = fabric.to_device((x.pin_memory(), y.pin_memory()))
    return x, y


def load_datasets(data_dir: str = "data/alpaca"):
    train_data = torch.load(os.path.join(data_dir, "train.pt"))
    val_data = torch.load(os.path.join(data_dir, "test.pt"))
    return train_data, val_data


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
