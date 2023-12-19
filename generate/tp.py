"""Tensor-parallel implementation adapted from https://github.com/pytorch-labs/gpt-fast/blob/14df27/tp.py"""

import sys
import time
from functools import partial
from pathlib import Path
from typing import Optional, Union

import lightning as L
import torch
import torch._dynamo.config
import torch._inductor.config
from torch.distributed._functional_collectives import all_reduce

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

import generate.base as generate_base
from lit_gpt import GPT, Config, Tokenizer
from lit_gpt.model import CausalSelfAttention, GptNeoxMLP, LLaMAMLP, LLaMAMoE
from lit_gpt.utils import check_valid_checkpoint_dir, get_default_supported_precision


def tensor_parallel_linear(fabric: L.Fabric, linear: torch.nn.Linear, style: str) -> None:
    world_size = fabric.world_size
    dim, attr = {"colwise": (0, "out_features"), "rowwise": (1, "in_features")}[style]
    size = getattr(linear, attr)
    if size % world_size != 0:
        raise ValueError(
            f"This linear's {attr} value ({size}) is not evenly divisible by the world size ({world_size})"
        )

    shard = torch.tensor_split(linear.weight, world_size, dim=dim)[fabric.global_rank]
    linear.weight = torch.nn.Parameter(shard, requires_grad=linear.weight.requires_grad)
    setattr(linear, attr, shard.size(dim))

    if linear.bias is not None and dim == 0:
        shard = torch.tensor_split(linear.bias, world_size)[fabric.global_rank]
        linear.bias = torch.nn.Parameter(shard, requires_grad=linear.bias.requires_grad)


def tensor_parallel_mlp(fabric: L.Fabric, mlp: Union[GptNeoxMLP, LLaMAMLP, LLaMAMoE]) -> None:
    if isinstance(mlp, LLaMAMLP):
        tensor_parallel_linear(fabric, mlp.fc_1, "colwise")
        tensor_parallel_linear(fabric, mlp.fc_2, "colwise")
        tensor_parallel_linear(fabric, mlp.proj, "rowwise")
        mlp.register_forward_hook(partial(all_reduce_output, fabric.world_size))
    elif isinstance(mlp, GptNeoxMLP):
        tensor_parallel_linear(fabric, mlp.fc, "colwise")
        tensor_parallel_linear(fabric, mlp.proj, "rowwise")
        mlp.register_forward_hook(partial(all_reduce_output, fabric.world_size))
    elif isinstance(mlp, LLaMAMoE):
        # we use expert slicing across ranks, alternatively, we could create a expert parallelism group
        # when the number of experts is a multiple of the world size
        for expert in mlp.experts:
            tensor_parallel_mlp(fabric, expert)
    else:
        raise NotImplementedError


def tensor_parallel_attn(fabric: L.Fabric, attn: CausalSelfAttention) -> None:
    tensor_parallel_linear(fabric, attn.attn, "colwise")
    tensor_parallel_linear(fabric, attn.proj, "rowwise")
    attn.register_forward_hook(partial(all_reduce_output, fabric.world_size))


def all_reduce_output(world_size: int, module: torch.nn.Module, ins, outs) -> torch.Tensor:
    return all_reduce(outs, "sum", list(range(world_size)))


def tensor_parallel(fabric: L.Fabric, model: GPT) -> GPT:
    for block in model.transformer.h:
        tensor_parallel_mlp(fabric, block.mlp)
        tensor_parallel_attn(fabric, block.attn)

    # update the config values to the shard sizes
    # this is only relevant for `tensor_parallel_attn`, but it needs to run only once
    world_size = fabric.world_size
    model.config.n_head //= world_size
    model.config.n_embd //= world_size
    model.config.n_query_groups //= world_size

    return model


@torch.inference_mode()
def main(
    prompt: str = "What food do llamas eat?",
    *,
    num_samples: int = 1,
    max_new_tokens: int = 50,
    top_k: Optional[int] = 200,
    temperature: float = 0.8,
    checkpoint_dir: Path = Path("checkpoints/stabilityai/stablelm-base-alpha-3b"),
    precision: Optional[str] = None,
    compile: bool = False,
) -> None:
    """Generates text samples based on a pre-trained model and tokenizer.

    Args:
        prompt: The prompt string to use for generating the samples.
        num_samples: The number of text samples to generate.
        max_new_tokens: The number of generation steps to take.
        top_k: The number of top most probable tokens to consider in the sampling process.
        temperature: A value controlling the randomness of the sampling process. Higher values result in more random
            samples.
        checkpoint_dir: The checkpoint directory to load.
        precision: Indicates the Fabric precision setting to use.
        compile: Whether to compile the model.
    """
    precision = precision or get_default_supported_precision(training=False)

    fabric = L.Fabric(devices="auto", strategy="ddp", precision=precision)
    fabric.launch()

    check_valid_checkpoint_dir(checkpoint_dir)

    config = Config.from_json(checkpoint_dir / "lit_config.json")

    model_file = "lit_model.pth"
    checkpoint_path = checkpoint_dir / model_file

    tokenizer = Tokenizer(checkpoint_dir)
    encoded = tokenizer.encode(prompt, device=fabric.device)
    prompt_length = encoded.size(0)
    max_returned_tokens = prompt_length + max_new_tokens

    fabric.print(f"Loading model {str(checkpoint_path)!r} with {config.__dict__}", file=sys.stderr)
    t0 = time.perf_counter()
    with fabric.init_module(empty_init=False), torch.device("meta"):
        model = GPT(config)
    fabric.print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.", file=sys.stderr)

    t0 = time.perf_counter()
    state_dict = torch.load(str(checkpoint_path), mmap=True)
    model.load_state_dict(state_dict, assign=True)
    fabric.print(f"Time to load the model weights: {time.perf_counter() - t0:.02f} seconds.", file=sys.stderr)

    t0 = time.perf_counter()
    model = tensor_parallel(fabric, model)
    fabric.print(f"Time to tensor-parallelize the model: {time.perf_counter() - t0:.02f} seconds.", file=sys.stderr)

    with fabric.init_tensor():
        # set the max_seq_length to limit the memory usage to what we need
        model.max_seq_length = max_returned_tokens
        # enable the kv cache
        model.set_kv_cache(batch_size=1)
    model.eval()

    model = fabric.to_device(model)

    if compile:
        torch._dynamo.config.automatic_dynamic_shapes = True
        torch._inductor.config.triton.unique_kernel_names = True
        torch._inductor.config.coordinate_descent_tuning = True
        generate_base.next_token = torch.compile(generate_base.next_token, mode="reduce-overhead")

    L.seed_everything(1234)
    for i in range(num_samples):
        t0 = time.perf_counter()
        y = generate_base.generate(
            model, encoded, max_returned_tokens, temperature=temperature, top_k=top_k, eos_id=tokenizer.eos_id
        )
        t = time.perf_counter() - t0
        for block in model.transformer.h:
            block.attn.kv_cache.reset_parameters()
        fabric.print(tokenizer.decode(y))
        tokens_generated = y.size(0) - prompt_length
        fabric.print(
            f"Time for inference {i + 1}: {t:.02f} sec total, {tokens_generated / t:.02f} tokens/sec", file=sys.stderr
        )
    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB", file=sys.stderr)


if __name__ == "__main__":
    from jsonargparse import CLI

    torch.set_float32_matmul_precision("high")
    CLI(main)
