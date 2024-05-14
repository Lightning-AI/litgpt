"""Tensor-parallel implementation adapted from https://github.com/pytorch-labs/gpt-fast/blob/14df27/tp.py"""

import logging
import sys
import time
from pathlib import Path
from typing import Literal, Optional, Union

import lightning as L
import torch
import torch._dynamo.config
import torch._inductor.config
from lightning.fabric.plugins import BitsandbytesPrecision
from lightning.fabric.strategies import ModelParallelStrategy
from lightning.fabric.utilities import rank_zero_only

import litgpt.generate.base as generate_base
from litgpt import GPT, Config, Tokenizer
from litgpt.model import CausalSelfAttention, GptNeoxMLP, LLaMAMLP, LLaMAMoE
from litgpt.utils import check_valid_checkpoint_dir, get_default_supported_precision
from torch.distributed._tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    SequenceParallel,
    parallelize_module,
)


def tensor_parallel_mlp(mesh, mlp: Union[GptNeoxMLP, LLaMAMLP, LLaMAMoE]) -> None:
    plan = {}
    if isinstance(mlp, LLaMAMLP):
        plan["fc_1"] = ColwiseParallel()
        plan["fc_2"] = ColwiseParallel()
        plan["proj"] = RowwiseParallel()
    elif isinstance(mlp, GptNeoxMLP):
        plan["fc"] = ColwiseParallel()
        plan["proj"] = RowwiseParallel()
    elif isinstance(mlp, LLaMAMoE):
        # we use expert slicing across ranks, alternatively, we could create a expert parallelism group
        # when the number of experts is a multiple of the world size
        for expert in mlp.experts:
            tensor_parallel_mlp(mesh, expert)
    else:
        raise NotImplementedError
    
    parallelize_module(mlp, mesh, plan)


def tensor_parallel_attn(mesh, attn: CausalSelfAttention) -> None:
    plan = {
        "attn": ColwiseParallel(),
        "proj": RowwiseParallel(),
    }
    parallelize_module(attn, mesh, plan)


def parallelize(model, device_mesh):
    tp_mesh = device_mesh["tensor_parallel"]
    dp_mesh = device_mesh["data_parallel"]
    
    assert tp_mesh.size() > 1
    assert dp_mesh.size() == 1

    plan = {
        "transformer.wte": RowwiseParallel(input_layouts=Replicate()),
        "transformer.ln_f": SequenceParallel(),
        "lm_head": ColwiseParallel(input_layouts=Shard(1), output_layouts=Replicate()),
    }
    parallelize_module(model, tp_mesh, plan)
    
    for block in model.transformer.h:
        plan = {
            # "norm_1": SequenceParallel(), 
            # "norm_2": SequenceParallel(),
        }
        # parallelize_module(block, tp_mesh, plan)
        tensor_parallel_mlp(tp_mesh, block.mlp)
        tensor_parallel_attn(tp_mesh, block.attn)

    # update the config values to the shard sizes
    # this is only relevant for `tensor_parallel_attn`, but it needs to run only once
    attrs = ["n_head", "n_embd", "n_query_groups"]
    for attr in attrs:
        size = getattr(model.config, attr)
        if size % tp_mesh.size() != 0:
            raise ValueError(f"This {attr} value ({size}) is not evenly divisible by the world size ({tp_mesh.size()})")
        setattr(model.config, attr, size // tp_mesh.size())

    return model


@torch.no_grad()
def main(
    prompt: str = "What food do llamas eat?",
    *,
    num_samples: int = 1,
    max_new_tokens: int = 50,
    top_k: Optional[int] = 50,
    top_p: float = 1.0,
    temperature: float = 0.8,
    checkpoint_dir: Path = Path("checkpoints/stabilityai/stablelm-base-alpha-3b"),
    quantize: Optional[Literal["bnb.nf4", "bnb.nf4-dq", "bnb.fp4", "bnb.fp4-dq"]] = None,
    precision: Optional[str] = None,
    compile: bool = False,
) -> None:
    """Generates text samples based on a pre-trained model and tokenizer.

    Args:
        prompt: The prompt string to use for generating the samples.
        num_samples: The number of text samples to generate.
        max_new_tokens: The number of generation steps to take.
        top_k: The number of top most probable tokens to consider in the sampling process.
        top_p: If specified, it represents the cumulative probability threshold to consider in the sampling process.
            In top-p sampling, the next token is sampled from the highest probability tokens
            whose cumulative probability exceeds the threshold `top_p`. When specified,
            it must be `0 <= top_p <= 1`. Here, `top_p=0` is equivalent
            to sampling the most probable token, while `top_p=1` samples from the whole distribution.
            It can be used in conjunction with `top_k` and `temperature` with the following order
            of application:

            1. `top_k` sampling
            2. `temperature` scaling
            3. `top_p` sampling

            For more details, see https://arxiv.org/abs/1904.09751
            or https://huyenchip.com/2024/01/16/sampling.html#top_p
        temperature: A value controlling the randomness of the sampling process. Higher values result in more random
            samples.
        checkpoint_dir: The checkpoint directory to load.
        quantize: Whether to quantize the model and using which method:
            - bnb.nf4, bnb.nf4-dq, bnb.fp4, bnb.fp4-dq: 4-bit quantization from bitsandbytes
            for more details, see https://github.com/Lightning-AI/litgpt/blob/main/tutorials/quantize.md
        precision: Indicates the Fabric precision setting to use.
        compile: Whether to compile the model.
    """
    precision = precision or get_default_supported_precision(training=False)

    plugins = None
    if quantize is not None:
        if compile:
            raise NotImplementedError  # untested
        if "mixed" in precision:
            raise ValueError("Quantization and mixed precision is not supported.")
        dtype = {"16-true": torch.float16, "bf16-true": torch.bfloat16, "32-true": torch.float32}[precision]
        bnb_logger = logging.getLogger("lightning.fabric.plugins.precision.bitsandbytes")
        bnb_logger.setLevel(logging.DEBUG)
        bnb_logger.debug = rank_zero_only(bnb_logger.debug)
        plugins = BitsandbytesPrecision(quantize[4:], dtype)
        precision = None

    strategy = ModelParallelStrategy(parallelize_fn=parallelize)
    fabric = L.Fabric(devices="auto", strategy=strategy, precision=precision, plugins=plugins)
    fabric.launch()

    check_valid_checkpoint_dir(checkpoint_dir)
    config = Config.from_file(checkpoint_dir / "model_config.yaml")

    model_file = "lit_model.pth"
    checkpoint_path = checkpoint_dir / model_file

    tokenizer = Tokenizer(checkpoint_dir)
    encoded = tokenizer.encode(prompt, device=fabric.device)
    prompt_length = encoded.size(0)
    max_returned_tokens = prompt_length + max_new_tokens

    fabric.print(f"Loading model {str(checkpoint_path)!r} with {config.__dict__}", file=sys.stderr)
    t0 = time.perf_counter()
    
    with fabric.init_module(empty_init=True):
        model = GPT(config)
    fabric.print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.", file=sys.stderr)

    model = fabric.setup(model)
    
    t0 = time.perf_counter()
    fabric.load_raw(checkpoint_path, model)
    fabric.print(f"Time to load the model weights: {time.perf_counter() - t0:.02f} seconds.", file=sys.stderr)

    with fabric.init_tensor():
        # set the max_seq_length to limit the memory usage to what we need
        model.max_seq_length = max_returned_tokens
        # the rope cache which is on meta device
        model.cos, model.sin = model.rope_cache()
        # # enable the kv cache
        model.set_kv_cache(batch_size=1)
    model.eval()

    t0 = time.perf_counter()

    if compile:
        torch._dynamo.config.automatic_dynamic_shapes = True
        torch._inductor.config.triton.unique_kernel_names = True
        torch._inductor.config.coordinate_descent_tuning = True
        generate_base.next_token = torch.compile(generate_base.next_token, mode="reduce-overhead")

    L.seed_everything(1234)
    for i in range(num_samples):
        t0 = time.perf_counter()
        y = generate_base.generate(
            model, encoded, max_returned_tokens, temperature=temperature, top_k=top_k, top_p=top_p, eos_id=tokenizer.eos_id
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
    
    torch.distributed.destroy_process_group()
