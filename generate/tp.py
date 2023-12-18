import sys
import time
from pathlib import Path
from typing import Optional, List

import lightning as L
import torch
import torch._dynamo.config
import torch._inductor.config
from torch.distributed import _functional_collectives as funcol

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_gpt import GPT, Config, Tokenizer
import generate.base as generate_base
from lit_gpt.model import CausalSelfAttention, LLaMAMLP
from lit_gpt.utils import (
    check_valid_checkpoint_dir,
    get_default_supported_precision,
    load_checkpoint,
)


def _apply_tp_linear(fabric: L.Fabric, linear: torch.nn.Linear, style: str, weight_splits: List[int] = []) -> None:
    world_size = fabric.world_size

    # Linear's weight matrix is transposed, and is of shape
    # (linear.out_features, linear.in_features)
    dim_lookup = {
        "colwise": (0, "out_features"),
        "rowwise": (1, "in_features")
    }
    assert style in dim_lookup
    shard_dim, size_attr = dim_lookup[style]

    # ensure we can shard evenly
    assert getattr(linear, size_attr) % world_size == 0
    def shard(x, dim):
        assert x.size(dim=dim) % world_size == 0
        return torch.tensor_split(x, world_size, dim=dim)[fabric.global_rank]
    
    def shard_qkv(qkv, dim, weight_splits):
        q, k, v = qkv.split(weight_splits, dim=dim)
        q = shard(q, dim)
        k = shard(k, dim)
        v = shard(v, dim)
        return torch.cat((q,k,v), dim=dim)

    # shard
    if weight_splits:
        assert len(weight_splits) == 3
        sharded_weight = shard_qkv(linear.weight, shard_dim, weight_splits)
    else:
        sharded_weight = shard(linear.weight, shard_dim)

    linear.weight = torch.nn.Parameter(sharded_weight, requires_grad=False)
    setattr(linear, size_attr, getattr(linear, size_attr) // world_size)


def _apply_tp_ffn(fabric: L.Fabric, mlp: LLaMAMLP) -> None:
    _apply_tp_linear(fabric, mlp.fc_1, "colwise")
    _apply_tp_linear(fabric, mlp.fc_2, "colwise")
    _apply_tp_linear(fabric, mlp.proj, "rowwise")

    mlp.register_forward_hook(lambda _module, _input, output: funcol.all_reduce(output, "sum", list(range(fabric.world_size))))


def _apply_tp_attn(fabric: L.Fabric, attn: CausalSelfAttention) -> None:
    kv_size = attn.config.n_query_groups * attn.config.head_size
    _apply_tp_linear(fabric, attn.attn, "colwise", [attn.config.n_embd, kv_size, kv_size])
    _apply_tp_linear(fabric, attn.proj, "rowwise")
    attn.register_forward_hook(lambda _module, _input, output: funcol.all_reduce(output[0], "sum", list(range(fabric.world_size))))


def tensor_parallel(fabric: L.Fabric, model: GPT) -> GPT:
    world_size = fabric.world_size
    #model.config.n_head //= world_size
    #model.config.n_embd //= world_size
    #model.config.n_query_groups //= world_size
    for block in model.transformer.h:
        #_apply_tp_attn(fabric, block.attn)
        _apply_tp_ffn(fabric, block.mlp)
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
    
    model = tensor_parallel(fabric, model)
    
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
        global next_token
        next_token = torch.compile(next_token, mode="reduce-overhead")

    L.seed_everything(1234)
    for i in range(num_samples):
        t0 = time.perf_counter()
        y = generate_base.generate(model, encoded, max_returned_tokens, temperature=temperature, top_k=top_k, eos_id=tokenizer.eos_id)
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
