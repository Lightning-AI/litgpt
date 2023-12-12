import itertools
import os
import sys
import time
from functools import partial
from pathlib import Path
from typing import Callable, Dict, Literal, Optional, Type
from warnings import filterwarnings

import lightning as L
import torch
from lightning.fabric.plugins import BitsandbytesPrecision
from lightning.fabric.utilities.throughput import _plugin_to_compute_dtype

from generate.base import generate
from lit_gpt import GPT, Config, Tokenizer
from lit_gpt.model import Block, build_mask_cache
from lit_gpt.utils import check_valid_checkpoint_dir, get_default_supported_precision

DEBUG = os.getenv("DEBUG", "0") == "1"


@torch.inference_mode()
def get_model(fabric: L.Fabric, config: Config, max_seq_length: int):
    if fabric.local_rank != fabric.global_rank:
        raise NotImplementedError("Multinode is not supported")

    with torch.device("meta"):
        model = GPT(config)

    assert model.config.n_layer % fabric.world_size == 0
    layers_per_rank = model.config.n_layer // fabric.world_size
    # dictates where each block should be instantiated
    mapping = layer_to_device(model, chunk_on=Block, chunks=layers_per_rank)
    if DEBUG:
        fabric.print(f"Layer mapping: {mapping}")
    # materialize each block on the appropriate rank (device)
    for layer_num, target_rank in mapping.items():
        path = f"transformer.h.{layer_num}"
        submodule = model.get_submodule(path)
        if fabric.local_rank == target_rank:
            if DEBUG:
                print(f"[{fabric.global_rank}] Materializing {path}")
            materialize_meta_tensors(submodule, fabric.device)
        # and build the kv cache
        submodule.attn.kv_cache = submodule.attn.build_kv_cache(
            1, max_seq_length, model.cos.size(-1), fabric.device if fabric.local_rank == target_rank else "meta"
        )
    # and everything that is not a block on rank 0
    if fabric.local_rank == 0:
        materialize_meta_tensors(model, fabric.device, skip_fn=lambda path: "transformer.h." in path)
    # rebuild odd ends on all ranks
    with fabric.device:
        # the rope cache which is on meta device
        model.max_seq_length = max_seq_length
        # the mask cache which cannot be created with `set_kv_cache` because that will set it for all layers
        model.mask_cache = build_mask_cache(max_seq_length)

    # quantize
    # FIXME

    if fabric.world_size > 1:
        # setup initial hook for the model input on non-zero ranks
        if fabric.local_rank != 0:
            model.register_forward_pre_hook(meta_gpt_input)

        # setup communication hooks to pipeline layers
        send_layers = [layers_per_rank * i - 1 for i in range(1, fabric.world_size + 1)]
        recv_layers = [layers_per_rank * i for i in range(1, fabric.world_size)]
        final_layer = max(send_layers)
        if DEBUG:
            fabric.print(f"{send_layers=}, {recv_layers=}, {final_layer=}")
        for layer_num, target_rank in mapping.items():
            path = f"transformer.h.{layer_num}"
            submodule = model.get_submodule(path)
            if fabric.local_rank == target_rank:
                if layer_num in send_layers:
                    dst = (target_rank + 1) % fabric.world_size
                    if DEBUG:
                        print(f"[{fabric.global_rank}] {path}: registered send_output to {dst}")
                    submodule.register_forward_hook(partial(send_block_output, dst))
                elif layer_num in recv_layers:
                    src = target_rank - 1
                    if DEBUG:
                        print(f"[{fabric.global_rank}] {path}: registered receive_input from {src}")
                    submodule.register_forward_pre_hook(partial(recv_block_input, src))
            if fabric.local_rank == 0 and layer_num == final_layer:
                src = fabric.world_size - 1
                if DEBUG:
                    print(f"[{fabric.global_rank}] {path}: registered replace_output from {src}")
                submodule.register_forward_hook(partial(recv_block_output, src))

        # setup final hook for the model output
        model.register_forward_hook(broadcast_gpt_output)

    if DEBUG:
        path_to_device = {k: str(v.device) for k, v in itertools.chain(model.named_parameters(), model.named_buffers())}
        print(f"[{fabric.global_rank}] {path_to_device}")

    return model


def layer_to_device(module: torch.nn.Module, chunk_on: Type[torch.nn.Module], chunks: int) -> Dict[int, int]:
    """Create a mapping from layer (block) number to device (rank)."""
    mapping = {}
    for name, submodule in module.named_modules():
        if isinstance(submodule, chunk_on):
            split = name.split(".")
            number = int(split[2])
            mapping[number] = number // chunks
    return mapping


def materialize(module: torch.nn.Module, device: torch.device) -> None:
    """Materialize a module."""
    module.to_empty(device=device, recurse=False)
    module.reset_parameters()


def materialize_meta_tensors(
    module: torch.nn.Module, device: torch.device, skip_fn: Optional[Callable[[str], bool]] = None
) -> None:
    """Materialize all tensors in a given module."""
    for path, module in module.named_modules():
        if skip_fn is not None and skip_fn(path):
            continue
        if any(t.is_meta for t in itertools.chain(module.parameters(recurse=False), module.buffers(recurse=False))):
            materialize(module, device)


def meta_gpt_input(module: torch.nn.Module, ins) -> Optional[torch.Tensor]:
    """``forward_pre_hook`` to replace the original GPT input."""
    tensor = ins[0]
    assert tensor.device.type == "cuda"
    tensor = tensor.to(device="meta")
    return (tensor,) + ins[1:]


def recv_block_input(src: int, module: torch.nn.Module, ins) -> Optional[torch.Tensor]:
    """``forward_pre_hook`` to receive a Block's input before forward."""
    tensor = ins[0]
    assert tensor.device.type == "meta"
    tensor = torch.empty_like(tensor, device="cuda")
    torch.distributed.recv(tensor, src)
    return (tensor,) + ins[1:]


def send_block_output(dst: int, module: torch.nn.Module, ins, outs) -> Optional[torch.Tensor]:
    """``forward_hook`` to send a Block's output after forward."""
    assert outs.device.type != "meta"
    torch.distributed.send(outs, dst)
    return torch.empty_like(outs, device="meta")


def recv_block_output(src: int, module: torch.nn.Module, ins, outs) -> Optional[torch.Tensor]:
    """``forward_hook`` to replace a Block's output after forward."""
    assert outs.device.type == "meta"
    outs = torch.empty_like(outs, device="cuda")
    torch.distributed.recv(outs, src)
    return outs


def broadcast_gpt_output(module: torch.nn.Module, ins, out) -> Optional[torch.Tensor]:
    """``forward_hook`` to replace the final GPT result."""
    if out.device.type == "meta":
        out = torch.empty_like(out, device="cuda")
    torch.distributed.broadcast(out, 0)
    return out


@torch.inference_mode()
def main(
    prompt: str = "What food do llamas eat?",
    *,
    num_samples: int = 1,
    max_new_tokens: int = 50,
    top_k: Optional[int] = 200,
    temperature: float = 0.8,
    checkpoint_dir: Path = Path("checkpoints/mistralai/Mistral-7B-Instruct-v0.1"),
    quantize: Optional[Literal["bnb.nf4", "bnb.nf4-dq", "bnb.fp4", "bnb.fp4-dq", "bnb.int8"]] = None,
    devices: int = "auto",
    precision: Optional[str] = None,
    # FIXME: compile
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
        quantize: Whether to quantize the model and using which method:
            - bnb.nf4, bnb.nf4-dq, bnb.fp4, bnb.fp4-dq: 4-bit quantization from bitsandbytes
            - bnb.int8: 8-bit quantization from bitsandbytes
            - gptq.int4: 4-bit quantization from GPTQ
            for more details, see https://github.com/Lightning-AI/lit-gpt/blob/main/tutorials/quantize.md
        devices: How many devices to use.
        precision: Indicates the Fabric precision setting to use.
    """
    precision = precision or get_default_supported_precision(training=False)

    plugins = None
    if quantize is not None:
        if "mixed" in precision:
            raise ValueError("Quantization and mixed precision is not supported.")
        dtype = {"16-true": torch.float16, "bf16-true": torch.bfloat16, "32-true": torch.float32}[precision]
        plugins = BitsandbytesPrecision(quantize[4:], dtype)
        precision = None

    fabric = L.Fabric(devices=devices, precision=precision, strategy="ddp", accelerator="cuda", plugins=plugins)
    # using Fabric as a launcher: we don't want to actually use DDP
    fabric.launch()

    dtype = _plugin_to_compute_dtype(fabric.strategy.precision)
    fabric.print(f"Using {dtype} as compute dtype", file=sys.stderr)
    torch.set_default_dtype(dtype)

    check_valid_checkpoint_dir(checkpoint_dir)
    tokenizer = Tokenizer(checkpoint_dir)
    encoded = tokenizer.encode(prompt, device=fabric.device)
    prompt_length = encoded.size(0)
    max_returned_tokens = prompt_length + max_new_tokens

    config = Config.from_json(checkpoint_dir / "lit_config.json")
    checkpoint_path = checkpoint_dir / "lit_model.pth"
    fabric.print(f"Loading model {str(checkpoint_path)!r} with {config.__dict__}", file=sys.stderr)
    model = get_model(fabric, config, max_returned_tokens)
    model.eval()

    # for this script, this warning is a false-positive
    filterwarnings("ignore", ".*copying from a non-meta parameter.*", module="torch.nn.modules.module")
    state_dict = torch.load(str(checkpoint_path), mmap=True)
    model.load_state_dict(state_dict)

    L.seed_everything(1234)
    for i in range(num_samples):
        t0 = time.perf_counter()
        y = generate(model, encoded, max_returned_tokens, temperature=temperature, top_k=top_k, eos_id=tokenizer.eos_id)
        t = time.perf_counter() - t0
        for block in model.transformer.h:
            block.attn.kv_cache.reset_parameters()
        fabric.print(tokenizer.decode(y))
        tokens_generated = y.size(0) - prompt_length
        fabric.print(
            f"Time for inference {i + 1}: {t:.02f} sec total, {tokens_generated / t:.02f} tokens/sec", file=sys.stderr
        )
    fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB", file=sys.stderr)


if __name__ == "__main__":
    from jsonargparse import CLI

    torch.set_float32_matmul_precision("high")
    CLI(main)
