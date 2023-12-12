import itertools
import os
from functools import partial
from typing import Dict, Type, Optional, Callable

import torch
from lightning import Fabric
from lightning.fabric.plugins import HalfPrecision
from lightning.fabric.utilities.throughput import _plugin_to_compute_dtype

import lit_gpt.model
from lit_gpt import GPT
from lit_gpt.model import Block, apply_rope

DEBUG = os.getenv("DEBUG", "0") == "1"


@torch.inference_mode()
def get_model(fabric: Fabric, name: str):
    if fabric.local_rank != fabric.global_rank:
        raise NotImplementedError("Multinode is not supported")

    dtype = _plugin_to_compute_dtype(fabric.strategy.precision)
    torch.set_default_dtype(dtype)

    with torch.device("meta"):
        model = GPT.from_name(name)

    # monkeypatch rope
    lit_gpt.model.apply_rope = meta_friendly_apply_rope

    assert model.config.n_layer % fabric.world_size == 0
    layers_per_rank = model.config.n_layer // fabric.world_size
    # dictates where each block should be instantiated
    mapping = layer_to_device(model, chunk_on=Block, chunks=layers_per_rank)
    if DEBUG:
        fabric.print(f"Layer mapping: {mapping}")
    # materialize each block on the appropriate rank (device)
    for layer_num, target_rank in mapping.items():
        if fabric.local_rank == target_rank:
            path = f"transformer.h.{layer_num}"
            if DEBUG:
                print(f"[{fabric.global_rank}] Materializing {path}")
            submodule = model.get_submodule(path)
            materialize_meta_tensors(submodule, fabric.device)
    # and everything that is not a block on rank 0
    if fabric.local_rank == 0:
        materialize_meta_tensors(model, fabric.device, skip_fn=lambda path: "transformer.h." in path)
    # rebuild the rope cache which is on meta device otherwise, all ranks need to do this
    with fabric.device:
        model.max_seq_length = model.max_seq_length

    # quantize
    # FIXME

    # setup communication hooks to pipeline layers
    send_layers = [layers_per_rank * i - 1 for i in range(1, fabric.world_size + 1)]
    recv_layers = [layers_per_rank * i for i in range(1, fabric.world_size)]
    final_layer = max(send_layers)
    if DEBUG:
        fabric.print(f"{send_layers=}, {recv_layers=}, {final_layer=}")
    if fabric.world_size > 1:
        for layer_num, target_rank in mapping.items():
            path = f"transformer.h.{layer_num}"
            submodule = model.get_submodule(path)
            if fabric.local_rank == target_rank:
                if layer_num in send_layers:
                    dst = (target_rank + 1) % fabric.world_size
                    if DEBUG:
                        print(f"[{fabric.global_rank}] {path}: registered send_output to {dst}")
                    submodule.register_forward_hook(partial(send_output, dst))
                elif layer_num in recv_layers:
                    src = target_rank - 1
                    if DEBUG:
                        print(f"[{fabric.global_rank}] {path}: registered receive_input from {src}")
                    submodule.register_forward_pre_hook(partial(receive_input, src))
            if fabric.local_rank == 0 and layer_num == final_layer:
                src = fabric.world_size - 1
                if DEBUG:
                    print(f"[{fabric.global_rank}] {path}: registered replace_output from {src}")
                submodule.register_forward_hook(partial(replace_output, src))

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


def receive_input(src: int, module: torch.nn.Module, ins) -> Optional[torch.Tensor]:
    """``forward_pre_hook`` to receive an input before forward."""
    tensor = ins[0]
    assert tensor.device.type == "meta"
    tensor = torch.empty_like(tensor, device="cuda")
    torch.distributed.recv(tensor, src)
    return (tensor,) + ins[1:]


def send_output(dst: int, module: torch.nn.Module, ins, outs) -> Optional[torch.Tensor]:
    """``forward_hook`` to send an output after forward."""
    assert outs.device.type != "meta"
    torch.distributed.send(outs, dst)
    tensor = torch.empty_like(outs, device="meta")
    return tensor


def replace_output(src: int, module: torch.nn.Module, ins, outs) -> Optional[torch.Tensor]:
    """``forward_hook`` to replace an output after forward."""
    assert outs.device.type == "meta"
    outs = torch.empty_like(outs, device="cuda")
    torch.distributed.recv(outs, src)
    return outs


def meta_friendly_apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """The RoPE cache is shared across layers. For the layers that are on meta-device, the cache needs to be moved."""
    if (device := x.device).type == "meta":
        return apply_rope(x, cos.to(device), sin.to(device))
    return apply_rope(x, cos, sin)


@torch.inference_mode()
def main(fabric: Fabric):
    model = get_model(fabric, name="Mistral-7B-Instruct-v0.1")

    if DEBUG:
        memory_before_fwd = torch.cuda.max_memory_allocated()

    model.eval()

    x = torch.randint(0, 10, (1, model.config.block_size), device=fabric.device if fabric.local_rank == 0 else "meta")
    y = model(x)
    print(f"[{fabric.global_rank}] {y.shape}, {y.sum()}")

    if DEBUG:
        memory_after_fwd = torch.cuda.max_memory_allocated()
        print(
            f"[{fabric.global_rank}] before: {memory_before_fwd / 1e9:.02f} GB, after: {memory_after_fwd / 1e9:.02f},"
            f" difference: {(memory_after_fwd - memory_before_fwd) / 1e9:.02f} GB"
        )


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    bnb = HalfPrecision()  # FIXME BitsandbytesPrecision(mode="nf4-dq", dtype=torch.bfloat16)
    fabric = Fabric(strategy="ddp", devices="auto", plugins=bnb)

    # using Fabric as a launcher
    fabric.launch(main)
