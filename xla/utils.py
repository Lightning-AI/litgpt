import itertools
from functools import partial
from pathlib import Path
from typing import Any, Callable

import lightning as L
import torch
from lightning.fabric.strategies.xla_fsdp import XLAFSDPStrategy, _activation_checkpointing_auto_wrapper
from lightning_utilities.core.rank_zero import rank_prefixed_message

from lit_gpt import GPT


def rank_print(fabric: L.Fabric, message: object, *, flush: bool = True, **kwargs: Any) -> None:
    if fabric.local_rank == 0:
        message = str(message)
        # let each host print, but only on rank 0
        message = rank_prefixed_message(message, fabric.global_rank)
        # TPU VM will only print when the script finishes if `flush=False`
        print(message, flush=flush, **kwargs)


def sequential_init(fabric: L.Fabric, get_model: Callable[[], GPT], checkpoint_path: Path) -> torch.nn.Module:
    assert fabric._launched
    # similar logic could be implemented for regular FSDP, but this implementation is specific to XLAFSDP
    assert isinstance(fabric.strategy, XLAFSDPStrategy)

    with fabric.init_module(empty_init=False), torch.device("meta"):
        model = get_model()

    # load the checkpoint on a single rank to limit the system memory usage
    state_dict = torch.load(checkpoint_path, map_location="cpu", mmap=True) if fabric.global_rank == 0 else {}
    fabric.barrier()

    fsdp_kwargs = fabric.strategy._fsdp_kwargs
    if "auto_wrapper_callable" in fsdp_kwargs:
        # includes activation checkpointing if configured
        wrap = fsdp_kwargs.pop("auto_wrapper_callable")
    else:
        wrap = partial(_activation_checkpointing_auto_wrapper, tuple())
    fsdp_kwargs.pop("auto_wrap_policy")  # we can ignore this
    assert not fsdp_kwargs  # these would be silently ignored

    for i, block in enumerate(model.transformer.h):
        # get the relevant piece of the state dict
        to_load = (
            {
                param_name: state_dict[key]
                for param_name, _ in block.named_parameters()
                if (key := f"transformer.h.{i}.{param_name}") in state_dict
            }
            if fabric.global_rank == 0
            else {}
        )
        # load the current block on all ranks
        to_load = fabric.broadcast(to_load)
        keys = block.load_state_dict(to_load, strict=False, assign=True)
        assert not keys.unexpected_keys

        # materialize any leftover meta parameters, regular FSDP does it automatically
        for param in itertools.chain(model.parameters(recurse=False), model.buffers(recurse=False)):
            if param.is_meta:
                # init on CPU, FSDP will shard and move it
                model.to_empty(device="cpu", recurse=False)
                model.reset_parameters()

        # shard the block
        wrapped_block = wrap(block)
        model.transformer.h[i] = wrapped_block

    # materialize all other modules, loading their state dict if applicable
    for module_name, module in model.named_modules():
        for param_name, param in itertools.chain(
            module.named_parameters(prefix=module_name, recurse=False), module.named_buffers(recurse=False)
        ):
            if param.is_meta:
                if fabric.broadcast(param_name in state_dict):
                    to_load = state_dict[param_name] if fabric.global_rank == 0 else {}
                    to_load = fabric.broadcast(to_load, src=0)
                    to_load = fabric.to_device(to_load)
                    keys = model.load_state_dict({param_name: to_load}, strict=False, assign=True)
                    assert not keys.unexpected_keys
                else:
                    module.to_empty(device=fabric.device, recurse=False)
                    module.reset_parameters()

    return model
