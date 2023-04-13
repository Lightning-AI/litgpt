"""Utility functions for training and inference."""

import functools
from pathlib import Path

import torch
import torch.utils._device
from lightning.fabric.strategies import DeepSpeedStrategy, FSDPStrategy
from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType


def save_model_checkpoint(fabric, model, file_path):
    """Handles boilerplate logic for retrieving and saving the state_dict.
    
    This will be upstreamed to Fabric soon.
    """
    file_path = Path(file_path)

    if isinstance(fabric.strategy, DeepSpeedStrategy):
        from deepspeed.utils.zero_to_fp32 import convert_zero_checkpoint_to_fp32_state_dict

        fabric.save(file_path, {"model": model})
        fabric.barrier()
        if fabric.global_rank == 0:
            # Create a consolidated checkpoint with the same name next to the deepspeed checkpoint
            convert_zero_checkpoint_to_fp32_state_dict(file_path, file_path.with_suffix(".pt"))
        return

    if isinstance(fabric.strategy, FSDPStrategy):
        save_policy = FullStateDictConfig(offload_to_cpu=(fabric.world_size > 1), rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
            state_dict = model._forward_module.state_dict()
    else:
        state_dict = model.state_dict()

    if fabric.global_rank == 0:
        torch.save(state_dict, file_path)
    fabric.barrier()


class EmptyInitOnDevice(torch.overrides.TorchFunctionMode):
    def __init__(self, device=None, dtype=None, quantization_mode=None):
        """
        Create tensors with given device and dtype and don't run initialization
           (but instead use "empty tensors", i.e. uninitialized memory).

            device: `torch.device` to work with
            dtype: `torch.dtype` to work with
            quantization_mode: optional string, quantization mode to work with, default `None`.
                 Available modes: `llm.int8` bitsnbytes LLM.int8 quantization (only on GPU)

        Example::
            with EmptyInitOnDevice("cuda", dtype=torch.bfloat16):
               model = LLaMA.from_name('7B')
            model.load_state_dict(torch.load('llama-lit/7B/state_dict.pth'))"""

        self.quantization_mode = quantization_mode
        self.quantized_linear_cls = None
        if self.quantization_mode == 'llm.int8':
            if device.type != "cuda":
                raise ValueError("Quantization is only supported on the GPU.")
            from .quantization import Linear8bitLt
            self.quantized_linear_cls = Linear8bitLt
        elif self.quantization_mode == 'gptq.int4':
            from .quantization import ColBlockQuantizedLinear
            self.quantized_linear_cls = functools.partial(ColBlockQuantizedLinear, bits=4, tile_cols=-1)
        elif self.quantization_mode == 'gptq.int8':
            from .quantization import ColBlockQuantizedLinear
            self.quantized_linear_cls = functools.partial(ColBlockQuantizedLinear, bits=8, tile_cols=-1)
        elif self.quantization_mode is not None:
            raise RuntimeError(f"unknown quantization mode {self.quantization_mode}")
        self.device = device
        self.dtype = dtype

    def __enter__(self):
        if self.quantized_linear_cls != None:
            self.torch_linear_cls = torch.nn.Linear
            torch.nn.Linear = self.quantized_linear_cls
        return super().__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.quantized_linear_cls != None:
            torch.nn.Linear = self.torch_linear_cls
        return super().__exit__(exc_type, exc_val, exc_tb)

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if getattr(func, "__module__", None) == "torch.nn.init":
            if "tensor" in kwargs:
                return kwargs["tensor"]
            else:
                return args[0]
        if (
            self.device is not None
            and func in torch.utils._device._device_constructors()
            and kwargs.get("device") is None
        ):
            kwargs["device"] = self.device
        if (
            self.dtype is not None
            and func in torch.utils._device._device_constructors()
            and kwargs.get("dtype") is None
        ):
            kwargs["dtype"] = self.dtype
        return func(*args, **kwargs)
