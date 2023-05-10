"""Utility functions for training and inference."""

import functools
from pathlib import Path
import pickle
import warnings
from io import BytesIO

import torch
import torch.utils._device
from lightning.fabric.strategies import DeepSpeedStrategy, FSDPStrategy
from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType


def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)


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
            convert_zero_checkpoint_to_fp32_state_dict(file_path, file_path.with_suffix(".pth"))
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
                                  `qptq.int4`, `gptq.int8`: GPTQ pre-quantized models

        Example::
            with EmptyInitOnDevice("cuda", dtype=torch.bfloat16):
               model = ...
            model.load_state_dict(torch.load('checkpoint.pth'))
        """

        self.quantization_mode = quantization_mode
        self.quantized_linear_cls = None
        if self.quantization_mode == "llm.int8":
            if device.type != "cuda":
                raise ValueError("Quantization is only supported on the GPU.")
            from .quantization import Linear8bitLt

            self.quantized_linear_cls = Linear8bitLt
        elif self.quantization_mode == "gptq.int4":
            from .quantization import ColBlockQuantizedLinear

            self.quantized_linear_cls = functools.partial(ColBlockQuantizedLinear, bits=4, tile_cols=-1)
        elif self.quantization_mode == "gptq.int8":
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


# this is taken from torchhacks https://github.com/lernapparat/torchhacks


class NotYetLoadedTensor:
    def __init__(self, metatensor, archiveinfo, storageinfo, rebuild_args):
        self.metatensor = metatensor
        self.archiveinfo = archiveinfo
        self.storageinfo = storageinfo
        self.rebuild_args = rebuild_args

    @classmethod
    def rebuild_from_type_v2(cls, func, new_type, args, state, *, archiveinfo=None):
        ret = func(*args)
        if isinstance(ret, NotYetLoadedTensor):
            old_lt = ret._load_tensor

            def _load_tensor():
                t = old_lt()
                return torch._tensor._rebuild_from_type_v2(lambda: t, new_type, (), state)

            ret._load_tensor = _load_tensor
            return ret
        return torch._tensor._rebuild_from_type_v2(func, new_type, args, state)

    @classmethod
    def rebuild_parameter(cls, data, requires_grad, backward_hooks, *, archiveinfo=None):
        if isinstance(data, NotYetLoadedTensor):
            old_lt = data._load_tensor

            def _load_tensor():
                t = old_lt()
                return torch._utils._rebuild_parameter(t, requires_grad, backward_hooks)

            data._load_tensor = _load_tensor
            return data
        return torch._utils._rebuild_parameter(data, requires_grad, backward_hooks)

    @classmethod
    def rebuild_tensor_v2(
        cls, storage, storage_offset, size, stride, requires_grad, backward_hooks, metadata=None, *, archiveinfo=None
    ):
        rebuild_args = (storage_offset, size, stride, requires_grad, backward_hooks, metadata)
        metatensor = torch._utils._rebuild_tensor_v2(
            storage, storage_offset, size, stride, requires_grad, backward_hooks, metadata
        )
        storageinfo = storage.archiveinfo
        return NotYetLoadedTensor(metatensor, archiveinfo, storageinfo, rebuild_args)

    def _load_tensor(self):
        name, storage_cls, fn, device, size = self.storageinfo
        dtype = self.metatensor.dtype

        uts = (
            self.archiveinfo.zipfile_context.zf.get_storage_from_record(
                f"data/{fn}", size * torch._utils._element_size(dtype), torch.UntypedStorage
            )
            ._typed_storage()
            ._untyped_storage
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            storage = torch.storage.TypedStorage(wrap_storage=uts, dtype=self.metatensor.dtype, _internal=True)
        tensor = torch._utils._rebuild_tensor_v2(storage, *self.rebuild_args)
        return tensor

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        loaded_args = [(a._load_tensor() if isinstance(a, NotYetLoadedTensor) else a) for a in args]
        res = func(*loaded_args, **kwargs)
        # gc.collect would be costly here, maybe do it optionally
        return res

    def __getattr__(self, name):
        # properties
        ## TODO: device, is_...??
        ## TODO: mH, mT, H, T, data, imag, real
        ## name ???
        if name in {
            "dtype",
            "grad",
            "grad_fn",
            "layout",
            "names",
            "ndim",
            "output_nr",
            "requires_grad",
            "retains_grad",
            "shape",
            "volatile",
        }:
            return getattr(self.metatensor, name)
        if name in {"size"}:
            return getattr(self.metatensor, name)
        # materializing with contiguous is needed for quantization
        if name in {"contiguous"}:
            return getattr(self._load_tensor(), name)

        raise AttributeError(f"{type(self)} does not have {name}")

    def __repr__(self):
        return f"NotYetLoadedTensor({repr(self.metatensor)})"


class LazyLoadingUnpickler(pickle.Unpickler):
    def __init__(self, file, zipfile_context):
        super().__init__(file)
        self.zipfile_context = zipfile_context

    def find_class(self, module, name):
        res = super().find_class(module, name)
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return functools.partial(NotYetLoadedTensor.rebuild_tensor_v2, archiveinfo=self)
        elif module == "torch._tensor" and name == "_rebuild_from_type_v2":
            return functools.partial(NotYetLoadedTensor.rebuild_from_type_v2, archiveinfo=self)
        elif module == "torch._utils" and name == "_rebuild_parameter":
            return functools.partial(NotYetLoadedTensor.rebuild_parameter, archiveinfo=self)
        return res

    def persistent_load(self, pid):
        name, cls, fn, device, size = pid
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s = torch.storage.TypedStorage(dtype=cls().dtype, device="meta")
        s.archiveinfo = pid
        return s


class lazy_load:
    def __init__(self, fn):
        self.zf = torch._C.PyTorchFileReader(str(fn))
        with BytesIO(self.zf.get_record("data.pkl")) as pkl:
            mup = LazyLoadingUnpickler(pkl, self)
            self.sd = mup.load()

    def __enter__(self):
        return self.sd

    def __exit__(self, exc_type, exc_val, exc_tb):
        del self.zf  # I don't think there is a way to force closing...
        self.zf = None


def check_valid_parrot_dir(checkpoint_dir: Path) -> None:
    if not checkpoint_dir.is_dir():
        raise OSError(f"`--checkpoint_dir {str(checkpoint_dir)!r} is not a directory.")
    
    if ((checkpoint_dir / "lit_model.pth").is_file()
        and (checkpoint_dir / "lit_config.json").is_file()):
        # we're good
        return

    raise OSError(
        f"`--checkpoint_dir {str(checkpoint_dir)!r} is not a valid Parrot checkpoint directory."
        " It must converted to Parron and contain the files: 'lit_model.pth', 'lit_config.json'."
        "\nPlease, follow the instructions at"
        " https://github.com/Lightning-AI/lit-parrot/blob/main/howto/download_stablelm.md\n"
    )

def check_valid_checkpoint_dir(checkpoint_dir: Path) -> None:
    if not checkpoint_dir.is_dir():
        raise OSError(f"`--checkpoint_dir {str(checkpoint_dir)!r} is not a directory.")
    if ( (checkpoint_dir / "tokenizer.json").is_file()
        and (checkpoint_dir / "tokenizer_config.json").is_file()
    ):
        # we're good
        return

    # list locally available checkpoints
    available = list(Path("checkpoints").glob("*/*"))
    if available:
        options = f"\n --checkpoint_dir ".join([""] + [repr(str(p.resolve())) for p in available])
        extra = f"\nYou have downloaded locally:{options}\n"
    else:
        extra = ""

    from lit_parrot.config import configs

    # list other possible checkpoints to download
    not_downloaded = [c for c in configs if not any(c in str(a) for a in available)]
    joined = "\n * ".join([""] + not_downloaded)
    supported = f"You can download:{joined}"

    raise OSError(
        f"`--checkpoint_dir {str(checkpoint_dir)!r} is not a valid checkpoint directory."
        " It must contain the files: 'lit_model.pth', 'lit_config.json', 'tokenizer.json' and 'tokenizer_config.json'."
        "\nPlease, follow the instructions at"
        " https://github.com/Lightning-AI/lit-parrot/blob/main/howto/download_stablelm.md\n"
        f"{extra}\n{supported}"
    )
