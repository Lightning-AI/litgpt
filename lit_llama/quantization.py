import os
from contextlib import contextmanager
import warnings

import torch

# configuration for bitsandbytes before import
os.environ["BITSANDBYTES_NOWELCOME"] = "1"
warnings.filterwarnings(
    "ignore", 
    message="MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization"
)
warnings.filterwarnings(
    "ignore", 
    message="The installed version of bitsandbytes was compiled without GPU support. 8-bit optimizers and GPU quantization are unavailable."
)
import bitsandbytes as bnb  # noqa: E402


class Linear8bitLt(bnb.nn.Linear8bitLt):
    """Wraps `bnb.nn.Linear8bitLt` and enables instantiation directly on the device and
    re-quantizaton when loading the state dict.
    
    
    This should only be used for inference. For training, use `bnb.nn.Linear8bitLt` directly.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, has_fp16_weights=False, threshold=6.0)
        # We quantize the initial weight here so we don't end up filling the device
        # memory with float32 weights which could lead to OOM.
        self._quantize_weight(self.weight.data)

    def _load_from_state_dict(self, local_state_dict, *args, **kwargs):
        # There is only one key that ends with `*.weight`, the other one is the bias
        weight_key = next(name for name in local_state_dict.keys() if name.endswith("weight"))

        # Load the weight from the state dict and re-quantize it
        weight = local_state_dict.pop(weight_key)
        self._quantize_weight(weight)

        # If there is a bias, let nn.Module load it
        if local_state_dict:
            super()._load_from_state_dict(local_state_dict, *args, **kwargs)
    
    def _quantize_weight(self, weight: torch.Tensor) -> None:
        # This code is taken and adapted from `bnb.nn.Int8Params.cuda()`
        B = weight.contiguous().half().cuda()
        CB, CBt, SCB, SCBt, coo_tensorB = bnb.functional.double_quant(B)
        del CBt
        del SCBt
        self.weight.data = CB
        setattr(self.weight, "CB", CB)
        setattr(self.weight, "SCB", SCB)


@contextmanager
def as_8_bit_quantized(device: torch.device, enabled: bool = True):
    """A context manager under which you can instantiate the model with 8-bit quantized tensors
    being created directly on the given device.
    """

    with torch.device(device):
        if not enabled:
            yield
            return

        if device.type != "cuda":
            raise ValueError("Quantization is only supported on the GPU.")

        torch_linear_cls = torch.nn.Linear
        torch.nn.Linear = Linear8bitLt
        yield
        torch.nn.Linear = torch_linear_cls
