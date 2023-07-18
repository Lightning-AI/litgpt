import os
import warnings

import torch
from lightning_utilities.core.imports import RequirementCache

# configuration for bitsandbytes before import
os.environ["BITSANDBYTES_NOWELCOME"] = "1"
warnings.filterwarnings("ignore", message=r".*bitsandbytes was compiled without GPU support.*")
_BITSANDBYTES_AVAILABLE = RequirementCache("bitsandbytes>=0.40.0")

if _BITSANDBYTES_AVAILABLE:
    warnings.filterwarnings(
        "ignore", message=r"MatMul8bitLt: inputs will be cast from .* to float16 during quantization"
    )
    import bitsandbytes as bnb

    class InferenceLinear8bitLt(bnb.nn.Linear8bitLt):
        """Wraps `bnb.nn.Linear8bitLt` and enables instantiation directly on the device and
        re-quantizaton when loading the state dict.


        This should only be used for inference. For training, use `bnb.nn.Linear8bitLt` directly.
        """

        def __init__(self, *args, **kwargs):
            super().__init__(
                *args,
                **kwargs,
                # true only for fine-tuning as the weights do not have to be converted back and forth for bwd.
                has_fp16_weights=False,
                threshold=6.0,
            )
            # We quantize the initial weight here so we don't end up filling the device
            # memory with float32 weights which could lead to OOM.
            self._quantize_weight(self.weight.data)

        def _load_from_state_dict(self, local_state_dict, *args, **kwargs):
            # There is only one key that ends with `*.weight`, the other one is the bias
            weight_key = next((name for name in local_state_dict if name.endswith("weight")), None)
            if weight_key is None:
                return

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

    class Linear4bit(bnb.modules.Linear4bit):
        def __init__(self, *args, device=None, **kwargs):
            super().__init__(*args, **kwargs)
            if device is None:
                device = torch.tensor(0).device
            if device.type == "cuda":
                # weight needs to be moved manually because it doesn't work with device as a context manager:
                # `weight.to()` gets skipped if `weight.data` is already on CUDA. we avoid it by moving it back
                # (inefficient). see condition:
                # https://github.com/TimDettmers/bitsandbytes/blob/817bdf6/bitsandbytes/nn/modules.py#L177
                self.weight.data = self.weight.data.to("cpu")
                warnings.filterwarnings("ignore", message=r".*Fabric.setup\(\)` has parameters on different devices.*")
                # we could manually move `self.weight.to(device)` here but that breaks checkpoint loading
                # bnb expects that the layers are moved to the device after loading

else:

    def __getattr__(name):
        raise ModuleNotFoundError(str(_BITSANDBYTES_AVAILABLE))
