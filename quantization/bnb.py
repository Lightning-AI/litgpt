import os
from typing import Tuple

import torch.nn as nn

os.environ["BITSANDBYTES_NOWELCOME"] = "1"
import bitsandbytes as bnb  # noqa: E402


def quantize(model: nn.Module, threshold: float = 6.0, skip: Tuple[str, ...] = ()) -> nn.Module:
    for name, module in model.named_children():
        if isinstance(module, nn.Linear) and name not in skip:
            model._modules[name] = bnb.nn.Linear8bitLt(
                module.in_features, module.out_features, bias=module.bias, has_fp16_weights=False, threshold=threshold
            )

        if module.children():
            quantize(module, threshold=threshold, skip=skip)
    return model
