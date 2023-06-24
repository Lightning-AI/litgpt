"""
Utility functions to extend the original GPT-Adapter method to GPT-Adapter v2,
This is a port from Lit-LLaMA based on the code prepared by @rasbt aka Sebastian Raschka
"""
from typing import Any

import torch

from lit_gpt.adapter import GPT


def adapter_filter(key: str, value: Any) -> bool:
    adapter_substrings = (
        # regular adapter v1 parameters
        "adapter_wte",
        "gating_factor",
        # adapter v2: new bias and scale used in Linear
        "adapter_scale",
        "adapter_bias",
        # adapter v2: Norm parameters are now trainable
        "norm_1",
        "norm_2",
        "ln_f",
    )
    return any(s in key for s in adapter_substrings)


def mark_only_adapter_v2_as_trainable(model: GPT) -> None:
    """Sets requires_grad=False for all non-adapter weights"""
    for name, param in model.named_parameters():
        param.requires_grad = adapter_filter(name, param)


def adapter_v2_new_forward(self, input: torch.Tensor) -> torch.Tensor:
    return self.adapter_scale * (torch.nn.functional.linear(input, self.weight, self.bias) + self.adapter_bias)


def adapter_v2_linear_with_bias_and_scale(layer):
    layer.adapter_bias = torch.nn.Parameter(
        torch.zeros(layer.weight.shape[0], dtype=layer.weight.dtype), requires_grad=False
    )
    layer.adapter_scale = torch.nn.Parameter(
        torch.ones(layer.weight.shape[0], dtype=layer.weight.dtype), requires_grad=False
    )
    bound_method = adapter_v2_new_forward.__get__(layer, layer.__class__)
    setattr(layer, "forward", bound_method)
    return layer


def add_adapter_v2_parameters_to_linear_layers(model):
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            adapter_v2_linear_with_bias_and_scale(module)
