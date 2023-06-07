"""
Utility functions to extend the original Parrot-Adapter method to Parrot-Adapter v2,
This is a port from Lit-LLaMA based on the code prepared by @rasbt aka Sebastian Raschka
"""
import contextlib
from functools import partial

import torch

from lit_parrot.adapter import Parrot


def get_adapter_substrings():
    return [
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
    ]


def mark_only_adapter_v2_as_trainable(model: Parrot) -> None:
    """Sets requires_grad=False for all non-adapter weights"""
    for name, param in model.named_parameters():
        param.requires_grad = any(s in name for s in get_adapter_substrings())


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


@contextlib.contextmanager
def adapter_state_only(module: torch.nn.Module):
    """Returns the model state dict with only the adapter weights for saving."""
    originals = {}
    substrings = get_adapter_substrings()

    def adapter_save(name, module, destination, prefix, keep_vars):
        if any(s in prefix for s in substrings) or "gating_factor" in module._parameters:
            original_fn = originals[name]
            return original_fn(destination, prefix, keep_vars)

    for name, submodule in module.named_modules():
        originals[name] = submodule._save_to_state_dict
        submodule._save_to_state_dict = partial(adapter_save, name, submodule)

    yield

    for name, module in module.named_modules():
        module._save_to_state_dict = originals[name]
