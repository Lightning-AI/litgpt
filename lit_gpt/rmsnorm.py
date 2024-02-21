# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import torch


class RMSNorm(torch.nn.Module):
    """Root Mean Square Layer Normalization.

    Derived from https://github.com/bzhangGo/rmsnorm/blob/master/rmsnorm_torch.py. BSD 3-Clause License:
    https://github.com/bzhangGo/rmsnorm/blob/master/LICENSE.
    """

    # TODO: make `add_unit_offset` be dependent by a config
    # def __init__(self, size: int, dim: int = -1, eps: float = 1e-5, add_unit_offset: bool = True) -> None:
    def __init__(self, size: int, dim: int = -1, eps: float = 1e-6, add_unit_offset: bool = True) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(size))
        self.eps = eps
        self.dim = dim
        self.add_unit_offset = add_unit_offset

    # NOTE: output now closer to the official gemma implementation
    # https://github.com/google/gemma_pytorch/blob/ca890c7abaa41ce7ab0eeda9aa8a52c0796b3a16/gemma/model.py#L170-L179
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        # NOTE: the original RMSNorm paper implementation is not equivalent
        norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        x_normed = x_normed.to(dtype=dtype)
        return x_normed * (self.add_unit_offset + self.weight)

    def reset_parameters(self) -> None:
        torch.nn.init.ones_(self.weight)
