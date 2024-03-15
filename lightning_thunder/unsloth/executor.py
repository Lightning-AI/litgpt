# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
from typing import Optional, Tuple

import thunder
import thunder.torch as ltorch
import torch
from thunder.core.proxies import TensorProxy
from thunder.core.transforms import get_grad, mean_backward, put_grads
from thunder.extend import OperatorExecutor, register_executor
from thunder.torch import ne, sum, true_divide

import lightning_thunder.unsloth.kernels as kernels
import litgpt

unsloth_ex = OperatorExecutor("unsloth_ex", version="0.1")
register_executor(unsloth_ex)


"""
====================
 Cross Entropy Loss
====================
"""


def unsloth_cross_entropy_meta(logits: TensorProxy, labels: TensorProxy) -> Tuple[TensorProxy, TensorProxy]:
    return (
        thunder.TensorProxy(
            shape=(logits.shape[0],),
            # the cross entropy kernel only supports float32
            dtype=thunder.dtypes.float32,
            device=logits.device,
            requires_grad=logits.requires_grad,
        ),
        thunder.TensorProxy(
            shape=(logits.shape[0],), dtype=thunder.dtypes.float32, device=logits.device, requires_grad=False
        ),
    )


unsloth_cross_entropy = unsloth_ex.register_operator(
    "unsloth_cross_entropy", meta=unsloth_cross_entropy_meta, fn=kernels.cross_entropy_loss._cross_entropy_forward_impl
)


def unsloth_cross_entropy_backward_impl(dlosses: torch.Tensor, logits: torch.Tensor, labels: torch.Tensor, logsumexp: torch.Tensor) -> torch.Tensor:
    # clone() because the kernel writes the grads in the logits.
    # If it works, we can remove this it, but it's not a thing we generally anticipate and support right now.
    return kernels.cross_entropy_loss._cross_entropy_backward_impl(dlosses, logits.clone(), logsumexp, labels)


def unsloth_cross_entropy_backward_meta(dlosses: TensorProxy, logits: TensorProxy, logsumexp: TensorProxy, labels: TensorProxy) -> TensorProxy:
    return thunder.TensorProxy(like=logits)


unsloth_cross_entropy_backward = unsloth_ex.register_operator(
    "unsloth_cross_entropy_backward", meta=unsloth_cross_entropy_backward_meta, fn=unsloth_cross_entropy_backward_impl
)


def unsloth_cross_entropy_checker(
    logits: TensorProxy,
    labels: TensorProxy,
    weight: Optional[TensorProxy] = None,
    size_average: Optional[bool] = None,
    ignore_index: int = -100,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
    label_smoothing: float = 0.0,
) -> bool:
    return (
        weight is None
        and size_average is None
        and reduce is None
        and reduction in ("none", "mean")
        and ignore_index == -100
        and label_smoothing == 0.0
        and logits.device.type == "cuda"
        and labels.device.type == "cuda"
    )


def cross_entropy_to_unsloth(
    logits: TensorProxy,
    labels: TensorProxy,
    weight: Optional[TensorProxy] = None,
    size_average: Optional[bool] = None,
    ignore_index: int = -100,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
    label_smoothing: float = 0.0,
) -> Tuple[TensorProxy, TensorProxy]:
    loss, logsumexp = unsloth_cross_entropy(logits, labels)
    if reduction == "mean":
        # "mean" reduction is not part of the kernel
        # TODO: this doesn't consider that all elements could be masked, causing a division by 0
        n_items = sum(ne(labels, -100))
        loss = true_divide(sum(loss), n_items)
    elif reduction != "none":
        raise NotImplementedError(reduction)
    return loss, logsumexp


def unsloth_cross_entropy_grad(
    logits: TensorProxy,
    labels: TensorProxy,
        weight: Optional[TensorProxy] = None,
        size_average: Optional[bool] = None,
        ignore_index: int = -100,
        reduce: Optional[bool] = None,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
) -> Tuple[TensorProxy, TensorProxy]:
    loss, logsumexp = cross_entropy_to_unsloth(**locals())

    grad = get_grad(loss)

    if reduction == "mean":
        grad = mean_backward(logsumexp.ndim, logsumexp.shape, (0,), grad)

    logits_grad = unsloth_cross_entropy_backward(grad, logits, labels, logsumexp)
    put_grads((logits,), (logits_grad,))

    return loss


# registers as cross entropy implementation, including the execution transform and now a grad transform
unsloth_ex.register_implementation(
    ltorch.cross_entropy,
    checker=unsloth_cross_entropy_checker,
    execution_transform=lambda *args: cross_entropy_to_unsloth(*args)[0],
    grad_transform=unsloth_cross_entropy_grad,
)


"""
=========
 RMSNorm
=========
"""

def rmsnorm_meta(x: TensorProxy, weight: TensorProxy, dim: int, eps: float, add_unit_offset: bool) -> TensorProxy:
    return TensorProxy(like=x)

# There's no `register_operator` for Modules and RMSNorm is not a torch symbol that we can register to. For now some
# duplication and monkey patching is required
def rmsnorm_forward(x: torch.Tensor, weight: torch.Tensor, dim: int, eps: float, add_unit_offset: bool) -> torch.Tensor:
    dtype = x.dtype
    x = x.float()
    # NOTE: the original RMSNorm paper implementation is not equivalent
    norm_x = (x * x).mean(dim=dim, keepdim=True)
    x_normed = x * (norm_x + eps).rsqrt()
    x_normed = x_normed.to(dtype=dtype)
    if add_unit_offset:
        # Gemma model requires a unit offset
        # https://github.com/google/gemma_pytorch/blob/main/gemma/model.py#L176
        return x_normed * (1 + weight)
    return x_normed * weight

litgpt_rmsnorm = unsloth_ex.register_operator('litgpt_rmsnorm', meta=rmsnorm_meta, fn=rmsnorm_forward)

from litgpt.model import RMSNorm as OriginalRMSNorm


class ThunderRMSNorm(OriginalRMSNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fn = litgpt_rmsnorm if thunder.core.interpreter.is_jitting() else rmsnorm_forward
        return fn(x, self.weight, self.dim, self.eps, self.add_unit_offset)

litgpt.model.RMSNorm = ThunderRMSNorm

def rmsnorm_forward_impl(X, W, eps):
    shape = X.shape
    dim = shape[-1]
    X = X.view(-1, dim)
    n_rows, n_cols = X.shape
    BLOCK_SIZE, num_warps = kernels.calculate_settings(n_cols)

    Y = torch.empty((n_rows, n_cols), dtype = X.dtype, device = "cuda")
    r = torch.empty(n_rows, dtype = torch.float32, device = "cuda")

    kernels._rms_layernorm_forward[(n_rows,)](
        Y, Y.stride(0),
        X, X.stride(0),
        W, W.stride(0),
        r, r.stride(0),
        n_cols, eps,
        BLOCK_SIZE = BLOCK_SIZE,
        num_warps  = num_warps,
    )
    return Y.view(*shape), (r, BLOCK_SIZE, num_warps)

def rmsnorm_forward_meta(X, W, eps):
    n_cols = X.shape[-1]
    n_rows = 1
    for i in X.shape[:-1]:
        n_rows *= i
    BLOCK_SIZE, num_warps = kernels.calculate_settings(n_cols)
    return (TensorProxy(like=X),
            (TensorProxy(shape=(n_rows,), device=X.device, dtype=thunder.dtypes.float32, requires_grad=False),
             BLOCK_SIZE,
             num_warps,
            )
           )

def rmsnorm_backward_impl(X, W, r, eps, BLOCK_SIZE, num_warps, dY):
    shape = dY.shape
    dim = shape[-1]
    dY = dY.view(-1, dim)
    n_rows, n_cols = dY.shape
    dW = X

    kernels._rms_layernorm_backward[(n_rows,)](
        dY, dY.stride(0),
        X,  X .stride(0),
        W,  W .stride(0),
        r,  r .stride(0),
        dW, dW.stride(0),
        n_cols, eps,
        GEMMA      = False,
        BLOCK_SIZE = BLOCK_SIZE,
        num_warps  = num_warps,
    )
    dX = dY.view(*shape)
    return dX

def rmsnorm_backward_meta(X, W, r, eps, BLOCK_SIZE, num_warps, dY):
    return TensorProxy(like=dY)


unsloth_rmsnorm_forward = unsloth_ex.register_operator('unsloth_rmsnorm_forward', meta=rmsnorm_forward_meta, fn=rmsnorm_forward_impl)
unsloth_rmsnorm_backward = unsloth_ex.register_operator('unsloth_rmsnorm_backward', meta=rmsnorm_backward_meta, fn=rmsnorm_backward_impl)


def rmsnorm_to_unsloth(x: TensorProxy, weight: TensorProxy, dim: int, eps: float, add_unit_offset: bool):
    assert dim == -1 and not add_unit_offset
    res, _ = unsloth_rmsnorm_forward(x, weight, eps)
    return res

def rmsnorm_to_unsloth_checker(x: TensorProxy, weight: TensorProxy, dim: int, eps: float, add_unit_offset: bool):
    if dim != -1 or add_unit_offset:
        return False
    return x.device.devicetype == thunder.devices.DeviceType.CUDA and weight.device.devicetype == thunder.devices.DeviceType.CUDA

unsloth_ex.register_implementation(litgpt_rmsnorm, checker=rmsnorm_to_unsloth_checker, execution_transform=rmsnorm_to_unsloth)



"""
========
 SwiGLU
========
"""


# FIXME


"""
======
 RoPE
======
"""


# FIXME
