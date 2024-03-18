# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
from functools import reduce
from typing import Optional, Tuple

import thunder
import thunder.torch as ltorch
from thunder.core.proxies import TensorProxy
from thunder.core.transforms import get_grad, mean_backward, put_grads
from thunder.extend import OperatorExecutor, register_executor
from thunder.torch import ne, sum, true_divide
from torch import Tensor

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
        TensorProxy(
            shape=(logits.shape[0],),
            # the cross entropy kernel only supports float32
            dtype=thunder.dtypes.float32,
            device=logits.device,
            requires_grad=logits.requires_grad,
        ),
        TensorProxy(shape=(logits.shape[0],), dtype=thunder.dtypes.float32, device=logits.device, requires_grad=False),
    )


unsloth_cross_entropy = unsloth_ex.register_operator(
    "unsloth_cross_entropy", meta=unsloth_cross_entropy_meta, fn=kernels.cross_entropy_loss._cross_entropy_forward_impl
)


def unsloth_cross_entropy_backward_impl(dlosses: Tensor, logits: Tensor, labels: Tensor, logsumexp: Tensor) -> Tensor:
    # clone() because the kernel writes the grads in the logits.
    # If it works, we can remove this it, but it's not a thing we generally anticipate and support right now.
    return kernels.cross_entropy_loss._cross_entropy_backward_impl(dlosses, logits.clone(), logsumexp, labels)


def unsloth_cross_entropy_backward_meta(
    dlosses: TensorProxy, logits: TensorProxy, logsumexp: TensorProxy, labels: TensorProxy
) -> TensorProxy:
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
) -> TensorProxy:
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


def rms_norm_meta(x: TensorProxy, weight: TensorProxy, dim: int, eps: float, add_unit_offset: bool) -> TensorProxy:
    return TensorProxy(like=x)


# There's no `register_operator` for Modules and rms_norm is not a torch symbol that we can register to. For now some
# duplication and monkey patching is required
def rms_norm_forward(x: Tensor, weight: Tensor, dim: int, eps: float, add_unit_offset: bool) -> Tensor:
    dtype = x.dtype
    x = x.float()
    # NOTE: the original rms_norm paper implementation is not equivalent
    norm_x = (x * x).mean(dim=dim, keepdim=True)
    x_normed = x * (norm_x + eps).rsqrt()
    x_normed = x_normed.to(dtype=dtype)
    if add_unit_offset:
        # Gemma model requires a unit offset
        # https://github.com/google/gemma_pytorch/blob/main/gemma/model.py#L176
        return x_normed * (1 + weight)
    return x_normed * weight


litgpt_rms_norm = unsloth_ex.register_operator("litgpt_rms_norm", meta=rms_norm_meta, fn=rms_norm_forward)

from litgpt.model import RMSNorm as OriginalRMSNorm


class ThunderRMSNorm(OriginalRMSNorm):
    def forward(self, x: Tensor) -> Tensor:
        fn = litgpt_rms_norm if thunder.core.interpreter.is_jitting() else rms_norm_forward
        return fn(x, self.weight, self.dim, self.eps, self.add_unit_offset)


litgpt.model.RMSNorm = ThunderRMSNorm


def unsloth_rms_norm_forward_meta(
    X: TensorProxy, W: TensorProxy, eps: float, gemma: bool
) -> Tuple[TensorProxy, TensorProxy]:
    Y = TensorProxy(like=X)
    # cannot use `.view` so reduce the dimension manually
    # n_rows = X.view(-1, X.shape[-1]).shape[0]
    n_rows = reduce(int.__mul__, X.shape[:-1])
    r = TensorProxy(shape=(n_rows,), dtype=thunder.dtypes.float32, device=X.device, requires_grad=False)
    return Y, r


def unsloth_rms_norm_backward_meta(X, W, r, eps, gemma, dY):
    return TensorProxy(like=dY)


unsloth_rms_norm_forward = unsloth_ex.register_operator(
    "unsloth_rms_norm_forward",
    meta=unsloth_rms_norm_forward_meta,
    fn=lambda *args: kernels._rms_layernorm_forward_impl(*args),
)
unsloth_rms_norm_backward = unsloth_ex.register_operator(
    "unsloth_rms_norm_backward", meta=unsloth_rms_norm_backward_meta, fn=kernels._rms_layernorm_backward_impl
)


def rms_norm_to_unsloth(
    x: TensorProxy, weight: TensorProxy, dim: int, eps: float, add_unit_offset: bool
) -> Tuple[TensorProxy, TensorProxy]:
    return unsloth_rms_norm_forward(x, weight, eps, add_unit_offset)


def rms_norm_to_unsloth_checker(
    x: TensorProxy, weight: TensorProxy, dim: int, eps: float, add_unit_offset: bool
) -> bool:
    return dim == -1 and x.device.type == "cuda" and weight.device.type == "cuda"


def unsloth_rms_norm_grad(
    x: TensorProxy, weight: TensorProxy, dim: int, eps: float, add_unit_offset: bool
) -> TensorProxy:
    Y, r = rms_norm_to_unsloth(**locals())

    dY = get_grad(Y)

    dY_grad = unsloth_rms_norm_backward(x, weight, r, eps, add_unit_offset, dY)
    # the kernel puts dX in dY
    put_grads((x,), (dY_grad,))

    return Y


unsloth_ex.register_implementation(
   litgpt_rms_norm,
   checker=rms_norm_to_unsloth_checker,
   execution_transform=lambda *args: rms_norm_to_unsloth(*args)[0] ,
   grad_transform=unsloth_rms_norm_grad,
)


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
