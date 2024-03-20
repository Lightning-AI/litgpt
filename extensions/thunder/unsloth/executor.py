# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
from typing import Optional, Tuple

import thunder
import thunder.torch as ltorch
from thunder.core.proxies import TensorProxy
from thunder.core.transforms import get_grad, mean_backward, put_grads
from thunder.extend import OperatorExecutor, register_executor
from thunder.torch import ne, sum, true_divide
from torch import Tensor

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
    "unsloth_cross_entropy",
    meta=unsloth_cross_entropy_meta,
    fn=extensions.lightning_thunder.unsloth.kernels.cross_entropy_loss._cross_entropy_forward_impl,
)


def unsloth_cross_entropy_backward_impl(dlosses: Tensor, logits: Tensor, labels: Tensor, logsumexp: Tensor) -> Tensor:
    # clone() because the kernel writes the grads in the logits.
    # If it works, we can remove this it, but it's not a thing we generally anticipate and support right now.
    return extensions.lightning_thunder.unsloth.kernels.cross_entropy_loss._cross_entropy_backward_impl(
        dlosses, logits.clone(), logsumexp, labels
    )


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

The RMSNorm kernel is not integrated because it's not numerically equal and it doesn't compute the gradient for the
weight, just for the input.
"""


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
