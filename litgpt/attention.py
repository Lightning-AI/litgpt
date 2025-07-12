# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import math
from typing import List, Optional, Tuple, Union

import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from litgpt.attention_utils import (
    attention_compute_scores,
    attention_compute_weighted_values,
    build_mask_cache,
    build_mask_slice,
    filter_sdpa_kernels,
)
from litgpt.config import Config

# Currently, `torch.nn.functional.scaled_dot_product_attention` does not
# properly support the case `enabla_gqa=True` (i.e., keys and values have
# less heads than queries). In this case, it is best to extend keys and
# values, which requires extra memory, but allows for efficient kernels to
# be used.
# Once PyTorch supports `enabla_gqa=True` properly at least with some fused
# kernels (such as flash attention), this flag can be switched to `False`.
FUSED_SDPA_DOES_NOT_SUPPORT_ENABLE_GQA = True


class KeysAndValues:
    """
    Object passed to :meth:`MultiHeadSelfAttention.__call__`. Allows to access
    keys or values, but (in general) not both at the same time. Implementations
    may use the same buffer to return them in the methods below.

    However, if :meth:`both_in_parallel` returns `True`, the tensors returned
    by :meth:`keys` and :meth:`values` may be used in parallel, since they are
    supported by separate buffers.

    """

    def keys(self) -> torch.Tensor:
        """
        Returns:
            keys tensor, shape `(eff_batch_size, n_query_groups, T, head_size)`,
            where `T <= cache_length` is the current cache length)

        """
        raise NotImplementedError()

    def values(self) -> torch.Tensor:
        """
        Returns:
            values tensor, shape `(eff_batch_size, n_query_groups, T, head_size)`,
            where `T <= cache_length` is the current cache length)

        """
        raise NotImplementedError()

    def both_in_parallel(self) -> bool:
        """
        Returns:
            Can use both `keys` and `values` in parallel? Otherwise, can only
            use one of them at the same time
        """
        return False


class DefaultKeysAndValues(KeysAndValues):
    def __init__(self, keys: torch.Tensor, values: torch.Tensor):
        # The final dimension of K and V can be different (in general)
        assert keys.shape[:-1] == values.shape[:-1] and keys.ndim == 4, (keys.shape, values.shape)
        self._keys = keys
        self._values = values

    def keys(self) -> torch.Tensor:
        return self._keys

    def values(self) -> torch.Tensor:
        return self._values

    def both_in_parallel(self) -> bool:
        """
        Keys and values are supported by different buffers, so they can be
        used at the same time.

        """
        return True


class MultiHeadSelfAttention:
    """
    Maintains code for the inner part of multi-head self-attention which is not
    parameterized. This is used both by :class:`CausalSelfAttention` and by the
    default KV cache implementation :class:`DefaultKVCache`.

    Kernels to be used for SDPA can be restricted by `sdpa_kernels`. By
    default, the choice is down to the method itself. If GPU memory is a
    concern (e.g., if MHA is used in training mode, to compute gradients),
    `sdpa_kernels=SDPBackend.EFFICIENT_ATTENTION` is recommended.

    If `sdpa_kernels` is used, their availabilities are checked upon the
    first call, and a warning is printed if some are not available.

    If `use_eager_sdpa_always=True`,
    `torch.nn.functional.scaled_dot_product_attention` is never used.

    """

    def __init__(
        self,
        config: Config,
        sdpa_kernels: Optional[Union[SDPBackend, List[SDPBackend]]] = None,
        use_eager_sdpa_always: bool = False,
    ) -> None:
        self.config = config
        self._sdpa_kernels = sdpa_kernels
        self._sdpa_kernels_filtered = False
        self.use_eager_sdpa_always = use_eager_sdpa_always

    @property
    def sdpa_kernels(self) -> Union[SDPBackend, List[SDPBackend]]:
        return self._sdpa_kernels if self._sdpa_kernels is not None else []

    def set_seq_length(
        self,
        value: int,
        device: torch.device,
    ) -> None:
        pass  # Currently, we don't use this

    def __call__(
        self,
        query: torch.Tensor,
        k_and_v: KeysAndValues,
        block_idx: int,
        input_pos: Optional[int] = None,
        return_attn_weights: bool = False,
        token_positions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """

        Args:
            query: Queries, shape `(batch_size, n_heads, q_len, head_size)`
            k_and_v: Access to keys and values, shape
                (batch_size, n_query_groups, kv_len, head_size)`
            block_idx: Index of block (or layer) in model
            input_pos: Position in input sequence. Defaults to 0
            return_attn_weights: If this is `True` and `input_pos > 0`, the
                attention weights (or scores) are returned as second argument
            token_positions: Required if `input_pos > 0`. Contains token
                positions in KV cache. This is needed to select the correct
                part of the mask matrix

        Returns:
            `attn_output, attn_weights`, where `attn_weights` is `None` if
            attention weights are not returned.

        """
        # We need the attention mask if there is sliding window attention
        for_prefill = input_pos == 0
        is_causal = input_pos is None or for_prefill
        if not is_causal and token_positions is None:
            raise ValueError("token_positions must be given if input_pos > 0")
        sliding_window_size = self._get_sliding_window_size(block_idx)
        B, _, T, _ = query.shape
        mask = None
        use_eager_sdpa = self._use_eager_sdpa(return_attn_weights, k_and_v)
        if use_eager_sdpa or sliding_window_size is not None or not is_causal:
            # Build attention mask
            mask_dtype = torch.float32 if use_eager_sdpa else query.dtype
            if is_causal:
                mask = (
                    build_mask_cache(
                        max_seq_length=T,
                        sliding_window_size=sliding_window_size,
                        dtype=mask_dtype,
                        device=query.device,
                    )
                    .view(1, 1, T, T)
                    .detach()
                )
            elif (not use_eager_sdpa) or T > 1:
                # We need a mask if T > 1, since inference needs to be causal
                # for the new tokens
                mask = build_mask_slice(
                    input_pos=input_pos,
                    num=T,
                    token_positions=token_positions,
                    n_head=self.config.n_head,
                    dtype=mask_dtype,
                    device=query.device,
                    sliding_window_size=sliding_window_size,
                ).detach()

        y, scores = self.scaled_dot_product_attention(
            query,
            k_and_v,
            mask,
            return_attn_weights,
        )
        # Re-assemble all head outputs side by side.
        y = y.reshape(B, T, -1)
        return y, scores

    def _get_sliding_window_size(self, block_idx: int) -> Optional[int]:
        apply_sliding_window_attention = (
            self.config.sliding_window_size is not None and self.config.sliding_window_indices[block_idx] == 1
        )
        return self.config.sliding_window_size if apply_sliding_window_attention else None

    def _use_eager_sdpa(
        self,
        return_attn_weights: bool,
        k_and_v: KeysAndValues,
    ) -> bool:
        return (
            return_attn_weights
            or self.use_eager_sdpa_always
            or self.config.attention_logit_softcapping is not None
            or not k_and_v.both_in_parallel()
        )

    def _filter_sdpa_kernels(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        dropout_p: float,
        is_causal: bool,
        enable_gqa: bool,
        **kwargs,
    ):
        if self._sdpa_kernels is not None and not self._sdpa_kernels_filtered:
            if isinstance(self._sdpa_kernels, list):
                kernels = self._sdpa_kernels
            else:
                kernels = [self._sdpa_kernels]
            new_kernels = filter_sdpa_kernels(
                sdpa_kernels=kernels,
                query=query,
                key=key,
                value=value,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                enable_gqa=enable_gqa,
            )
            self._sdpa_kernels = new_kernels if new_kernels else None
            self._sdpa_kernels_filtered = True

    def _get_scale_factor(self):
        return 1.0 / math.sqrt(self.config.attention_scores_scalar or self.config.head_size)

    def scaled_dot_product_attention(
        self,
        query: torch.Tensor,
        k_and_v: KeysAndValues,
        mask: Optional[torch.Tensor] = None,
        return_scores: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        scale = self._get_scale_factor()
        # We cannot call PyTorch scaled_dot_product_attention if:
        # - Attention scores need to be returned; or
        # - Logit softcapping is required; or
        # - We cannot access keys and values from `k_and_v` in parallel
        if self._use_eager_sdpa(return_scores, k_and_v):
            assert mask is not None or query.shape[2] == 1
            y, scores = scaled_dot_product_attention(
                query=query,
                k_and_v=k_and_v,
                scale=scale,
                mask=mask,
                attention_logit_softcapping=self.config.attention_logit_softcapping,
            )
            if not return_scores:
                scores = None
        else:
            # We need `key` and `value` at the same time here. For the training
            # use case, this will be the case, since `k_and_v` is the default
            # in this case.
            key = k_and_v.keys()
            value = k_and_v.values()
            is_causal = mask is None
            enable_gqa = self.config.n_query_groups < self.config.n_head
            if enable_gqa and FUSED_SDPA_DOES_NOT_SUPPORT_ENABLE_GQA:
                # Some efficient kernels have not implemented
                # `enabla_gqa=True`. It is better to extend keys, values in
                # this case.
                q_per_kv = self.config.n_head // self.config.n_query_groups
                key = key.repeat_interleave(q_per_kv, dim=1)
                value = value.repeat_interleave(q_per_kv, dim=1)
                enable_gqa = False
            kwargs = dict(
                query=query,
                key=key,
                value=value,
                attn_mask=mask,
                dropout_p=0.0,
                scale=scale,
                is_causal=is_causal,
                enable_gqa=enable_gqa,
            )
            self._filter_sdpa_kernels(**kwargs)
            if self._sdpa_kernels is not None:
                with sdpa_kernel(self._sdpa_kernels):
                    y = F.scaled_dot_product_attention(**kwargs)
            else:
                y = F.scaled_dot_product_attention(**kwargs)
            scores = None
        return y.transpose(1, 2), scores


def do_softcapping(x: torch.Tensor, thresh: Optional[float]) -> torch.Tensor:
    if thresh is not None:
        return torch.tanh(x / thresh) * thresh
    else:
        return x


def scaled_dot_product_attention(
    query: torch.Tensor,
    k_and_v: KeysAndValues,
    scale: float,
    mask: Optional[torch.Tensor] = None,
    attention_logit_softcapping: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    dtype = query.dtype
    key = k_and_v.keys().to(torch.float32)
    query = query.to(torch.float32)
    scores = attention_compute_scores(query, key) * scale
    scores = do_softcapping(scores, attention_logit_softcapping)
    if mask is not None:
        scores = scores + mask.to(torch.float32)
    scores = F.softmax(scores, dim=-1)
    value = k_and_v.values().to(torch.float32)
    return attention_compute_weighted_values(scores, value).to(dtype), scores.to(dtype)
