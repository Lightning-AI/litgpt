# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

from typing import List, Optional

import torch
from torch.backends.cuda import (
    can_use_cudnn_attention,
    can_use_efficient_attention,
    can_use_flash_attention,
)
from torch.nn.attention import SDPAParams, SDPBackend


def filter_sdpa_kernels(
    sdpa_kernels: List[SDPBackend],
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor],
    dropout_p: float,
    is_causal: bool,
    enable_gqa: bool,
    **kwargs,
) -> List[SDPBackend]:
    params = SDPAParams(query, key, value, attn_mask, dropout_p, is_causal, enable_gqa)
    new_kernels = []
    for kernel in sdpa_kernels:
        if kernel == SDPBackend.FLASH_ATTENTION and not can_use_flash_attention(params):
            continue
        elif kernel == SDPBackend.EFFICIENT_ATTENTION and not can_use_efficient_attention(params):
            continue
        elif kernel == SDPBackend.CUDNN_ATTENTION and not can_use_cudnn_attention(params):
            continue
        new_kernels.append(kernel)
    return new_kernels


def attention_compute_scores(
    query: torch.Tensor,
    key: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    assert query.ndim == key.ndim == 4
    assert query.shape[0] == key.shape[0] and query.shape[3] == key.shape[3]
    nh_q = query.shape[1]
    nh_k = key.shape[1]
    assert nh_q % nh_k == 0
    # - query: (bs, nh_q, T_q, hs)
    # - key: (bs, nh_k, T_k, hs)
    q_per_kv = nh_q // nh_k
    key_transposed = key.mT  # (bs, nh_k, hs, T_k)
    if q_per_kv == 1:
        out = torch.matmul(query, key_transposed, out=out)
    else:
        assert q_per_kv > 1
        q_shape = query.shape[:1] + (nh_k, q_per_kv) + query.shape[2:]
        _query = query.view(*q_shape)
        key_transposed = key_transposed.unsqueeze(2)
        # At this point:
        # - _query: (bs, nh_k, q_per_kv, T_q, hs)
        # - key_transposed: (bs, nh_k, 1, hs, T_k)
        # - scores: (bs, nh_k, q_per_kv, T_q, T_k)
        if out is not None:
            out = out.view(_query.shape[:-1] + (key.shape[2],))
        out = torch.matmul(_query, key_transposed, out=out)
        s_shape = query.shape[:-1] + (key.shape[2],)
        out = out.view(*s_shape)
    return out


def attention_compute_weighted_values(
    scores: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    assert scores.ndim == value.ndim == 4
    assert scores.shape[0] == scores.shape[0] and scores.shape[3] == value.shape[2]
    nh_q = scores.shape[1]
    nh_k = value.shape[1]
    assert nh_q % nh_k == 0
    # - scores: (bs, nh_q, T_q, T_k)
    # - value: (bs, nh_k, T_k, hs)
    q_per_kv = nh_q // nh_k
    if q_per_kv == 1:
        return scores @ value
    else:
        s_shape = scores.shape[:1] + (nh_k, q_per_kv) + scores.shape[2:]
        _scores = scores.view(*s_shape)
        _value = value.unsqueeze(2)
        # At this point:
        # - _scores: (bs, nh_k, q_per_kv, T_q, T_k)
        # - _value: (bs, nh_k, 1, T_k, hs)
        # - result: (bs, nh_k, q_per_kv, T_q, hs)
        result = torch.matmul(_scores, _value)
        r_shape = scores.shape[:-1] + (value.shape[-1],)
        return result.view(*r_shape)


def minus_infinity(dtype: torch.dtype) -> float:
    return torch.finfo(dtype).min


def mask_cache_bool(
    max_seq_length: int,
    sliding_window_size: Optional[int],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    # Usual causal mask:
    mask = torch.ones(
        max_seq_length,
        max_seq_length,
        device=device,
        dtype=dtype,
    ).triu(diagonal=1)
    if sliding_window_size is not None:
        mask += torch.ones_like(mask).tril(diagonal=-sliding_window_size)
    return mask


def build_mask_cache(
    max_seq_length: int,
    sliding_window_size: Optional[int],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
          Global Window              Sliding window             Sliding window
          attention mask      +            bias          =      attention mask
    ┌────────────────────────┐  ┌───────────────────────┐  ┌─────────────────────────┐
    │ True False False False │  │ True  True  True True │  │ True  False False False │
    │ True True  False False │  │ True  True  True True │  │ True  True  False False │
    │ True True  True  False │  │ False True  True True │  │ False True  True  False │
    │ True True  True  True  │  │ False False True True │  │ False False True  True  │
    └────────────────────────┘  └───────────────────────┘  └─────────────────────────┘
    """
    mask = mask_cache_bool(max_seq_length, sliding_window_size, device, dtype)
    mask.masked_fill_(mask.bool(), minus_infinity(dtype))
    return mask


def mask_slice_bool(
    input_pos: int,
    num: int,
    token_positions: torch.Tensor,
    n_head: int,
    device: torch.device,
    sliding_window_size: Optional[int] = None,
) -> torch.Tensor:
    # Build boolean mask, then map False -> 0, True -> -infty
    # If (i, j) indexes the complete (seq_len, seq_len) mask matrix,
    # causality is given by I(i < j). If `sliding_window_size` is given,
    # this translates to I(i >= j + sws) if sws = sliding_window_size.
    assert token_positions.ndim == 3
    tp_dtype = token_positions.dtype
    batch_size, n_query_groups, _ = token_positions.shape
    assert n_head % n_query_groups == 0 and n_head >= n_query_groups
    token_positions = (
        token_positions.to(device=device)
        .unsqueeze(2)
        .expand(
            -1,
            -1,
            num,
            -1,
        )
    )
    kwargs = dict(device=device, dtype=tp_dtype)
    bool_mask = (
        torch.arange(
            input_pos,
            input_pos + num,
            **kwargs,
        )
        .view(1, 1, -1, 1)
        .expand_as(token_positions)
        < token_positions
    )
    if sliding_window_size is not None:
        extra_mask = (
            torch.arange(
                input_pos - sliding_window_size,
                input_pos + num - sliding_window_size,
                **kwargs,
            )
            .view(1, 1, -1, 1)
            .expand_as(token_positions)
            >= token_positions
        )
        bool_mask |= extra_mask
    if n_head != n_query_groups:
        q_per_kv = n_head // n_query_groups
        bool_mask = (
            bool_mask.unsqueeze(2)
            .expand(
                -1,
                -1,
                q_per_kv,
                -1,
                -1,
            )
            .reshape(batch_size, n_head, num, -1)
        )
    return bool_mask


def build_mask_slice(
    input_pos: int,
    num: int,
    token_positions: torch.Tensor,
    n_head: int,
    dtype: torch.dtype,
    device: torch.device,
    sliding_window_size: Optional[int] = None,
) -> torch.Tensor:
    """
    Returns mask for case `input_pos > 0` in :class:`MultiHeadSelfAttention`.

    Args:
        input_pos: Position in input sequence, must be positive
        num: Length of query argument `q_len`
        token_positions: Token positions in KV cache, shape
            `(eff_batch_size, n_query_groups, cache_length)`
        n_head: Number of attention heads, must be multiple of
            `n_query_groups`
        dtype: Data type of the output mask
        device: Device of the output mask
        sliding_window_size: Size of sliding window (if any)

    Returns:
        Mask tensor, shape `(eff_batch_size, n_head, num, cache_length)`

    """
    bool_mask = mask_slice_bool(
        input_pos,
        num,
        token_positions,
        n_head,
        device,
        sliding_window_size,
    )
    mask = torch.zeros(bool_mask.shape, dtype=dtype, device=device)
    mask.masked_fill_(bool_mask, minus_infinity(dtype))
    return mask
