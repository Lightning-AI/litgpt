import math
from typing import Optional, Tuple

import torch
from torch.nn import functional as F

from litgpt.config import Config
from litgpt.utils import batched_index_select


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

    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.mask_cache: Optional[torch.Tensor] = None

    def set_seq_length(
        self,
        value: int,
        device: torch.device,
    ) -> None:
        if self.mask_cache is None or self.mask_cache.shape[-1] < value:
            self.mask_cache = build_mask_cache(
                value,
                sliding_window_size=self.config.sliding_window_size,
                device=device,
            )

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
                part of `mask_cache`.

        Returns:
            `attn_output, attn_weights`, where `attn_weights` is `None` if
            attention weights are not returned.

        """
        # We need the attention mask if there is sliding window attention,
        # or if `input_pos > 0` and T > 1.
        for_prefill = input_pos == 0
        is_causal = input_pos is None or for_prefill
        if not is_causal and token_positions is None:
            raise ValueError("token_positions must be given if input_pos > 0")
        apply_sliding_window_attention = (
            self.config.sliding_window_size is not None and block_idx % self.config.sliding_window_layer_stride == 0
        )
        B, _, T, _ = query.shape
        use_mask = apply_sliding_window_attention or (not is_causal and T > 1)
        mask = None
        if use_mask:
            # Special case requires building a mask. `mask_cache` is only needed
            # then.
            assert (
                self.mask_cache is not None
            ), "mask_cache must be given if sliding window attention is used, or if input_pos given and T > 1"
            if is_causal:
                mask = self.mask_cache[:T, :T].view(1, 1, T, T)
                is_causal = False
            else:
                # We need a mask if T > 1, since inference needs to be causal
                # for the new tokens
                mask = batched_index_select(
                    self.mask_cache[input_pos : (input_pos + T), :],
                    dim=1,
                    idx=token_positions,
                )
                # mask has shape (B, n_query_groups, T, kv_len), must have
                # shape (B, n_head, T, kv_len)
                nh_q = self.config.n_head
                nh_k = self.config.n_query_groups
                q_per_kv = nh_q // nh_k
                if q_per_kv > 1:
                    mask = mask.unsqueeze(2).expand(
                        -1, -1, q_per_kv, -1, -1
                    ).reshape(B, nh_q, T, -1)

        # Efficient attention using Flash Attention CUDA kernels.
        # NOTE: efficient implementation is disabled if `mask` is not None or softcapping is enabled.
        # ↓ (B, nh, T, hs) @ (B, nh, T, hs).mT --> (B, nh, T, T) @ (B, nh, T, hs) --> (B, nh, T, hs)
        return_scores = not (input_pos is None or for_prefill) and return_attn_weights
        y, scores = self.scaled_dot_product_attention(
            query,
            k_and_v,
            mask,
            is_causal,
            return_scores,
        )
        # Re-assemble all head outputs side by side.
        y = y.reshape(B, T, -1)
        return y, scores

    def scaled_dot_product_attention(
        self,
        query: torch.Tensor,
        k_and_v: KeysAndValues,
        mask: Optional[torch.Tensor] = None,
        is_causal: bool = True,
        return_scores: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        assert mask is None or not is_causal, "Cannot have mask and is_causal=True"
        scale = 1.0 / math.sqrt(self.config.attention_scores_scalar or self.config.head_size)

        # We cannot call PyTorch scaled_dot_product_attention if:
        # - Attention scores need to be returned; or
        # - Logit softcapping is required; or
        # - We cannot access keys and values from `k_and_v` in parallel (this
        #   never happens if `is_causal == True`)
        if return_scores or self.config.attention_logit_softcapping is not None or not k_and_v.both_in_parallel():
            y, scores = scaled_dot_product_attention(
                query=query,
                k_and_v=k_and_v,
                scale=scale,
                mask=mask,
                attention_logit_softcapping=self.config.attention_logit_softcapping,
                is_causal=is_causal,
            )
            if not return_scores:
                scores = None
        else:
            # We need `key` and `value` at the same time here. For the training
            # use case, this will be the case, since `k_and_v` is the default
            # in this case.
            key = k_and_v.keys()
            value = k_and_v.values()
            y = F.scaled_dot_product_attention(
                query=query,
                key=key,
                value=value,
                attn_mask=mask,
                dropout_p=0.0,
                scale=scale,
                is_causal=is_causal,
                enable_gqa=self.config.n_query_groups < self.config.n_head,
            )
            scores = None
        return y.transpose(1, 2), scores


def scaled_dot_product_attention(
    query: torch.Tensor,
    k_and_v: KeysAndValues,
    scale: float,
    mask: Optional[torch.Tensor] = None,
    attention_logit_softcapping: Optional[float] = None,
    is_causal: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    dtype = query.dtype
    key = k_and_v.keys()
    scores = _attention_compute_scores(query, key) * scale
    scores = do_softcapping(scores, attention_logit_softcapping)
    if mask is None and is_causal:
        T = query.shape[2]
        assert key.size(2) == T, "is_causal=True only if query, key have same size"
        mask = torch.ones(T, T, dtype=dtype, device=query.device).triu(diagonal=1)
        mask.masked_fill_(mask.bool(), torch.finfo(query.dtype).min)
        mask = mask.view(1, 1, T, T)
    if mask is not None:
        scores = scores + mask
    scores = F.softmax(scores, dim=-1, dtype=torch.float).to(dtype=dtype)
    value = k_and_v.values()
    return _attention_compute_weighted_values(scores, value), scores


def _attention_compute_scores(
    query: torch.Tensor,
    key: torch.Tensor,
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
        return query @ key_transposed
    else:
        assert q_per_kv > 1
        q_shape = query.shape[:1] + (nh_k, q_per_kv) + query.shape[2:]
        _query = query.view(*q_shape)
        key_transposed = key_transposed.unsqueeze(2)
        # At this point:
        # - _query: (bs, nh_k, q_per_kv, T_q, hs)
        # - key_transposed: (bs, nh_k, 1, hs, T_k)
        # - scores: (bs, nh_k, q_per_kv, T_q, T_k)
        scores = torch.matmul(_query, key_transposed)
        s_shape = query.shape[:-1] + (key.shape[2],)
        return scores.view(*s_shape)


def _attention_compute_weighted_values(
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


def build_mask_cache(
    max_seq_length: int, sliding_window_size: Optional[int], device: Optional[torch.device] = None
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
    # Usual causal mask:
    mask = torch.ones(max_seq_length, max_seq_length, device=device).triu(diagonal=1)
    mask.masked_fill_(mask.bool(), float("-inf"))
    if sliding_window_size is not None:
        sliding_window_bias = torch.ones_like(mask).tril(diagonal=-sliding_window_size)
        sliding_window_bias.masked_fill_(sliding_window_bias.bool(), float("-inf"))
        mask += sliding_window_bias
    return mask


def do_softcapping(x: torch.Tensor, thresh: Optional[float]) -> torch.Tensor:
    if thresh is not None:
        return torch.tanh(x / thresh) * thresh
    else:
        return x
