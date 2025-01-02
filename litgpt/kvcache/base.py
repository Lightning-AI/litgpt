from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

from litgpt.attention import (
    DefaultKeysAndValues,
    KeysAndValues,
    MultiHeadSelfAttention,
)
from litgpt.config import Config


@dataclass(frozen=True)
class KVCacheParams:
    batch_size: int
    n_query_groups: int
    cache_length: int
    head_size: int
    n_head: int
    device: Optional[torch.device]
    dtype: Optional[torch.dtype]

    @staticmethod
    def from_config(
        config: Config,
        batch_size: int,
        cache_length: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        head_size: Optional[int] = None,
    ) -> "KVCacheParams":
        if head_size is None:
            head_size = config.n_embd // config.n_head
        return KVCacheParams(
            batch_size=batch_size,
            n_query_groups=config.n_query_groups,
            cache_length=cache_length,
            head_size=head_size,
            n_head=config.n_head,
            device=device,
            dtype=dtype,
        )


class KVCache(torch.nn.Module):
    """
    Base class for key-value (KV) caches.

    Buffers have shapes
    `(batch_size, config.n_query_groups, cache_length, head_size)`, where
    `head_size` is a parameter. Caching can be used for
    batch size `1 <= eff_batch_size <= batch_size`, which is determined in
    prefill calls (`input_pos=0`) of :meth:`forward`.

    Note: In general, query and key tensors need to be position-encoded
    (e.g., RoPE).

    """

    def __init__(
        self,
        config: Config,
        batch_size: int,
        cache_length: int,
        block_idx: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        head_size: Optional[int] = None,
    ):
        """
        Note that `batch_size` is the maximum batch size the cache can be used
        with. The effective batch size is determined when calling
        :meth:`forward` with `input_pos=0`, and can change with any such prefill
        call. If this is smaller than `batch_size`, then in general only parts
        of the buffers are used.

        Args:
            config: Model config
            batch_size: Inference batch size (maximum)
            cache_length: Number of slots in cache
            block_idx: Index of model block (or layer). Multi-head attention
                needs to know this.
            device: Device for buffers
            dtype: Data type for buffers
            head_size: Size of final dimension of buffers. Defaults to head
                size of model
        """
        super().__init__()
        if cache_length <= 0:
            raise ValueError("cache_length must be positive")
        self.batch_size = batch_size
        self._n_query_groups = config.n_query_groups
        self._cache_length = cache_length
        if head_size is None:
            head_size = config.head_size
        self.head_size = head_size
        self.n_head = config.n_head
        self._device = device
        self._dtype = dtype
        self.block_idx = block_idx
        # TODO: Remove once HuggingFace bug is fixed
        # https://github.com/huggingface/transformers/issues/35233
        # https://github.com/huggingface/transformers/pull/35901
        self._work_around_hf_bug = config.rope_n_elem == 1

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def cache_length(self) -> Optional[int]:
        return self._cache_length

    @property
    def n_query_groups(self) -> int:
        return self._n_query_groups

    @property
    def next_token_pos(self) -> Optional[int]:
        """
        Returns:
            Input position for next token to be generated, or `None` if cache
            has not been initialized yet (call of :meth:`prefill`).
        """
        raise NotImplementedError()

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        token_idx: torch.Tensor,
        input_pos: int,
    ) -> torch.Tensor:
        """
        Given query, key, value tensors, this method extends the KV cache with
        `key`, `value`, then computes multi-head self attention. There are two
        cases:

        * Prefill (`input_pos == 0`): Starts a generation loop by passing key
          and value tensors. The KV cache is reset. The length must be
          `num <= max_prefill_length`. The effective batch size must be
          `eff_batch_size <= batch_size`. This batch size is then fixed for
          subsequent calls of :meth:`forward`.
        * Update (`input_pos > 0`): Continues a generation loop (or processing
          of large prompt). The length must be `num <= max_tokens_forward`.

        If the cache makes eviction decisions based on scores which require
        attention weights, scores for the next :meth:`forward` call need to
        be computed here.

        If a sequence is generated token by token, updates always use `num=1`.
        The case `num > 1` arises if large prompts are to be ingested with more
        than `max_prefill_length` tokens. Note that if the cache makes eviction
        decisions by scoring in :meth:`update`, then large `num` may lead to
        worse decisions. On the other hand, ingesting prompts with larger `num`
        is faster.

        Args:
            query: New queries,
                `(eff_batch_size, n_query_groups, num, head_size)`. Here,
                `num <= max_tokens_forward` if `input_pos > 0`, and
                `num <= max_prefill_length` if `input_pos == 0`. Must be
                position encoded.
            key: New keys, `(eff_batch_size, n_query_groups, num, head_size)`.
                Must be position encoded.
            value: New values, `(eff_batch_size, n_query_groups, num, head_size)`
            token_idx: Token indices of input sequence, `(eff_batch_size, num)`.
                Some KV caches make use of this information.
            input_pos: Token position of the new chunk in the full input
                sequence.

        Returns:
            Multi-head self-attention outputs before final linear map,
            `(eff_batch_size, n_head, num, head_size)`

        """
        raise NotImplementedError()

    def get_keys_values(self) -> Optional[KeysAndValues]:
        """
        Returns:
            :class:`KeysAndValues` object, providing access to currently stored
            keys and values tensors. If the cache is empty or has not been
            initialized, `None` is returned.

        """
        raise NotImplementedError()

    @property
    def max_tokens_forward(self) -> int:
        """
        Note that this limit may change during the course of the generation
        for certain caches.

        Returns:
            Maximum number of token positions which can be treated in
            :meth:`forward` with `input_pos > 0`. Depends on cache, but is
            `<= cache_length`

        """
        raise NotImplementedError()

    @property
    def max_prefill_length(self) -> Optional[int]:
        """
        Returns:
            Maximum sequence length for `key`, `value` tensors passed to
            :meth:`forward` if `input_pos == 0`. If there is no such maximum
            length, `None` is returned.

        """
        raise NotImplementedError()

    def get_params(self) -> KVCacheParams:
        return KVCacheParams(
            batch_size=self.batch_size,
            n_query_groups=self.n_query_groups,
            cache_length=self.cache_length,
            head_size=self.head_size,
            n_head=self.n_head,
            device=self.device,
            dtype=self.dtype,
        )

    def token_positions(self) -> torch.Tensor:
        """
        Returns:
            Token positions in slots of the cache, shape
            `(eff_batch_size, n_query_groups, T)`.where `T <= cache_length`
            is the current cache length.
        """
        raise NotImplementedError()

    def size_estimate(self) -> Tuple[int, Dict[str, int]]:
        """
        This is an estimate of the main buffers (which should all be allocated
        up front), it does not cover temporary storage used in the methods
        (make sure these are small compared to the main buffers). Also, real
        memory usage may be larger due to alignment issues.

        Returns:
            num_bits_total, bits_by_part (unit is bit)

        """
        raise NotImplementedError()

    @classmethod
    def size_estimate_apriori(cls, params: KVCacheParams, **kwargs) -> Tuple[int, Dict[str, int]]:
        """
        Same semantics as :meth:`size_estimate`, but can be called without a
        cache being created. Results may not be exactly the same, but should
        be very close.

        Args:
            params: KV cache parameters
            **kwargs: Extra arguments (optional)

        Returns:
            num_bits_total, bits_by_part (unit is bit)

        """
        raise NotImplementedError()

    def reset_parameters(self) -> None:
        pass


class DefaultKVCache(KVCache):
    """
    Default implementation of :class:`KVCache`, which implements :meth:`forward`
    using scaled dot product attention. Most KV caches will inherit from this
    class.

    """

    def __init__(
        self,
        config: Config,
        batch_size: int,
        cache_length: int,
        block_idx: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        head_size: Optional[int] = None,
    ):
        super().__init__(
            config=config,
            batch_size=batch_size,
            cache_length=cache_length,
            block_idx=block_idx,
            device=device,
            dtype=dtype,
            head_size=head_size,
        )
        self.mha = MultiHeadSelfAttention(config)

    @property
    def eff_batch_size(self) -> Optional[int]:
        raise NotImplementedError()

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        token_idx: torch.Tensor,
        input_pos: int,
    ) -> torch.Tensor:
        for_prefill = input_pos == 0
        if query.ndim != 4:
            raise ValueError("query, key, value must be 4D tensors")
        eff_batch_size, _, num, _ = query.shape
        if for_prefill:
            if not (1 <= eff_batch_size <= self.batch_size):
                raise ValueError(f"query.shape[0] = {eff_batch_size}, must be in [1, {self.batch_size}]")
            if self.max_prefill_length is not None and not (1 <= num <= self.max_prefill_length):
                raise ValueError(f"query.shape[2] = {num}, must be in [1, {self.max_prefill_length}]")
        else:
            if eff_batch_size != self.eff_batch_size:
                raise ValueError(f"query.shape[0] = {eff_batch_size} != eff_batch_size = {self.eff_batch_size}")
            if not (1 <= num <= self.max_tokens_forward):
                raise ValueError(f"query.shape[2] = {num}, must be in [1, {self.max_tokens_forward}]")
        q_shape = (eff_batch_size, self.n_head, num, self.head_size)
        if query.shape != q_shape:
            raise ValueError(f"query.shape = {query.shape}, must be {q_shape}")
        k_shape = (eff_batch_size, self.n_query_groups, num, self.head_size)
        if key.shape != k_shape:
            raise ValueError(f"key.shape = {key.shape}, must be {k_shape}")
        if value.shape != k_shape:
            raise ValueError(f"value.shape = {value.shape}, must be {k_shape}")
        t_shape = (eff_batch_size, num)
        if token_idx.shape != t_shape:
            raise ValueError(f"token_idx.shape = {token_idx.shape}, must be {t_shape}")
        self.mha.set_seq_length(input_pos + num, device=query.device)

        # Call :meth:`_forward` or :meth:`_prefill`, depending on `for_prefill`
        if for_prefill:
            self._prefill(key, value, token_idx)
            # In this case, `k_and_v` can vend both keys and values at the same
            # time.
            k_and_v = DefaultKeysAndValues(key, value)
        else:
            # Extend KV cache and retrieve key, value tensors to be used.
            # Instead of asking for the key and value tensors as such,
            # `k_and_v` allows access to them. Since they are never needed at
            # the same time, this can save memory.
            k_and_v = self._forward(key, value, token_idx)

        # Multi-head self-attention main computation
        return_attn_weights = self.update_requires_attn_weights()
        y, scores = self.mha(
            query=query,
            k_and_v=k_and_v,
            block_idx=self.block_idx,
            input_pos=input_pos,
            return_attn_weights=return_attn_weights,
            token_positions=self.token_positions(),
        )
        if scores is not None and return_attn_weights:
            # Pass attention weights to KV cache
            self._update(attn_weights=scores)

        return y

    def _forward(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        token_idx: torch.Tensor,
    ) -> KeysAndValues:
        """
        Implements part of :meth:`forward` if `input_pos > 0`. Namely, `key`
        and `value` are written into the cache, possibly evicting slots. Then,
        an object is returned which provides read access to the full keys and
        values buffers.

        Args:
            key: New keys, `(eff_batch_size, n_query_groups, num, head_size)`,
                where `1 <= num <= max_tokens_forward`
            value: New values, `(eff_batch_size, n_query_groups, num, head_size)`
            token_idx: Token indices of input sequence, `(eff_batch_size, num)`.
                Some KV caches make use of this information.

        Returns:
            key_cached, value_cached, `(eff_batch_size, n_query_groups, T,
                head_size)`, where `T <= cache_length` is the current cache
                length

        """
        raise NotImplementedError()

    def _update(self, *args, **kwargs):
        """
        Method called in :meth:`forward` if `input_pos > 0`, passing extra
        information depending on the subclass. In general, this method updates
        internal scores and takes a decision which slot is evicted upon the
        next :meth:`forward` call, if the cache is full.

        One important example are KV caches based on the Heavy Hitter Oracle
        (H2O) proposal. These require the attention weights from the current
        MLA computation to be passed, and :meth:`update_requires_attn_weights`
        has to return `True`.

        Note: The extra information typically scales with `num`, the number of
        tokens :meth:`forward` was called for.

        Args:
            *args: Depends on subclass
            **kwargs: Depends on subclass

        """
        raise NotImplementedError()

    def update_requires_attn_weights(self) -> bool:
        """
        Attention weights are required for KV caches following the Heavy
        Hitter Oracle (H2O) proposal.

        Returns:
            If `True`, :meth:`update` requires argument `attn_weights`, which
            passes current attention weights as
            `(eff_batch_size, n_query_groups, num, T)` tensor, where
            `T <= cache_length` is the current cache length, and `num` is the
            number of tokens in the last recent :meth:`forward` call.

        """
        return False

    def _prefill(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        token_idx: torch.Tensor,
    ):
        """
        Implements :meth:`forward` for `input_pos=0`.
        Starts a generation loop by passing key and value tensors coming from
        a prefill with embeddings coming from the prompts. The length must be
        `T <= max_prefill_length`. The effective batch size must be
        `eff_batch_size <= batch_size`. This batch size is then fixed for
        subsequent calls of :meth:`forward` and :meth:`update`.

        Args:
            key: Prefill keys, `(eff_batch_size, n_query_groups, T, head_size)`
            value: Prefill values, `(eff_batch_size, n_query_groups, T, head_size)`
            token_idx: Token indices of input sequence, `(eff_batch_size, T)`.
                Some KV caches make use of this information.

        """
        raise NotImplementedError()
