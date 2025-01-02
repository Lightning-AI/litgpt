# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

from typing import Dict, Optional, Tuple

import torch

from litgpt.attention import DefaultKeysAndValues, KeysAndValues
from litgpt.config import Config
from litgpt.kvcache import DefaultKVCache, KVCacheParams
from litgpt.kvcache.utils import bits_for_torch_dtype, bitsize_of


class DenseKVCache(DefaultKVCache):
    """
    Key-value cache for dense attention. Key and value tensors for all
    past tokens are maintained. The cache length is the maximum sequence
    length. This cache requires a lot of memory, it can only be used for
    moderate cache lengths.

    Note: If the cache is full, :meth:`forward` raises an exception. The cache
    buffers are allocated up front and are not enlarged later on.

    """

    def __init__(
        self,
        config: Config,
        batch_size: int,
        block_idx: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        max_sequence_length: Optional[int] = None,
        head_size: Optional[int] = None,
        **base_kwargs,
    ):
        """
        Args:
            config: Model config
            batch_size: Inference batch size
            device: Device for buffers
            dtype: Data type for buffers
            max_sequence_length: Cache length. If not given, we use
            `config.block_size`
            head_size: Size of final dimension of buffers. Defaults to head
                size of model

        """
        if max_sequence_length is None:
            max_sequence_length = config.block_size
        super().__init__(
            config=config,
            batch_size=batch_size,
            cache_length=max_sequence_length,
            block_idx=block_idx,
            dtype=dtype,
            head_size=head_size,
            **base_kwargs,
        )
        shape = (batch_size, self.n_query_groups, max_sequence_length, self.head_size)
        self.register_buffer("v", torch.zeros(shape, device=device, dtype=dtype), persistent=False)
        # TODO: Remove once HF bug fixed
        if self._work_around_hf_bug:
            shape = shape[:-1] + (self.head_size + 1,)
        self.register_buffer("k", torch.zeros(shape, device=device, dtype=dtype), persistent=False)
        self.next_position = None
        self._eff_batch_size = None

    @property
    def device(self) -> torch.device:
        return self.k.device

    @property
    def eff_batch_size(self) -> Optional[int]:
        return self._eff_batch_size

    @property
    def next_token_pos(self) -> Optional[int]:
        return self.next_position

    @property
    def max_tokens_forward(self) -> int:
        return self.cache_length

    @property
    def max_prefill_length(self) -> Optional[int]:
        return self.cache_length

    @property
    def current_length(self) -> int:
        return self.next_position

    def get_keys_values(self) -> Optional[KeysAndValues]:
        if self.eff_batch_size is None or self.next_position is None:
            return None
        else:
            return DefaultKeysAndValues(
                self.k[: self.eff_batch_size, :, : self.next_position, :],
                self.v[: self.eff_batch_size, :, : self.next_position, :],
            )

    def _forward(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        token_idx: torch.Tensor,
    ) -> KeysAndValues:
        if self.next_position is None:
            raise IndexError("Cache needs to be initialized with 'prefill' before being used")
        num = key.shape[2]
        if not 1 <= num <= self.max_tokens_forward:
            raise ValueError(f"key.shape[2] = {num}, must be in [1, {self.max_tokens_forward}]")
        np = self.next_position
        if np + num > self.cache_length:
            raise IndexError(f"Cache has at most {self.cache_length - np} free slots, cannot add {num} entries")
        shape = (self.eff_batch_size, self.n_query_groups, num, self.head_size)
        # TODO: Remove once HF bug fixed
        if self._work_around_hf_bug:
            assert value.shape == shape
            shape = shape[:-1] + (self.head_size + 1,)
            assert key.shape == shape
        elif key.shape != shape or value.shape != shape:
            raise ValueError(
                f"Shapes of key, value must be {shape}, but key.shape = {key.shape}, value.shape = {value.shape}"
            )
        if key.dtype != value.dtype:
            raise ValueError(f"key.dtype = {key.dtype} != {value.dtype} = value.dtype")
        # Move the buffer to the activation dtype for when AMP is used
        # TODO: Is this needed? Other KV caches do not support changing
        # `dtype` after creation.
        if key.dtype != self.dtype:
            self._dtype = key.dtype
            self.k = self.k.to(self.dtype)
            self.v = self.v.to(self.dtype)
        # Append new content to cache
        self.k[: self.eff_batch_size, :, np : (np + num), :] = key
        self.v[: self.eff_batch_size, :, np : (np + num), :] = value
        self.next_position += num
        return self.get_keys_values()

    def _update(self, *args, **kwargs):
        pass

    def _prefill(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        token_idx: torch.Tensor,
    ):
        if key.dim() != 4:
            raise ValueError("key must have 4 dimensions")
        init_length = key.shape[2]
        if init_length > self.cache_length:
            raise ValueError(f"key.shape[2] = {init_length}, must be at most {self.cache_length}")
        eff_batch_size = key.shape[0]
        if eff_batch_size > self.batch_size:
            raise ValueError(f"key.shape[0] = {eff_batch_size} must be at most batch_size = {self.batch_size}")
        shape = (eff_batch_size, self.n_query_groups, init_length, self.head_size)
        # TODO: Remove once HF bug fixed
        if self._work_around_hf_bug:
            assert value.shape == shape
            shape = shape[:-1] + (self.head_size + 1,)
            assert key.shape == shape
        elif key.shape != shape or value.shape != shape:
            raise ValueError(
                f"Shapes of key, value must be {shape}, but key.shape = {key.shape}, value.shape = {value.shape}"
            )
        # Initialize cache content
        self.k = self.k.to(key.dtype)
        self.v = self.v.to(value.dtype)
        self.k[:eff_batch_size, :, :init_length, :] = key
        self.v[:eff_batch_size, :, :init_length, :] = value
        self.next_position = init_length
        self._eff_batch_size = eff_batch_size

    def resize(self, new_length: int):
        """
        Shortens the cache content to length `current_length`, removing the
        most recently inserted content. Note that this method is currently
        supported only for specific KV caches; the cost for supporting it
        generally would be high.

        Args:
            new_length: New length, must be <= current length

        """
        if not (0 <= new_length <= self.next_position):
            raise ValueError(f"current_length = {new_length}, must be in [0, {self.next_position}]")
        self.next_position = new_length

    def token_positions(self) -> torch.Tensor:
        return (
            torch.arange(self.next_position, device=self.device)
            .reshape(1, 1, -1)
            .expand(self.eff_batch_size, self.n_query_groups, -1)
        )

    def size_estimate(self) -> Tuple[int, Dict[str, int]]:
        sz_buffs = bitsize_of(self.k) + bitsize_of(self.v)
        return sz_buffs, dict(buffers=sz_buffs)

    @classmethod
    def size_estimate_apriori(cls, params: KVCacheParams, **kwargs) -> Tuple[int, Dict[str, int]]:
        cache_length = params.cache_length
        dtype = params.dtype
        if dtype is None:
            raise ValueError("params.dtype must be provided")
        numel = params.batch_size * params.n_query_groups * cache_length * params.head_size
        sz_buffs = 2 * numel * bits_for_torch_dtype(dtype)
        return sz_buffs, dict(buffers=sz_buffs)


class LastRecentlyInsertedKVCache(DefaultKVCache):
    """
    Baseline key-value cache which stores the last recently inserted
    `cache_length` key, value tensors.
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
        **base_kwargs,
    ):
        super().__init__(
            config=config,
            batch_size=batch_size,
            cache_length=cache_length,
            block_idx=block_idx,
            dtype=dtype,
            head_size=head_size,
            **base_kwargs,
        )
        shape = (batch_size, self.n_query_groups, cache_length, self.head_size)
        self.register_buffer("v", torch.zeros(shape, device=device, dtype=dtype), persistent=False)
        # TODO: Remove once HF bug fixed
        if self._work_around_hf_bug:
            shape = shape[:-1] + (self.head_size + 1,)
        self.register_buffer("k", torch.zeros(shape, device=device, dtype=dtype), persistent=False)
        self.register_buffer("token_pos", torch.zeros(cache_length, device=device, dtype=torch.int), persistent=False)
        self.next_position = None
        self._eff_batch_size = None
        self.current_length = None
        self._next_token_pos = None

    @property
    def device(self) -> torch.device:
        return self.k.device

    @property
    def eff_batch_size(self) -> Optional[int]:
        return self._eff_batch_size

    @property
    def next_token_pos(self) -> Optional[int]:
        return self._next_token_pos

    @property
    def max_tokens_forward(self) -> int:
        return self.cache_length

    @property
    def max_prefill_length(self) -> Optional[int]:
        return None

    def get_keys_values(self) -> Optional[KeysAndValues]:
        if self.eff_batch_size is None or self.current_length is None:
            return None
        else:
            return DefaultKeysAndValues(
                self.k[: self.eff_batch_size, :, : self.current_length, :],
                self.v[: self.eff_batch_size, :, : self.current_length, :],
            )

    def _forward(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        token_idx: torch.Tensor,
    ) -> KeysAndValues:
        if self.next_position is None:
            raise IndexError("Cache needs to be initialized with 'prefill' before being used")
        if key.ndim != 4:
            raise ValueError(f"key must be a 4D tensor, but has shape {key.shape}")
        num = key.shape[2]
        if not 1 <= num <= self.max_tokens_forward:
            raise ValueError(f"key.shape[2] = {num}, must be in [1, {self.max_tokens_forward}]")
        shape = (self.eff_batch_size, self.n_query_groups, num, self.head_size)
        # TODO: Remove once HF bug fixed
        if self._work_around_hf_bug:
            assert value.shape == shape
            shape = shape[:-1] + (self.head_size + 1,)
            assert key.shape == shape
        elif key.shape != shape or value.shape != shape:
            raise ValueError(
                f"Shapes of key, value must be {shape}, but key.shape = {key.shape}, value.shape = {value.shape}"
            )
        # Move the buffer to the activation dtype for when AMP is used
        self.k = self.k.to(key.dtype)
        self.v = self.v.to(value.dtype)
        # Append new content to cache
        np = self.next_position
        num1 = min(num, self.cache_length - np)
        self.k[: self.eff_batch_size, :, np : (np + num1), :] = key[:, :, :num1, :]
        self.v[: self.eff_batch_size, :, np : (np + num1), :] = value[:, :, :num1, :]
        ntp = self._next_token_pos
        self.token_pos[np : (np + num1)] = torch.arange(ntp, ntp + num1, device=self.device, dtype=torch.int)
        if num1 < num:
            diff = num - num1
            self.k[: self.eff_batch_size, :, :diff, :] = key[:, :, num1:, :]
            self.v[: self.eff_batch_size, :, :diff, :] = value[:, :, num1:, :]
            self.token_pos[:diff] = torch.arange(ntp + num1, ntp + num, device=self.device, dtype=torch.int)
        self.next_position = (np + num) % self.cache_length
        self.current_length = min(self.current_length + num, self.cache_length)
        self._next_token_pos += num
        return self.get_keys_values()

    def _update(self, *args, **kwargs):
        pass

    def _prefill(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        token_idx: torch.Tensor,
    ):
        if key.dim() != 4:
            raise ValueError("key must have 4 dimensions")
        init_length = key.shape[2]
        eff_init_length = min(init_length, self.cache_length)
        eff_batch_size = key.shape[0]
        if eff_batch_size > self.batch_size:
            raise ValueError(f"key.shape[0] = {eff_batch_size} must be at most batch_size = {self.batch_size}")
        shape = (eff_batch_size, self.n_query_groups, init_length, self.head_size)
        # TODO: Remove once HF bug fixed
        if self._work_around_hf_bug:
            assert value.shape == shape
            shape = shape[:-1] + (self.head_size + 1,)
            assert key.shape == shape
        elif key.shape != shape or value.shape != shape:
            raise ValueError(
                f"Shapes of key, value must be {shape}, but key.shape = {key.shape}, value.shape = {value.shape}"
            )
        # Initialize cache content
        self.k = self.k.to(key.dtype)
        self.v = self.v.to(value.dtype)
        self.k[:eff_batch_size, :, :eff_init_length, :] = key[:, :, -eff_init_length:, :]
        self.v[:eff_batch_size, :, :eff_init_length, :] = value[:, :, -eff_init_length:, :]
        self.token_pos[:eff_init_length] = torch.arange(
            init_length - eff_init_length,
            init_length,
            dtype=self.token_pos.dtype,
            device=self.token_pos.device,
        )
        self.current_length = eff_init_length
        self._next_token_pos = init_length
        self.next_position = eff_init_length % self.cache_length
        self._eff_batch_size = eff_batch_size

    def token_positions(self) -> torch.Tensor:
        return (
            self.token_pos[: self.current_length].reshape(1, 1, -1).expand(self.eff_batch_size, self.n_query_groups, -1)
        )

    def size_estimate(self) -> Tuple[int, Dict[str, int]]:
        sz_buffs = bitsize_of(self.k) + bitsize_of(self.v)
        sz_pos = bitsize_of(self.token_pos)
        return sz_buffs + sz_pos, dict(buffers=sz_buffs, token_pos=sz_pos)

    @classmethod
    def size_estimate_apriori(cls, params: KVCacheParams, **kwargs) -> Tuple[int, Dict[str, int]]:
        cache_length = params.cache_length
        dtype = params.dtype
        if dtype is None:
            raise ValueError("params.dtype must be provided")
        numel = params.batch_size * params.n_query_groups * cache_length * params.head_size
        k_and_v = 2 * numel * bits_for_torch_dtype(dtype)
        tk_p = cache_length * bits_for_torch_dtype(torch.int)
        return k_and_v + tk_p, dict(buffers=k_and_v, token_pos=tk_p)
