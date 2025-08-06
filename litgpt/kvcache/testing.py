from typing import Tuple

import torch

from litgpt.config import Config
from litgpt.kvcache.base import KVCache, KVCacheParams
from litgpt.kvcache.baselines import DenseKVCache, LastRecentlyInsertedKVCache

KV_CACHE_NAMES = (
    "dense-default",
    "lastrec-default",
)


def create_kv_cache(
    name: str,
    params: KVCacheParams,
    block_idx: int = 0,
) -> KVCache:
    config = Config(
        n_embd=params.n_head * params.head_size,
        n_head=params.n_head,
        n_query_groups=params.n_query_groups,
    )
    from_config_kwargs = dict(
        config=config,
        batch_size=params.batch_size,
        block_idx=block_idx,
        device=params.device,
        dtype=params.dtype,
    )

    result = None
    if name == "dense-default":
        result = DenseKVCache(**from_config_kwargs)
    elif name == "lastrec-default":
        result = LastRecentlyInsertedKVCache(**from_config_kwargs, cache_length=params.cache_length)

    if result is None:
        raise ValueError(f"name = {name} not supported")
    return result


def tensor_is_simple(x: torch.Tensor) -> bool:
    assert x.ndim > 1
    x = x.view(-1, x.shape[-1])
    other = x[0].unsqueeze(0).expand(*x.shape)
    return x.equal(other)


def random_tensor(
    params: KVCacheParams,
    num: int,
) -> torch.Tensor:
    shape = (params.batch_size, params.n_query_groups, num, params.head_size)
    return torch.randn(*shape, device=params.device, dtype=params.dtype)


def random_keys_values(
    params: KVCacheParams,
    num: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    keys = random_tensor(params, num)
    values = random_tensor(params, num)
    return keys, values


def random_attn_weights(
    params: KVCacheParams,
    num: int,
) -> torch.Tensor:
    attn_weights = torch.randn(
        (params.batch_size, params.n_head, num),
        device=params.device,
        dtype=params.dtype,
    )
    return torch.nn.functional.softmax(attn_weights, dim=-1)
