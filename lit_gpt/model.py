"""Full definition of a GPT NeoX Language Model, all of it in this single file.

Based on the nanoGPT implementation: https://github.com/karpathy/nanoGPT and
https://github.com/EleutherAI/gpt-neox/tree/main/megatron/model.
"""
import math
from typing import List, Optional, Tuple, Any

import torch
import torch.nn as nn
from typing_extensions import Self

from lit_gpt.config import Config
from lit_gpt.rmsnorm import RMSNorm

RoPECache = Tuple[torch.Tensor, torch.Tensor]
KVCache = Tuple[torch.Tensor, torch.Tensor]


class GPT(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        assert config.padded_vocab_size is not None
        self.config = config

        self.lm_head = nn.Linear(config.n_embd, config.padded_vocab_size, bias=False)
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.padded_vocab_size, config.n_embd),
                h=nn.ModuleList(Block(config) for _ in range(config.n_layer)),
                ln_f=config.norm_class(config.n_embd, eps=config.norm_eps),
            )
        )
        self.rope_cache: Optional[RoPECache] = None
        self.mask_cache: Optional[torch.Tensor] = None
        self.kv_caches: List[KVCache] = []

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
            module.eps = self.config.norm_eps
        elif isinstance(module, RMSNorm):
            torch.nn.init.ones_(module.weight)
            module.eps = self.config.norm_eps

    def reset_cache(self) -> None:
        self.kv_caches.clear()
        if self.mask_cache is not None and self.mask_cache.device.type == "xla":
            # https://github.com/Lightning-AI/lit-gpt/pull/83#issuecomment-1558150179
            self.rope_cache = None
            self.mask_cache = None

    def forward(
        self, idx: torch.Tensor, max_seq_length: Optional[int] = None, input_pos: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, T = idx.size()
        use_kv_cache = input_pos is not None

        block_size = self.config.block_size
        if max_seq_length is None:
            max_seq_length = block_size
        if use_kv_cache:  # not relevant otherwise
            assert (
                max_seq_length >= T
            ), f"Cannot forward sequence of length {T}, max seq length is only {max_seq_length}"
        assert max_seq_length <= block_size, f"Cannot attend to {max_seq_length}, block size is only {block_size}"
        assert block_size >= T, f"Cannot forward sequence of length {T}, block size is only {block_size}"

        if self.rope_cache is None:
            self.rope_cache = self.build_rope_cache(idx)
        # passing `attn_mask` to SDPA downgrades it to use the inefficient implementation. since we only need the mask
        # for the kv-cache support (only during inference), we only create it in that situation
        # this will be resolved by https://github.com/pytorch/pytorch/issues/96099
        if use_kv_cache and self.mask_cache is None:
            self.mask_cache = self.build_mask_cache(idx)

        cos, sin = self.rope_cache
        if use_kv_cache:
            cos = cos.index_select(0, input_pos)
            sin = sin.index_select(0, input_pos)
            mask = self.mask_cache.index_select(2, input_pos)
            mask = mask[:, :, :, :max_seq_length]
        else:
            cos = cos[:T]
            sin = sin[:T]
            mask = None

        # forward the model itself
        x = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)

        if not use_kv_cache:
            for block in self.transformer.h:
                x, *_ = block(x, (cos, sin), max_seq_length)
        else:
            self.kv_caches = self.kv_caches or self.build_kv_caches(x, max_seq_length, cos.size(-1))
            for i, block in enumerate(self.transformer.h):
                x, self.kv_caches[i] = block(x, (cos, sin), max_seq_length, mask, input_pos, self.kv_caches[i])

        x = self.transformer.ln_f(x)

        return self.lm_head(x)  # (b, t, vocab_size)

    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> Self:
        return cls(Config.from_name(name, **kwargs))

    def build_rope_cache(self, idx: torch.Tensor) -> RoPECache:
        return build_rope_cache(
            seq_len=self.config.block_size,
            n_elem=int(self.config.rotary_percentage * self.config.head_size),
            dtype=torch.get_default_dtype(),
            device=idx.device,
            condense_ratio=self.config.condense_ratio,
        )

    def build_mask_cache(self, idx: torch.Tensor) -> torch.Tensor:
        ones = torch.ones((self.config.block_size, self.config.block_size), device=idx.device, dtype=torch.bool)
        return torch.tril(ones).unsqueeze(0).unsqueeze(0)

    def build_kv_caches(self, idx: torch.Tensor, max_seq_length: int, rope_cache_length: int) -> List[KVCache]:
        B = idx.size(0)
        heads = 1 if self.config.n_query_groups == 1 else self.config.n_head
        k_cache_shape = (
            B,
            heads,
            max_seq_length,
            rope_cache_length + self.config.head_size - int(self.config.rotary_percentage * self.config.head_size),
        )
        v_cache_shape = (B, heads, max_seq_length, self.config.head_size)
        device = idx.device
        return [
            (torch.zeros(k_cache_shape, device=device), torch.zeros(v_cache_shape, device=device))
            for _ in range(self.config.n_layer)
        ]


class Block(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.norm_1 = config.norm_class(config.n_embd, eps=config.norm_eps)
        self.attn = CausalSelfAttention(config)
        if not config.shared_attention_norm:
            self.norm_2 = config.norm_class(config.n_embd, eps=config.norm_eps)
        self.mlp = config.mlp_class(config)

        self.config = config

    def forward(
        self,
        x: torch.Tensor,
        rope: RoPECache,
        max_seq_length: int,
        mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        n_1 = self.norm_1(x)
        h, new_kv_cache = self.attn(n_1, rope, max_seq_length, mask, input_pos, kv_cache)
        if self.config.parallel_residual:
            n_2 = n_1 if self.config.shared_attention_norm else self.norm_2(x)
            x = x + h + self.mlp(n_2)
        else:
            if self.config.shared_attention_norm:
                raise NotImplementedError(
                    "No checkpoint amongst the ones we support uses this configuration"
                    " (non-parallel residual and shared attention norm)."
                )
            x = x + h
            x = x + self.mlp(self.norm_2(x))
        return x, new_kv_cache


class CausalSelfAttention(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        shape = (config.n_head + 2 * config.n_query_groups) * config.head_size
        # key, query, value projections for all heads, but in a batch
        self.attn = nn.Linear(config.n_embd, shape, bias=config.bias)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.config = config

    def forward(
        self,
        x: torch.Tensor,
        rope: RoPECache,
        max_seq_length: int,
        mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        qkv = self.attn(x)

        # assemble into a number of query groups to support MHA, MQA and GQA together (see `config.n_query_groups`)
        q_per_kv = self.config.n_head // self.config.n_query_groups
        total_qkv = q_per_kv + 2  # each group has 1+ queries, 1 key, and 1 value
        qkv = qkv.view(B, T, self.config.n_query_groups, total_qkv, self.config.head_size)
        qkv = qkv.permute(0, 2, 3, 1, 4)  # (B, n_query_groups, total_qkv, T, hs)

        # split batched computation into three
        q, k, v = qkv.split((q_per_kv, 1, 1), dim=2)

        # repeat k and v if necessary
        if self.config.n_query_groups != 1:  # doing this would require a full kv cache with MQA (inefficient!)
            # for MHA this is a no-op
            k = k.repeat_interleave(q_per_kv, dim=2)
            v = v.repeat_interleave(q_per_kv, dim=2)

        q = q.reshape(B, -1, T, self.config.head_size)  # (B, nh_q, T, hs)
        k = k.view(B, -1, T, self.config.head_size)  # (B, nh_k, T, hs)
        v = v.view(B, -1, T, self.config.head_size)  # (B, nh_v, T, hs)

        n_elem = int(self.config.rotary_percentage * self.config.head_size)

        cos, sin = rope
        q_roped = apply_rope(q[..., :n_elem], cos, sin)
        k_roped = apply_rope(k[..., :n_elem], cos, sin)
        q = torch.cat((q_roped, q[..., n_elem:]), dim=-1)
        k = torch.cat((k_roped, k[..., n_elem:]), dim=-1)

        if kv_cache is not None:
            cache_k, cache_v = kv_cache
            cache_k, cache_v = cache_k.to(dtype=k.dtype), cache_v.to(dtype=v.dtype)
            # check if reached token limit
            if input_pos[-1] >= max_seq_length:
                input_pos = torch.tensor(max_seq_length - 1, device=input_pos.device)
                # shift 1 position to the left
                cache_k = torch.roll(cache_k, -1, dims=2)
                cache_v = torch.roll(cache_v, -1, dims=2)
            k = cache_k.index_copy_(2, input_pos, k)
            v = cache_v.index_copy_(2, input_pos, v)
            kv_cache = k, v

        # efficient attention using Flash Attention CUDA kernels
        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=0.0, scale=1.0 / math.sqrt(self.config.head_size), is_causal=mask is None
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.proj(y)

        return y, kv_cache


class GptNeoxMLP(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.fc = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        self.proj = nn.Linear(config.intermediate_size, config.n_embd, bias=config.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = torch.nn.functional.gelu(x)
        return self.proj(x)


class LLaMAMLP(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.fc_1 = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        self.fc_2 = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        self.proj = nn.Linear(config.intermediate_size, config.n_embd, bias=config.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fc_1 = self.fc_1(x)
        x_fc_2 = self.fc_2(x)
        x = torch.nn.functional.silu(x_fc_1) * x_fc_2
        return self.proj(x)


def build_rope_cache(
    seq_len: int, n_elem: int, dtype: torch.dtype, device: torch.device, base: int = 10000, condense_ratio: int = 1
) -> RoPECache:
    """Enhanced Transformer with Rotary Position Embedding.

    Derived from: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/
    transformers/rope/__init__.py. MIT License:
    https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/license.
    """
    # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
    theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, device=device) / n_elem))

    # Create position indexes `[0, 1, ..., seq_len - 1]`
    seq_idx = torch.arange(seq_len, device=device) / condense_ratio

    # Calculate the product of position index and $\theta_i$
    idx_theta = torch.outer(seq_idx, theta).repeat(1, 2)

    cos, sin = torch.cos(idx_theta), torch.sin(idx_theta)

    # this is to mimic the behaviour of complex32, else we will get different results
    if dtype in (torch.float16, torch.bfloat16, torch.int8):
        return cos.half(), sin.half()
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    head_size = x.size(-1)
    x1 = x[..., : head_size // 2]  # (B, nh, T, hs/2)
    x2 = x[..., head_size // 2 :]  # (B, nh, T, hs/2)
    rotated = torch.cat((-x2, x1), dim=-1)  # (B, nh, T, hs)
    roped = (x * cos) + (rotated * sin)
    return roped.type_as(x)
