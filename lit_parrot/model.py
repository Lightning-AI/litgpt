"""Full definition of a GPT NeoX Language Model, all of it in this single file.

Based on the nanoGPT implementation: https://github.com/karpathy/nanoGPT and
https://github.com/EleutherAI/gpt-neox/tree/main/megatron/model.
"""
import math
from typing import List, Optional, Tuple, Any, Union

import torch
import torch.nn as nn
from torch.nn import functional as F
from typing_extensions import Self

from lit_parrot.config import Config

RoPECache = Tuple[torch.Tensor, torch.Tensor]
KVCache = Tuple[torch.Tensor, torch.Tensor]


class Parrot(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        assert config.padded_vocab_size is not None
        self.config = config

        self.lm_head = nn.Linear(config.n_embd, config.padded_vocab_size, bias=False)
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.padded_vocab_size, config.n_embd),
                h=nn.ModuleList(Block(config) for _ in range(config.n_layer)),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        self.rope_cache: Optional[RoPECache] = None
        self.mask_cache: Optional[torch.Tensor] = None
        self.kv_caches: List[KVCache] = []

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            # https://huggingface.co/stabilityai/stablelm-base-alpha-3b/blob/main/config.json#L10
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
            # https://huggingface.co/stabilityai/stablelm-base-alpha-3b/blob/main/config.json#L12
            module.eps = 1e-5

    def reset_cache(self) -> None:
        self.kv_caches.clear()
        if self.mask_cache is not None and self.mask_cache.device.type == "xla":

            # https://github.com/Lightning-AI/lit-parrot/pull/83#issuecomment-1558150179
            self.rope_cache = None
            self.mask_cache = None

    def forward(
        self, idx: torch.Tensor, max_seq_length: Optional[int] = None, input_pos: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[KVCache]]]:
        B, T = idx.size()

        block_size = self.config.block_size
        if max_seq_length is None:
            max_seq_length = block_size
        assert T <= max_seq_length, f"Cannot forward sequence of length {T}, max seq length is only {max_seq_length}"
        assert max_seq_length <= block_size, f"Cannot attend to {max_seq_length}, block size is only {block_size}"
        assert T <= block_size, f"Cannot forward sequence of length {T}, block size is only {block_size}"

        if self.rope_cache is None:
            self.rope_cache = self.build_rope_cache(idx)
        if self.mask_cache is None:
            self.mask_cache = self.build_mask_cache(idx)

        cos, sin = self.rope_cache
        if input_pos is not None:
            cos = cos.index_select(0, input_pos)
            sin = sin.index_select(0, input_pos)
            mask = self.mask_cache.index_select(2, input_pos)
            mask = mask[:, :, :, :max_seq_length]
        else:
            cos = cos[:T]
            sin = sin[:T]
            mask = self.mask_cache[:, :, :T, :T]

        # forward the model itself
        x = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)

        if input_pos is None:  # proxy for use_cache=False
            for block in self.transformer.h:
                x, *_ = block(x, (cos, sin), mask, max_seq_length)
        else:
            if not self.kv_caches:
                head_size = self.config.n_embd // self.config.n_head
                k_cache_shape = (
                    B,
                    self.config.n_head,
                    max_seq_length,
                    cos.size(-1) + head_size - int(self.config.rotary_percentage * head_size),
                )
                v_cache_shape = (B, self.config.n_head, max_seq_length, head_size)
                self.kv_caches = [
                    (torch.zeros(k_cache_shape, device=x.device, dtype=x.dtype),
                     torch.zeros(v_cache_shape, device=x.device, dtype=x.dtype))
                    for _ in range(self.config.n_layer)
                ]
            for i, block in enumerate(self.transformer.h):
                x, self.kv_caches[i] = block(x, (cos, sin), mask, max_seq_length, input_pos, self.kv_caches[i])

        x = self.transformer.ln_f(x)

        logits = self.lm_head(x)  # (b, t, vocab_size)

        return logits

    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> Self:
        return cls(Config.from_name(name, **kwargs))

    def build_rope_cache(self, idx: torch.Tensor) -> RoPECache:
        return build_rope_cache(
            seq_len=self.config.block_size,
            n_elem=int(self.config.rotary_percentage * (self.config.n_embd // self.config.n_head)),
            dtype=idx.dtype,
            device=idx.device,
        )

    def build_mask_cache(self, idx: torch.Tensor) -> torch.Tensor:
        ones = torch.ones((self.config.block_size, self.config.block_size), device=idx.device, dtype=torch.bool)
        return torch.tril(ones).unsqueeze(0).unsqueeze(0)


class Block(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.norm_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.norm_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

        self.parallel_residual = config.parallel_residual

    def forward(
        self,
        x: torch.Tensor,
        rope: RoPECache,
        mask: torch.Tensor,
        max_seq_length: int,
        input_pos: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        h, new_kv_cache = self.attn(self.norm_1(x), rope, mask, max_seq_length, input_pos, kv_cache)
        if self.parallel_residual:
            x = x + h + self.mlp(self.norm_2(x))
        else:
            x = x + h
            x = x + self.mlp(self.norm_2(x))
        return x, new_kv_cache


class CausalSelfAttention(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # key, query, value projections for all heads, but in a batch
        self.attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=True)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=True)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.block_size = config.block_size
        self.rotary_percentage = config.rotary_percentage

    def forward(
        self,
        x: torch.Tensor,
        rope: RoPECache,
        mask: torch.Tensor,
        max_seq_length: int,
        input_pos: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        qkv = self.attn(x)
        head_size = C // self.n_head
        qkv = qkv.view(B, T, self.n_head, 3 * head_size).transpose(1, 2)
        q, k, v = qkv.split(head_size, dim=-1)  # (B, nh, T, hs)

        n_elem = int(self.rotary_percentage * head_size)

        cos, sin = rope
        q_roped = apply_rope(q[..., :n_elem], cos, sin)
        k_roped = apply_rope(k[..., :n_elem], cos, sin)
        q = torch.cat((q_roped, q[..., n_elem:]), dim=-1)
        k = torch.cat((k_roped, k[..., n_elem:]), dim=-1)

        if input_pos is not None and kv_cache is not None:
            cache_k, cache_v = kv_cache
            # check if reached token limit
            if input_pos[-1] >= max_seq_length:
                input_pos = torch.tensor(max_seq_length - 1, device=input_pos.device)
                # shift 1 position to the left
                cache_k = torch.roll(cache_k, -1, dims=2)
                cache_v = torch.roll(cache_v, -1, dims=2)
            k = cache_k.index_copy(2, input_pos, k)
            v = cache_v.index_copy(2, input_pos, v)
            kv_cache = k, v

        # efficient attention using Flash Attention CUDA kernels
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, scale=1.0 / math.sqrt(head_size))

        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.proj(y)

        return y, kv_cache


class MLP(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        hidden_dim = 4 * config.n_embd
        self.fc = nn.Linear(config.n_embd, hidden_dim, bias=True)
        self.proj = nn.Linear(hidden_dim, config.n_embd, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # gpt-neox style MLP
        x = self.fc(x)
        x = F.gelu(x)
        x = self.proj(x)
        return x


def build_rope_cache(
    seq_len: int, n_elem: int, dtype: torch.dtype, device: torch.device, base: int = 10000
) -> RoPECache:
    """Enhanced Transformer with Rotary Position Embedding.

    Derived from: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/
    transformers/rope/__init__.py. MIT License:
    https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/license.
    """
    # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
    theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, dtype=dtype, device=device) / n_elem))

    # Create position indexes `[0, 1, ..., seq_len - 1]`
    seq_idx = torch.arange(seq_len, dtype=dtype, device=device)

    # Calculate the product of position index and $\theta_i$
    idx_theta = torch.outer(seq_idx, theta).float().repeat(1, 2)

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
