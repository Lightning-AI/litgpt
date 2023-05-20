"""Implementation of the paper:

LLaMA-Adapter: Efficient Fine-tuning of Language Models with Zero-init Attention
https://arxiv.org/abs/2303.16199

Port for Lit-Parrot
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Any, List, Union

import torch
import torch.nn as nn
from torch.nn import functional as F
from typing_extensions import Self

from lit_parrot.config import Config as BaseConfig
from lit_parrot.model import MLP, Parrot as BaseModel, build_rope_cache, apply_rope, RoPECache, KVCache


@dataclass
class Config(BaseConfig):
    adapter_prompt_length: int = 10
    adapter_start_layer: int = 2


class CausalSelfAttention(nn.Module):
    """A modification of `lit_parrot.model.CausalSelfAttention` that adds the attention
    over the adaption prompt."""

    def __init__(self, config: Config, block_idx: int) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # key, query, value projections for all heads, but in a batch
        self.attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=True)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=True)

        if block_idx >= config.adapter_start_layer:
            # adapter embedding layer
            self.adapter_wte = nn.Embedding(config.adapter_prompt_length, config.n_embd)
            # gate for adaption
            self.gating_factor = torch.nn.Parameter(torch.zeros(1))

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.block_size = config.block_size
        self.rotary_percentage = config.rotary_percentage
        self.block_idx = block_idx
        self.adapter_prompt_length = config.adapter_prompt_length
        self.adapter_start_layer = config.adapter_start_layer

    def forward(
        self,
        x: torch.Tensor,
        rope: RoPECache,
        mask: torch.Tensor,
        max_seq_length: int,
        input_pos: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        adapter_kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.Tensor, Optional[KVCache], Optional[KVCache]]:
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

        if kv_cache is not None:
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

        if self.block_idx >= self.adapter_start_layer:
            if adapter_kv_cache is not None:
                ak, av = adapter_kv_cache
            else:
                prefix = self.adapter_wte.weight.reshape(1, self.adapter_prompt_length, self.n_embd)
                aT = prefix.size(1)
                _, ak, av = self.attn(prefix).split(self.n_embd, dim=2)  # mayby dim=2
                ak = ak.view(1, aT, self.n_head, head_size).repeat(B, 1, 1, 1).transpose(1, 2)
                av = av.view(1, aT, self.n_head, head_size).repeat(B, 1, 1, 1).transpose(1, 2)
                adapter_kv_cache = (ak, av)

            amask = torch.ones(q.shape[-2], ak.shape[-2], dtype=torch.bool, device=x.device)
            ay = F.scaled_dot_product_attention(q, ak, av, attn_mask=amask, dropout_p=0.0, is_causal=False)
            y = y + self.gating_factor * ay

        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.proj(y)

        return y, kv_cache, adapter_kv_cache


class Block(nn.Module):
    """The implementation is identical to `lit_parrot.model.Block` with the exception that
    we replace the attention layer where adaption is implemented."""

    def __init__(self, config: Config, block_idx: int) -> None:
        super().__init__()
        self.norm_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config, block_idx)
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
        adapter_kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.Tensor, Optional[KVCache], Optional[KVCache]]:
        h, new_kv_cache, new_adapter_kv_cache = self.attn(self.norm_1(x), rope, mask, max_seq_length, input_pos, kv_cache, adapter_kv_cache)
        if self.parallel_residual:
            x = x + h + self.mlp(self.norm_2(x))
        else:
            x = x + h
            x = x + self.mlp(self.norm_2(x))
        return x, new_kv_cache, new_adapter_kv_cache


class Parrot(BaseModel):
    """The implementation is identical to `lit_parrot.model.Parrot` with the exception that
    the `Block` saves the layer index and passes it down to the attention layer."""

    def __init__(self, config: Config) -> None:
        nn.Module.__init__(self)
        assert config.padded_vocab_size is not None
        self.config = config

        self.lm_head = nn.Linear(config.n_embd, config.padded_vocab_size, bias=False)
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.padded_vocab_size, config.n_embd),
                h=nn.ModuleList(Block(config, i) for i in range(config.n_layer)),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )

        self.rope_cache: Optional[RoPECache] = None
        self.mask_cache: Optional[torch.Tensor] = None
        self.kv_caches: List[KVCache] = []
        self.adapter_kv_caches: List[KVCache] = []

    def reset(self) -> None:
        self.rope_cache = None
        self.mask_cache = None
        self.kv_caches = []
        self.adapter_kv_caches = []

    def forward(
        self, idx: torch.Tensor, max_seq_length: Optional[int] = None, input_pos: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[KVCache], List[KVCache]]]:
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
            if not self.adapter_kv_caches:
                self.adapter_kv_caches = [None for _ in range(self.config.n_layer)]
            for i, block in enumerate(self.transformer.h):
                x, self.kv_caches[i], self.adapter_kv_caches[i] = block(x, (cos, sin), mask, max_seq_length, input_pos, self.kv_caches[i], self.adapter_kv_caches[i])

        x = self.transformer.ln_f(x)

        logits = self.lm_head(x)  # (b, t, vocab_size)

        return logits

    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> Self:
        return cls(Config.from_name(name, **kwargs))


def mark_only_adapter_as_trainable(model: Parrot) -> None:
    """Sets `requires_grad=False` for all non-adapter weights."""
    for name, param in model.named_parameters():
        param.requires_grad = "adapter_wte" in name or "gating_factor" in name


def adapter_state_from_state_dict(state_dict: dict) -> dict:
    """Returns the model state dict with only the adapter weights for saving."""
    return {name: param for name, param in state_dict.items() if "adapter_wte" in name or "gating_factor" in name}
