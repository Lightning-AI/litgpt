"""Implementation of the paper:

LLaMA-Adapter: Efficient Fine-tuning of Language Models with Zero-init Attention
https://arxiv.org/abs/2303.16199

Port for Lit-GPT
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from typing_extensions import Self

from lit_gpt.config import Config as BaseConfig
from lit_gpt.model import GPT as BaseModel
from lit_gpt.model import CausalSelfAttention as BaseCausalSelfAttention
from lit_gpt.model import KVCache, RoPECache, apply_rope


@dataclass
class Config(BaseConfig):
    adapter_prompt_length: int = 10
    adapter_start_layer: int = 2


class GPT(BaseModel):
    """The implementation is identical to `lit_gpt.model.GPT` with the exception that
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
                ln_f=config.norm_class(config.n_embd, eps=config.norm_eps),
            )
        )

        self.rope_cache: Optional[RoPECache] = None
        self.mask_cache: Optional[torch.Tensor] = None
        self.kv_caches: List[KVCache] = []
        self.adapter_kv_caches: List[KVCache] = []

    def reset_cache(self) -> None:
        super().reset_cache()
        self.adapter_kv_caches.clear()

    def forward(
        self,
        idx: torch.Tensor,
        max_seq_length: Optional[int] = None,
        input_pos: Optional[torch.Tensor] = None,
        lm_head_chunk_size: int = 0,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
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
            self.adapter_kv_caches = self.adapter_kv_caches or [None for _ in range(self.config.n_layer)]
            for i, block in enumerate(self.transformer.h):
                x, self.kv_caches[i], self.adapter_kv_caches[i] = block(
                    x, (cos, sin), max_seq_length, mask, input_pos, self.kv_caches[i], self.adapter_kv_caches[i]
                )

        x = self.transformer.ln_f(x)

        if lm_head_chunk_size > 0:
            # chunk the lm head logits to reduce the peak memory used by autograd
            return [self.lm_head(x_i) for x_i in x.split(lm_head_chunk_size, dim=1)]
        return self.lm_head(x)  # (b, t, vocab_size)

    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> Self:
        return cls(Config.from_name(name, **kwargs))

    def _init_weights(self, module: nn.Module) -> None:
        """Meant to be used with `gpt.apply(gpt._init_weights)`. Unused method left for completeness."""
        super()._init_weights(module)
        if isinstance(module, CausalSelfAttention):
            module.reset_parameters()


class Block(nn.Module):
    """The implementation is identical to `lit_gpt.model.Block` with the exception that
    we replace the attention layer where adaption is implemented."""

    def __init__(self, config: Config, block_idx: int) -> None:
        super().__init__()
        self.norm_1 = config.norm_class(config.n_embd, eps=config.norm_eps)
        self.attn = CausalSelfAttention(config, block_idx)
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
        adapter_kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.Tensor, Optional[KVCache], Optional[KVCache]]:
        n_1 = self.norm_1(x)
        h, new_kv_cache, new_adapter_kv_cache = self.attn(
            n_1, rope, max_seq_length, mask, input_pos, kv_cache, adapter_kv_cache
        )
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
        return x, new_kv_cache, new_adapter_kv_cache


class CausalSelfAttention(BaseCausalSelfAttention):
    """A modification of `lit_gpt.model.CausalSelfAttention` that adds the attention
    over the adaption prompt."""

    def __init__(self, config: Config, block_idx: int) -> None:
        super().__init__(config)
        if block_idx >= config.adapter_start_layer:
            # adapter embedding layer
            self.adapter_wte = nn.Embedding(config.adapter_prompt_length, config.n_embd)
            # gate for adaption
            self.gating_factor = torch.nn.Parameter(torch.zeros(1, 1, config.n_head, 1))
            self.reset_parameters()
        self.block_idx = block_idx

    def forward(
        self,
        x: torch.Tensor,
        rope: RoPECache,
        max_seq_length: int,
        mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        adapter_kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.Tensor, Optional[KVCache], Optional[KVCache]]:
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
            k = k.expand(B, self.config.n_query_groups, q_per_kv, T, self.config.head_size)
            v = v.expand(B, self.config.n_query_groups, q_per_kv, T, self.config.head_size)

        q = q.reshape(B, -1, T, self.config.head_size)  # (B, nh_q, T, hs)
        k = k.reshape(B, -1, T, self.config.head_size)  # (B, nh_k, T, hs)
        v = v.reshape(B, -1, T, self.config.head_size)  # (B, nh_v, T, hs)

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

        y = self.scaled_dot_product_attention(q, k, v, mask=mask)

        if self.block_idx >= self.config.adapter_start_layer:
            aT = self.config.adapter_prompt_length
            if adapter_kv_cache is not None:
                ak, av = adapter_kv_cache
            else:
                prefix = self.adapter_wte.weight.reshape(1, aT, C)
                aqkv = self.attn(prefix)
                aqkv = aqkv.view(1, aT, self.config.n_query_groups, q_per_kv + 2, self.config.head_size)
                aqkv = aqkv.permute(0, 2, 3, 1, 4)
                _, ak, av = aqkv.split((q_per_kv, 1, 1), dim=2)
                if self.config.n_query_groups != 1:
                    # for MHA this is a no-op
                    ak = ak.repeat_interleave(q_per_kv, dim=2)
                    av = av.repeat_interleave(q_per_kv, dim=2)
                ak = ak.view(1, -1, aT, self.config.head_size)  # (1, nh_ak, aT, hs)
                av = av.view(1, -1, aT, self.config.head_size)  # (1, nh_av, aT, hs)
                adapter_kv_cache = (ak, av)

            amask = torch.ones(T, aT, dtype=torch.bool, device=x.device)
            ay = self.scaled_dot_product_attention(q, ak, av, amask)
            y = y + self.gating_factor * ay

        y = y.reshape(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.proj(y)

        return y, kv_cache, adapter_kv_cache

    def reset_parameters(self) -> None:
        torch.nn.init.zeros_(self.gating_factor)

    def _load_from_state_dict(self, state_dict: Dict, prefix: str, *args: Any, **kwargs: Any) -> None:
        """For compatibility with older checkpoints."""
        if (key := prefix + "gating_factor") in state_dict and state_dict[key].size(1) == self.config.n_head:
            state_dict[key] = state_dict[key].permute(0, 2, 1, 3)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


def mark_only_adapter_as_trainable(model: GPT) -> None:
    """Sets `requires_grad=False` for all non-adapter weights."""
    for name, param in model.named_parameters():
        param.requires_grad = adapter_filter(name, param)


def adapter_filter(key: str, value: Any) -> bool:
    return "adapter_wte" in key or "gating_factor" in key
