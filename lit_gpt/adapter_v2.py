"""Implementation of the paper:

LLaMA-Adapter V2: Parameter-Efficient Visual Instruction Model
https://arxiv.org/abs/2304.15010

Port for Lit-GPT
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type

import torch
import torch.nn as nn
from typing_extensions import Self

import lit_gpt
from lit_gpt.adapter import GPT as BaseModel
from lit_gpt.adapter import Block as BaseBlock
from lit_gpt.adapter import Config as BaseConfig
from lit_gpt.adapter import KVCache, RoPECache
from lit_gpt.model import CausalSelfAttention as BaseCausalSelfAttention
from lit_gpt.model import apply_rope
from lit_gpt.utils import map_old_state_dict_weights


@dataclass
class Config(BaseConfig):
    @property
    def mlp_class(self) -> Type:
        return getattr(lit_gpt.adapter_v2, self._mlp_class)


def adapter_filter(key: str, value: Any) -> bool:
    adapter_substrings = (
        # regular adapter v1 parameters
        "adapter_wte",
        "gating_factor",
        # adapter v2: new bias and scale used in Linear
        "adapter_scale",
        "adapter_bias",
        # adapter v2: Norm parameters are now trainable
        "norm_1",
        "norm_2",
        "ln_f",
    )
    return any(s in key for s in adapter_substrings)


class AdapterV2Linear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, **kwargs) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features, **kwargs)
        self.adapter_bias = torch.nn.Parameter(torch.zeros(out_features), requires_grad=False)
        self.adapter_scale = torch.nn.Parameter(torch.ones(out_features), requires_grad=False)
        self.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.adapter_scale * (self.linear(x) + self.adapter_bias)

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.adapter_bias)
        nn.init.ones_(self.adapter_scale)


class GPT(BaseModel):
    def __init__(self, config: Config) -> None:
        # Skip the parent class __init__ altogether and replace it to avoid useless allocations
        nn.Module.__init__(self)
        assert config.padded_vocab_size is not None
        self.config = config

        self.lm_head = AdapterV2Linear(config.n_embd, config.padded_vocab_size, bias=False)
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

    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> Self:
        return cls(Config.from_name(name, **kwargs))

    def _init_weights(self, module: nn.Module) -> None:
        """Meant to be used with `gpt.apply(gpt._init_weights)`. Unused method left for completeness."""
        super()._init_weights(module)
        if isinstance(module, CausalSelfAttention):
            module.reset_parameters()
        if isinstance(module, AdapterV2Linear):
            module.reset_parameters()

    def _load_from_state_dict(self, state_dict: Dict, prefix: str, *args: Any, **kwargs: Any) -> None:
        """For compatibility with base checkpoints."""
        mapping = {"lm_head.weight": "lm_head.linear.weight"}
        state_dict = map_old_state_dict_weights(state_dict, mapping, prefix)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


class Block(BaseBlock):
    """The implementation is identical to `lit_gpt.model.Block` with the exception that
    we replace the attention layer where adaption is implemented."""

    def __init__(self, config: Config, block_idx: int) -> None:
        # Skip the parent class __init__ altogether and replace it to avoid useless allocations
        nn.Module.__init__(self)
        self.norm_1 = config.norm_class(config.n_embd, eps=config.norm_eps)
        self.attn = CausalSelfAttention(config, block_idx)
        if not config.shared_attention_norm:
            self.norm_2 = config.norm_class(config.n_embd, eps=config.norm_eps)
        self.mlp = config.mlp_class(config)

        self.config = config


class CausalSelfAttention(BaseCausalSelfAttention):
    def __init__(self, config: Config, block_idx: int) -> None:
        """Causal self-attention with calculating qkv matrices with a single matrix* and Low Ranking Adaptation for
        parameter-efficient fine-tuning.

        *Instead of creating multiple heads and concatenating the result (in addition to creating separate matrices for
        query, key and value for each head) we can do this in a single pass with a single weight matrix.
        """
        # Skip the parent class __init__ altogether and replace it to avoid useless allocations
        nn.Module.__init__(self)
        shape = (config.n_head + 2 * config.n_query_groups) * config.head_size
        # key, query, value projections for all heads, but in a batch
        self.attn = AdapterV2Linear(in_features=config.n_embd, out_features=shape, bias=config.bias)
        # output projection
        self.proj = AdapterV2Linear(config.n_embd, config.n_embd, bias=config.bias)
        if block_idx >= config.adapter_start_layer:
            # adapter embedding layer
            self.adapter_wte = nn.Embedding(config.adapter_prompt_length, config.n_embd)
            # gate for adaption
            self.gating_factor = torch.nn.Parameter(torch.zeros(1, 1, config.n_head, 1))
            self.reset_parameters()
        self.block_idx = block_idx

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
        """For compatibility with base checkpoints."""
        mapping = {
            "attn.weight": "attn.linear.weight",
            "attn.bias": "attn.linear.bias",
            "proj.weight": "proj.linear.weight",
            "proj.bias": "proj.linear.bias",
        }
        state_dict = map_old_state_dict_weights(state_dict, mapping, prefix)
        # For compatibility with older checkpoints
        if (key := prefix + "gating_factor") in state_dict and state_dict[key].size(1) == self.config.n_head:
            state_dict[key] = state_dict[key].permute(0, 2, 1, 3)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


class GptNeoxMLP(lit_gpt.model.GptNeoxMLP):
    def __init__(self, config: Config) -> None:
        nn.Module.__init__(self)
        self.fc = AdapterV2Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        self.proj = AdapterV2Linear(config.intermediate_size, config.n_embd, bias=config.bias)

    def _load_from_state_dict(self, state_dict: Dict, prefix: str, *args: Any, **kwargs: Any) -> None:
        """For compatibility with base checkpoints."""
        mapping = {
            "fc.weight": "fc.linear.weight",
            "fc.bias": "fc.linear.bias",
            "proj.weight": "proj.linear.weight",
            "proj.bias": "proj.linear.bias",
        }
        state_dict = map_old_state_dict_weights(state_dict, mapping, prefix)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


class LLaMAMLP(lit_gpt.model.LLaMAMLP):
    def __init__(self, config: Config) -> None:
        nn.Module.__init__(self)
        self.fc_1 = AdapterV2Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        self.fc_2 = AdapterV2Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        self.proj = AdapterV2Linear(config.intermediate_size, config.n_embd, bias=config.bias)

    def _load_from_state_dict(self, state_dict: Dict, prefix: str, *args: Any, **kwargs: Any) -> None:
        """For compatibility with base checkpoints."""
        mapping = {
            "fc_1.weight": "fc_1.linear.weight",
            "fc_1.bias": "fc_1.linear.bias",
            "fc_2.weight": "fc_2.linear.weight",
            "fc_2.bias": "fc_2.linear.bias",
            "proj.weight": "proj.linear.weight",
            "proj.bias": "proj.linear.bias",
        }
        state_dict = map_old_state_dict_weights(state_dict, mapping, prefix)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


def mark_only_adapter_v2_as_trainable(model: GPT) -> None:
    """Sets requires_grad=False for all non-adapter weights"""
    for name, param in model.named_parameters():
        param.requires_grad = adapter_filter(name, param)
