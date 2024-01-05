# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

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
from lit_gpt.model import Block as BaseBlock
from lit_gpt.model import CausalSelfAttention as BaseCausalSelfAttention


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

        self.lm_head = nn.Linear(config.n_embd, config.padded_vocab_size, bias=config.lm_head_bias)
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.padded_vocab_size, config.n_embd),
                h=nn.ModuleList(Block(config, i) for i in range(config.n_layer)),
                ln_f=config.norm_class(config.n_embd, eps=config.norm_eps),
            )
        )
        self.max_seq_length = self.config.block_size
        self.mask_cache: Optional[torch.Tensor] = None

    def forward(
        self, idx: torch.Tensor, input_pos: Optional[torch.Tensor] = None, lm_head_chunk_size: int = 0
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        T = idx.size(1)
        if self.max_seq_length < T:
            raise ValueError(f"Cannot forward sequence of length {T}, max seq length is only {self.max_seq_length}.")

        if input_pos is not None:  # use the kv cache
            cos = self.cos.index_select(0, input_pos)
            sin = self.sin.index_select(0, input_pos)
            if self.mask_cache is None:
                raise TypeError("You need to call `gpt.set_kv_cache()`")
            mask = self.mask_cache.index_select(2, input_pos)
        else:
            cos = self.cos[:T]
            sin = self.sin[:T]
            mask = None

        x = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        for block in self.transformer.h:
            x = block(x, cos, sin, mask, input_pos)
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
    """A modification of `lit_gpt.model.CausalSelfAttention` that adds the attention
    over the adaption prompt."""

    def __init__(self, config: Config, block_idx: int) -> None:
        super().__init__(config)
        if block_idx >= config.adapter_start_layer:
            # adapter embedding layer
            self.adapter_wte = nn.Embedding(config.adapter_prompt_length, config.n_embd)
            # gate for adaption
            self.gating_factor = torch.nn.Parameter(torch.zeros(1, 1, config.n_head, 1))
            # kv cache for inference
            self.adapter_kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self.block_idx = block_idx

    def scaled_dot_product_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        y = super().scaled_dot_product_attention(q, k, v, mask)
        if self.block_idx < self.config.adapter_start_layer:
            return y

        aT = self.config.adapter_prompt_length
        if self.adapter_kv_cache is not None:
            # since this uses the wte weights as the prefix and the kv cache is only used during inference, ak and av
            # are the same every call
            ak, av = self.adapter_kv_cache
        else:
            prefix = self.adapter_wte.weight.reshape(1, aT, self.config.n_embd)
            aqkv = self.attn(prefix)
            q_per_kv = self.config.n_head // self.config.n_query_groups
            aqkv = aqkv.view(1, aT, self.config.n_query_groups, q_per_kv + 2, self.config.head_size)
            aqkv = aqkv.permute(0, 2, 3, 1, 4)
            _, ak, av = aqkv.split((q_per_kv, 1, 1), dim=2)
            if self.config.n_query_groups != 1:
                # for MHA this is a no-op
                ak = ak.repeat_interleave(q_per_kv, dim=2)
                av = av.repeat_interleave(q_per_kv, dim=2)
            ak = ak.view(1, -1, aT, self.config.head_size)  # (1, nh_ak, aT, hs)
            av = av.view(1, -1, aT, self.config.head_size)  # (1, nh_av, aT, hs)
            self.adapter_kv_cache = (ak, av)

        T = q.size(2)
        amask = torch.ones(T, aT, dtype=torch.bool, device=q.device)
        ay = super().scaled_dot_product_attention(q, ak, av, amask)
        return y + self.gating_factor * ay

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
