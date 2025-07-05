# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

"""Implementation of the paper:

LLaMA-Adapter: Efficient Fine-tuning of Language Models with Zero-init Attention
https://arxiv.org/abs/2303.16199

Port for LitGPT
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from typing_extensions import Self

from litgpt.attention import DefaultKeysAndValues, MultiHeadSelfAttention
from litgpt.config import Config as BaseConfig
from litgpt.kvcache.base import KVCache
from litgpt.model import GPT as BaseModel
from litgpt.model import Block as BaseBlock
from litgpt.model import CausalSelfAttention as BaseCausalSelfAttention


@dataclass
class Config(BaseConfig):
    adapter_prompt_length: int = 10
    adapter_start_layer: int = 2


class GPT(BaseModel):
    # Copy & paste from :class:`model.GPT`. Note that :class:`Block` is new here.
    def __init__(self, config: Config, **mha_kwargs) -> None:
        nn.Module.__init__(self)
        assert config.padded_vocab_size is not None
        self.config = config

        self.lm_head = nn.Linear(
            config.n_embd,
            config.padded_vocab_size,
            bias=config.lm_head_bias,
        )
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.padded_vocab_size, config.n_embd),
                h=nn.ModuleList(Block(config, block_idx) for block_idx in range(config.n_layer)),
                ln_f=config.norm_class(config.n_embd, eps=config.norm_eps),
            )
        )
        self.mha = MultiHeadSelfAttention(config, **mha_kwargs)
        self.max_seq_length = self.config.block_size
        self._start_of_layer_hook = config.start_of_layer_hook
        # Have dense KV caches been created by `set_kv_caches`?
        self._default_kv_cache = False

    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> Self:
        return cls(Config.from_name(name, **kwargs))

    def _init_weights(self, module: nn.Module) -> None:
        """Meant to be used with `gpt.apply(gpt._init_weights)`. Unused method left for completeness."""
        super()._init_weights(module)
        if isinstance(module, CausalSelfAttention):
            module.reset_parameters()


class Block(BaseBlock):
    def __init__(
        self,
        config: Config,
        block_idx: int,
        kv_cache: Optional[KVCache] = None,
    ) -> None:
        super().__init__(config, block_idx, kv_cache)
        self.attn = CausalSelfAttention(config, block_idx, kv_cache=kv_cache)


class CausalSelfAttention(BaseCausalSelfAttention):
    """A modification of `litgpt.model.CausalSelfAttention` that adds the attention
    over the adaption prompt."""

    def __init__(
        self,
        config: Config,
        block_idx: int,
        kv_cache: Optional[KVCache] = None,
    ) -> None:
        super().__init__(
            config=config,
            block_idx=block_idx,
            kv_cache=kv_cache,
        )
        self._extend_forward = block_idx >= config.adapter_start_layer
        if self._extend_forward:
            # adapter embedding layer
            self.adapter_wte = nn.Embedding(config.adapter_prompt_length, config.n_embd)
            # gate for adaption
            self.gating_factor = torch.nn.Parameter(torch.zeros(1, 1, config.n_head, 1))
            # kv cache for inference
            self.adapter_kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    def _transform_output(
        self,
        y: torch.Tensor,
        query: torch.Tensor,
        mha: MultiHeadSelfAttention,
    ) -> torch.Tensor:
        if self._extend_forward:
            B, T, _ = y.shape
            y = y.view(B, T, self.config.n_head, self.config.head_size)
            aT = self.config.adapter_prompt_length
            if self.adapter_kv_cache is not None:
                # since this uses the wte weights as the prefix and the kv cache is only used during inference, ak and av
                # are the same every call
                ak, av = self.adapter_kv_cache
            else:
                prefix = self.adapter_wte.weight.reshape(1, aT, self.config.n_embd)
                aqkv = self.qkv(prefix)
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

            amask = torch.ones(T, aT, dtype=torch.bool, device=query.device)
            a_k_and_v = DefaultKeysAndValues(keys=ak, values=av)
            ay, _ = mha.scaled_dot_product_attention(
                query=query,
                k_and_v=a_k_and_v,
                mask=amask,
            )
            y = (y + self.gating_factor * ay).view(B, T, -1)

        return y

    def reset_parameters(self) -> None:
        if hasattr(self, "gating_factor"):
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
