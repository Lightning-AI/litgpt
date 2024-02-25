# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

"""Full definition of a decoder-only transformer-based language model, all of it in this single file.

Based on the nanoGPT implementation: https://github.com/karpathy/nanoGPT and
https://github.com/EleutherAI/gpt-neox/tree/main/megatron/model.
"""

import math
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
from typing_extensions import Self

from lit_gpt.config import Config
import copy


# class IntentionGPT(nn.Module):
#     def __init__(self, config: Config) -> None:
#         super().__init__()
#         assert config.padded_vocab_size is not None
#         self.config = config

#         self.lm_head = nn.Linear(config.n_embd, config.padded_vocab_size, bias=config.lm_head_bias)

#         encoder_layer_num = 1
#         self.encoder_layer_num = encoder_layer_num
#         n_action_embd = config.n_embd
#         self.sentence_action = True
#         self.finetune = False
#         self.state_encoder = nn.ModuleDict(
#             dict(
#                 wte=nn.Embedding(config.padded_vocab_size, config.n_embd),
#                 h=nn.ModuleList(Block(config) for _ in range(encoder_layer_num)),
#                 ln_f=config.norm_class(config.n_embd, eps=config.norm_eps),
#             )
#         )
#         self.action_encoder = nn.ModuleDict(
#             dict(
#                 wte=nn.Embedding(config.padded_vocab_size, config.n_embd),
#                 h=nn.ModuleList(Block(config) for _ in range(encoder_layer_num)),
#                 ln_f=config.norm_class(config.n_embd, eps=config.norm_eps),
#             )
#         )
#         self.mean_layer = nn.Linear(config.n_embd, n_action_embd)
#         self.logvar_layer = nn.Linear(config.n_embd, n_action_embd)
#         concat_block_config = copy.deepcopy(config)
#         concat_block_config.input_n_embd = concat_block_config.n_embd + n_action_embd
#         self.concat_block = Block(concat_block_config)
#         decoder_layer_num = config.n_layer - encoder_layer_num
#         self.decoder = nn.ModuleDict(
#             dict(
#                 h=nn.ModuleList(Block(config) for _ in range(decoder_layer_num)),
#                 ln_f=config.norm_class(config.n_embd, eps=config.norm_eps),
#             )
#         )
#         self.max_seq_length = self.config.block_size
#         self.mask_cache: Optional[torch.Tensor] = None
        
#         if self.finetune:
#             self.decoder.eval()
#             self.state_encoder.eval()

#     @property
#     def max_seq_length(self) -> int:
#         return self._max_seq_length

#     @max_seq_length.setter
#     def max_seq_length(self, value: int) -> None:
#         """
#         When doing inference, the sequences used might be shorter than the model's context length.
#         This allows setting a smaller number to avoid allocating unused memory
#         """
#         if value > self.config.block_size:
#             raise ValueError(f"Cannot attend to {value}, block size is only {self.config.block_size}")
#         self._max_seq_length = value
#         if not hasattr(self, "cos"):
#             # first call
#             cos, sin = self.rope_cache()
#             self.register_buffer("cos", cos, persistent=False)
#             self.register_buffer("sin", sin, persistent=False)
#         # override
#         elif value != self.cos.size(0):
#             self.cos, self.sin = self.rope_cache(device=self.cos.device)
#         # the mask and kv cache size will get updated on `set_kv_cache`. we cannot update it here because we don't know
#         # if the kv cache is expected

#     def reset_parameters(self) -> None:
#         # Trigger resetting the rope-cache
#         self.cos, self.sin = self.rope_cache()

#     def _init_weights(self, module: nn.Module) -> None:
#         """Meant to be used with `gpt.apply(gpt._init_weights)`."""
#         if isinstance(module, nn.Linear):
#             torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
#             if module.bias is not None:
#                 torch.nn.init.zeros_(module.bias)
#         elif isinstance(module, nn.Embedding):
#             torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

#     def reparameterization(self, mean, var):
#         epsilon = torch.randn_like(var)
#         z = mean + var*epsilon
#         return z
    
#     def forward(self, idx: torch.Tensor, input_pos: Optional[torch.Tensor] = None, train_mode=False) -> torch.Tensor:
#         T = idx.size(1)
#         if self.max_seq_length < T:
#             raise ValueError(f"Cannot forward sequence of length {T}, max seq length is only {self.max_seq_length}.")

#         if input_pos is not None:  # use the kv cache
#             cos = self.cos.index_select(0, input_pos)
#             sin = self.sin.index_select(0, input_pos)
#             if self.mask_cache is None:
#                 raise TypeError("You need to call `gpt.set_kv_cache()`")
#             mask = self.mask_cache.index_select(2, input_pos)
#         else:
#             cos = self.cos[:T]
#             sin = self.sin[:T]
#             mask = None

#         x = self.state_encoder.wte(idx)  # token embeddings of shape (b, t, n_embd)
#         action_x = self.action_encoder.wte(idx)
#         for block_state, block_action in zip(self.state_encoder.h, self.action_encoder.h):
#             x = block_state(x, cos, sin, mask, input_pos)
#             aciton_x = block_action(action_x, cos, sin, mask, input_pos)
            
#         if self.sentence_action:
#             action_x = action_x[:, -1:, :].repeat(1, T, 1)
#         else:
#             action_x[:, :-1] = action_x[:, 1:]

#         mean, logvar = self.mean_layer(aciton_x), self.logvar_layer(aciton_x)
#         z = self.reparameterization(mean, logvar)

#         x = torch.cat([x, z], dim=-1)
#         for block in enumerate(self.decoder.h):
#             x = block(x, cos, sin, mask, input_pos)
#         x = self.decoder.ln_f(x)
#         # TODO need to return action
#         if not train_mode:
#             return self.lm_head(x)
        
#         return self.lm_head(x), {"mean": mean, "logvar": logvar, "z": z}  # (b, t, vocab_size)
    
#     def load_from_gpt(self, gpt_model):
#         self.state_encoder.wte.load_state_dict(gpt_model.transformer.wte.state_dict())
#         for idx, block in enumerate(self.state_encoder.h):
#             block.load_state_dict(gpt_model.transformer.h[idx].state_dict())
            
#         for idx, block in enumerate(self.decoder.h):
#             block.load_state_dict(gpt_model.transformer.h[idx + self.encoder_layer_num].state_dict())
#         self.decoder.ln_f.load_state_dict(gpt_model.transformer.ln_f.state_dict())
#         self.lm_head.load_state_dict(gpt_model.lm_head.state_dict()) 

#     @classmethod
#     def from_name(cls, name: str, **kwargs: Any) -> Self:
#         return cls(Config.from_name(name, **kwargs))

#     def rope_cache(self, device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
#         return build_rope_cache(
#             seq_len=self.max_seq_length,
#             n_elem=self.config.rope_n_elem,
#             device=device,
#             condense_ratio=self.config.rope_condense_ratio,
#             base=self.config.rope_base,
#         )

#     def set_kv_cache(
#         self,
#         batch_size: int,
#         rope_cache_length: Optional[int] = None,
#         device: Optional[torch.device] = None,
#         dtype: Optional[torch.dtype] = None,
#     ) -> None:
#         if rope_cache_length is None:
#             rope_cache_length = self.cos.size(-1)
#         max_seq_length = self.max_seq_length

#         # initialize the kv cache for all blocks
#         for block in self.transformer.h:
#             block.attn.kv_cache = block.attn.build_kv_cache(
#                 batch_size, max_seq_length, rope_cache_length, device, dtype
#             )

#         if self.mask_cache is None or self.mask_cache.size(3) != max_seq_length:
#             # passing `attn_mask` to SDPA disables the flash implementation. since we only need the mask
#             # for the kv-cache support (only during inference), we only create it in that situation
#             self.mask_cache = build_mask_cache(max_seq_length, device)

#     def clear_kv_cache(self) -> None:
#         self.mask_cache = None
#         for block in self.transformer.h:
#             block.attn.kv_cache = None


class IntentionGPT(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        assert config.padded_vocab_size is not None
        self.config = config

        self.lm_head = nn.Linear(config.n_embd, config.padded_vocab_size, bias=config.lm_head_bias)
        
        self.enc_layer_num = 1
        self.dyna_layer_num = 1
        self.transformer_enc = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.padded_vocab_size, config.n_embd),
                h=nn.ModuleList(Block(config) for _ in range(self.enc_layer_num)),
            )
        )
        self.transformer_act = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.padded_vocab_size, config.n_embd),
                h=nn.ModuleList(Block(config) for _ in range(self.enc_layer_num)),
            )
        )
        self.mean_layer = nn.Linear(config.n_embd, config.n_embd)
        self.logvar_layer = nn.Linear(config.n_embd, config.n_embd)
        self.concat_layer = nn.Linear(config.n_embd + config.n_embd, config.n_embd)
        
        self.transformer_dyna = nn.ModuleDict(
            dict(
                h=nn.ModuleList(Block(config) for _ in range(self.dyna_layer_num)),
            )
        )
        self.transformer_dec = nn.ModuleDict(
            dict(
                h=nn.ModuleList(Block(config) for _ in range(config.n_layer - self.enc_layer_num - self.dyna_layer_num)),
                ln_f=config.norm_class(config.n_embd, eps=config.norm_eps),
            )
        )
        
        self.max_seq_length = self.config.block_size
        self.mask_cache: Optional[torch.Tensor] = None

    @property
    def max_seq_length(self) -> int:
        return self._max_seq_length

    @max_seq_length.setter
    def max_seq_length(self, value: int) -> None:
        """
        When doing inference, the sequences used might be shorter than the model's context length.
        This allows setting a smaller number to avoid allocating unused memory
        """
        if value > self.config.block_size:
            raise ValueError(f"Cannot attend to {value}, block size is only {self.config.block_size}")
        self._max_seq_length = value
        if not hasattr(self, "cos"):
            # first call
            cos, sin = self.rope_cache()
            self.register_buffer("cos", cos, persistent=False)
            self.register_buffer("sin", sin, persistent=False)
        # override
        elif value != self.cos.size(0):
            self.cos, self.sin = self.rope_cache(device=self.cos.device)
        # the mask and kv cache size will get updated on `set_kv_cache`. we cannot update it here because we don't know
        # if the kv cache is expected

    def reset_parameters(self) -> None:
        # Trigger resetting the rope-cache
        self.cos, self.sin = self.rope_cache()

    def _init_weights(self, module: nn.Module) -> None:
        """Meant to be used with `gpt.apply(gpt._init_weights)`."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var)
        z = mean + var*epsilon
        return z

    def forward(self, idx: torch.Tensor, input_pos: Optional[torch.Tensor] = None, train_mode=False) -> torch.Tensor:
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

        x = self.transformer_enc.wte(idx)  # token embeddings of shape (b, t, n_embd)
        x_act = self.transformer_act.wte(idx)  # token embeddings of shape (b, t, n_embd)
        for block, block_a in zip(self.transformer_enc.h, self.transformer_act.h):
            x = block(x, cos, sin, mask, input_pos)
            x_act = block_a(x_act, cos, sin, mask, input_pos)
            
        # x_act = x_act[:, -1:, :]
        # mean, logvar = self.mean_layer(x_act), self.logvar_layer(x_act)
        # z = self.reparameterization(mean, torch.exp(logvar))
        # x = torch.cat([z, x], dim=1)
        # cos_ = torch.cat([torch.zeros_like(cos[:1]), cos], dim=0)
        # sin_ = torch.cat([torch.zeros_like(sin[:1]), sin], dim=0)
        # mask_ = torch.cat([torch.ones_like(mask[:1]), mask], dim=0) if mask is not None else None
        # input_pos_ = torch.cat([torch.ones_like(input_pos[:1]), input_pos], dim=0) if input_pos is not None else None
        # for block in self.transformer_dyna.h:
        #     x = block(x, cos_, sin_, mask_, input_pos_)
        # x = x[:, 1:]
        
        mean, logvar = self.mean_layer(x_act), self.logvar_layer(x_act)
        z = self.reparameterization(mean, torch.exp(logvar))
        
        x = torch.cat([x, z], dim=-1)
        x = self.concat_layer(x)
        for block in self.transformer_dec.h:
            x = block(x, cos, sin, mask, input_pos)
        
        for block in self.transformer_dec.h:
            x = block(x, cos, sin, mask, input_pos)
            
        x = self.transformer_dec.ln_f(x)
        if not train_mode:
            return self.lm_head(x)
        
        return self.lm_head(x), {"mean": mean, "logvar": logvar, "z": z}  # (b, t, vocab_size)

    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> Self:
        return cls(Config.from_name(name, **kwargs))

    def rope_cache(self, device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        return build_rope_cache(
            seq_len=self.max_seq_length,
            n_elem=self.config.rope_n_elem,
            device=device,
            condense_ratio=self.config.rope_condense_ratio,
            base=self.config.rope_base,
        )

    def set_kv_cache(
        self,
        batch_size: int,
        rope_cache_length: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        if rope_cache_length is None:
            rope_cache_length = self.cos.size(-1)
        max_seq_length = self.max_seq_length

        # initialize the kv cache for all blocks
        for block in self.transformer.h:
            block.attn.kv_cache = block.attn.build_kv_cache(
                batch_size, max_seq_length, rope_cache_length, device, dtype
            )

        if self.mask_cache is None or self.mask_cache.size(3) != max_seq_length:
            # passing `attn_mask` to SDPA disables the flash implementation. since we only need the mask
            # for the kv-cache support (only during inference), we only create it in that situation
            self.mask_cache = build_mask_cache(max_seq_length, device)

    def clear_kv_cache(self) -> None:
        self.mask_cache = None
        for block in self.transformer.h:
            block.attn.kv_cache = None



class GPT(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        assert config.padded_vocab_size is not None
        self.config = config

        self.lm_head = nn.Linear(config.n_embd, config.padded_vocab_size, bias=config.lm_head_bias)
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.padded_vocab_size, config.n_embd),
                h=nn.ModuleList(Block(config) for _ in range(config.n_layer)),
                ln_f=config.norm_class(config.n_embd, eps=config.norm_eps),
            )
        )
        self.max_seq_length = self.config.block_size
        self.mask_cache: Optional[torch.Tensor] = None

    @property
    def max_seq_length(self) -> int:
        return self._max_seq_length

    @max_seq_length.setter
    def max_seq_length(self, value: int) -> None:
        """
        When doing inference, the sequences used might be shorter than the model's context length.
        This allows setting a smaller number to avoid allocating unused memory
        """
        if value > self.config.block_size:
            raise ValueError(f"Cannot attend to {value}, block size is only {self.config.block_size}")
        self._max_seq_length = value
        if not hasattr(self, "cos"):
            # first call
            cos, sin = self.rope_cache()
            self.register_buffer("cos", cos, persistent=False)
            self.register_buffer("sin", sin, persistent=False)
        # override
        elif value != self.cos.size(0):
            self.cos, self.sin = self.rope_cache(device=self.cos.device)
        # the mask and kv cache size will get updated on `set_kv_cache`. we cannot update it here because we don't know
        # if the kv cache is expected

    def reset_parameters(self) -> None:
        # Trigger resetting the rope-cache
        self.cos, self.sin = self.rope_cache()

    def _init_weights(self, module: nn.Module) -> None:
        """Meant to be used with `gpt.apply(gpt._init_weights)`."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, input_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
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
        return self.lm_head(x)  # (b, t, vocab_size)

    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> Self:
        return cls(Config.from_name(name, **kwargs))

    def rope_cache(self, device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        return build_rope_cache(
            seq_len=self.max_seq_length,
            n_elem=self.config.rope_n_elem,
            device=device,
            condense_ratio=self.config.rope_condense_ratio,
            base=self.config.rope_base,
        )

    def set_kv_cache(
        self,
        batch_size: int,
        rope_cache_length: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        if rope_cache_length is None:
            rope_cache_length = self.cos.size(-1)
        max_seq_length = self.max_seq_length

        # initialize the kv cache for all blocks
        for block in self.transformer.h:
            block.attn.kv_cache = block.attn.build_kv_cache(
                batch_size, max_seq_length, rope_cache_length, device, dtype
            )

        if self.mask_cache is None or self.mask_cache.size(3) != max_seq_length:
            # passing `attn_mask` to SDPA disables the flash implementation. since we only need the mask
            # for the kv-cache support (only during inference), we only create it in that situation
            self.mask_cache = build_mask_cache(max_seq_length, device)

    def clear_kv_cache(self) -> None:
        self.mask_cache = None
        for block in self.transformer.h:
            block.attn.kv_cache = None


class Block(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.norm_1 = config.norm_class(config.n_embd, eps=config.norm_eps)
        self.attn = CausalSelfAttention(config)
        self.norm_2 = None if config.shared_attention_norm else config.norm_class(config.n_embd, eps=config.norm_eps)
        self.mlp = config.mlp_class(config)

        self.config = config

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        n_1 = self.norm_1(x)
        h = self.attn(n_1, cos, sin, mask, input_pos)
        if self.config.parallel_residual:
            n_2 = n_1 if self.config.shared_attention_norm else self.norm_2(x)
            x = self.mlp(n_2) + h + x
        else:
            if self.config.shared_attention_norm:
                raise NotImplementedError(
                    "No checkpoint amongst the ones we support uses this configuration"
                    " (non-parallel residual and shared attention norm)."
                )
            x = h + x
            x = self.mlp(self.norm_2(x)) + x
        return x


class CausalSelfAttention(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        shape = (config.n_head + 2 * config.n_query_groups) * config.head_size
        # key, query, value projections for all heads, but in a batch
        self.attn = nn.Linear(config.n_embd, shape, bias=config.bias)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # disabled by default
        self.kv_cache: Optional[KVCache] = None

        self.config = config

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        qkv = self.attn(x)

        # assemble into a number of query groups to support MHA, MQA and GQA together (see `config.n_query_groups`)
        q_per_kv = self.config.n_head // self.config.n_query_groups
        total_qkv = q_per_kv + 2  # each group has 1+ queries, 1 key, and 1 value
        qkv = qkv.view(B, T, self.config.n_query_groups, total_qkv, self.config.head_size)
        qkv = qkv.permute(0, 2, 3, 1, 4)  # (B, n_query_groups, total_qkv, T, hs)

        # split batched computation into three
        q, k, v = qkv.split((q_per_kv, 1, 1), dim=2)

        # maybe repeat k and v if for the non multi-head attention cases
        # training: flash attention requires it
        # inference: multi-query would require a full kv cache so avoid it to limit its memory usage
        if self.config.n_query_groups != self.config.n_head and (input_pos is None or self.config.n_query_groups != 1):
            k = k.expand(B, self.config.n_query_groups, q_per_kv, T, self.config.head_size)
            v = v.expand(B, self.config.n_query_groups, q_per_kv, T, self.config.head_size)

        q = q.reshape(B, -1, T, self.config.head_size)  # (B, nh_q, T, hs)
        k = k.reshape(B, -1, T, self.config.head_size)  # (B, nh_k, T, hs)
        v = v.reshape(B, -1, T, self.config.head_size)  # (B, nh_v, T, hs)

        q_roped = apply_rope(q[..., : self.config.rope_n_elem], cos, sin)
        k_roped = apply_rope(k[..., : self.config.rope_n_elem], cos, sin)
        q = torch.cat((q_roped, q[..., self.config.rope_n_elem :]), dim=-1)
        k = torch.cat((k_roped, k[..., self.config.rope_n_elem :]), dim=-1)

        if input_pos is not None:
            if not isinstance(self.kv_cache, KVCache):
                raise TypeError("You need to call `gpt.set_kv_cache()`")
            k, v = self.kv_cache(input_pos, k, v)

        y = self.scaled_dot_product_attention(q, k, v, mask)

        y = y.reshape(B, T, self.config.n_embd)  # re-assemble all head outputs side by side

        # output projection
        return self.proj(y)

    def scaled_dot_product_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        scale = 1.0 / math.sqrt(self.config.head_size)
        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=0.0, scale=scale, is_causal=mask is None
        )
        return y.transpose(1, 2)

    def build_kv_cache(
        self,
        batch_size: int,
        max_seq_length: int,
        rope_cache_length: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "KVCache":
        heads = 1 if self.config.n_query_groups == 1 else self.config.n_head
        v_shape = (batch_size, heads, max_seq_length, self.config.head_size)
        if rope_cache_length is None:
            if self.config.rotary_percentage != 1.0:
                raise TypeError("Please pass the `rope_cache_length=gpt.cos.size(-1)` value")
            k_shape = v_shape
        else:
            k_shape = (
                batch_size,
                heads,
                max_seq_length,
                rope_cache_length + self.config.head_size - self.config.rope_n_elem,
            )
        return KVCache(k_shape, v_shape, device=device, dtype=dtype)


class GptNeoxMLP(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.fc = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        self.proj = nn.Linear(config.intermediate_size, config.n_embd, bias=config.bias)

        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = torch.nn.functional.gelu(x, approximate=self.config.gelu_approximate)
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


class LLaMAMoE(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.gate = nn.Linear(config.n_embd, config.n_expert, bias=False)
        self.experts = nn.ModuleList(LLaMAMLP(config) for _ in range(config.n_expert))

        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Derived from: https://github.com/mistralai/mistral-src/blob/b46d6/moe_one_file_ref.py#L203-L219
        See also figure 1 in https://arxiv.org/abs/2211.15841
        """
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)
        x = x.view(-1, C)  # (B*T, C)
        router = self.gate(x)  # (B*T, n_expert)
        probs, indices = torch.topk(router, self.config.n_expert_per_token)  # (B*T, n_expert_per_token)
        probs = probs.softmax(dim=1, dtype=torch.float).to(dtype=x.dtype)
        masks = indices.unsqueeze(-1) == torch.arange(self.config.n_expert, device=x.device)
        masks = masks.permute(2, 0, 1)  # (n_expert, B*T, n_expert_per_token)
        y = torch.zeros_like(x)  # (B*T, C)
        for mask, expert in zip(masks, self.experts):
            token_idx, expert_idx = torch.where(mask)
            y[token_idx] += probs[token_idx, expert_idx, None] * expert(x[token_idx])
        return y.view(B, T, C)


def build_rope_cache(
    seq_len: int, n_elem: int, device: Optional[torch.device] = None, base: int = 10000, condense_ratio: int = 1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Enhanced Transformer with Rotary Position Embedding.

    Derived from: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/
    transformers/rope/__init__.py. MIT License:
    https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/license.
    """
    # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
    theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, device=device).float() / n_elem))

    # Create position indexes `[0, 1, ..., seq_len - 1]`
    seq_idx = torch.arange(seq_len, device=device) / condense_ratio

    # Calculate the product of position index and $\theta_i$
    idx_theta = torch.outer(seq_idx, theta).repeat(1, 2)

    return torch.cos(idx_theta), torch.sin(idx_theta)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    head_size = x.size(-1)
    x1 = x[..., : head_size // 2]  # (B, nh, T, hs/2)
    x2 = x[..., head_size // 2 :]  # (B, nh, T, hs/2)
    rotated = torch.cat((-x2, x1), dim=-1)  # (B, nh, T, hs)
    roped = (x * cos) + (rotated * sin)
    return roped.to(dtype=x.dtype)


class KVCache(nn.Module):
    def __init__(
        self,
        k_shape: Tuple[int, int, int, int],
        v_shape: Tuple[int, int, int, int],
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.register_buffer("k", torch.zeros(k_shape, device=device, dtype=dtype), persistent=False)
        self.register_buffer("v", torch.zeros(v_shape, device=device, dtype=dtype), persistent=False)

    def forward(self, input_pos: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # move the buffer to the activation dtype for when AMP is used
        self.k = self.k.to(k.dtype)
        self.v = self.v.to(v.dtype)
        # update the cache
        k = self.k.index_copy_(2, input_pos, k)
        v = self.v.index_copy_(2, input_pos, v)
        return k, v

    def reset_parameters(self) -> None:
        torch.nn.init.zeros_(self.k)
        torch.nn.init.zeros_(self.v)


def build_mask_cache(max_seq_length: int, device: Optional[torch.device] = None) -> torch.Tensor:
    ones = torch.ones((max_seq_length, max_seq_length), device=device, dtype=torch.bool)
    return torch.tril(ones).unsqueeze(0).unsqueeze(0)
