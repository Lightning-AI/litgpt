# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# BSD 3-Clause License
#
# Copyright (c) 2022, Tri Dao, trid@cs.stanford.edu.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from einops import rearrange
from transformers import PretrainedConfig, PreTrainedModel
from transformers.activations import ACT2FN
from transformers.modeling_outputs import CausalLMOutputWithPast


class MixFormerSequentialConfig(PretrainedConfig):
    """MixFormer (sequential for DeepSpeed) configuration."""

    model_type = "mixformer-sequential"

    attribute_map = {
        "max_position_embeddings": "n_positions",
        "hidden_size": "n_embd",
        "num_attention_heads": "n_head",
        "num_hidden_layers": "n_layer",
        "input_emb_layer": "embd_layer",  # `input_emb_layer` key is for backward compatibility
        "blocks": "architecture",  # `blocks` key is for backward compatibility
    }

    def __init__(
        self,
        vocab_size: Optional[int] = 50304,
        n_positions: Optional[int] = 2048,
        n_embd: Optional[int] = 1024,
        n_layer: Optional[int] = 20,
        n_inner: Optional[int] = None,
        n_head: Optional[int] = 16,
        rotary_dim: Optional[int] = 32,
        activation_function: Optional[str] = "gelu_new",
        embd_layer: Optional[str] = "default",
        architecture: Union[Dict[str, Any], List[Dict[str, Any]]] = None,
        embd_pdrop: Optional[float] = 0.0,
        resid_pdrop: Optional[float] = 0.0,
        layer_norm_epsilon: Optional[float] = 1e-5,
        initializer_range: Optional[float] = 0.02,
        tie_word_embeddings: Optional[bool] = False,
        pad_vocab_size_multiple: Optional[int] = 64,
        **kwargs,
    ) -> None:
        self.vocab_size = int(math.ceil(vocab_size / pad_vocab_size_multiple) * pad_vocab_size_multiple)
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_inner = n_inner
        self.n_head = n_head
        self.rotary_dim = min(rotary_dim, n_embd // n_head)
        self.activation_function = activation_function
        self.embd_layer = embd_layer
        self.architecture = architecture
        self.embd_pdrop = embd_pdrop
        self.resid_pdrop = resid_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


@dataclass
class InferenceParams:
    """Inference parameters that are passed to the main model in order
    to efficienly calculate and store the context during inference.
    Adapted from https://github.com/Dao-AILab/flash-attention."""

    max_sequence_len: int
    max_batch_size: int
    sequence_len_offset: int = 0
    batch_size_offset: int = 0
    key_value_memory_dict: dict = field(default_factory=dict)
    fused_ft_kernel: bool = False
    lengths_per_sample: Optional[torch.Tensor] = None


class Embedding(nn.Module):
    """Token embedding with dropout."""

    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__()

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)

    def forward(self, input_ids: torch.LongTensor) -> torch.FloatTensor:
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        hidden_states = self.wte(input_ids)
        return self.drop(hidden_states)


class RotaryEmbedding(nn.Module):
    """PyTorch implementation of `flash-attn` RotaryEmbedding layer.
    Adapted from https://github.com/Dao-AILab/flash-attention."""

    def __init__(
        self,
        dim: int,
        base: Optional[int] = 10000,
        scale_base: Optional[float] = None,
        device: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()

        if scale_base is not None:
            raise NotImplementedError

        # Generate and save the inverse frequency buffer (non-trainable)
        self.dim = dim
        self.base = base
        self.scale_base = scale_base
        self.device = device

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq)

        scale = (
            (torch.arange(0, dim, 2, device=device, dtype=torch.float32) + 0.4 * dim) / (1.4 * dim)
            if scale_base is not None
            else None
        )
        self.register_buffer("scale", scale)

        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
        self._cos_k_cached = None
        self._sin_k_cached = None

    def _update_cos_sin_cache(self, x: torch.FloatTensor, seqlen_offset: Optional[int] = 0) -> None:
        # Reset the tables if the sequence length has changed,
        # or if we're on a new device (possibly due to tracing for instance)
        seqlen = x.shape[1] + seqlen_offset

        # Re-generate the inverse frequency buffer if it's not fp32
        # (for instance if model.half() was called)
        if self.inv_freq.dtype != "torch.float32":
            self.inv_freq = 1.0 / (
                self.base ** (torch.arange(0, self.dim, 2, device=self.device, dtype=torch.float32) / self.dim)
            )

        if seqlen > self._seq_len_cached or self._cos_cached.device != x.device or self._cos_cached.dtype != x.dtype:
            self._seq_len_cached = seqlen
            t = torch.arange(seqlen, device=x.device, dtype=torch.float32)

            # Don't do einsum, it converts fp32 to fp16
            # freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            freqs = torch.outer(t, self.inv_freq.to(device=t.device, dtype=torch.float32))
            if self.scale is None:
                self._cos_cached = torch.cos(freqs).to(x.dtype)
                self._sin_cached = torch.sin(freqs).to(x.dtype)
            else:
                power = (
                    torch.arange(seqlen, dtype=self.scale.dtype, device=self.scale.device) - seqlen // 2
                ) / self.scale_base
                scale = self.scale.to(device=power.device) ** rearrange(power, "s -> s 1")

                # We want the multiplication by scale to happen in fp32
                self._cos_cached = (torch.cos(freqs) * scale).to(x.dtype)
                self._sin_cached = (torch.sin(freqs) * scale).to(x.dtype)
                self._cos_k_cached = (torch.cos(freqs) / scale).to(x.dtype)
                self._sin_k_cached = (torch.sin(freqs) / scale).to(x.dtype)

    def apply_rotary_emb_qkv(
        self,
        qkv: torch.FloatTensor,
        sin: torch.FloatTensor,
        cos: torch.FloatTensor,
        sin_k: Optional[torch.FloatTensor] = None,
        cos_k: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        _, seqlen, three, _, headdim = qkv.shape
        assert three == 3

        rotary_seqlen, rotary_dim = cos.shape
        rotary_dim *= 2
        assert rotary_dim <= headdim
        assert seqlen <= rotary_seqlen

        cos_k = cos if cos_k is None else cos_k
        sin_k = sin if sin_k is None else sin_k
        assert sin.shape == cos_k.shape == sin_k.shape == (rotary_seqlen, rotary_dim // 2)

        q_rot = qkv[:, :, 0, :, :rotary_dim]
        q_pass = qkv[:, :, 0, :, rotary_dim:]

        k_rot = qkv[:, :, 1, :, :rotary_dim]
        k_pass = qkv[:, :, 1, :, rotary_dim:]

        # Splits the queries and keys in half
        q1, q2 = q_rot.chunk(2, dim=-1)
        k1, k2 = k_rot.chunk(2, dim=-1)
        c, s = rearrange(cos[:seqlen], "s d -> s 1 d"), rearrange(sin[:seqlen], "s d -> s 1 d")

        # Casts to fp32 are necessary to prevent fp16 overflow issues
        q1, q2, k1, k2, c, s = [t.to(dtype=torch.float32) for t in [q1, q2, k1, k2, c, s]]

        # Computes the new keys and queries, recasting to original dtype
        q_rot = torch.cat([q1 * c - q2 * s, q1 * s + q2 * c], axis=-1).to(qkv.dtype)

        k_rot = torch.cat([k1 * c - k2 * s, k1 * s + k2 * c], axis=-1).to(qkv.dtype)

        return torch.cat(
            [
                torch.cat([q_rot, q_pass], axis=-1).unsqueeze(2),
                torch.cat([k_rot, k_pass], axis=-1).unsqueeze(2),
                qkv[:, :, 2:3, :, :],
            ],
            axis=2,
        )

    def forward(self, qkv: torch.Tensor, seqlen_offset: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform the forward pass.

        Args:
            qkv: Query, key and value tensors of shape (batch, seqlen, nheads, headdim) or (batch, seqlen, 3, nheads, headdim).
            seqlen_offset: Used in generation where the passed `qkv` is only the last token in the batch.

        Returns:
            New `qkv` and the cached sinusoids.

        """

        self._update_cos_sin_cache(qkv, seqlen_offset)

        return self.apply_rotary_emb_qkv(qkv, self._sin_cached[seqlen_offset:], self._cos_cached[seqlen_offset:])


def _update_kv_cache(kv, inference_params, layer_idx):
    """kv: (batch_size, seqlen, 2, nheads, head_dim) or (batch_size, 1, 2, nheads, head_dim)
    Adapted from https://github.com/Dao-AILab/flash-attention."""
    # Pre-allocate memory for key-values for inference.
    num_heads, head_dim = kv.shape[-2:]
    if layer_idx not in inference_params.key_value_memory_dict:
        kv_cache = torch.empty(
            inference_params.max_batch_size,
            inference_params.max_sequence_len,
            2,
            num_heads,
            head_dim,
            dtype=kv.dtype,
            device=kv.device,
        )
        inference_params.key_value_memory_dict[layer_idx] = kv_cache
    else:
        kv_cache = inference_params.key_value_memory_dict[layer_idx]

    # Adjust key and value for inference
    batch_start = inference_params.batch_size_offset
    batch_end = batch_start + kv.shape[0]
    sequence_start = inference_params.sequence_len_offset
    sequence_end = sequence_start + kv.shape[1]
    assert batch_end <= (kv_cache.shape[0] if kv_cache is not None else v_cache.shape[0])
    assert sequence_end <= (kv_cache.shape[1] if kv_cache is not None else v_cache.shape[2])

    assert kv_cache is not None
    kv_cache[batch_start:batch_end, sequence_start:sequence_end, ...] = kv
    return kv_cache[batch_start:batch_end, :sequence_end, ...]


class MLP(nn.Module):
    """Multi-Layer Perceptron.

    Reference:
        Attention Is All You Need.
        https://arxiv.org/pdf/1706.03762.pdf.

    """

    def __init__(self, config: PretrainedConfig, n_inner: Optional[int] = None, act_fn: Optional[str] = None) -> None:
        super().__init__()

        act_fn = config.activation_function if act_fn is None else act_fn
        assert act_fn in ACT2FN, f"`act_fn` must be one of: {ACT2FN.keys()}."

        n_inner = getattr(config, "n_inner", None) if n_inner is None else n_inner
        n_inner = n_inner if n_inner is not None else 4 * config.n_embd

        self.fc1 = nn.Linear(config.n_embd, n_inner)
        self.fc2 = nn.Linear(n_inner, config.n_embd)
        self.act = ACT2FN[act_fn]

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        old_keys = [prefix + "fc_in.weight", prefix + "fc_out.weight", prefix + "fc_in.bias", prefix + "fc_out.bias"]
        new_keys = [prefix + "fc1.weight", prefix + "fc2.weight", prefix + "fc1.bias", prefix + "fc2.bias"]

        if all(k in state_dict for k in old_keys) and not all(k in state_dict for k in new_keys):
            # Older version of `MLP` saved with different key names.
            for old_key, new_key in zip(old_keys, new_keys):
                state_dict[new_key] = state_dict.pop(old_key)

        return super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.act(hidden_states)
        return self.fc2(hidden_states)


class FusedMLP(nn.Module):
    """Fused Multi-Layer Perceptron from `flash-attn`.

    Reference:
        https://github.com/HazyResearch/flash-attention/blob/main/flash_attn/ops/fused_dense.py.

    """

    def __init__(
        self,
        config: PretrainedConfig,
        n_inner: Optional[int] = None,
        act_fn: Optional[str] = None,
        raise_on_missing: bool = False,
    ) -> None:
        super().__init__()

        act_fn = config.activation_function if act_fn is None else act_fn
        assert act_fn in ACT2FN, f"`act_fn` must be one of: {ACT2FN.keys()}."

        n_inner = getattr(config, "n_inner", None) if n_inner is None else n_inner
        n_inner = n_inner if n_inner is not None else 4 * config.n_embd

        self.mlp = MLP(config, n_inner=n_inner, act_fn=act_fn)

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        return self.mlp(hidden_states)


class SelfAttention(nn.Module):
    """Implement the scaled dot product attention with softmax.
    Adapted from https://github.com/Dao-AILab/flash-attention.
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.0)
    """

    def __init__(self, causal=False, softmax_scale=None, attention_dropout=0.0):
        super().__init__()
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.drop = nn.Dropout(attention_dropout)

    def forward(self, qkv, causal=None, key_padding_mask=None):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            qkv: The tensor containing the query, key, and value. (B, S, 3, H, D)
            causal: if passed, will override self.causal
            key_padding_mask: boolean mask to apply to the attention weights. True means to keep,
                False means to mask out. (B, S)
        """
        batch_size, seqlen = qkv.shape[0], qkv.shape[1]
        causal = self.causal if causal is None else causal
        q, k, v = qkv.unbind(dim=2)

        softmax_scale = self.softmax_scale or 1.0 / math.sqrt(q.shape[-1])
        scores = torch.einsum("bthd,bshd->bhts", q, k * softmax_scale)
        if key_padding_mask is not None:
            padding_mask = torch.full((batch_size, seqlen), -10000.0, dtype=scores.dtype, device=scores.device)
            padding_mask.masked_fill_(key_padding_mask, 0.0)
            # TD [2022-09-30]: Adding is faster than masked_fill_ (idk why, just better kernel I guess)
            scores = scores + rearrange(padding_mask, "b s -> b 1 1 s")
        if causal:
            # "triu_tril_cuda_template" not implemented for 'BFloat16'
            # So we have to construct the mask in float
            causal_mask = torch.triu(torch.full((seqlen, seqlen), -10000.0, device=scores.device), 1)
            # TD [2022-09-30]: Adding is faster than masked_fill_ (idk why, just better kernel I guess)
            scores = scores + causal_mask.to(dtype=scores.dtype)
        attention = torch.softmax(scores, dim=-1, dtype=v.dtype)
        attention_drop = self.drop(attention)
        return torch.einsum("bhts,bshd->bthd", attention_drop, v)


class CrossAttention(nn.Module):
    """Implement the scaled dot product attention with softmax.
    Adapted from https://github.com/Dao-AILab/flash-attention.
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.0)
    """

    def __init__(self, causal=False, softmax_scale=None, attention_dropout=0.0):
        super().__init__()
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.drop = nn.Dropout(attention_dropout)

    def forward(self, q, kv, causal=None, key_padding_mask=None):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            q: The tensor containing the query. (B, Sq, H, D)
            kv: The tensor containing the key and value. (B, Sk, 2, H, D)
            causal: if passed, will override self.causal
            key_padding_mask: boolean mask to apply to the attention weights. True means to keep,
                False means to mask out. (B, Sk)
        """
        batch_size, seqlen_q = q.shape[0], q.shape[1]
        causal = self.causal if causal is None else causal
        seqlen_k = kv.shape[1]
        assert kv.shape[0] == batch_size
        assert kv.shape[3] == q.shape[2]
        assert kv.shape[4] == q.shape[3]
        k, v = kv.unbind(dim=2)
        softmax_scale = self.softmax_scale or 1.0 / math.sqrt(q.shape[-1])
        scores = torch.einsum("bthd,bshd->bhts", q, k * softmax_scale)
        if key_padding_mask is not None:
            padding_mask = torch.full((batch_size, seqlen_k), -10000.0, dtype=scores.dtype, device=scores.device)
            padding_mask.masked_fill_(key_padding_mask, 0.0)
            # TD [2022-09-30]: Adding is faster than masked_fill_ (idk why, just better kernel I guess)
            scores = scores + rearrange(padding_mask, "b s -> b 1 1 s")
        if causal:
            # "triu_tril_cuda_template" not implemented for 'BFloat16'
            # So we have to construct the mask in float
            causal_mask = torch.triu(torch.full((seqlen_q, seqlen_k), -10000.0, device=scores.device), 1)
            # TD [2022-09-30]: Adding is faster than masked_fill_ (idk why, just better kernel I guess)
            scores = scores + causal_mask.to(dtype=scores.dtype)
        attention = torch.softmax(scores, dim=-1, dtype=v.dtype)
        attention_drop = self.drop(attention)
        return torch.einsum("bhts,bshd->bthd", attention_drop, v)


def find_mha_dims(
    config: PretrainedConfig, n_head: Optional[int] = None, head_dim: Optional[int] = None
) -> Tuple[int, int]:
    """Validate and return the number of heads and head dimension for multi-head attention.

    Args:
        config: Model configuration.
        n_head: Number of heads.
        head_dim: Head dimension.

    Returns:
        Number of heads and head dimension.

    """

    assert all(
        hasattr(config, attr) for attr in ["n_embd", "n_head"]
    ), "`config` must have `n_embd` and `n_head` attributes."

    if head_dim is None:
        assert (
            config.n_embd % config.n_head == 0
        ), f"Hidden size ({config.n_embd}) must be divisible by the number of heads ({config.n_head})."

    if n_head is None and head_dim is None:
        head_dim = config.n_embd // config.n_head
        n_head = config.n_head
    elif n_head is None or head_dim is None:
        raise ValueError("`n_head` and `head_dim` must be both specified or `None`.")

    return n_head, head_dim


class MHA(nn.Module):
    """Multi-head attention layer.
    Adapted from https://github.com/Dao-AILab/flash-attention."""

    def __init__(
        self,
        config: PretrainedConfig,
        rotary_dim: Optional[int] = None,
        n_head: Optional[int] = None,
        head_dim: Optional[int] = None,
        bias: Optional[bool] = True,
        dropout: Optional[float] = 0.0,
        softmax_scale: Optional[float] = None,
        causal: Optional[bool] = True,
        layer_idx: Optional[int] = None,
        rotary_emb_scale_base: Optional[float] = None,
        return_residual: Optional[bool] = False,
        checkpointing: Optional[bool] = False,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        fused_dense: Optional[bool] = True,
        flash_attn: Optional[bool] = True,
        cutlass_attn: Optional[bool] = False,
        flash_rotary: Optional[bool] = True,
        raise_on_missing: Optional[bool] = False,
    ) -> None:
        super().__init__()

        factory_kwargs = {"device": device, "dtype": dtype}
        n_head, head_dim = find_mha_dims(config, n_head, head_dim)

        self.hidden_size = config.n_embd
        self.n_head = n_head
        self.head_dim = head_dim
        self.op_size = n_head * head_dim

        self.causal = causal
        self.layer_idx = layer_idx
        self.rotary_emb_dim = rotary_dim if rotary_dim is not None else getattr(config, "rotary_dim", 0)
        self.fused_dense = fused_dense
        self.flash_attn = flash_attn
        self.cutlass_attn = cutlass_attn
        self.flash_rotary = flash_rotary
        self.return_residual = return_residual
        self.checkpointing = checkpointing

        if self.rotary_emb_dim > 0:
            rotary_kwargs = {"device": device}
            if rotary_emb_scale_base is not None and rotary_emb_scale_base > 0.0:
                rotary_kwargs["scale_base"] = rotary_emb_scale_base

            self.rotary_emb = RotaryEmbedding(self.rotary_emb_dim, **rotary_kwargs)
        else:
            pass

        self.Wqkv = nn.Linear(self.hidden_size, 3 * self.op_size, bias=bias, **factory_kwargs)
        self.out_proj = nn.Linear(self.op_size, self.hidden_size, bias=bias, **factory_kwargs)

        self.inner_attn = SelfAttention(causal=causal, softmax_scale=softmax_scale, attention_dropout=dropout)
        self.inner_cross_attn = CrossAttention(causal=causal, softmax_scale=softmax_scale, attention_dropout=dropout)

    def _update_kv_cache(self, kv: torch.FloatTensor, inference_params: InferenceParams) -> None:
        """kv: (batch_size, seqlen, 2, nheads, head_dim) or (batch_size, 1, 2, nheads, head_dim)
        Adapted from https://github.com/Dao-AILab/flash-attention."""

        assert self.layer_idx is not None, "Generation requires layer_idx in the constructor"

        return _update_kv_cache(kv, inference_params, self.layer_idx)

    def forward(
        self,
        x: torch.FloatTensor,
        x_kv: Optional[torch.FloatTensor] = None,
        key_padding_mask: Optional[torch.BoolTensor] = None,
        cu_seqlens: Optional[torch.LongTensor] = None,
        max_seqlen: Optional[int] = None,
        mixer_subset: Optional[torch.LongTensor] = None,
        past_cache: Optional[InferenceParams] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Perform the forward pass.

        Args:
            x: (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim) if
                cu_seqlens is None and max_seqlen is None, else (total, hidden_dim) where total
                is the is the sum of the sequence lengths in the batch.
            x_kv: (batch, seqlen, hidden_dim), only applicable for cross-attention. If None, use x.
            key_padding_mask: boolean mask, True means to keep, False means to mask out.
                (batch, seqlen). Only applicable when not using FlashAttention.
            cu_seqlens: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
                of the sequences in the batch, used to index into x. Only applicable when using
                FlashAttention.
            max_seqlen: int. Maximum sequence length in the batch.
            mixer_subset: for cross-attention only. If not None, will take a subset of x
                before applying the query projection. Useful for e.g., ViT where we only care
                about the CLS token in the last layer.
            past_cache: For generation only.

        Returns:
            (batch, seqlen, hidden_dim) if cu_seqlens is None and max_seqlen is None,
                else (total, hidden_dim) where total is the is the sum of the sequence lengths
                in the batch.

        """

        if cu_seqlens is not None:
            assert max_seqlen is not None
            assert key_padding_mask is None
            assert self.flash_attn
            assert self.rotary_emb_dim == 0

        if key_padding_mask is not None:
            assert cu_seqlens is None
            assert max_seqlen is None
            assert not self.flash_attn

        if past_cache is not None:
            assert key_padding_mask is None
            assert cu_seqlens is None
            assert max_seqlen is None

        attn_kwargs = {"key_padding_mask": key_padding_mask}

        assert x_kv is None
        assert mixer_subset is None

        qkv = self.Wqkv(x)

        q, k, v = qkv.split(self.op_size, -1)

        print("THEIRS Block qkv", qkv.sum(), qkv.shape, q.sum(), k.sum(), v.sum(), q.shape)

        qkv = rearrange(qkv, "... (three h d) -> ... three h d", three=3, d=self.head_dim)

        if past_cache is None:
            if self.rotary_emb_dim > 0:
                q, k, v = qkv.unbind(dim=2)
                print("THEIRS before rotary", qkv.sum(), q.sum(), k.sum(), v.sum(), q.shape)

                qkv = self.rotary_emb(qkv)

                q, k, v = qkv.unbind(dim=2)

                context = self.inner_attn(qkv, **attn_kwargs)

        else:
            if self.rotary_emb_dim > 0:
                qkv = self.rotary_emb(qkv, seqlen_offset=past_cache.sequence_len_offset)
            q = qkv[:, :, 0]
            kv = self._update_kv_cache(qkv[:, :, 1:], past_cache)
            # If we're processing the prompt, causal=None (use self.causal).
            # If we're decoding, then causal=False.
            causal = None if past_cache.sequence_len_offset == 0 else False
            context = self.inner_cross_attn(q, kv, causal=causal)

        out = rearrange(context, "... h d -> ... (h d)")
        out = self.out_proj(out)

        return out if not self.return_residual else (out, x)


class ParallelBlock(nn.Module):
    """Parallel block.

    This block applies parallel mixer and MLP layers to the input (used in GPT-J and CodeGen).

    """

    def __init__(
        self,
        config: PretrainedConfig,
        mixer: Optional[Dict[str, Any]] = None,
        mlp: Optional[Dict[str, Any]] = None,
        block_idx: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.ln = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.block_idx = block_idx

        self.mixer = MHA(config=config, **mixer, layer_idx=block_idx)
        mlp_cls = mlp.pop("mlp_cls")
        if mlp_cls == "fused_mlp":
            self.mlp = FusedMLP(config=config, **mlp)
        else:
            self.mlp = MLP(config=config, **mlp)

    def forward(
        self, hidden_states: torch.FloatTensor, past_cache: Optional[torch.FloatTensor] = None
    ) -> torch.FloatTensor:
        residual = hidden_states
        hidden_states = self.ln(hidden_states)

        attn_outputs = self.mixer(hidden_states, past_cache=past_cache)
        if isinstance(attn_outputs, tuple):
            attn_outputs = attn_outputs[0]

        attn_outputs = self.resid_dropout(attn_outputs)

        feed_forward_hidden_states = self.resid_dropout(self.mlp(hidden_states))

        return attn_outputs + feed_forward_hidden_states + residual


class CausalLMHead(nn.Module):
    """Causal Language Modeling head.

    Reference:
        Improving Language Understanding by Generative Pre-Training.
        https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf.

    """

    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__()

        self.ln = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.linear = nn.Linear(config.n_embd, config.vocab_size)

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        hidden_states = self.ln(hidden_states)
        return self.linear(hidden_states).to(torch.float32)


class CausalLMLoss(nn.Module):
    """Causal Language Modeling loss.

    Reference:
        Improving Language Understanding by Generative Pre-Training.
        https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf.

    """

    def __init__(self, shift_labels: Optional[bool] = True) -> None:
        super().__init__()

        self.shift_labels = shift_labels
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, logits: torch.FloatTensor, labels: torch.LongTensor) -> torch.FloatTensor:
        if self.shift_labels:
            logits = logits[..., :-1, :].contiguous()
            labels = labels[..., 1:].contiguous()

        return self.loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))


class MixFormerSequentialPreTrainedModel(PreTrainedModel):
    """MixFormer (sequential for DeepSpeed) pre-trained model."""

    config_class = MixFormerSequentialConfig
    base_model_prefix = "transformer"
    supports_gradient_checkpointing = True

    def __init__(self, *inputs, **kwargs) -> None:
        super().__init__(*inputs, **kwargs)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs) -> Dict[str, Any]:
        if "use_cache" in kwargs and not kwargs["use_cache"]:
            return {"input_ids": input_ids}

        if past_key_values is None or not (isinstance(past_key_values, InferenceParams)):
            past_key_values = InferenceParams(
                max_batch_size=input_ids.shape[0],
                max_sequence_len=self.config.n_positions,
                sequence_len_offset=0,
                batch_size_offset=0,
                fused_ft_kernel=False,
                key_value_memory_dict={},
            )
        else:
            # assume past_key_values has cached all but last token in input_ids
            past_key_values.sequence_len_offset = len(input_ids[0]) - 1
            input_ids = input_ids[:, -1].unsqueeze(-1)

        return {"input_ids": input_ids, "past_key_values": past_key_values, **kwargs}


class MixFormerSequentialForCausalLM(MixFormerSequentialPreTrainedModel):
    """MixFormer (sequential for DeepSpeed) for Causal Language Modeling."""

    _keys_to_ignore_on_load_missing = [""]
    _keys_to_ignore_on_load_unexpected = [r"layers\.\d+\.mlp.(fc_in|fc_out)\.(weight|bias)"]
    _no_split_modules = ["ParallelBlock"]

    def __init__(self, config: MixFormerSequentialConfig) -> None:
        super().__init__(config)

        modules = [Embedding(config)]
        block_config = config.architecture

        if not isinstance(block_config, list):
            block_config = [block_config for _ in range(config.n_layer)]

        if config.n_layer != len(block_config):
            config.n_layer = len(block_config)

        for block_idx, block in enumerate(block_config):
            # `block_cls` with `legacy` value is for backward compatibility
            # `path` key is for backward compatibility
            block = copy.deepcopy(block) or {"block_cls": "parallel"}
            block.pop("path", None) or block.pop("block_cls", None)

            block["block_idx"] = block_idx
            modules.append(ParallelBlock(config, **block))

        modules.append(CausalLMHead(config))

        self.layers = nn.Sequential(*modules)
        self.loss = CausalLMLoss()

        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.layers[0].wte

    def set_input_embeddings(self, new_embeddings: nn.Embedding) -> None:
        self.layers[0].wte = new_embeddings

    def get_output_embeddings(self) -> nn.Linear:
        return self.layers[-1].linear

    def set_output_embeddings(self, new_embeddings: nn.Linear) -> None:
        self.layers[-1].linear = new_embeddings

    def forward(
        self,
        input_ids: torch.LongTensor,
        labels: Optional[torch.LongTensor] = None,
        past_key_values: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        if not past_key_values:
            lm_logits = self.layers(input_ids)
        else:
            hidden_layer = self.layers[0](input_ids)
            for module in self.layers[1:-1]:
                hidden_layer = module(hidden_layer, past_cache=past_key_values)
            lm_logits = self.layers[-1](hidden_layer)

        loss = None
        if labels is not None:
            loss = self.loss(lm_logits, labels)

        return CausalLMOutputWithPast(loss=loss, logits=lm_logits, past_key_values=past_key_values)
