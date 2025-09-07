import math
from typing import Callable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from litgpt.model import MultiheadLatentAttention


# This script contains the DeepseekV3Attention module and its dependencies
# extracted into a self-contained file for standalone use.

class DeepseekV3Config:
    """Simplified configuration class for DeepseekV3Attention."""
    def __init__(self, **kwargs):
        # Attention-specific parameters
        self.hidden_size = kwargs.get("hidden_size", 1024)
        self.num_attention_heads = kwargs.get("num_attention_heads", 16)
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 16)
        self.head_dim = self.hidden_size // self.num_attention_heads
        
        # LoRA and special head dimensions
        self.q_lora_rank = kwargs.get("q_lora_rank", None) # Set to None for standard projection
        self.kv_lora_rank = kwargs.get("kv_lora_rank", 64)
        self.qk_rope_head_dim = kwargs.get("qk_rope_head_dim", 64)
        self.v_head_dim = kwargs.get("v_head_dim", 128)
        self.qk_nope_head_dim = kwargs.get("qk_nope_head_dim", self.head_dim - self.qk_rope_head_dim)
        self.qk_head_dim = self.qk_rope_head_dim + self.qk_nope_head_dim
        
        # Other settings
        self.attention_bias = kwargs.get("attention_bias", False)
        self.attention_dropout = kwargs.get("attention_dropout", 0.0)
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)
        
        # RoPE (Rotary Position Embedding) parameters
        self.rope_theta = kwargs.get("rope_theta", 10000.0)
        self.rope_scaling = kwargs.get("rope_scaling", None)
        self.rope_interleave = kwargs.get("rope_interleave", False)
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 4096)
        self._attn_implementation = "eager" # Use the eager implementation for simplicity

class DeepseekV3RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

def rotate_half(x):
    """Rotates half the hidden dimensions of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def apply_rotary_pos_emb_interleave(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Interleaved Rotary Position Embedding."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    
    b, h, s, d = q.shape
    q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
    
    b, h, s, d = k.shape
    k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def yarn_get_mscale(scale=1, mscale=1):
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0

class DeepseekV3RotaryEmbedding(nn.Module):
    """Computes rotary position embeddings for query and key tensors."""
    def __init__(self, config: DeepseekV3Config, device=None):
        super().__init__()
        self.dim = config.qk_rope_head_dim
        self.max_position_embeddings = config.max_position_embeddings
        self.base = config.rope_theta
        self.attention_scaling = 1.0
        
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len_cached = 0

    def forward(self, x, position_ids):
        # x: [bs, seq_len, num_heads, head_dim]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
        
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeats key-value heads to match the number of query heads in GQA."""
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def eager_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    num_key_value_groups: int,
    scaling: float,
    dropout: float = 0.0,
    training: bool = False,
):
    """Eager (standard) attention implementation."""

    key_states = repeat_kv(key, num_key_value_groups)
    value_states = repeat_kv(value, num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output


class DeepseekV3Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper, adapted for DeepseekV3."""

    def __init__(self, config: DeepseekV3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.attention_dropout = config.attention_dropout
        self.num_heads = config.num_attention_heads
        self.q_lora_rank = config.q_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_head_dim = config.qk_head_dim

        # Projections for Q, K, V
        if self.q_lora_rank is None:
            self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.qk_head_dim, bias=False)
        else:
            self.q_a_proj = nn.Linear(config.hidden_size, config.q_lora_rank, bias=config.attention_bias)
            self.q_a_layernorm = DeepseekV3RMSNorm(config.q_lora_rank, eps=config.rms_norm_eps)
            self.q_b_proj = nn.Linear(config.q_lora_rank, self.num_heads * self.qk_head_dim, bias=False)

        self.kv_a_proj_with_mqa = nn.Linear(
            config.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=config.attention_bias,
        )
        self.kv_a_layernorm = DeepseekV3RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            config.num_key_value_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )

        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )

        # Scaling factor for attention scores
        self.scaling = self.qk_head_dim ** (-0.5)
        if self.config.rope_scaling is not None:
            mscale_all_dim = self.config.rope_scaling.get("mscale_all_dim", 0)
            scaling_factor = self.config.rope_scaling["factor"]
            if mscale_all_dim:
                mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
                self.scaling = self.scaling * mscale * mscale
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_length, _ = hidden_states.shape
        
        # Shapes for reshaping tensors
        query_shape = (batch_size, seq_length, self.num_heads, self.qk_head_dim)
        # Note: num_key_value_heads is used for K/V projections
        kv_shape = (batch_size, seq_length, self.config.num_key_value_heads, -1)

        # 1. Project hidden_states to Q, K, V
        if self.q_lora_rank is None:
            q_states = self.q_proj(hidden_states)
        else:
            q_states = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        
        q_states = q_states.view(query_shape).transpose(1, 2)
        q_pass, q_rot = torch.split(q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        k_pass, k_rot = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pass = self.kv_b_proj(self.kv_a_layernorm(k_pass)).view(kv_shape).transpose(1, 2)

        k_pass, value_states = torch.split(k_pass, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        k_rot = k_rot.view(batch_size, 1, seq_length, self.qk_rope_head_dim)

        # 2. Apply Rotary Position Embeddings (RoPE)
        cos, sin = position_embeddings
        if self.config.rope_interleave:
            q_rot, k_rot = apply_rotary_pos_emb_interleave(q_rot, k_rot, cos, sin)
        else:
            q_rot, k_rot = apply_rotary_pos_emb(q_rot, k_rot, cos, sin)
        
        k_rot = k_rot.expand(*k_pass.shape[:-1], -1)

        # 3. Concatenate RoPE-applied and non-RoPE parts
        query_states = torch.cat((q_pass, q_rot), dim=-1)
        key_states = torch.cat((k_pass, k_rot), dim=-1)

        # 4. Compute Attention
        attn_output = eager_attention_forward(
            query=query_states,
            key=key_states,
            value=value_states,
            attention_mask=attention_mask,
            num_key_value_groups=self.num_key_value_groups,
            scaling=self.scaling,
            dropout=self.attention_dropout,
            training=self.training
        )
        
        # 5. Reshape and final projection
        attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        
        return attn_output


def sync_weights(litgpt_model: MultiheadLatentAttention, hf_model: DeepseekV3Attention):
    """Copies weights from lit-gpt model to HF model."""
    print("Synchronizing weights...")
    with torch.no_grad():
        hf_model.q_a_proj.weight.copy_(litgpt_model.q_a_proj.weight)
        hf_model.q_a_layernorm.weight.copy_(litgpt_model.q_a_norm.weight)
        hf_model.q_b_proj.weight.copy_(litgpt_model.q_b_proj.weight)
        hf_model.kv_a_proj_with_mqa.weight.copy_(litgpt_model.kv_a_proj_with_mqa.weight)
        hf_model.kv_a_layernorm.weight.copy_(litgpt_model.kv_a_norm.weight)
        hf_model.kv_b_proj.weight.copy_(litgpt_model.kv_b_proj.weight)
        hf_model.o_proj.weight.copy_(litgpt_model.proj.weight)
    print("Synchronization complete.")