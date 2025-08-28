# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import pytest
import torch

from litgpt import Config, GPT
from litgpt.model import MultiheadLatentAttention


@torch.inference_mode()
@pytest.mark.parametrize("batch_size", (1, 2))
@pytest.mark.parametrize("seq_len", (8, 16))
@pytest.mark.parametrize("device", [torch.device("cpu")])
def test_multihead_latent_attention_forward(batch_size, seq_len, device):
    """Test basic forward pass of MultiheadLatentAttention"""
    config = Config(
        n_embd=64,
        n_head=4,
        n_query_groups=4,
        head_size=16,
        latent_attention={
            "q_lora_rank": 32,
            "kv_lora_rank": 16,
            "qk_rope_head_dim": 8,
            "qk_nope_head_dim": 8,
            "v_head_dim": 16,
        },
    )
    
    mla = MultiheadLatentAttention(config, block_idx=0).to(device)
    
    # Create input tensor
    x = torch.randn(batch_size, seq_len, config.n_embd, device=device)
    cos = torch.randn(1, seq_len, config.qk_rope_head_dim, device=device)
    sin = torch.randn(1, seq_len, config.qk_rope_head_dim, device=device)
    
    # Forward pass
    output = mla(x, cos, sin)
    
    # Check output shape
    assert output.shape == (batch_size, seq_len, config.n_embd)
    assert output.dtype == x.dtype


@torch.inference_mode()
def test_multihead_latent_attention_kv_cache():
    """Test KV cache functionality"""
    config = Config(
        block_size=32,
        n_embd=64,
        n_head=4,
        n_query_groups=4,
        head_size=16,
        latent_attention={
            "q_lora_rank": 32,
            "kv_lora_rank": 16,
            "qk_rope_head_dim": 8,
            "qk_nope_head_dim": 8,
            "v_head_dim": 16,
        },
    )
    
    mla = MultiheadLatentAttention(config, block_idx=0)
    
    # Build KV cache
    kv_cache = mla.build_kv_cache(
        batch_size=2,
        max_seq_length=32,
        device=torch.device("cpu"),
        dtype=torch.float32
    )
    
    # Check cache shapes
    assert kv_cache.k.shape == (2, config.n_head, 32, config.qk_head_dim)
    assert kv_cache.v.shape == (2, config.n_head, 32, config.v_head_dim)


@torch.inference_mode()
def test_multihead_latent_attention_with_mask():
    """Test attention with causal mask"""
    config = Config(
        n_embd=64,
        n_head=4,
        n_query_groups=4,
        head_size=16,
        latent_attention={
            "q_lora_rank": 32,
            "kv_lora_rank": 16,
            "qk_rope_head_dim": 8,
            "qk_nope_head_dim": 8,
            "v_head_dim": 16,
        },
    )
    
    mla = MultiheadLatentAttention(config, block_idx=0)
    
    batch_size, seq_len = 1, 8
    x = torch.randn(batch_size, seq_len, config.n_embd)
    cos = torch.randn(1, seq_len, config.qk_rope_head_dim)
    sin = torch.randn(1, seq_len, config.qk_rope_head_dim)
    
    # Create causal mask
    mask = torch.ones(seq_len, seq_len, dtype=x.dtype).triu(diagonal=1)
    mask.masked_fill_(mask.bool(), float("-inf"))
    mask = mask.view(1, 1, seq_len, seq_len)
    
    # Forward pass with mask
    output = mla(x, cos, sin, mask=mask)
    
    assert output.shape == (batch_size, seq_len, config.n_embd)


@torch.inference_mode()
def test_multihead_latent_attention_integration():
    """Test MLA integration with full GPT model"""
    config = Config(
        block_size=16,
        vocab_size=100,
        n_layer=1,
        n_head=4,
        n_embd=64,
        head_size=16,
        latent_attention={
            "q_lora_rank": 32,
            "kv_lora_rank": 16,
            "qk_rope_head_dim": 8,
            "qk_nope_head_dim": 8,
            "v_head_dim": 16,
        },
    )
    
    model = GPT(config)
    
    batch_size, seq_len = 2, 8
    input_ids = torch.randint(0, config.padded_vocab_size, (batch_size, seq_len))
    
    # Forward pass through full model
    output = model(input_ids)
    
    # Check output shape
    assert output.shape == (batch_size, seq_len, config.padded_vocab_size)
    
    # Verify the attention layer is MLA
    assert isinstance(model.transformer.h[0].attn, MultiheadLatentAttention)