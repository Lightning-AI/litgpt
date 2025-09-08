# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import pytest
import torch
from transformers.models.deepseek_v3 import DeepseekV3Config, DeepseekV3ForCausalLM

from litgpt import Config
from litgpt.model import MultiheadLatentAttention


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
    kv_cache = mla.build_kv_cache(batch_size=2, max_seq_length=32, device=torch.device("cpu"), dtype=torch.float32)

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
@pytest.mark.parametrize("batch_size", (1, 2))
@pytest.mark.parametrize("seq_len", (8, 16))
@pytest.mark.parametrize("device", [torch.device("cpu")])
def test_multihead_latent_attention_litgpt_vs_hf(batch_size, seq_len, device):
    """Test MLA litgpt vs hf"""
    config_litgpt = Config(
        n_embd=64,
        n_head=4,
        n_query_groups=4,
        head_size=16,
        norm_eps=1e-6,
        bias=False,
        latent_attention={
            "q_lora_rank": 32,
            "kv_lora_rank": 16,
            "qk_rope_head_dim": 8,
            "qk_nope_head_dim": 8,
            "v_head_dim": 16,
        },
    )

    config_hf = DeepseekV3Config(
        padded_vocab_size=10000,
        num_hidden_layers=1,
        vocab_size=10000,
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=4,
        q_lora_rank=32,
        kv_lora_rank=16,
        qk_rope_head_dim=8,
        qk_nope_head_dim=8,
        v_head_dim=16,
        rope_interleave=False,
    )

    mla_litgpt = MultiheadLatentAttention(config_litgpt, block_idx=0).to(device)
    model_hf = DeepseekV3ForCausalLM(config_hf).to(device)
    mla_hf = model_hf.model.layers[0].self_attn

    mla_litgpt.eval()
    mla_hf.eval()

    sync_weights(mla_litgpt, mla_hf)

    hidden_states = torch.randn(batch_size, seq_len, config_litgpt.n_embd, device=device)

    # Prepare RoPE sin/cos tables
    rope_head_dim = config_litgpt.latent_attention["qk_rope_head_dim"]
    cos = torch.randn(batch_size, seq_len, rope_head_dim, device=device, dtype=hidden_states.dtype)
    sin = torch.randn(batch_size, seq_len, rope_head_dim, device=device, dtype=hidden_states.dtype)

    causal_mask = torch.triu(
        torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=hidden_states.dtype), diagonal=1
    )
    attention_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)

    # Run forward passes
    output_litgpt = mla_litgpt(hidden_states, cos, sin)
    output_hf = mla_hf(hidden_states, position_embeddings=(cos, sin), attention_mask=attention_mask)[0]

    assert torch.allclose(output_litgpt, output_hf, atol=1e-5)


def sync_weights(litgpt_model, hf_model):
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
