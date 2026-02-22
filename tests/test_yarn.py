# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import pytest
import torch
from transformers.models.deepseek_v3 import DeepseekV3Config, DeepseekV3ForCausalLM

from litgpt import Config
from litgpt.model import Block


@torch.inference_mode()
@pytest.mark.parametrize("batch_size", (1, 2))
@pytest.mark.parametrize("seq_len", (8, 16))
@pytest.mark.parametrize("device", [torch.device("cpu")])
def test_deepseek_v3_block_with_yarn(batch_size, seq_len, device):
    """Test DeepSeek V3 block (attention + MLP + norms) with YaRN RoPE scaling - litgpt vs hf"""
    # Use layer_idx=0 to test dense MLP instead of MoE
    layer_idx = 0

    # YaRN configuration
    yarn_config = dict(
        factor=8.0,
        beta_fast=32.0,
        beta_slow=1.0,
        original_max_seq_len=4096,
        mscale=1.0,
        mscale_all_dim=0.8,
    )

    config_litgpt = Config(
        n_embd=64,
        n_head=4,
        n_query_groups=4,
        head_size=16,
        norm_eps=1e-6,
        norm_class_name="RMSNorm",
        bias=False,
        parallel_residual=False,
        mlp_class_name="LLaMAMoE",
        intermediate_size=128,
        rope_interleave=True,
        rope_adjustments=yarn_config,  # YaRN config
        latent_attention={
            "q_lora_rank": 32,
            "kv_lora_rank": 16,
            "qk_rope_head_dim": 8,
            "qk_nope_head_dim": 8,
            "v_head_dim": 16,
        },
        first_k_dense_replace=3,  # Use dense MLP for first 3 layers
    )

    # HF config with YaRN
    rope_parameters = {
        "type": "yarn",
        "rope_theta": 10000.0,
        "factor": yarn_config["factor"],
        "beta_fast": yarn_config["beta_fast"],
        "beta_slow": yarn_config["beta_slow"],
        "original_max_position_embeddings": yarn_config["original_max_seq_len"],
        "mscale": yarn_config["mscale"],
        "mscale_all_dim": yarn_config["mscale_all_dim"],
    }

    config_hf = DeepseekV3Config(
        padded_vocab_size=10000,
        num_hidden_layers=1,
        vocab_size=10000,
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=4,
        q_lora_rank=32,
        kv_lora_rank=16,
        qk_rope_head_dim=8,
        qk_nope_head_dim=8,
        v_head_dim=16,
        rope_interleave=True,
        first_k_dense_replace=3,
        rms_norm_eps=1e-6,
        rope_scaling=rope_parameters,  # YaRN config
    )

    # Debug: Check if HF config has rope_parameters
    print("\n=== HF Config Debug ===")
    print(f"config_hf.rope_parameters: {config_hf.rope_scaling}")

    block_litgpt = Block(config_litgpt, block_idx=layer_idx).to(device)
    model_hf = DeepseekV3ForCausalLM(config_hf).to(device)
    block_hf = model_hf.model.layers[layer_idx]

    block_litgpt.eval()
    block_hf.eval()

    sync_block_weights(block_litgpt, block_hf)

    hidden_states = torch.randn(batch_size, seq_len, config_litgpt.n_embd, device=device)

    # Prepare RoPE sin/cos tables using YaRN computation
    from litgpt.model import build_rope_cache

    rope_head_dim = config_litgpt.latent_attention["qk_rope_head_dim"]

    # Build YaRN RoPE cache for LitGPT
    cos_litgpt, sin_litgpt = build_rope_cache(
        seq_len=seq_len,
        n_elem=rope_head_dim,
        device=device,
        base=config_litgpt.rope_base,
        extra_config={
            "factor": yarn_config["factor"],
            "beta_fast": yarn_config["beta_fast"],
            "beta_slow": yarn_config["beta_slow"],
            "original_max_seq_len": yarn_config["original_max_seq_len"],
            "mscale": yarn_config["mscale"],
            "mscale_all_dim": yarn_config["mscale_all_dim"],
        },
    )

    # Get YaRN RoPE embeddings from HF (rotary_emb is on model level, not layer level)
    rotary_emb = model_hf.model.rotary_emb
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
    cos_hf, sin_hf = rotary_emb(hidden_states, position_ids)

    # Expand dimensions for batch and broadcast
    cos_litgpt = cos_litgpt.unsqueeze(0).expand(batch_size, -1, -1)
    sin_litgpt = sin_litgpt.unsqueeze(0).expand(batch_size, -1, -1)

    # Compare RoPE embeddings first
    print("\n=== RoPE Embeddings Comparison ===")
    print(f"LitGPT cos/sin shape: {cos_litgpt.shape}, {sin_litgpt.shape}")
    print(f"HF cos/sin shape: {cos_hf.shape}, {sin_hf.shape}")
    print(f"Cos max diff: {(cos_litgpt - cos_hf).abs().max()}")
    print(f"Sin max diff: {(sin_litgpt - sin_hf).abs().max()}")
    print(f"\nLitGPT cos sample [0,0,:4]: {cos_litgpt[0, 0, :4]}")
    print(f"HF cos sample [0,0,:4]: {cos_hf[0, 0, :4]}")
    print(f"LitGPT cos min/max: {cos_litgpt.min():.4f} / {cos_litgpt.max():.4f}")
    print(f"HF cos min/max: {cos_hf.min():.4f} / {cos_hf.max():.4f}")

    # Check inv_freq from both
    print("\n=== Checking inv_freq ===")
    print(f"HF rotary_emb.inv_freq shape: {rotary_emb.inv_freq.shape}")
    print(f"HF inv_freq: {rotary_emb.inv_freq}")
    print(f"HF attention_scaling: {rotary_emb.attention_scaling}")

    # Use the same embeddings for both (LitGPT's)
    cos = cos_litgpt
    sin = sin_litgpt

    causal_mask = torch.triu(
        torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=hidden_states.dtype), diagonal=1
    )
    attention_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)

    # Run forward passes
    output_litgpt = block_litgpt(hidden_states, cos, sin)
    output_hf = block_hf(hidden_states, position_embeddings=(cos, sin), attention_mask=attention_mask)
    if isinstance(output_hf, tuple):
        output_hf = output_hf[0]

    max_diff = (output_litgpt - output_hf).abs().max()
    print("\n=== DEBUG INFO ===")
    print(f"Max diff: {max_diff}")
    print(f"Output litgpt mean: {output_litgpt.mean()}, std: {output_litgpt.std()}")
    print(f"Output hf mean: {output_hf.mean()}, std: {output_hf.std()}")
    print(f"Cos/sin shape: {cos.shape}, {sin.shape}")
    print(f"Hidden states shape: {hidden_states.shape}")

    # Check if the issue is in attention or MLP
    if hasattr(output_litgpt, "shape") and hasattr(output_hf, "shape"):
        if output_litgpt.shape != output_hf.shape:
            print(f"Shape mismatch! litgpt: {output_litgpt.shape}, hf: {output_hf.shape}")

    assert torch.allclose(output_litgpt, output_hf, atol=1e-5, rtol=1e-4), f"FAILED: Max diff: {max_diff}"


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


def sync_block_weights(block_litgpt, block_hf):
    """Synchronize all weights from LitGPT block to HF block."""
    print("Synchronizing block weights...")
    with torch.no_grad():
        # Sync attention weights
        sync_weights(block_litgpt.attn, block_hf.self_attn)

        # Sync MLP weights (assumes dense MLP, not MoE)
        block_hf.mlp.gate_proj.weight.copy_(block_litgpt.mlp.fc_1.weight)
        block_hf.mlp.up_proj.weight.copy_(block_litgpt.mlp.fc_2.weight)
        block_hf.mlp.down_proj.weight.copy_(block_litgpt.mlp.proj.weight)

        # Sync normalization layers
        block_hf.input_layernorm.weight.copy_(block_litgpt.norm_1.weight)
        block_hf.post_attention_layernorm.weight.copy_(block_litgpt.norm_2.weight)

    print("Block synchronization complete.")
