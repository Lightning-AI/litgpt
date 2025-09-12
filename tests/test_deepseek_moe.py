# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import pytest
import torch
from transformers.models.deepseek_v3 import DeepseekV3Config, DeepseekV3ForCausalLM

from litgpt import Config
from litgpt.model import GPT, LLaMAMLP


@torch.inference_mode()
@pytest.mark.parametrize("batch_size", (1, 2))
@pytest.mark.parametrize("seq_len", (8, 16))
@pytest.mark.parametrize("device", [torch.device("cpu")])
def test_deepseek_moe_litgpt_vs_hf(batch_size, seq_len, device):
    """Test MLA litgpt vs hf"""
    config_litgpt = Config(
        padded_vocab_size=10000,
        n_layer=2,
        vocab_size=10000,
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
        n_expert=16,
        n_shared_expert=1,
        n_expert_per_token=2,
        n_expert_groups=4,
        n_topk_groups=2,
        n_topk_scores_per_group=2,  # Note: Deepseek hardcodes this to `2`
        first_k_dense_replace=1,
        routed_scaling_factor=2.5,
        norm_topk_prob=True,
        moe_intermediate_size=20,
        mlp_class_name="LLaMAMoE",
    )

    config_hf = DeepseekV3Config(
        padded_vocab_size=10000,
        num_hidden_layers=2,
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
        first_k_dense_replace=1,
        routed_scaling_factor=2.5,
        norm_topk_prob=True,
        n_routed_experts=config_litgpt.n_expert,
        n_shared_experts=config_litgpt.n_shared_expert,
        num_experts_per_tok=config_litgpt.n_expert_per_token,
        n_group=config_litgpt.n_expert_groups,
        topk_group=config_litgpt.n_topk_groups,
        moe_intermediate_size=config_litgpt.moe_intermediate_size,
    )

    model_litgpt = GPT(config_litgpt).to(device)
    model_litgpt.apply(model_litgpt._init_weights)

    mlp_litgpt = model_litgpt.transformer.h[0].mlp
    assert isinstance(mlp_litgpt, LLaMAMLP)  # Test first_k_dense_replace (k=1)

    moe_litgpt = model_litgpt.transformer.h[1].mlp
    model_hf = DeepseekV3ForCausalLM(config_hf).to(device)
    moe_hf = model_hf.model.layers[1].mlp

    moe_litgpt.eval()
    moe_hf.eval()

    sync_weights(moe_litgpt, moe_hf)

    hidden_states = torch.randn(batch_size, seq_len, config_litgpt.n_embd, device=device)

    output_litgpt = moe_litgpt(hidden_states)
    output_hf = moe_hf(hidden_states)

    assert torch.allclose(output_litgpt, output_hf, atol=1e-5)


def sync_weights(litgpt_model, hf_model):
    print("Synchronizing MoE weights...")

    with torch.no_grad():
        if hasattr(litgpt_model, "gate"):
            if hasattr(litgpt_model.gate, "weight"):
                hf_model.gate.weight.copy_(litgpt_model.gate.weight)
            if hasattr(litgpt_model.gate, "e_score_correction_bias"):
                hf_model.gate.e_score_correction_bias.copy_(litgpt_model.gate.e_score_correction_bias)

        for i, (litgpt_expert, hf_expert) in enumerate(zip(litgpt_model.experts, hf_model.experts)):
            hf_expert.gate_proj.weight.copy_(litgpt_expert.fc_1.weight)
            hf_expert.up_proj.weight.copy_(litgpt_expert.fc_2.weight)
            hf_expert.down_proj.weight.copy_(litgpt_expert.proj.weight)

        if hasattr(litgpt_model, "shared_experts") and hasattr(hf_model, "shared_experts"):
            hf_model.shared_experts.gate_proj.weight.copy_(litgpt_model.shared_experts.fc_1.weight)
            hf_model.shared_experts.up_proj.weight.copy_(litgpt_model.shared_experts.fc_2.weight)
            hf_model.shared_experts.down_proj.weight.copy_(litgpt_model.shared_experts.proj.weight)

    print("MoE weight synchronization complete.")
