# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import pytest
import torch
from transformers.models.deepseek_v3 import DeepseekV3Config, DeepseekV3ForCausalLM

from litgpt import Config
from litgpt.model import GPT, GroupedTopkRouter, LLaMAMLP, LLaMAMoE


@torch.inference_mode()
@pytest.mark.parametrize("batch_size", (1, 2))
@pytest.mark.parametrize("seq_len", (8, 16))
@pytest.mark.parametrize("device", [torch.device("cpu")])
def test_deepseek_moe_litgpt_vs_hf(batch_size, seq_len, device):
    """Test MOE litgpt vs hf"""
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


@pytest.mark.parametrize(
    ("e_score_correction_bias", "expected_experts"),
    [
        # Every eligible corrected score is negative: masking excluded experts with zero used to let them win.
        ([-0.6, -0.7, -1.0, -1.1], [0, 1]),
        # An eligible corrected score of exactly zero must still beat the excluded experts.
        ([-0.5, -0.7, -1.0, -1.1], [0, 1]),
        # A positive eligible score mixed with a negative one: only experts from the selected group may contribute.
        ([-0.4, -0.6, -1.0, -1.1], [0, 1]),
        # A rejected group may hold the highest individual corrected score; it must still be masked out.
        ([-0.6, -0.7, 0.3, -2.4], [0, 1]),
    ],
)
def test_grouped_topk_router_masks_excluded_experts(e_score_correction_bias, expected_experts):
    """Rejected groups must never contribute, even when they hold the highest score or eligible scores are <= 0."""
    torch.manual_seed(0)
    config = Config(
        n_layer=1,
        n_head=2,
        n_embd=8,
        vocab_size=16,
        mlp_class_name="LLaMAMoE",
        moe_intermediate_size=4,
        n_expert=4,
        n_expert_groups=2,
        n_topk_groups=1,
        n_topk_scores_per_group=2,
        n_expert_per_token=2,
        norm_topk_prob=True,
    )
    moe = LLaMAMoE(config)
    assert isinstance(moe.gate, GroupedTopkRouter)
    with torch.no_grad():
        # A zero router gives every token the same uncorrected score, so the bias alone decides the selection.
        moe.gate.weight.zero_()
        moe.gate.e_score_correction_bias.copy_(torch.tensor(e_score_correction_bias))

    x = torch.ones(1, 1, config.n_embd, requires_grad=True)
    y = moe(x)

    mixture_weights, topk_indices = moe.gate(x.detach().view(-1, config.n_embd))
    assert sorted(topk_indices[0].tolist()) == expected_experts
    # The correction bias must not leak into the mixture weights: all uncorrected sigmoid scores are 0.5.
    expected_weights = torch.full((len(expected_experts),), 0.5)
    torch.testing.assert_close(mixture_weights, expected_weights.unsqueeze(0))

    # Independently weight only the experts of the known selected group, using the actual LLaMAMLP experts.
    x_ref = torch.ones_like(x, requires_grad=True)
    y_ref = sum(weight * moe.experts[expert](x_ref) for weight, expert in zip(expected_weights, expected_experts))
    torch.testing.assert_close(y, y_ref)

    (grad,) = torch.autograd.grad(y.sum(), x)
    (grad_ref,) = torch.autograd.grad(y_ref.sum(), x_ref)
    torch.testing.assert_close(grad, grad_ref)


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
