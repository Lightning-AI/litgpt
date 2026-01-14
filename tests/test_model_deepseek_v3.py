# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.


import pytest
import torch
from transformers.integrations.finegrained_fp8 import FP8Linear
from transformers.models.deepseek_v3 import DeepseekV3Config, DeepseekV3ForCausalLM

from litgpt import GPT, Config
from litgpt.scripts.convert_hf_checkpoint import (
    copy_weights_deepseek_v3,
)
from litgpt.utils import _RunIf


@torch.inference_mode()
@pytest.mark.parametrize("model_name", ["DeepSeek-V3"])
@pytest.mark.parametrize(
    ("device", "dtype"),
    [
        (torch.device("cpu"), torch.float32),
        pytest.param(
            torch.device("cuda"),
            torch.float16,
            marks=[
                # the reference does softmax upscaled to fp32 during attention. additionally, the final layernorm input
                # is slightly different
                pytest.mark.xfail(raises=AssertionError, strict=False),
                _RunIf(min_cuda_gpus=1),
            ],
        ),
    ],
)
def test_against_original_deepseek_v3(model_name, device, dtype):
    torch.set_default_dtype(dtype)

    T = 20
    ours_config = Config.from_name(
        model_name,
        block_size=T,
        n_layer=2,
        n_head=16,
        n_embd=32,
        n_query_groups=4,
        intermediate_size=86,
        moe_intermediate_size=20,
        n_expert=4,
        n_shared_expert=1,
        n_expert_per_token=2,
        n_expert_groups=2,
        n_topk_groups=2,
        n_topk_scores_per_group=2,  # hardcoded in DeepseekV3ForCausalLM
        first_k_dense_replace=1,
        latent_attention=dict(
            q_lora_rank=16,
            kv_lora_rank=16,
            qk_rope_head_dim=8,
            qk_nope_head_dim=8,
            v_head_dim=16,
        ),
    )
    theirs_config = DeepseekV3Config(
        vocab_size=ours_config.padded_vocab_size,
        hidden_size=ours_config.n_embd,
        head_dim=ours_config.head_size,
        num_attention_heads=ours_config.n_head,
        num_hidden_layers=ours_config.n_layer,
        intermediate_size=ours_config.intermediate_size,
        moe_intermediate_size=ours_config.moe_intermediate_size,
        max_position_embeddings=ours_config.block_size,
        rms_norm_eps=ours_config.norm_eps,
        num_key_value_heads=ours_config.n_query_groups,
        rope_theta=ours_config.rope_base,
        tie_word_embeddings=False,
        num_experts_per_tok=ours_config.n_expert_per_token,
        norm_topk_prob=True,
        n_routed_experts=ours_config.n_expert,  # 256
        n_shared_experts=ours_config.n_shared_expert,  # 1
        n_group=ours_config.n_expert_groups,
        topk_group=ours_config.n_topk_groups,
        routed_scaling_factor=ours_config.routed_scaling_factor,  # 2.5
        first_k_dense_replace=ours_config.first_k_dense_replace,
        qk_nope_head_dim=ours_config.latent_attention["qk_nope_head_dim"],  # 128
        qk_rope_head_dim=ours_config.latent_attention["qk_rope_head_dim"],
        v_head_dim=ours_config.latent_attention["v_head_dim"],
        q_lora_rank=ours_config.latent_attention["q_lora_rank"],
        kv_lora_rank=ours_config.latent_attention["kv_lora_rank"],
    )

    theirs_model = DeepseekV3ForCausalLM(theirs_config).to(device)
    theirs_state_dict = theirs_model.state_dict()
    state_dict = {}
    copy_weights_deepseek_v3(ours_config, {}, state_dict, theirs_state_dict)
    ours_model = GPT(ours_config)
    ours_model = patch_deepseek_v3(ours_model)
    ours_model.to(device)
    ours_model.load_state_dict(state_dict)

    # test end to end
    x = torch.randint(low=0, high=ours_config.padded_vocab_size, size=(T,), device=device).unsqueeze(0)
    assert x.size(1) == T
    ours_y = ours_model(x)
    theirs_y = theirs_model(x)["logits"].to(dtype)  # HF converts logits to float
    torch.testing.assert_close(ours_y, theirs_y)


def patch_deepseek_v3(model: GPT):
    to_replace = [
        "attn.q_a_proj",
        "attn.q_b_proj",
        "attn.kv_a_proj_with_mqa",
        "attn.kv_b_proj",
        "attn.proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    for name, module in model.named_modules():
        new_module = None
        with torch.device("meta"):
            if isinstance(module, nn.Linear) and any(to_replace) in name:
                new_module = FP8Linear(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    bias=module.bias is not None,
                    activation_scheme="dynamic",
                    block_size=(128, 128),
                )
            if new_module is not None:
                model.replace_module(name, new_module)
    return model
