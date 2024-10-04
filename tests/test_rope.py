# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import torch
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXRotaryEmbedding
from transformers.models.gpt_neox.modeling_gpt_neox import apply_rotary_pos_emb as apply_rotary_pos_emb_gptneo
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb as apply_rotary_pos_emb_llama
from transformers.models.llama.configuration_llama import LlamaConfig

from litgpt.model import apply_rope, build_rope_cache


@torch.inference_mode()
def test_rope_gptneox():
    bs, seq_len, n_head, n_embed = 1, 6, 2, 8
    head_size = n_embed // n_head
    x = torch.randint(0, 10000, size=(bs, n_head, seq_len, head_size)).float()
    position_ids = torch.arange(seq_len).unsqueeze(0)

    theirs_rot_emb = GPTNeoXRotaryEmbedding(head_size, seq_len)
    theirs_cos, theirs_sin = theirs_rot_emb(x, position_ids)

    ours_cos_cached, ours_sin_cached = build_rope_cache(seq_len, head_size, device=x.device)
    # their rope cache has 2 added dimensions and the cos/sin is duplicated
    torch.testing.assert_close(ours_cos_cached, theirs_cos.squeeze())
    torch.testing.assert_close(ours_sin_cached, theirs_sin.squeeze())

    ours_x_rope = apply_rope(x, ours_cos_cached, ours_sin_cached)
    theirs_x_rope, _ = apply_rotary_pos_emb_gptneo(x, x, theirs_cos, theirs_sin, position_ids)
    torch.testing.assert_close(ours_x_rope, theirs_x_rope)


@torch.inference_mode()
def test_rope_llama_2():
    head_dim = 64
    rope_theta = 10_000

    ##################################
    # Compare cos and sin
    ##################################
    # transformer rope
    rot_emb = LlamaRotaryEmbedding(head_dim, scaling_factor=None, base=rope_theta)
    batch_size, seq_len = 1, 10
    qk_tensor = torch.randn(batch_size, seq_len, head_dim)
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    theirs_cos, theirs_sin = rot_emb(qk_tensor, position_ids)

    # our rope
    ours_cos, ours_sin = build_rope_cache(seq_len, n_elem=head_dim, base=rope_theta)
    torch.testing.assert_close(theirs_cos.squeeze(0), ours_cos)
    torch.testing.assert_close(theirs_sin.squeeze(0), ours_sin)

    ##################################
    # Compare rotated tensors
    ##################################
    # Settings
    num_heads = 4

    # Dummy query and key tensors
    torch.manual_seed(123)
    queries = torch.randn(batch_size, num_heads, seq_len, head_dim)
    keys = torch.randn(batch_size, num_heads, seq_len, head_dim)

    ours_q_rot = apply_rope(queries, ours_cos, ours_sin)
    ours_k_rot = apply_rope(keys, ours_cos, ours_sin)
    theirs_q_rot, theirs_k_rot = apply_rotary_pos_emb_llama(queries, keys, theirs_cos, theirs_sin)
    torch.testing.assert_close(theirs_q_rot, ours_q_rot)
    torch.testing.assert_close(theirs_k_rot, ours_k_rot)


# See https://huggingface.co/meta-llama/Meta-Llama-3-8B/blob/main/config.json for settings
@torch.inference_mode()
def test_rope_llama_3():
    head_dim = 64
    rope_theta = 50_000

    ##################################
    # Compare cos and sin
    ##################################
    # transformer rope
    rot_emb = LlamaRotaryEmbedding(head_dim, scaling_factor=None, base=rope_theta)
    batch_size, seq_len = 1, 10
    qk_tensor = torch.randn(batch_size, seq_len, head_dim)
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    theirs_cos, theirs_sin = rot_emb(qk_tensor, position_ids)

    # our rope
    ours_cos, ours_sin = build_rope_cache(seq_len, n_elem=head_dim, base=rope_theta)
    torch.testing.assert_close(theirs_cos.squeeze(0), ours_cos)
    torch.testing.assert_close(theirs_sin.squeeze(0), ours_sin)

    ##################################
    # Compare rotated tensors
    ##################################
    # Settings
    num_heads = 4

    # Dummy query and key tensors
    torch.manual_seed(123)
    queries = torch.randn(batch_size, num_heads, seq_len, head_dim)
    keys = torch.randn(batch_size, num_heads, seq_len, head_dim)

    ours_q_rot = apply_rope(queries, ours_cos, ours_sin)
    ours_k_rot = apply_rope(keys, ours_cos, ours_sin)
    theirs_q_rot, theirs_k_rot = apply_rotary_pos_emb_llama(queries, keys, theirs_cos, theirs_sin)
    torch.testing.assert_close(theirs_q_rot, ours_q_rot)
    torch.testing.assert_close(theirs_k_rot, ours_k_rot)


# See https://huggingface.co/meta-llama/Llama-3.1-8B/blob/main/config.json for settings
@torch.inference_mode()
def test_rope_llama_3_1():
    head_dim = 128
    rope_theta = 50_000

    their_rope_config = {
        "factor": 8.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position_embeddings": 8192,
        "rope_type": "llama3"
     }

    our_rope_config = {
        "factor": 8.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_seq_len": 8192
     }

    config = LlamaConfig(
        rope_theta=rope_theta,
        rope_scaling=their_rope_config
    )

    ##################################
    # Compare cos and sin
    ##################################
    # transformer rope
    rot_emb = LlamaRotaryEmbedding(head_dim, base=rope_theta, config=config, rope_type="llama3")
    batch_size, seq_len = 1, 10
    qk_tensor = torch.randn(batch_size, seq_len, head_dim)
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    theirs_cos, theirs_sin = rot_emb(qk_tensor, position_ids)

    # our rope
    ours_cos, ours_sin = build_rope_cache(seq_len, n_elem=head_dim, base=rope_theta, extra_config=our_rope_config)
    torch.testing.assert_close(theirs_cos.squeeze(0), ours_cos)
    torch.testing.assert_close(theirs_sin.squeeze(0), ours_sin)

    ##################################
    # Compare rotated tensors
    ##################################
    # Settings
    num_heads = 4

    # Dummy query and key tensors
    torch.manual_seed(123)
    queries = torch.randn(batch_size, num_heads, seq_len, head_dim)
    keys = torch.randn(batch_size, num_heads, seq_len, head_dim)

    ours_q_rot = apply_rope(queries, ours_cos, ours_sin)
    ours_k_rot = apply_rope(keys, ours_cos, ours_sin)
    theirs_q_rot, theirs_k_rot = apply_rotary_pos_emb_llama(queries, keys, theirs_cos, theirs_sin)
    torch.testing.assert_close(theirs_q_rot, ours_q_rot)
    torch.testing.assert_close(theirs_k_rot, ours_k_rot)


# See https://huggingface.co/meta-llama/Llama-3.2-3B/blob/main/config.json for settings
@torch.inference_mode()
def test_rope_llama_3_2():
    head_dim = 128
    rope_theta = 50_000

    their_rope_config = {
        "factor": 32.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position_embeddings": 8192,
        "rope_type": "llama3"
     }

    our_rope_config = {
        "factor": 32.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_seq_len": 8192
     }

    config = LlamaConfig(
        rope_theta=rope_theta,
        rope_scaling=their_rope_config
    )

    ##################################
    # Compare cos and sin
    ##################################
    # transformer rope
    rot_emb = LlamaRotaryEmbedding(head_dim, base=rope_theta, config=config, rope_type="llama3")
    batch_size, seq_len = 1, 10
    qk_tensor = torch.randn(batch_size, seq_len, head_dim)
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    theirs_cos, theirs_sin = rot_emb(qk_tensor, position_ids)

    # our rope
    ours_cos, ours_sin = build_rope_cache(seq_len, n_elem=head_dim, base=rope_theta, extra_config=our_rope_config)
    torch.testing.assert_close(theirs_cos.squeeze(0), ours_cos)
    torch.testing.assert_close(theirs_sin.squeeze(0), ours_sin)

    ##################################
    # Compare rotated tensors
    ##################################
    # Settings
    num_heads = 4

    # Dummy query and key tensors
    torch.manual_seed(123)
    queries = torch.randn(batch_size, num_heads, seq_len, head_dim)
    keys = torch.randn(batch_size, num_heads, seq_len, head_dim)

    ours_q_rot = apply_rope(queries, ours_cos, ours_sin)
    ours_k_rot = apply_rope(keys, ours_cos, ours_sin)
    theirs_q_rot, theirs_k_rot = apply_rotary_pos_emb_llama(queries, keys, theirs_cos, theirs_sin)
    torch.testing.assert_close(theirs_q_rot, ours_q_rot)
    torch.testing.assert_close(theirs_k_rot, ours_k_rot)