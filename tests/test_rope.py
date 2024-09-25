# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import torch
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXRotaryEmbedding
from transformers.models.gpt_neox.modeling_gpt_neox import apply_rotary_pos_emb as apply_rotary_pos_emb_gptneo
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb as apply_rotary_pos_emb_llama

from litgpt.model import apply_rope, build_rope_cache


@torch.inference_mode()
def test_rope_gptneox():
    bs, seq_len, n_head, n_embed = 1, 6, 2, 8
    head_size = n_embed // n_head
    x = torch.randint(0, 10000, size=(bs, n_head, seq_len, head_size)).float()
    position_ids = torch.arange(seq_len).unsqueeze(0)

    theirs = GPTNeoXRotaryEmbedding(head_size, seq_len)
    ours_cos_cached, ours_sin_cached = build_rope_cache(seq_len, head_size, device=x.device)
    # their rope cache has 2 added dimensions and the cos/sin is duplicated
    torch.testing.assert_close(ours_cos_cached, theirs.cos_cached.squeeze())
    torch.testing.assert_close(ours_sin_cached, theirs.sin_cached.squeeze())

    ours_x_rope = apply_rope(x, ours_cos_cached, ours_sin_cached)
    theirs_x_rope, _ = apply_rotary_pos_emb_gptneo(x, x, theirs.cos_cached, theirs.sin_cached, position_ids)
    torch.testing.assert_close(ours_x_rope, theirs_x_rope)


@torch.inference_mode()
def test_rope_llama():
    head_dim = 64

    ##################################
    # Compare cos and sin
    ##################################
    # transformer rope
    rot_emb = LlamaRotaryEmbedding(head_dim)
    batch_size, seq_len = 1, 10
    qk_tensor = torch.randn(batch_size, seq_len, head_dim)
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    theirs_cos, theirs_sin = rot_emb(qk_tensor, position_ids)

    # our rope
    ours_cos, ours_sin = build_rope_cache(seq_len, n_elem=head_dim)
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
