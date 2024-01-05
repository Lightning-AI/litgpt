# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import torch


@torch.inference_mode()
def test_rope():
    from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXRotaryEmbedding, apply_rotary_pos_emb

    from lit_gpt.model import apply_rope, build_rope_cache

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
    theirs_x_rope, _ = apply_rotary_pos_emb(x, x, theirs.cos_cached, theirs.sin_cached, position_ids)
    torch.testing.assert_close(ours_x_rope, theirs_x_rope)
