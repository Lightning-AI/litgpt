import torch
from transformers.models.gpt_neox.modeling_gpt_neox import RotaryEmbedding, apply_rotary_pos_emb


@torch.inference_mode()
def test_rope(lit_parrot):
    bs, seq_len, n_head, n_embed = 1, 6, 2, 8
    head_size = n_embed // n_head
    x = torch.randint(0, 10000, size=(bs, n_head, seq_len, head_size)).float()

    theirs = RotaryEmbedding(head_size, seq_len)
    ours_cos_cached, ours_sin_cached = lit_parrot.build_rope_cache(seq_len, head_size, device=x.device, dtype=x.dtype)
    # their rope cache has 2 added dimensions and the cos/sin is duplicated
    torch.testing.assert_close(ours_cos_cached, theirs.cos_cached.squeeze())
    torch.testing.assert_close(ours_sin_cached, theirs.sin_cached.squeeze())

    ours_x_rope = lit_parrot.apply_rope(x, ours_cos_cached, ours_sin_cached)
    theirs_x_rope, _ = apply_rotary_pos_emb(x, x, theirs.cos_cached, theirs.sin_cached)
    torch.testing.assert_close(ours_x_rope, theirs_x_rope)
