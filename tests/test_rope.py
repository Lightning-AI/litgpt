import torch


def build_rope_cache_old(seq_len: int, n_elem: int, dtype: torch.dtype, base: int = 10000) -> torch.Tensor:
    """This is the `build_rope_cache` implementation we initially intended to use, but it is numerically not
    exactly equivalent to the one in the Meta model. We keep it here for posterity.

    Derived from:mers/rope/__init__.py
    https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/license MIT License:
    """  # noqa: E501
    # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
    theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, dtype=dtype) / n_elem))

    # Create position indexes `[0, 1, ..., seq_len - 1]`
    seq_idx = torch.arange(seq_len, dtype=dtype)

    # Calculate the product of position index and $\theta_i$
    idx_theta = torch.outer(seq_idx, theta)

    # Concatenate so that for row $m$ we have
    # $[m \theta_0, m \theta_1, ..., m \theta_{\frac{d}{2}}, m \theta_0, m \theta_1, ..., m \theta_{\frac{d}{2}}]$
    idx_theta2 = torch.cat([idx_theta, idx_theta], dim=1)

    # Cache them
    cos_cache = idx_theta2.cos()[None, None, :, :]
    sin_cache = idx_theta2.sin()[None, None, :, :]

    return torch.stack((cos_cache, sin_cache), dim=0)


def rotate_neg_half(x: torch.Tensor) -> torch.Tensor:
    # $\frac{d}{2}$
    d_2 = x.shape[-1] // 2
    # Calculate $[-x^{(\frac{d}{2} + 1)}, -x^{(\frac{d}{2} + 2)}, ..., -x^{(d)}, x^{(1)}, x^{(2)}, ..., x^{(\frac{d}{2})}]$  # noqa: E501
    return torch.cat([-x[:, :, :, d_2:], x[:, :, :, :d_2]], dim=-1)


def apply_rope_old(x: torch.Tensor, rope_cache: torch.Tensor) -> torch.Tensor:
    """This is the `apply_rope` implementation we initially intended to use, but it is numerically not exactly
    equivalent to the one in the Meta model.

    We keep it here for posterity.
    """
    neg_half_x = rotate_neg_half(x)
    cos, sin = rope_cache
    # truncate to support variable sizes
    T = x.size(2)
    cos = cos[:, :, :T]
    sin = sin[:, :, :T]
    return (x * cos) + (neg_half_x * sin)


@torch.no_grad()
def test_rope(lit_llama, orig_llama) -> None:
    bs, seq_len, n_head, n_embed = 1, 6, 2, 8
    x = torch.randint(0, 10000, size=(bs, seq_len, n_head, n_embed // n_head)).float()

    freqs_cis = orig_llama.precompute_freqs_cis(n_embed // n_head, seq_len)
    llama_rope_cache = lit_llama.build_rope_cache(seq_len, n_embed // n_head, dtype=x.dtype, device=x.device)
    assert torch.equal(freqs_cis, llama_rope_cache)

    llama_x_rope = lit_llama.apply_rope(x.transpose(1, 2), llama_rope_cache).transpose(1, 2)
    orig_llama_x_rope, _ = orig_llama.apply_rotary_emb(x, x, freqs_cis)

    assert torch.equal(llama_x_rope, orig_llama_x_rope)

    # For posterity, we show here that our older implementation we initially wanted to use
    # is not numerically equivalent to Meta's rope implementation
    llama_rope_cache_old = build_rope_cache_old(seq_len, n_embed // n_head, dtype=x.dtype)
    llama_x_rope_old = apply_rope_old(x, llama_rope_cache_old)
    assert not torch.allclose(llama_x_rope_old, orig_llama_x_rope)
