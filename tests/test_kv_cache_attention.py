# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

from unittest.mock import patch

import pytest
import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from litgpt import GPT, Config


def make_model(n_query_groups=4, **kwargs):
    return GPT(
        Config(
            block_size=12,
            n_layer=1,
            n_embd=32,
            n_head=4,
            n_query_groups=n_query_groups,
            padded_vocab_size=32,
            **kwargs,
        )
    ).eval()


@pytest.mark.parametrize("n_query_groups", [1, 2, 4])
@torch.inference_mode()
def test_trimmed_single_token_attention(n_query_groups):
    torch.manual_seed(123)
    model = make_model(n_query_groups)
    tokens = torch.randint(0, 32, (2, 6))
    expected = model(tokens)
    model.set_kv_cache(2)
    model(tokens[:, :3], torch.arange(3), input_pos_maxp1=3)
    for pos in range(3, 6):
        # Nonzero future slots make accidental access to an unsliced cache visible.
        cache = model.transformer.h[0].attn.kv_cache
        cache.k[:, :, pos + 1 :] = 10
        cache.v[:, :, pos + 1 :] = 10
        with patch("litgpt.model.F.scaled_dot_product_attention", wraps=F.scaled_dot_product_attention) as sdpa:
            actual = model(tokens[:, pos : pos + 1], torch.tensor([pos]), input_pos_maxp1=pos + 1)
        assert sdpa.call_args.kwargs["attn_mask"] is None
        assert sdpa.call_args.kwargs["is_causal"] is False
        assert sdpa.call_args.args[1].size(2) == pos + 1
        torch.testing.assert_close(actual, expected[:, pos : pos + 1], atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("case", ["unsliced", "batched", "chunk", "window", "softcap"])
@torch.inference_mode()
def test_keep_cache_mask(case):
    kwargs = {}
    if case == "window":
        kwargs = {"sliding_window_size": 3, "sliding_window_indices": [1]}
    elif case == "softcap":
        kwargs = {"attention_logit_softcapping": 10.0}
    model = make_model(**kwargs)
    model.set_kv_cache(2)
    tokens = torch.randint(0, 32, (2, 5))
    model(tokens[:, :3], torch.arange(3))
    positions = torch.tensor([[2], [3]]) if case == "batched" else torch.tensor([3, 4] if case == "chunk" else [3])
    length = positions.shape[-1]
    attn = model.transformer.h[0].attn
    with patch.object(attn, "scaled_dot_product_attention", wraps=attn.scaled_dot_product_attention) as sdpa:
        model(
            tokens[:, 3 : 3 + length],
            positions,
            input_pos_maxp1=None if case == "unsliced" else int(positions.max()) + 1,
        )
    assert sdpa.call_args.args[3] is not None


@torch.inference_mode()
def test_trimmed_decode_fullgraph():
    model = make_model()
    model.set_kv_cache(1)
    tokens = torch.randint(0, 32, (1, 4))
    model(tokens[:, :3], torch.arange(3))
    expected = model(tokens[:, 3:], torch.tensor([3]))
    compiled = torch.compile(model, backend="eager", fullgraph=True)
    actual = compiled(tokens[:, 3:], torch.tensor([3]), input_pos_maxp1=4)
    torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA Flash Attention")
@torch.inference_mode()
def test_trimmed_decode_flash_attention():
    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("requires an Ampere or newer GPU")
    model = make_model().to(device="cuda", dtype=torch.float16)
    model.set_kv_cache(1, device="cuda", dtype=torch.float16)
    tokens = torch.randint(0, 32, (1, 4), device="cuda")
    positions = torch.arange(4, device="cuda")
    model(tokens[:, :3], positions[:3])
    with sdpa_kernel(SDPBackend.MATH):
        expected = model(tokens[:, 3:], positions[3:])
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        actual = model(tokens[:, 3:], positions[3:], input_pos_maxp1=4)
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)
