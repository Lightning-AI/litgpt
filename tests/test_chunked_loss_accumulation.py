# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import math
from copy import deepcopy

import pytest
import torch
import torch.nn.functional as F

from litgpt import GPT, Config
from litgpt.utils import chunked_cross_entropy

DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)


def as_chunks(logits, chunked_head):
    return list(logits.split(128, dim=1)) if chunked_head else logits


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("chunked_head", [False, True])
@pytest.mark.parametrize("masked", [False, True])
def test_finite_mean_when_token_loss_sum_exceeds_half_range(dtype, chunked_head, masked):
    logits = torch.zeros(2, 8192, 128, dtype=dtype, requires_grad=True)
    targets = torch.zeros(2, 8192, dtype=torch.long)
    if masked:
        targets[:, ::8] = -1
    valid = targets != -1
    count = valid.sum().item()
    assert math.log(128) * count > torch.finfo(torch.float16).max

    loss = chunked_cross_entropy(as_chunks(logits, chunked_head), targets, chunk_size=128, ignore_index=-1)
    assert loss.dtype == dtype
    assert torch.isfinite(loss)
    torch.testing.assert_close(loss, torch.tensor(math.log(128), dtype=dtype))
    loss.backward()
    expected = torch.full(logits.shape, 1 / (128 * count), dtype=torch.float64)
    expected[..., 0] -= 1 / count
    expected[~valid] = 0
    torch.testing.assert_close(logits.grad, expected.to(dtype), rtol=0.015, atol=6e-8)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("chunked_head", [False, True])
def test_normalization_with_more_tokens_than_half_max(dtype, chunked_head):
    logits = torch.zeros(2, 32768, 4, dtype=dtype, requires_grad=True)
    targets = torch.zeros(2, 32768, dtype=torch.long)
    loss = chunked_cross_entropy(as_chunks(logits, chunked_head), targets, chunk_size=256)
    assert loss.dtype == dtype
    torch.testing.assert_close(loss, torch.tensor(math.log(4), dtype=dtype))
    loss.backward()
    expected = torch.tensor([-0.75, 0.25, 0.25, 0.25], dtype=dtype) / targets.numel()
    torch.testing.assert_close(logits.grad, expected.expand_as(logits), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("chunked_head", [False, True])
def test_all_ignored_chunked_loss_keeps_zero_gradients(dtype, chunked_head):
    logits = torch.randn(2, 257, 8, dtype=dtype, requires_grad=True)
    targets = torch.full((2, 257), -7, dtype=torch.long)
    loss = chunked_cross_entropy(as_chunks(logits, chunked_head), targets, chunk_size=128, ignore_index=-7)
    assert loss.dtype == dtype
    assert loss.item() == 0
    loss.backward()
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits), rtol=0, atol=0)


@pytest.mark.parametrize("chunked_head", [False, True])
def test_valid_nan_is_not_hidden_by_accumulation(chunked_head):
    logits = torch.zeros(1, 257, 8, dtype=torch.float16)
    logits[0, 0, 0] = torch.nan
    targets = torch.zeros(1, 257, dtype=torch.long)
    assert chunked_cross_entropy(as_chunks(logits, chunked_head), targets, chunk_size=128).isnan()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("chunked_head", [False, True])
def test_native_gpt_half_training_matches_float_loss_reference(dtype, chunked_head):
    torch.manual_seed(19)
    config = Config(
        block_size=16,
        vocab_size=128,
        padded_vocab_size=128,
        n_layer=1,
        n_head=2,
        n_embd=8,
        intermediate_size=16,
        rotary_percentage=1.0,
    )
    model = GPT(config).to(dtype=dtype)
    reference = deepcopy(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1, momentum=0.9)
    generator = torch.Generator().manual_seed(31)
    initial = deepcopy(model.state_dict())
    for _ in range(2):
        inputs = torch.randint(0, 128, (1024, 16), generator=generator)
        targets = torch.randint(0, 128, (1024, 16), generator=generator)
        optimizer.zero_grad()
        reference_optimizer.zero_grad()
        logits = model(inputs, lm_head_chunk_size=8 if chunked_head else 0)
        expected_logits = reference(inputs)
        actual = chunked_cross_entropy(logits, targets, chunk_size=128)
        expected = F.cross_entropy(expected_logits.float().flatten(0, 1), targets.flatten())
        assert torch.isfinite(actual)
        torch.testing.assert_close(actual.float(), expected, rtol=0.008, atol=0.008)
        actual.backward()
        expected.backward()
        for parameter, expected_parameter in zip(model.parameters(), reference.parameters()):
            assert torch.isfinite(parameter.grad).all()
            torch.testing.assert_close(parameter.grad, expected_parameter.grad, rtol=0.04, atol=0.0003)
        optimizer.step()
        reference_optimizer.step()
        torch.testing.assert_close(model.state_dict(), reference.state_dict(), rtol=0.008, atol=0.004)
    assert any(not torch.equal(parameter, initial[name]) for name, parameter in model.state_dict().items())
