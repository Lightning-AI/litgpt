# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

from unittest.mock import patch

import pytest
import torch

from litgpt.generate.base import multinomial_num_samples_1, sample


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_zero_probability_token_cannot_win_after_exponential_underflow(dtype):
    probs = torch.tensor([1.0, 0.0], dtype=dtype)

    def exponential_samples(tensor, rate=1):
        # A valid positive exponential sample rounds to zero in FP16.
        tensor.copy_(torch.tensor([1.0, 1e-8], dtype=torch.float64))
        return tensor

    with (
        patch("torch._dynamo.is_compiling", return_value=True),
        patch.object(torch.Tensor, "exponential_", exponential_samples),
    ):
        token = multinomial_num_samples_1(probs)

    assert token.shape == (1,)
    assert token.dtype == torch.int64
    assert token.item() == 0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("compiling", [False, True])
def test_top_k_one_preserves_sampling_support(dtype, compiling):
    # A large vocabulary makes FP16 exponential underflow observable on CPU.
    logits = torch.zeros((1, 1, 50304), dtype=dtype)
    logits[0, 0, 253] = 1
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]), patch("torch._dynamo.is_compiling", return_value=compiling):
            torch.manual_seed(20)
            token = sample(logits, temperature=1.0, top_k=1)
        assert token.shape == (1,)
        assert token.dtype == torch.int64
        assert token.item() == 253
    finally:
        torch.set_num_threads(previous_threads)
