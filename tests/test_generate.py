# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import subprocess
import sys
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock
from unittest.mock import ANY, Mock, call

import pytest
import torch
import yaml

import litgpt.generate.base as generate
from litgpt import GPT, Config
from litgpt.generate.base import sample


@pytest.mark.parametrize(
    "max_seq_length", (pytest.param(10, marks=pytest.mark.xfail(raises=NotImplementedError, strict=True)), 20 + 5)
)
def test_generate(max_seq_length):
    T = 5
    input_idx = torch.randint(10, size=(T,))

    config = Config(block_size=128, vocab_size=16, n_layer=1, n_head=4, n_embd=8)
    model = GPT(config)
    model.max_seq_length = max_seq_length
    model.set_kv_cache(batch_size=1)
    max_new_tokens = 20

    multinomial_results = []

    def multinomial(*args, **kwargs):
        out = torch.multinomial(*args, **kwargs, num_samples=1)
        multinomial_results.append(out)
        return out

    with mock.patch("litgpt.generate.base.multinomial_num_samples_1", multinomial):
        out = next(generate.generate(model, input_idx, T + max_new_tokens, top_k=4))

    assert out.size(0) == T + max_new_tokens
    multinomial_results = torch.hstack(multinomial_results)
    expected = torch.cat((input_idx, multinomial_results))
    assert out.shape == expected.shape
    torch.testing.assert_close(out, expected)


@pytest.mark.parametrize(
    "max_returned_tokens",
    [15, 25, 20]
)
def test_generate_stream(max_returned_tokens):
    T = 5
    prompt = torch.randint(10, size=(T,))
    config = Config(block_size=128, vocab_size=16, n_layer=1, n_head=4, n_embd=8)
    model = GPT(config)
    model.max_seq_length = 30
    max_new_tokens = max_returned_tokens - T

    model.set_kv_cache(batch_size=1)

    multinomial_results = []

    def multinomial(*args, **kwargs):
        out = torch.multinomial(*args, **kwargs, num_samples=1)
        multinomial_results.append(out)
        return out

    with mock.patch("litgpt.generate.base.multinomial_num_samples_1", multinomial):
        token_generator = generate.generate(model, prompt, max_returned_tokens, stream=True)
        generated_tokens = list(token_generator)

    expected_length = min(max_new_tokens, len(multinomial_results))
    assert len(generated_tokens) == expected_length


def test_main(fake_checkpoint_dir, monkeypatch, tensor_like):
    config_path = fake_checkpoint_dir / "model_config.yaml"
    config = {"block_size": 128, "vocab_size": 50, "n_layer": 2, "n_head": 4, "n_embd": 8, "rotary_percentage": 1}
    config_path.write_text(yaml.dump(config))

    module_mock = Mock()
    module_mock.config.block_size = 128
    module_mock.max_seq_length = 150
    load_mock = Mock()
    load_mock.return_value = module_mock
    monkeypatch.setattr(generate, "load_checkpoint", load_mock)
    tokenizer_mock = Mock()
    tokenizer_mock.return_value.encode.return_value = torch.tensor([1, 2, 3])
    tokenizer_mock.return_value.decode.return_value = "foo bar baz"
    monkeypatch.setattr(generate, "Tokenizer", tokenizer_mock)

    def generate_mock(model, prompt, max_returned_tokens, *, temperature, top_k, top_p, eos_id, stream):
        if stream:
            for i in range(max_returned_tokens - prompt.size(0)):
                yield torch.tensor([3, 2, 1][i % 3])
        else:
            yield torch.cat([prompt] + [torch.tensor([3, 2, 1])] * (max_returned_tokens - prompt.size(0)))

    generate_function_mock = Mock()
    generate_function_mock.side_effect = generate_mock
    monkeypatch.setattr(generate, "generate", generate_function_mock)

    num_samples = 2
    out, err = StringIO(), StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        generate.main(temperature=2.0, top_k=2, top_p=0.9, num_samples=num_samples, checkpoint_dir=fake_checkpoint_dir)

    assert len(tokenizer_mock.return_value.decode.mock_calls) == num_samples
    assert out.getvalue().strip().split('\n') == ["foo bar baz"] * num_samples
    assert "'padded_vocab_size': 512, 'n_layer': 2, 'n_head': 4" in err.getvalue()


def test_cli():
    args = ["litgpt", "generate", "base", "-h"]
    output = subprocess.check_output(args)
    output = str(output.decode())
    assert "Generates text samples" in output


@pytest.mark.parametrize("temperature", (0.0, 1.0, 0.5))
def test_sample(temperature):
    # shape: 2x3x5
    logits = torch.tensor(
        [
            [[24, 4, 98, 77, 47], [65, 70, 32, 67, 24], [92, 32, 88, 36, 62]],
            [[85, 79, 57, 68, 50], [89, 46, 72, 45, 32], [68, 96, 68, 24, 36]],
        ],
        dtype=torch.float32,
    )
    token = sample(logits, temperature=temperature, top_p=0.8)

    assert token.shape == (1,)
    # sample is batch size 1 only for now - this should be [0, 1] once batched generation is supported
    assert token.tolist() == [0]


def test_generate_different_results_with_different_top_p():
    config = Config(block_size=128, vocab_size=16, n_layer=1, n_head=4, n_embd=8)
    model = GPT(config)
    model.max_seq_length = 50
    model.set_kv_cache(batch_size=1)

    torch.manual_seed(123)
    input_idx = torch.randint(10, size=(1,))

    torch.manual_seed(123)
    output1 = next(generate.generate(model, input_idx, 20, top_p=1.0))
    torch.manual_seed(123)
    output2 = next(generate.generate(model, input_idx, 20, top_p=0.1))

    assert not torch.equal(output1, output2)