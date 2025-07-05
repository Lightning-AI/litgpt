# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import os
import re
import subprocess
import sys
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from unittest import mock
from unittest.mock import ANY, Mock, call

import pytest
import torch
import yaml

import litgpt.generate.base as generate
from litgpt import GPT, Config
from litgpt.generate.base import batched_sample

skip_in_ci_on_macos = pytest.mark.skipif(
    sys.platform == "darwin" and os.getenv("GITHUB_ACTIONS") == "true",
    reason="Skipped on macOS in CI environment because CI machine does not have enough memory to run this test.",
)


def test_generate():
    import lightning as L

    L.seed_everything(1234)

    T = 5
    input_idx = torch.arange(0, T)

    config = Config(
        block_size=128,
        vocab_size=16,
        n_layer=1,
        n_head=4,
        n_embd=8,
    )
    model = GPT(config)
    max_new_tokens = 20
    model.max_seq_length = T + max_new_tokens
    model.set_kv_caches(batch_size=1)

    multinomial_results = []

    def multinomial(*args, **kwargs):
        if args:
            probs = args[0]
        else:
            probs = kwargs.get("probs")
        out = torch.multinomial(probs, num_samples=1)
        multinomial_results.append(out)
        return out

    with mock.patch("litgpt.generate.base.multinomial_num_samples_1", multinomial):
        out = generate.generate(
            model=model,
            prompt=input_idx,
            max_returned_tokens=T + max_new_tokens,
            top_k=1,
        )

    assert out.size(0) == T + max_new_tokens, (out.size(0), T + max_new_tokens)
    multinomial_results = torch.hstack(multinomial_results)
    print(f"input_idx {input_idx.shape}, multinomial_results: {multinomial_results.shape}")
    expected = torch.cat((input_idx, multinomial_results.squeeze(0)))
    assert out.shape == expected.shape, (out.shape, expected.shape)
    torch.testing.assert_close(out, expected)


@skip_in_ci_on_macos
def test_main(fake_checkpoint_dir, monkeypatch, tensor_like):
    config_path = fake_checkpoint_dir / "model_config.yaml"
    config = {
        "block_size": 128,
        "vocab_size": 50,
        "n_layer": 2,
        "n_head": 4,
        "n_embd": 8,
        "rotary_percentage": 1,
    }
    config_path.write_text(yaml.dump(config))

    module_mock = Mock()
    module_mock.config.block_size = config["block_size"]
    load_mock = Mock()
    load_mock.return_value = load_mock
    monkeypatch.setattr(generate, "load_checkpoint", load_mock)
    tokenizer_mock = Mock()
    tokenizer_mock.return_value.encode.return_value = torch.tensor([1, 2, 3])
    tokenizer_mock.return_value.decode.return_value = "foo bar baz"
    monkeypatch.setattr(generate, "Tokenizer", tokenizer_mock)
    generate_mock = Mock()
    # fmt: off
    generate_mock.return_value = torch.tensor([
        1, 2, 3, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0,
        0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0,
    ])
    # fmt: on
    len_return_value = generate_mock.return_value.numel()
    monkeypatch.setattr(generate, "generate", generate_mock)

    num_samples = 2
    out, err = StringIO(), StringIO()
    sample_kwargs = dict(
        temperature=2.0,
        top_k=2,
        top_p=0.9,
    )
    with redirect_stdout(out), redirect_stderr(err):
        generate.main(
            **sample_kwargs,
            num_samples=num_samples,
            checkpoint_dir=fake_checkpoint_dir,
        )

    assert len(tokenizer_mock.return_value.decode.mock_calls) == num_samples
    assert torch.allclose(
        tokenizer_mock.return_value.decode.call_args[0][0].to(torch.device("cpu")), generate_mock.return_value
    )
    expected_call = call(
        model=ANY,
        prompt=tensor_like,
        prompt_chunksize=16,
        max_returned_tokens=len_return_value,
        **sample_kwargs,
        eos_id=tokenizer_mock.return_value.eos_id,
    )
    assert generate_mock.mock_calls == [expected_call] * num_samples
    expected_output = "foo bar baz\n" * num_samples
    # Allow for the config to be printed before the expected repeated strings.
    pattern = rf".*^{re.escape(expected_output.strip())}$.*"
    assert re.match(pattern, out.getvalue().strip(), re.DOTALL | re.MULTILINE)

    err_value = err.getvalue()
    expected_parts = [
        "'padded_vocab_size': 512",
        "'n_layer': 2",
        "'n_head': 4",
    ]
    assert all(part in err_value for part in expected_parts)


def test_cli():
    args = ["litgpt", "generate", "-h"]
    output = subprocess.check_output(args)
    output = str(output.decode())
    assert "Default generation option" in output


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
    # Note: Both `sample` and `batched_sample` create only 1 sample, not 3.
    # It is like passing `logits[:, 1-:, :]`
    token = batched_sample(logits, kwargs=dict(temperature=temperature, top_p=0.8))

    assert token.shape == (2, 1)
    # sample is batch size 1 only for now - this should be [0, 1] once batched generation is supported
    assert token[0, -1].item() == 0


def test_generate_different_results_with_different_top_p():
    config = Config(
        block_size=128,
        vocab_size=16,
        n_layer=1,
        n_head=4,
        n_embd=8,
        rotary_percentage=1,
    )
    model = GPT(config)
    model.max_seq_length = 50
    model.set_kv_caches(batch_size=1)

    torch.manual_seed(123)
    input_idx = torch.randint(10, size=(1,))

    torch.manual_seed(123)
    output1 = generate.generate(
        model=model,
        prompt=input_idx,
        max_returned_tokens=20,
        top_p=1.0,
    )
    torch.manual_seed(123)
    output2 = generate.generate(
        model=model,
        prompt=input_idx,
        max_returned_tokens=20,
        top_p=0.1,
    )

    assert not torch.equal(output1, output2)
