# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import copy
import re
import subprocess
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from unittest.mock import ANY, Mock, call

import pytest
import torch
import yaml
from torch import nn

import litgpt.generate.speculative_decoding as generate
from litgpt import GPT, Config
from litgpt.utils import _RunIf

# The two ways of asking for greedy decoding: `sample` treats both as a point mass on the argmax.
GREEDY_SAMPLE_KWARGS = (
    pytest.param(dict(temperature=0.0, top_p=1.0, top_k=None), id="temperature-0"),
    pytest.param(dict(temperature=1.0, top_p=0.0, top_k=None), id="top-p-0"),
)


def _tiny_model_pair(prompt: torch.Tensor, draft_agrees: bool) -> tuple[GPT, GPT]:
    """A tiny target model and a draft model that either mirrors it or prefers a different token."""
    with torch.random.fork_rng(), torch.inference_mode():
        torch.manual_seed(0)
        config = Config(block_size=16, vocab_size=16, padded_vocab_size=16, n_layer=1, n_head=2, n_embd=16)
        draft_model = GPT(config)
        target_model = copy.deepcopy(draft_model)
        if draft_agrees:
            return draft_model, target_model

        # Make the target prefer the draft's second choice, so the draft's greedy token is not the target's
        preferred, runner_up = draft_model(prompt.view(1, -1))[0, -1].topk(2).indices.tolist()
        with torch.no_grad():
            for param in (target_model.lm_head.weight, target_model.lm_head.bias):
                if param is not None:
                    param[[preferred, runner_up]] = param[[runner_up, preferred]].clone()
    return draft_model, target_model


def _target_only_greedy(model: GPT, prompt: torch.Tensor, num_tokens: int) -> torch.Tensor:
    """Reference decoder: greedy decoding with the target model alone, without a KV cache and without `sample`."""
    tokens = prompt.clone()
    generated = []
    for _ in range(num_tokens):
        next_token = model(tokens.view(1, -1))[0, -1].argmax()
        generated.append(next_token)
        tokens = torch.cat((tokens, next_token.view(1)))
    return torch.stack(generated)


@torch.inference_mode()
@pytest.mark.parametrize("sample_kwargs", GREEDY_SAMPLE_KWARGS)
@pytest.mark.parametrize("speculative_k", (1, 3))
@pytest.mark.parametrize("draft_agrees", (True, False))
def test_speculative_decoding_greedy_matches_target_only(sample_kwargs, speculative_k, draft_agrees):
    prompt = torch.tensor([1, 4, 2, 7, 3])
    with torch.random.fork_rng():
        draft_model, target_model = _tiny_model_pair(prompt, draft_agrees)
        # The reference continuation, starting with the token the target picks after the prompt
        expected = _target_only_greedy(copy.deepcopy(target_model), prompt, speculative_k + 2)

        for model in (draft_model, target_model):
            model.set_kv_cache(batch_size=1)
        prompt_pos = torch.arange(0, prompt.size(0))
        draft_model(prompt.view(1, -1), prompt_pos, input_pos_maxp1=prompt.size(0))
        target_logits = target_model(prompt.view(1, -1), prompt_pos, input_pos_maxp1=prompt.size(0))

        token = target_logits[0, -1].argmax().view(1)
        output = generate.speculative_decoding(
            draft_model,
            target_model,
            token=token,
            input_pos=torch.tensor([prompt.size(0)]),
            input_pos_maxp1=prompt.size(0) + 1,
            speculative_k=speculative_k,
            **sample_kwargs,
        )

    # Sanity check of the reference: the prefilled target model picks the same token as the target-only decoder
    assert torch.equal(token, expected[:1])
    # Greedy speculative decoding has to reproduce the target model's own greedy continuation: accepted
    # draft tokens are exactly the tokens the target would have picked, and so is the resampled token.
    # A single returned token when the draft and the target disagree means the draft token was rejected.
    expected_length = speculative_k + 1 if draft_agrees else 1
    assert output.size(0) == expected_length
    assert torch.equal(output, expected[1 : 1 + expected_length])


def test_speculative_decoding_target_never_accepts_draft_tokens():
    class DraftModel(nn.Module):
        def forward(self, **kwargs):
            return torch.tensor([1, 2, 3, 4, 5, 0, 0, 0, 0, 0], dtype=torch.float)[None, None, ...]  # (B, T, C)

    class TargetModel(nn.Module):
        def forward(self, idx, **kwargs):
            _, T = idx.shape
            return torch.tensor([[0, 0, 0, 0, 0, 6, 7, 8, 9, 10]] * T, dtype=torch.float)[None, ...]  # (B, T, C)

    draft_model = DraftModel()
    target_model = TargetModel()

    token = torch.tensor([-1])
    input_pos = torch.tensor([0])
    sample_kwargs = dict(top_k=None, top_p=0.0, temperature=0.0)  # to make sampling consistent
    output = generate.speculative_decoding(
        draft_model, target_model, token, input_pos, input_pos, speculative_k=3, **sample_kwargs
    )

    # target model never accepts draft model's output, thus the output of the `speculative_decoding`
    # is a single token sampled from the target model
    assert len(output) == 1
    assert output > 5


def test_speculative_decoding_target_always_accepts_draft_tokens():
    class DraftModel(nn.Module):
        def forward(self, **kwargs):
            return torch.tensor([0, 0, 3, 4, 5, 6, 7, 8, 0, 0], dtype=torch.float)[None, None, ...]  # (B, T, C)

    class TargetModel(nn.Module):
        def forward(self, idx, **kwargs):
            _, T = idx.shape
            return torch.tensor([[0, 0, 3, 4, 5, 6, 7, 8, 0, 0]] * T, dtype=torch.float)[None, ...]  # (B, T, C)

    draft_model = DraftModel()
    target_model = TargetModel()

    token = torch.tensor([-1])
    input_pos = torch.tensor([0])
    sample_kwargs = dict(top_k=None, top_p=0.0, temperature=0.0)  # to make sampling consistent
    output = generate.speculative_decoding(
        draft_model, target_model, token, input_pos, input_pos, speculative_k=3, **sample_kwargs
    )

    # target model always accepts draft model's output, thus the output of the `speculative_decoding`
    # is 4 tokens (3 accepted draft tokens + 1 sampled from target model's output)
    assert len(output) == 4
    assert torch.all((output >= 3) & (output <= 8))


def test_speculative_decoding_target_sometimes_accepts_draft_tokens():
    class DraftModel(nn.Module):
        def forward(self, **kwargs):
            return torch.tensor([0, 0, 3, 4, 10, 9, 7, 8, 0, 0], dtype=torch.float)[None, None, ...]  # (B, T, C)

    class TargetModel(nn.Module):
        def forward(self, idx, **kwargs):
            return torch.tensor(
                [
                    [0, 0, 0, 0, 10, 9, 0, 0, 0, 0],
                    [0, 0, 0, 0, 10, 9, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 10],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 10],
                ],
                dtype=torch.float,
            )[None, ...]  # (B, T, C)

    draft_model = DraftModel()
    target_model = TargetModel()

    token = torch.tensor([-1])
    input_pos = torch.tensor([0])
    sample_kwargs = dict(top_k=None, top_p=0.0, temperature=0.0)  # to make sampling consistent
    output = generate.speculative_decoding(
        draft_model, target_model, token, input_pos, input_pos, speculative_k=3, **sample_kwargs
    )

    # target model accepts only 2 out of 3 draft model's output, thus the output of the `speculative_decoding`
    # is 3 tokens (2 accepted draft tokens + 1 sampled from adjusted distribution)
    assert len(output) == 3
    assert torch.equal(output, torch.tensor([4, 4, 9]))


@pytest.mark.parametrize("max_seq_length", (10, 15, 20, 25))
@pytest.mark.parametrize("speculative_k", (1, 2, 3))
def test_generate(max_seq_length, speculative_k):
    # create a prompt
    T = 5
    input_idx = torch.arange(0, T)
    max_new_tokens = max_seq_length - T

    # prepare models
    draft_model = GPT(Config(vocab_size=16, block_size=64, n_layer=1, n_head=4, n_embd=8))
    target_model = GPT(Config(vocab_size=16, block_size=128, n_layer=2, n_head=8, n_embd=16))
    for model in (draft_model, target_model):
        model.max_seq_length = max_seq_length
        model.set_kv_cache(batch_size=1)

    # generate tokens
    out, acceptance_rate = generate.generate(
        draft_model, target_model, input_idx, T + max_new_tokens, top_k=1, speculative_k=speculative_k
    )

    # validate
    assert out.size(0) == T + max_new_tokens - 1, (out.size(0), T + max_new_tokens - 1)
    assert 0.0 <= acceptance_rate <= 1.0


@_RunIf(min_cuda_gpus=1)  # speculative decoding makes sense only on a GPU
def test_main(fake_checkpoint_dir, monkeypatch, tensor_like):
    # prepare configs for draft and target models
    draft_model_dir = fake_checkpoint_dir / "draft_model"
    draft_model_dir.mkdir()
    target_model_dir = fake_checkpoint_dir / "target_model"
    target_model_dir.mkdir()

    draft_model_config = dict(vocab_size=16, block_size=64, n_layer=1, n_head=4, n_embd=8)
    target_model_config = dict(vocab_size=16, block_size=128, n_layer=2, n_head=8, n_embd=16)

    (draft_model_dir / "model_config.yaml").write_text(yaml.dump(draft_model_config))
    (target_model_dir / "model_config.yaml").write_text(yaml.dump(target_model_config))

    # create empty files required for validation
    for model_dir in (draft_model_dir, target_model_dir):
        (model_dir / "tokenizer.json").touch()
        (model_dir / "tokenizer_config.json").touch()
        (model_dir / "lit_model.pth").touch()

    # moke functions
    module_mock = Mock()
    module_mock.config.block_size = 128
    load_mock = Mock()
    load_mock.return_value = load_mock
    monkeypatch.setattr(generate, "load_checkpoint", load_mock)
    tokenizer_mock = Mock()
    tokenizer_mock.return_value.encode.return_value = torch.tensor([1, 2, 3])
    tokenizer_mock.return_value.decode.return_value = "foo bar baz"
    monkeypatch.setattr(generate, "Tokenizer", tokenizer_mock)
    generate_mock = Mock()
    generated_tokens = torch.tensor([3, 2, 1])
    acceptance_rate = 0.0
    generate_mock.return_value = (generated_tokens, acceptance_rate)
    monkeypatch.setattr(generate, "generate", generate_mock)

    # do the sampling
    num_samples = 2
    out, err = StringIO(), StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        generate.main(
            draft_model_checkpoint_dir=draft_model_dir,
            target_model_checkpoint_dir=target_model_dir,
            temperature=2.0,
            top_k=2,
            top_p=0.9,
            num_samples=num_samples,
        )

    assert len(tokenizer_mock.return_value.decode.mock_calls) == num_samples
    assert torch.allclose(tokenizer_mock.return_value.decode.call_args[0][0], generate_mock.return_value[0])
    assert (
        generate_mock.mock_calls
        == [
            call(
                ANY,
                ANY,
                tensor_like,
                53,
                temperature=2.0,
                top_k=2,
                top_p=0.9,
                stop_tokens=[tokenizer_mock.return_value.eos_id],
                speculative_k=3,
            )
        ]
        * num_samples
    )
    expected_output = "foo bar baz\nAcceptance rate: 0.00%\n" * num_samples
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
    args = ["litgpt", "generate_speculatively", "-h"]
    output = subprocess.check_output(args)
    output = str(output.decode())
    assert "Default generation option" in output
