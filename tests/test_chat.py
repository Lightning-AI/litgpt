# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
import os
import re
import subprocess
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from itertools import repeat
from pathlib import Path
from unittest.mock import ANY, MagicMock, Mock, call, patch
import sys
from typing import Iterable

import pytest
import torch
import yaml
from lightning.fabric import Fabric

import litgpt.chat.base as chat
import litgpt.generate.base as generate
from litgpt import Config, Tokenizer
from litgpt.utils import save_config, auto_download_checkpoint


skip_in_ci_on_macos = pytest.mark.skipif(
    sys.platform == "darwin" and os.getenv("GITHUB_ACTIONS") == "true",
    reason="Skipped on macOS in CI environment because CI machine does not have enough memory to run this test."
)


@pytest.mark.parametrize(
    ("generated", "stop_tokens", "expected"),
    [
        (repeat(1), (), [1] * 8),
        ([1, 2, 3, 0], ([0],), [1, 2, 3]),
        ([1, 2, 3, 0], ([9], [2, 4], [1, 2, 3, 0]), []),
        ([1, 2, 3, 0, 0], ([0, 0, 0], [0, 0]), [1, 2, 3]),
        ([3, 1, 2], ([1, 2], [3]), []),
        ([1, 2, 3, 0, 3, 2, 1, 0], ([4, 3, 2, 1], [2, 4]), [1, 2, 3, 0, 3, 2, 1, 0]),
    ],
)
def test_generate(monkeypatch, generated, stop_tokens, expected):
    import lightning as L
    L.seed_everything(1234)

    input_idx = torch.tensor([5, 3])
    max_returned_tokens = len(input_idx) + 8
    model = MagicMock()
    model.config.block_size = 100
    model.max_seq_length = 100
    it = iter(generated)

    def multinomial(*_, **__):
        out = next(it)
        return torch.tensor([out])

    monkeypatch.setattr(generate, "multinomial_num_samples_1", multinomial)
    actual = chat.generate(model, input_idx, max_returned_tokens, stop_tokens=stop_tokens)
    actual = list(actual)

    assert len(actual) == len(expected), (actual, expected)
    if not actual:
        assert actual == expected, (actual, expected)
    else:
        for t in actual:
            assert t.dtype == torch.long, t.dtype
        actual_list = torch.cat(actual).tolist()
        assert actual_list == expected, (actual_list, expected)


def test_decode():
    checkpoint_dir = auto_download_checkpoint("EleutherAI/pythia-14m")
    tokenizer = Tokenizer(checkpoint_dir)

    text = ("Hello World! This a bunch of text. Lorem ipsum dolor sit amet, "
            "consectetur adipiscing elit, sed do eiusmod tempor incididunt "
            "ut labore et dolore magna aliqua.")

    encoded: torch.Tensor = tokenizer.encode(text)
    encoded_stream: Iterable[torch.Tensor] = torch.tensor_split(encoded, encoded.shape[0], dim=0)

    decoded_stream: Iterator[str] = tokenizer.decode_stream(encoded_stream)
    decoded: str = "".join(decoded_stream)

    # Note that encoded and decoded text will not always be character for character identical.abs
    # Indeed, sometimes it is not. But that tends to be because of special cases, and this is not
    # one of those.
    assert text == decoded, (text, decoded)


@skip_in_ci_on_macos
@patch("litgpt.chat.base.input")
@pytest.mark.parametrize("stop_iteration", [KeyboardInterrupt, ""])
def test_main(mocked_input, stop_iteration, fake_checkpoint_dir, monkeypatch, tensor_like):
    # these values will be iteratively provided for each `input()` call
    mocked_input.side_effect = ["Hello", stop_iteration]

    config_path = fake_checkpoint_dir / "model_config.yaml"
    config = {
        "name": "Llama 3",
        "block_size": 128,
        "vocab_size": 50,
        "n_layer": 2,
        "n_head": 4,
        "n_embd": 8,
        "rotary_percentage": 1,
    }
    config_path.write_text(yaml.dump(config))

    load_mock = Mock()
    load_mock.return_value = load_mock
    monkeypatch.setattr(chat, "load_checkpoint", load_mock)
    tokenizer_mock = Mock()
    tokenizer_mock.return_value.backend = "sentencepiece"
    tokenizer_mock.return_value.encode.return_value = torch.tensor([1, 2, 3])
    tokenizer_mock.return_value.decode_stream.return_value = "foo bar baz"
    monkeypatch.setattr(chat, "Tokenizer", tokenizer_mock)
    generate_mock = MagicMock()
    generate_mock.__iter__.return_value = [torch.tensor([3, 2, 1])]
    monkeypatch.setattr(chat, "generate", generate_mock)

    out, err = StringIO(), StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        chat.main(temperature=2.0, max_new_tokens=10, top_k=2, top_p=0.9, checkpoint_dir=fake_checkpoint_dir)

    # decoding is done per each generated item
    assert len(tokenizer_mock.return_value.decode_stream.mock_calls) == 1
    assert tokenizer_mock.return_value.decode_stream.call_args[0][0] is generate_mock.return_value # Now a Mock

    # Assert that the generated result is printed to stdout
    assert re.match(r".*Now chatting with Llama 3.*>> .*Reply: foo bar baz", out.getvalue(), re.DOTALL), out.getvalue()


def test_cli():
    args = ["litgpt", "chat", "-h"]
    output = subprocess.check_output(args)
    output = str(output.decode())
    assert "Chat with a model" in output


@skip_in_ci_on_macos
@patch("litgpt.chat.base.input")
@patch("litgpt.chat.base.merge_lora")
def test_merge_lora_if_needed(mocked_merge_lora, mocked_input, fake_checkpoint_dir, monkeypatch, tensor_like):
    # these values will be iteratively provided for each `input()` call
    mocked_input.side_effect = [""]

    # pretend there is an unmerged LORA checkpoint
    os.rename(fake_checkpoint_dir / "lit_model.pth", fake_checkpoint_dir / "lit_model.pth.lora")
    mocked_merge_lora.side_effect = lambda _: Path(fake_checkpoint_dir / "lit_model.pth").touch()

    config = Config.from_name("pythia-14m")
    save_config(config, fake_checkpoint_dir)
    monkeypatch.setattr(chat, "load_checkpoint", Mock())
    monkeypatch.setattr(chat, "Tokenizer", Mock())

    out, err = StringIO(), StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        chat.main(checkpoint_dir=fake_checkpoint_dir)

    assert re.match(r".*Merging LoRA weights with the base model\..*", out.getvalue(), re.DOTALL)
    mocked_merge_lora.assert_called_once()


@skip_in_ci_on_macos
def test_litgpt_chat_endtoend():
    from litgpt.chat.base import main

    checkpoint_dir = auto_download_checkpoint("EleutherAI/pythia-14m")

    # Patch input() and redirect stdout. Raise to exit the repl.
    simulated_input = Mock(side_effect=["input", KeyboardInterrupt])
    captured_output = StringIO()
    with patch('builtins.input', simulated_input):
        with redirect_stdout(captured_output):
            try:
                main(checkpoint_dir=checkpoint_dir, max_new_tokens=256, top_k=1)
            except KeyboardInterrupt:
                pass

    # pythia-14m is not instruct-tuned, so it does not give an "answer" per se, but a continuation.
    assert ">> Reply: !" in captured_output.getvalue(), f"Expected output not found. Got:\n{captured_output.getvalue()}"
    assert simulated_input.call_count == 2


@skip_in_ci_on_macos
def test_litgpt_generate_endtoend():
    from litgpt.generate.base import main

    checkpoint_dir = auto_download_checkpoint("EleutherAI/pythia-14m")

    captured_output = StringIO()
    with redirect_stdout(captured_output):
        try:
            main(checkpoint_dir=checkpoint_dir, prompt="Hello World", max_new_tokens=256, top_k=1)
        except KeyboardInterrupt:
            pass

    # pythia-14m is not instruct-tuned, so it does not give an "answer" per se, but a continuation.
    assert "Hello World!" in captured_output.getvalue(), f"Expected output not found. Got:\n{captured_output.getvalue()}"
