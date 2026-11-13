# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import re
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from unittest.mock import ANY, Mock, call

import torch
import yaml


def test_main(fake_checkpoint_dir, monkeypatch, tensor_like):
    import litgpt.generate.full as generate

    config_path = fake_checkpoint_dir / "model_config.yaml"
    config = {"block_size": 128, "vocab_size": 50, "n_layer": 2, "n_head": 4, "n_embd": 8, "rotary_percentage": 1}
    config_path.write_text(yaml.dump(config))

    monkeypatch.setattr(generate, "load_checkpoint", Mock())
    tokenizer_mock = Mock()
    tokenizer_mock.return_value.encode.return_value = torch.tensor([1, 2, 3])
    # no "### Response:" marker: the checkpoint has no prompt_style.yaml, so a non-Alpaca prompt style is used
    tokenizer_mock.return_value.decode.return_value = "foo bar baz"
    monkeypatch.setattr(generate, "Tokenizer", tokenizer_mock)
    generate_mock = Mock()
    generate_mock.return_value = torch.tensor([1, 2, 3, 7, 8, 9])
    monkeypatch.setattr(generate, "generate", generate_mock)

    out, err = StringIO(), StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        generate.main(
            temperature=2.0,
            top_k=2,
            top_p=0.9,
            checkpoint_dir=fake_checkpoint_dir,
            finetuned_path=fake_checkpoint_dir / "lit_model.pth",
        )

    # only the generated tokens are decoded, not the prompt
    assert torch.equal(tokenizer_mock.return_value.decode.call_args[0][0], torch.tensor([7, 8, 9]))
    assert generate_mock.mock_calls == [call(ANY, tensor_like, 103, temperature=2.0, top_k=2, top_p=0.9, eos_id=ANY)]
    assert re.search(r"^foo bar baz$", out.getvalue(), re.MULTILINE)
