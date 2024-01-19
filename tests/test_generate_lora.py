# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import json
import subprocess
import sys
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import ANY, Mock, call

import torch


def test_main(fake_checkpoint_dir, monkeypatch, tensor_like):
    import generate.lora as generate

    config_path = fake_checkpoint_dir / "lit_config.json"
    config = {
        "block_size": 128,
        "vocab_size": 50,
        "n_layer": 2,
        "n_head": 4,
        "n_embd": 8,
        "rotary_percentage": 1,
        "to_query": False,
        "to_value": False,
        "to_projection": True,
    }
    config_path.write_text(json.dumps(config))

    monkeypatch.setattr(generate, "lazy_load", Mock())
    monkeypatch.setattr(generate.GPT, "load_state_dict", Mock())
    tokenizer_mock = Mock()
    tokenizer_mock.return_value.encode.return_value = torch.tensor([[1, 2, 3]])
    tokenizer_mock.return_value.decode.return_value = "### Response:foo bar baz"
    monkeypatch.setattr(generate, "Tokenizer", tokenizer_mock)
    generate_mock = Mock()
    generate_mock.return_value = torch.tensor([[3, 2, 1]])
    monkeypatch.setattr(generate, "generate", generate_mock)

    num_samples = 1
    out, err = StringIO(), StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        generate.main(temperature=2.0, top_k=2, checkpoint_dir=fake_checkpoint_dir)

    assert len(tokenizer_mock.return_value.decode.mock_calls) == num_samples
    assert torch.allclose(tokenizer_mock.return_value.decode.call_args[0][0], generate_mock.return_value)
    assert generate_mock.mock_calls == [call(ANY, tensor_like, 101, temperature=2.0, top_k=2, eos_id=ANY)] * num_samples
    # only the generated result is printed to stdout
    assert out.getvalue() == "foo bar baz\n" * num_samples

    assert "'padded_vocab_size': 512, 'n_layer': 2, 'n_head': 4, 'n_embd': 8" in err.getvalue()


def test_lora_variables_exist():
    import generate.lora as generate

    for lora_argument in ("r", "alpha", "dropout", "query", "key", "value", "projection", "mlp", "head"):
        assert getattr(generate, f"lora_{lora_argument}", None) is not None


def test_lora_is_enabled():
    import generate.lora as generate

    lora_arguments = ("query", "key", "value", "projection", "mlp", "head")
    assert any(getattr(generate, f"lora_{lora_argument}") for lora_argument in lora_arguments)


def test_cli():
    cli_path = Path(__file__).parent.parent / "generate" / "lora.py"
    output = subprocess.check_output([sys.executable, cli_path, "-h"])
    output = str(output.decode())
    assert "Generates a response" in output
