import functools
import subprocess
import sys
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock
from unittest.mock import Mock, call, ANY

import pytest
import torch

wd = Path(__file__).parent.parent.absolute()


@functools.lru_cache(maxsize=1)
def load_generate_script():
    sys.path.append(str(wd))

    import generate

    return generate


@pytest.mark.parametrize("B", (1, 2))
def test_generate(B):
    generate = load_generate_script()

    T, C = 5, 3
    logits = torch.randn(B, T, C)
    input_idx = torch.randint(10, size=(B, T))

    model = Mock(return_value=logits)
    max_new_tokens = 20

    multinomial_results = []
    original_multinomial = torch.multinomial

    def multinomial(*args, **kwargs):
        out = original_multinomial(*args, **kwargs)
        multinomial_results.append(out)
        return out

    with mock.patch("torch.multinomial", multinomial):
        out = generate.generate(model, input_idx, max_new_tokens, max_seq_length=10)

    assert out.shape == (B, T + max_new_tokens)
    multinomial_results = torch.hstack(multinomial_results)
    expected = torch.cat((input_idx, multinomial_results), dim=1)
    assert out.shape == expected.shape
    torch.testing.assert_close(out, expected)


@mock.patch("torch.cuda.is_bf16_supported", return_value=False)
def test_main(tmp_path, monkeypatch):
    generate = load_generate_script()

    checkpoint_path = tmp_path / "ckpt"
    checkpoint_path.touch()
    tokenizer_path = tmp_path / "tokenizer"
    tokenizer_path.touch()

    class FabricMock(Mock):
        @property
        def device(self):
            return torch.device("cpu")

    monkeypatch.setattr(generate.L, "Fabric", FabricMock)
    model_mock = Mock()
    monkeypatch.setattr(generate.LLaMA, "from_name", model_mock)
    load_mock = Mock()
    monkeypatch.setattr(generate.torch, "load", load_mock)
    tokenizer_mock = Mock()
    tokenizer_mock.return_value.encode.return_value = torch.tensor([[1, 2, 3]])
    tokenizer_mock.return_value.decode.return_value = "foo bar baz"
    monkeypatch.setattr(generate, "Tokenizer", tokenizer_mock)
    generate_mock = Mock()
    generate_mock.return_value = torch.tensor([[3, 2, 1]])
    monkeypatch.setattr(generate, "generate", generate_mock)

    num_samples = 2
    out = StringIO()
    with redirect_stdout(out):
        generate.main(
            checkpoint_path=checkpoint_path,
            tokenizer_path=tokenizer_path,
            model_size="1T",
            temperature=2.0,
            top_k=2,
            num_samples=num_samples,
        )

    model_mock.assert_called_once_with("1T")
    load_mock.assert_called_once_with(checkpoint_path)
    tokenizer_mock.assert_called_once_with(tokenizer_path)
    assert len(tokenizer_mock.return_value.decode.mock_calls) == num_samples
    assert torch.allclose(tokenizer_mock.return_value.decode.call_args[0][0], generate_mock.return_value)
    assert generate_mock.mock_calls == [call(ANY, ANY, 50, ANY, temperature=2.0, top_k=2)] * num_samples
    # only the generated result is printed to stdout
    assert out.getvalue() == "foo bar baz\n" * num_samples


def test_cli():
    cli_path = wd / "generate.py"
    output = subprocess.check_output([sys.executable, cli_path, "-h"])
    output = str(output.decode())
    assert "Generates text samples" in output
