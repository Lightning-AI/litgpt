import functools
import subprocess
import sys
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock
from unittest.mock import Mock, call, ANY

import torch

wd = Path(__file__).parent.parent.absolute()


@functools.lru_cache(maxsize=1)
def load_generate_script():
    sys.path.append(str(wd))

    import generate

    return generate


def test_generate():
    generate = load_generate_script()

    from lit_stablelm import StableLM, StableLMConfig

    T, C = 5, 3
    input_idx = torch.randint(10, size=(T,))

    config = StableLMConfig(block_size=128, vocab_size=16, n_layer=1, n_head=4, n_embd=8)
    model = StableLM(config)
    max_new_tokens = 20

    multinomial_results = []
    original_multinomial = torch.multinomial

    def multinomial(*args, **kwargs):
        out = original_multinomial(*args, **kwargs)
        multinomial_results.append(out)
        return out

    with mock.patch("torch.multinomial", multinomial):
        out = generate.generate(model, input_idx, max_new_tokens, max_seq_length=10, top_k=4)

    assert out.size(0) == T + max_new_tokens
    multinomial_results = torch.hstack(multinomial_results)
    expected = torch.cat((input_idx, multinomial_results))
    assert out.shape == expected.shape
    torch.testing.assert_close(out, expected)


@mock.patch("torch.cuda.is_bf16_supported", return_value=False)
def test_main(_, tmp_path, monkeypatch):
    generate = load_generate_script()

    config_path = tmp_path / "config"
    config_path.write_text("{}")

    class FabricMock(Mock):
        @property
        def device(self):
            return torch.device("cpu")

    monkeypatch.setattr(generate.L, "Fabric", FabricMock)
    load_mock = Mock()
    load_mock.return_value = load_mock
    load_mock.__enter__ = Mock()
    load_mock.__exit__ = Mock()
    monkeypatch.setattr(generate, "lazy_load", load_mock)
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
        generate.main(temperature=2.0, top_k=2, num_samples=num_samples, config_path=config_path)

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
