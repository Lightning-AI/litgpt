import functools
import subprocess
import sys
from itertools import repeat
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

wd = Path(__file__).parent.parent.absolute()


@functools.lru_cache(maxsize=1)
def load_script():
    sys.path.append(str(wd))

    import chat

    return chat


@pytest.mark.parametrize(
    ("generated", "stop_tokens", "expected"),
    [
        (repeat(1), tuple(), [1] * 6),
        ([1, 2, 3, 0], ([0],), [1, 2, 3]),
        ([1, 2, 3, 0], ([9], [2, 4], [1, 2, 3, 0]), []),
        ([1, 2, 3, 0, 0], ([0, 0, 0], [0, 0]), [1, 2, 3]),
    ],
)
def test_generate(generated, stop_tokens, expected):
    chat = load_script()

    input_idx = torch.tensor([5, 3])
    max_seq_len = 8
    model = MagicMock()

    original_multinomial = torch.multinomial
    it = iter(generated)

    def multinomial(*_, **__):
        out = next(it)
        return torch.tensor([out])

    chat.torch.multinomial = multinomial
    actual = chat.generate(model, input_idx, max_seq_len, stop_tokens=stop_tokens)
    actual = list(actual)
    chat.torch.multinomial = original_multinomial

    # the function can yield a 0-dim tensor or a 1-dim tensor. For decoding it's okay because we support it, but
    # here we need to unsqueeze
    actual = [t.unsqueeze(0) if t.ndim == 0 else t for t in actual]
    if len(actual):
        actual = torch.cat(actual).tolist()
    assert actual == expected


def test_cli():
    cli_path = wd / "chat.py"
    output = subprocess.check_output([sys.executable, cli_path, "-h"])
    output = str(output.decode())
    assert "Starts a conversation" in output
