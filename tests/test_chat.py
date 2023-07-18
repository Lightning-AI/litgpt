import subprocess
import sys
from itertools import repeat
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch


@pytest.mark.parametrize(
    ("generated", "stop_tokens", "expected"),
    [
        (repeat(1), (), [1] * 8),
        ([1, 2, 3, 0], ([0],), [1, 2, 3]),
        ([1, 2, 3, 0], ([9], [2, 4], [1, 2, 3, 0]), []),
        ([1, 2, 3, 0, 0], ([0, 0, 0], [0, 0]), [1, 2, [3]]),
    ],
)
def test_generate(generated, stop_tokens, expected):
    import chat.base as chat

    input_idx = torch.tensor([5, 3])
    max_returned_tokens = len(input_idx) + 8
    model = MagicMock()
    model.config.block_size = 100

    original_multinomial = torch.multinomial
    it = iter(generated)

    def multinomial(*_, **__):
        out = next(it)
        return torch.tensor([out])

    chat.torch.multinomial = multinomial
    actual = chat.generate(model, input_idx, max_returned_tokens, max_returned_tokens, stop_tokens=stop_tokens)
    actual = list(actual)
    chat.torch.multinomial = original_multinomial

    for t in actual:
        assert t.dtype == torch.long
    assert [t.tolist() for t in actual] == expected


def test_cli():
    cli_path = Path(__file__).parent.parent / "chat" / "base.py"
    output = subprocess.check_output([sys.executable, cli_path, "-h"])
    output = str(output.decode())
    assert "Starts a conversation" in output
