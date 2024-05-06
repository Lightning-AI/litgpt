import sys
from contextlib import redirect_stdout
from io import StringIO
from unittest import mock

import pytest
from packaging.version import Version

from litgpt.__main__ import main


def test_cli():
    out = StringIO()
    with pytest.raises(SystemExit), redirect_stdout(out), mock.patch("sys.argv", ["litgpt", "-h"]):
        main()
    out = out.getvalue()
    assert "usage: litgpt" in out
    assert "{download,chat,finetune,pretrain,generate,convert,merge_lora,evaluate,serve}" in out
    assert (
        """Available subcommands:
    download            Download weights or tokenizer data from the Hugging
                        Face Hub.
    chat                Chat with a model."""
        in out
    )
    assert """evaluate            Evaluate a model with the LM Evaluation Harness.""" in out
    assert """serve               Serve and deploy a model with LitServe.""" in out
    out = StringIO()
    with pytest.raises(SystemExit), redirect_stdout(out), mock.patch("sys.argv", ["litgpt", "finetune", "lora", "-h"]):
        main()
    out = out.getvalue()
    assert (
        """--lora_alpha LORA_ALPHA
                        The LoRA alpha. (type: int, default: 16)"""
        in out
    )

    if Version(f"{sys.version_info.major}.{sys.version_info.minor}") < Version("3.9"):
        # python 3.8 prints `Union[int, null]` instead of `Optional[int]`
        return

    out = StringIO()
    with pytest.raises(SystemExit), redirect_stdout(out), mock.patch("sys.argv", ["litgpt", "pretrain", "-h"]):
        main()
    out = out.getvalue()
    print(out)
    assert (
        """--train.max_tokens MAX_TOKENS
                        Total number of tokens to train on (type:
                        Optional[int], default: 3000000000000)"""
        in out
    )


def test_rewrite_finetune_command():
    out1 = StringIO()
    with pytest.raises(SystemExit), redirect_stdout(out1), mock.patch("sys.argv", ["litgpt", "fineune", "-h"]):
        main()
    out2 = StringIO()
    with pytest.raises(SystemExit), redirect_stdout(out2), mock.patch("sys.argv", ["litgpt", "fineune", "lora", "-h"]):
        main()
    assert out1.getvalue() == out2.getvalue()
