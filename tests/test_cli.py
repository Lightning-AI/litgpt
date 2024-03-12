from contextlib import redirect_stdout
from io import StringIO
from unittest import mock

import pytest


def test_cli(tmp_path):
    from litgpt.__main__ import main

    out = StringIO()
    with pytest.raises(SystemExit), redirect_stdout(out), mock.patch("sys.argv", ["litgpt", "-h"]):
        main()
    out = out.getvalue()
    assert "usage: litgpt" in out
    assert "{download,chat,finetune,pretrain,generate,convert,merge_lora}" in out
    assert """Available subcommands:
    download            Download weights or tokenizer data from the Hugging
                        Face Hub.
    chat                Chat with a model.""" in out

    out = StringIO()
    with pytest.raises(SystemExit), redirect_stdout(out), mock.patch("sys.argv", ["litgpt", "finetune", "-h"]):
        main()
    out = out.getvalue()
    assert """Available subcommands:
    lora                Finetune a model with LoRA.
    full                Finetune a model.""" in out

    out = StringIO()
    with pytest.raises(SystemExit), redirect_stdout(out), mock.patch("sys.argv", ["litgpt", "finetune", "lora", "-h"]):
        main()
    out = out.getvalue()
    assert """ARG:   --lora_alpha LORA_ALPHA
  ENV:   LITGPT_FINETUNE__LORA__LORA_ALPHA
                        (type: int, default: 16)""" in out

    out = StringIO()
    with pytest.raises(SystemExit), redirect_stdout(out), mock.patch("sys.argv", ["litgpt", "pretrain", "-h"]):
        main()
    out = out.getvalue()
    assert """ARG:   --train.max_tokens MAX_TOKENS
  ENV:   LITGPT_PRETRAIN__TRAIN__MAX_TOKENS""" in out
