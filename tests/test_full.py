# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import os
from contextlib import redirect_stdout
from io import StringIO
from unittest import mock
from unittest.mock import Mock

import torch


@mock.patch.dict(os.environ, {"LT_ACCELERATOR": "cpu"})
def test_full_script(tmp_path, fake_checkpoint_dir, monkeypatch, alpaca_path):
    import finetune.full as module
    from lit_gpt.args import EvalArgs, TrainArgs
    from lit_gpt.data import Alpaca
    from lit_gpt.config import name_to_config

    model_config = dict(block_size=128, n_layer=2, n_embd=8, n_head=4, padded_vocab_size=8)
    monkeypatch.setitem(name_to_config, "tmp", model_config)
    monkeypatch.setattr(module, "load_checkpoint", Mock())

    tokenizer_mock = Mock()
    tokenizer_mock.return_value = tokenizer_mock
    tokenizer_mock.encode = lambda *_, **__: torch.tensor([3, 2, 1])
    monkeypatch.setattr(module, "Tokenizer", tokenizer_mock)

    stdout = StringIO()
    with redirect_stdout(stdout):
        module.setup(
            data=Alpaca(
                download_dir=alpaca_path.parent,
                file_name=alpaca_path.name,
                test_split_fraction=0.5,
                num_workers=0
            ),
            checkpoint_dir=fake_checkpoint_dir,
            out_dir=tmp_path,
            precision="32-true",
            train=TrainArgs(global_batch_size=1, save_interval=2, epochs=1, max_steps=6, micro_batch_size=1),
            eval=EvalArgs(interval=2, max_iters=2, max_new_tokens=1),
        )

    assert {p.name for p in tmp_path.glob("*.pth")} == {
        "step-000002.pth",
        "step-000004.pth",
        "step-000006.pth",
        "lit_model_finetuned.pth",
    }
    assert (tmp_path / "version_0" / "metrics.csv").is_file()

    logs = stdout.getvalue()
    assert logs.count("optimizer.step") == 6
    assert logs.count("val loss") == 3
    assert "of trainable parameters: 1,888" in logs
