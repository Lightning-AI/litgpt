# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import os
from contextlib import redirect_stdout
from io import StringIO
from unittest import mock
from unittest.mock import Mock

import torch
import yaml

import litgpt.finetune.full as module
from litgpt.args import EvalArgs, TrainArgs
from litgpt.data import Alpaca
from litgpt.utils import CLI


@mock.patch.dict(os.environ, {"LT_ACCELERATOR": "cpu"})
def test_full_script(tmp_path, fake_checkpoint_dir, monkeypatch, alpaca_path):
    model_config = dict(block_size=128, n_layer=2, n_embd=8, n_head=4, padded_vocab_size=8)
    (fake_checkpoint_dir / "model_config.yaml").write_text(yaml.dump(model_config))
    monkeypatch.setattr(module, "load_checkpoint", Mock())

    tokenizer_mock = Mock()
    tokenizer_mock.return_value = tokenizer_mock
    tokenizer_mock.encode = lambda *_, **__: torch.tensor([3, 2, 1])
    monkeypatch.setattr(module, "Tokenizer", tokenizer_mock)

    out_dir = tmp_path / "out"
    setup_kwargs = dict(
        data=Alpaca(download_dir=alpaca_path.parent, file_name=alpaca_path.name, val_split_fraction=0.5, num_workers=0),
        checkpoint_dir=fake_checkpoint_dir,
        out_dir=out_dir,
        precision="32-true",
        train=TrainArgs(global_batch_size=1, save_interval=2, epochs=1, max_steps=6, micro_batch_size=1),
        eval=EvalArgs(interval=2, max_iters=2, max_new_tokens=1),
    )
    stdout = StringIO()
    with redirect_stdout(stdout), mock.patch("sys.argv", ["full.py"]):
        module.setup(**setup_kwargs)

    out_dir_contents = set(os.listdir(out_dir))
    checkpoint_dirs = {"step-000002", "step-000004", "step-000006", "final"}
    assert checkpoint_dirs.issubset(out_dir_contents)
    assert all((out_dir / p).is_dir() for p in checkpoint_dirs)
    for checkpoint_dir in checkpoint_dirs:
        assert set(os.listdir(out_dir / checkpoint_dir)) == {
            "lit_model.pth",
            "model_config.yaml",
            "tokenizer_config.json",
            "tokenizer.json",
            "hyperparameters.yaml",
            "prompt_style.yaml",
        }
    assert (out_dir / "logs" / "csv" / "version_0" / "metrics.csv").is_file()

    logs = stdout.getvalue()
    assert logs.count("(step)") == 6
    assert logs.count("val loss") == 4  # 3 validations + 1 final validation
    assert logs.count("Final evaluation") == 1
    assert "of trainable parameters: 1,888" in logs

    # Resume training and do 2 steps more
    setup_kwargs["train"].max_steps = 8
    setup_kwargs["resume"] = True
    stdout = StringIO()
    with redirect_stdout(stdout), mock.patch("sys.argv", ["full.py"]):
        module.setup(**setup_kwargs)
    logs = stdout.getvalue()
    assert f"Resuming training from {out_dir / 'step-000006' / 'lit_model.pth'}" in logs
    assert logs.count("(step)") == 2
    assert out_dir / "step-000008" in set(out_dir.iterdir())


@mock.patch.dict(os.environ, {"LT_ACCELERATOR": "cpu"})
def test_full_longlora_script(tmp_path, fake_checkpoint_dir, monkeypatch, alpaca_path):
    model_config = dict(block_size=128, n_layer=2, n_embd=8, n_head=4, padded_vocab_size=8)
    (fake_checkpoint_dir / "model_config.yaml").write_text(yaml.dump(model_config))
    monkeypatch.setattr(module, "load_checkpoint", Mock())

    tokenizer_mock = Mock()
    tokenizer_mock.return_value = tokenizer_mock
    tokenizer_mock.encode = lambda *_, **__: torch.tensor([3, 2, 1])
    monkeypatch.setattr(module, "Tokenizer", tokenizer_mock)

    out_dir = tmp_path / "out"
    setup_kwargs = dict(
        data=Alpaca(download_dir=alpaca_path.parent, file_name=alpaca_path.name, val_split_fraction=0.5, num_workers=0),
        checkpoint_dir=fake_checkpoint_dir,
        out_dir=out_dir,
        precision="32-true",
        train=TrainArgs(global_batch_size=1, save_interval=2, epochs=1, max_steps=6, micro_batch_size=1),
        eval=EvalArgs(interval=2, max_iters=2, max_new_tokens=1),
    )
    stdout = StringIO()
    with redirect_stdout(stdout), mock.patch(
        # Needed to save_hyperparameters function saves correctly LongLora params to be used when resuming
        "sys.argv",
        [
            "full.py",
            "--data=litgpt.data.Alpaca",
            "--data.download_dir=" + str(alpaca_path.parent),
            "--data.file_name=" + str(alpaca_path.name),
            "--data.val_split_fraction=0.5",
            "--data.num_workers=0",
            "--checkpoint_dir=" + str(fake_checkpoint_dir),
            "--out_dir=" + str(out_dir),
            "--precision=32-true",
            "--train.global_batch_size=1",
            "--train.save_interval=2",
            "--train.epochs=1",
            "--train.max_steps=6",
            "--train.micro_batch_size=1",
            "--eval.interval=2",
            "--eval.max_iters=2",
            "--eval.max_new_tokens=1",
            "--longlora.use_longlora=True",
            "--longlora.n_groups=4",
            "--longlora.context_length=256",
        ],
    ):
        CLI(module.setup)

    out_dir_contents = set(os.listdir(out_dir))
    checkpoint_dirs = {"step-000002", "step-000004", "step-000006", "final"}
    assert checkpoint_dirs.issubset(out_dir_contents)
    assert all((out_dir / p).is_dir() for p in checkpoint_dirs)
    for checkpoint_dir in checkpoint_dirs:
        assert set(os.listdir(out_dir / checkpoint_dir)) == {
            "lit_model.pth",
            "model_config.yaml",
            "tokenizer_config.json",
            "tokenizer.json",
            "hyperparameters.yaml",
            "prompt_style.yaml",
        }
    assert (out_dir / "logs" / "csv" / "version_0" / "metrics.csv").is_file()

    logs = stdout.getvalue()
    assert logs.count("(step)") == 6
    assert logs.count("val loss") == 4  # 3 validations + 1 final validation
    assert logs.count("Final evaluation") == 1
    assert "of trainable parameters: 1,888" in logs
    assert "The model context length has been increased from 128 to 256" in logs
    assert "The 'rope_condense_ratio' has been adapted from 1 to 2.0" in logs

    # Resume training and do 2 steps more
    setup_kwargs["train"].max_steps = 8
    setup_kwargs["resume"] = True
    stdout = StringIO()
    with redirect_stdout(stdout), mock.patch("sys.argv", ["full.py"]):
        module.setup(**setup_kwargs)
    logs = stdout.getvalue()
    assert f"Resuming training from {out_dir / 'step-000006' / 'lit_model.pth'}" in logs
    assert logs.count("(step)") == 2
    assert out_dir / "step-000008" in set(out_dir.iterdir())
    assert "The model context length has been increased from 128 to 256" in logs
    assert "The 'rope_condense_ratio' has been adapted from 1 to 2.0" in logs

