# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import os
from contextlib import redirect_stdout
from io import StringIO
from unittest import mock
from unittest.mock import Mock

import pytest
import torch
import yaml

import litgpt.finetune.full as module
from litgpt import GPT, Config
from litgpt.args import EvalArgs, TrainArgs
from litgpt.data import Alpaca


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
    setup_args = (fake_checkpoint_dir,)
    setup_kwargs = dict(
        data=Alpaca(download_dir=alpaca_path.parent, file_name=alpaca_path.name, val_split_fraction=0.5, num_workers=0),
        out_dir=out_dir,
        precision="32-true",
        train=TrainArgs(global_batch_size=1, save_interval=2, epochs=1, max_steps=6, micro_batch_size=1),
        eval=EvalArgs(interval=2, max_iters=2, max_new_tokens=1),
    )
    stdout = StringIO()
    with redirect_stdout(stdout), mock.patch("sys.argv", ["full.py", str(fake_checkpoint_dir)]):
        module.setup(*setup_args, **setup_kwargs)

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
    with redirect_stdout(stdout), mock.patch("sys.argv", ["full.py", str(fake_checkpoint_dir)]):
        module.setup(*setup_args, **setup_kwargs)
    logs = stdout.getvalue()
    assert f"Resuming training from {out_dir / 'step-000006' / 'lit_model.pth'}" in logs
    assert logs.count("(step)") == 2
    assert out_dir / "step-000008" in set(out_dir.iterdir())


@pytest.mark.parametrize(
    ("checkpoint_tied", "tie_embeddings", "expect_tied"),
    [(True, None, True), (False, None, False), (True, False, False), (False, True, True)],
)
@mock.patch.dict(os.environ, {"LT_ACCELERATOR": "cpu"})
def test_full_script_tied_embeddings(
    tmp_path, fake_checkpoint_dir, monkeypatch, alpaca_path, checkpoint_tied, tie_embeddings, expect_tied
):
    model_config = dict(block_size=128, n_layer=2, n_embd=8, n_head=4, padded_vocab_size=8)
    (fake_checkpoint_dir / "model_config.yaml").write_text(yaml.dump(model_config))
    initial = GPT(Config(**model_config))
    if checkpoint_tied:  # as `convert_hf_checkpoint` writes Qwen3-1.7B, Llama 3.2 1B, SmolLM2, Gemma, ...
        initial.lm_head.weight = initial.transformer.wte.weight
    torch.save(initial.state_dict(), fake_checkpoint_dir / "lit_model.pth")

    tokenizer_mock = Mock()
    tokenizer_mock.return_value = tokenizer_mock
    tokenizer_mock.encode = lambda *_, **__: torch.tensor([3, 2, 1])
    monkeypatch.setattr(module, "Tokenizer", tokenizer_mock)

    out_dir = tmp_path / "out"

    def run(max_steps, tie, resume=False):
        stdout = StringIO()
        with redirect_stdout(stdout), mock.patch("sys.argv", ["full.py", str(fake_checkpoint_dir)]):
            module.setup(
                fake_checkpoint_dir,
                data=Alpaca(
                    download_dir=alpaca_path.parent, file_name=alpaca_path.name, val_split_fraction=0.5, num_workers=0
                ),
                out_dir=out_dir,
                precision="32-true",
                resume=resume,
                train=TrainArgs(
                    global_batch_size=1,
                    micro_batch_size=1,
                    epochs=1,
                    max_steps=max_steps,
                    lr_warmup_steps=1,
                    save_interval=2,
                    tie_embeddings=tie,
                ),
                eval=EvalArgs(interval=100, max_iters=1, max_new_tokens=1, final_validation=False),
            )
        # 1,888 parameters untied; the 8x8 output projection is shared when tied
        assert f"of trainable parameters: {'1,824' if expect_tied else '1,888'}" in stdout.getvalue()
        final = torch.load(out_dir / "final" / "lit_model.pth", weights_only=True)["model"]
        lm_head, wte = final["lm_head.weight"], final["transformer.wte.weight"]
        assert (lm_head.data_ptr() == wte.data_ptr()) is expect_tied
        assert torch.equal(lm_head, wte) is expect_tied
        return stdout.getvalue(), wte

    _, wte = run(max_steps=2, tie=tie_embeddings)
    assert not torch.equal(wte, initial.transformer.wte.weight)  # trained

    # without the flag, a resumed run follows the step checkpoint's layout, so that its optimizer state still loads
    logs, _ = run(max_steps=3, tie=None, resume=True)
    assert f"Resuming training from {out_dir / 'step-000002' / 'lit_model.pth'}" in logs
