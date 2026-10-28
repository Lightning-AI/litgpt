# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import os
from contextlib import redirect_stdout
from io import StringIO
from unittest import mock
from unittest.mock import MagicMock, Mock

import pytest
import torch
import yaml
from lightning import Fabric

import litgpt.finetune.full as module
from litgpt.args import EvalArgs, TrainArgs
from litgpt.data import Alpaca, get_sft_collate_fn
from litgpt.model import GPT, Config


@pytest.mark.parametrize("loss_normalization", ["micro_batch", "token"])
def test_full_fit_loss_normalization(loss_normalization, monkeypatch, tmp_path):
    monkeypatch.setattr(module, "Tokenizer", Mock())
    monkeypatch.setattr(module, "generate_example", Mock())

    # 4 micro-batches with 2, 15, 4 and 7 target tokens after the shift, accumulated into one optimizer step
    torch.manual_seed(0)
    samples = []
    for length, prompt_length in ((16, 14), (16, 1), (5, 0), (10, 3)):
        input_ids = torch.randint(0, 16, (length,))
        labels = torch.where(torch.arange(length) < prompt_length, -100, input_ids)
        samples.append(
            {"input_ids": input_ids, "labels": labels, "token_counts": {"raw": 0, "raw_plus_prompt_template": 0}}
        )
    collate_fn = get_sft_collate_fn()
    dataloader = torch.utils.data.DataLoader(samples, batch_size=1, collate_fn=collate_fn)
    train = TrainArgs(
        global_batch_size=4, micro_batch_size=1, epochs=1, max_steps=1, loss_normalization=loss_normalization
    )

    model = GPT(Config(n_layer=2, n_head=2, n_embd=8, block_size=16, padded_vocab_size=16))

    logger = Mock()
    fabric = Fabric(accelerator="cpu", devices=1, loggers=logger)
    # a mock optimizer keeps the accumulated gradients around for inspection
    state = {
        "model": fabric.setup(model),
        "optimizer": Mock(),
        "scheduler": MagicMock(),
        "iter_num": 0,
        "step_count": 0,
    }
    module.fit(
        fabric=fabric,
        state=state,
        train_dataloader=dataloader,
        val_dataloader=dataloader,
        devices=1,
        resume=False,
        checkpoint_dir=tmp_path,
        out_dir=tmp_path,
        train=train,
        eval=EvalArgs(interval=1, max_iters=3),
        data=Mock(),
    )
    grads = {name: param.grad.clone() for name, param in model.named_parameters() if param.grad is not None}
    model.zero_grad()

    # the objective of the whole batch in a single forward pass
    batch = collate_fn(samples)
    logits = model(batch["input_ids"])[:, :-1]
    targets = batch["labels"][:, 1:]
    losses = torch.nn.functional.cross_entropy(logits.flatten(0, 1), targets.flatten(), reduction="none").view_as(
        targets
    )
    num_targets = (targets != -100).sum(dim=1)
    expected_loss = {"token": losses.sum() / num_targets.sum(), "micro_batch": (losses.sum(dim=1) / num_targets).mean()}
    assert not torch.allclose(expected_loss["token"], expected_loss["micro_batch"])
    expected_loss[loss_normalization].backward()
    expected_grads = {name: param.grad for name, param in model.named_parameters() if param.grad is not None}
    torch.testing.assert_close(grads, expected_grads)

    # the logged training loss is the running mean over the micro-batches so far, normalized like the objective: at the
    # optimizer step, it is the objective of the step
    sample_losses = losses.detach().sum(dim=1)
    expected_logged_losses = {
        "token": sample_losses.cumsum(0) / num_targets.cumsum(0),
        "micro_batch": (sample_losses / num_targets).cumsum(0) / torch.arange(1, 5),
    }
    metrics = [call.kwargs["metrics"] for call in logger.log_metrics.call_args_list]
    logged_losses = torch.tensor([m["loss"] for m in metrics if "loss" in m])
    torch.testing.assert_close(logged_losses, expected_logged_losses[loss_normalization])

    # the validation after the optimizer step is normalized in the same way, over at most `eval.max_iters` batches
    (val_loss,) = [m["val_loss"] for m in metrics if "val_loss" in m]
    expected_val_loss = {
        "token": sample_losses[:3].sum() / num_targets[:3].sum(),
        "micro_batch": (sample_losses[:3] / num_targets[:3]).mean(),
    }
    torch.testing.assert_close(torch.tensor(val_loss), expected_val_loss[loss_normalization])


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
