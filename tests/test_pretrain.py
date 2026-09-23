# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import os
from contextlib import redirect_stdout
from io import StringIO
from unittest import mock
from unittest.mock import ANY, Mock

import pytest
import torch
from lightning.fabric.strategies import FSDPStrategy, SingleDeviceStrategy
from torch.utils.data import DataLoader

from litgpt import pretrain
from litgpt.args import EvalArgs, TrainArgs
from litgpt.config import Config
from litgpt.pretrain import initialize_weights
from litgpt.utils import _RunIf


@pytest.mark.parametrize(
    ("max_steps", "initial_steps", "expected_steps"),
    [(2, 0, 2), (2, 1, 2), (2, 2, 2), (None, 0, 4), (6, 0, 4)],
)
def test_fit_max_steps(tmp_path, monkeypatch, max_steps, initial_steps, expected_steps):
    fabric = pretrain.L.Fabric(accelerator="cpu", devices=1)
    model = pretrain.GPT(Config(block_size=2, n_layer=1, n_embd=4, n_head=2, padded_vocab_size=8))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    model, optimizer = fabric.setup(model, optimizer)
    dataloader = DataLoader(torch.tensor([[0, 1, 2], [3, 4, 5]]))
    state = {
        "model": model,
        "optimizer": optimizer,
        "iter_num": initial_steps * 2,
        "step_count": initial_steps,
    }
    monkeypatch.setattr(pretrain, "measure_flops", Mock(return_value=0))
    monkeypatch.setattr(pretrain, "save_hyperparameters", Mock())
    pretrain.fit(
        fabric,
        1,
        state,
        dataloader,
        dataloader,
        tmp_path,
        None,
        TrainArgs(
            global_batch_size=2,
            micro_batch_size=1,
            max_tokens=16,
            max_steps=max_steps,
            save_interval=2,
            max_norm=1.0,
            lr_warmup_steps=0,
        ),
        EvalArgs(interval=100, max_iters=1, final_validation=False),
    )
    assert state["step_count"] == expected_steps
    assert state["iter_num"] == expected_steps * 2
    expected_dirs = {f"step-{step:08d}" for step in range(initial_steps + 1, expected_steps + 1) if step % 2 == 0}
    assert {path.name for path in tmp_path.iterdir()} == expected_dirs
    for name in expected_dirs:
        checkpoint = torch.load(tmp_path / name / "lit_model.pth", weights_only=True)
        assert checkpoint["step_count"] == int(name.removeprefix("step-"))
        assert (tmp_path / name / "model_config.yaml").is_file()


@_RunIf(min_cuda_gpus=1, standalone=True)
@mock.patch("litgpt.pretrain.save_hyperparameters")
def test_optimizer_args(_, tmp_path):
    model_config = Config(block_size=2, n_layer=2, n_embd=4, n_head=2, padded_vocab_size=8)

    dataset = torch.tensor([[0, 1, 2], [3, 4, 5], [0, 1, 2]])
    dataloader = DataLoader(dataset)
    pretrain.get_dataloaders = Mock(return_value=(dataloader, dataloader))

    for i in ("AdamW", "SGD", "RMSprop"):
        pretrain.setup(
            "pythia-14m",
            devices=1,
            optimizer="RMSprop",
            model_config=model_config,
            out_dir=tmp_path,
            train=TrainArgs(global_batch_size=2, max_tokens=16, save_interval=1, micro_batch_size=1, max_norm=1.0),
            eval=EvalArgs(interval=1, max_iters=1, final_validation=False),
        )


@_RunIf(min_cuda_gpus=2, standalone=True)
# If we were to use `save_hyperparameters()`, we would have to patch `sys.argv` or otherwise
# the CLI would capture pytest args, but unfortunately patching would mess with subprocess
# launching, so we need to mock `save_hyperparameters()`
@mock.patch("litgpt.pretrain.save_hyperparameters")
# todo: it expects exactly 2 GPUs and has strange failing for validated 4 # GPUs, so we temporarily mark it as xfail
@pytest.mark.xfail(condition=torch.cuda.device_count() != 2, reason="This test is flaky, expects exactly 2 GPUs")
def test_pretrain(_, tmp_path):
    model_config = Config(block_size=2, n_layer=2, n_embd=8, n_head=4, padded_vocab_size=8)

    dataset = torch.tensor([[0, 1, 2], [3, 4, 5], [0, 1, 2]])
    dataloader = DataLoader(dataset)
    pretrain.get_dataloaders = Mock(return_value=(dataloader, dataloader))

    out_dir = tmp_path / "out"
    stdout = StringIO()
    with redirect_stdout(stdout):
        pretrain.setup(
            "pythia-14m",
            devices=2,
            model_config=model_config,
            out_dir=out_dir,
            train=TrainArgs(global_batch_size=2, max_tokens=16, save_interval=1, micro_batch_size=1, max_norm=1.0),
            eval=EvalArgs(interval=1, max_iters=1, final_validation=False),
        )

    if torch.distributed.get_rank() == 0:
        # tmp_path is not the same across all ranks, run assert only on rank 0
        out_dir_contents = set(os.listdir(out_dir))
        checkpoint_dirs = {"step-00000001", "step-00000002", "step-00000003", "step-00000004", "final"}
        assert checkpoint_dirs.issubset(out_dir_contents)
        assert all((out_dir / p).is_dir() for p in checkpoint_dirs)
        for checkpoint_dir in checkpoint_dirs:
            # the `tokenizer_dir` is None by default, so only 'lit_model.pth' shows here
            assert set(os.listdir(out_dir / checkpoint_dir)) == {"lit_model.pth", "model_config.yaml"}

        assert (out_dir / "logs" / "tensorboard" / "version_0").is_dir()

        # logs only appear on rank 0
        logs = stdout.getvalue()
        assert logs.count("(step)") == 4
        assert logs.count("val loss") == 4
        assert "Total parameters: 1,888" in logs

    torch.distributed.barrier()


@_RunIf(min_cuda_gpus=2, standalone=True)
@mock.patch("litgpt.pretrain.L.Fabric.load_raw")
# See comment in `test_pretrain` why we need to mock `save_hyperparameters()`
@mock.patch("litgpt.pretrain.save_hyperparameters")
def test_initial_checkpoint_dir(_, load_mock, tmp_path):
    model_config = Config(block_size=2, n_layer=2, n_embd=8, n_head=4, padded_vocab_size=8)

    dataset = torch.tensor([[0, 1, 2], [3, 4, 5], [0, 1, 2]])
    dataloader = DataLoader(dataset)
    pretrain.get_dataloaders = Mock(return_value=(dataloader, dataloader))
    pretrain.fit = Mock()

    pretrain.setup(
        "pythia-14m",
        initial_checkpoint_dir=tmp_path,
        devices=torch.cuda.device_count(),
        model_config=model_config,
        out_dir=tmp_path,
    )

    load_mock.assert_called_once_with(tmp_path / "lit_model.pth", ANY)


@pytest.mark.parametrize(("strategy", "expected"), [(SingleDeviceStrategy, True), (FSDPStrategy, False)])
def test_initialize_weights(strategy, expected):
    fabric_mock = Mock()
    fabric_mock.strategy = Mock(spec=strategy)

    class Child(torch.nn.Module):
        pass

    class Parent(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.child = Child()

    model = Parent()
    model.reset_parameters = Mock()
    model.child.reset_parameters = Mock()

    initialize_weights(fabric_mock, model, n_layer=2, n_embd=8)
    assert model.reset_parameters.call_count == int(expected)
    assert model.child.reset_parameters.call_count == int(expected)
