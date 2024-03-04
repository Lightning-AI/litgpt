# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import os
from contextlib import redirect_stdout
from io import StringIO
from unittest import mock
from unittest.mock import Mock

import torch
from conftest import RunIf
from torch.utils.data import DataLoader


@RunIf(min_cuda_gpus=2, standalone=True)
# Set CUDA_VISIBLE_DEVICES for FSDP hybrid-shard, if fewer GPUs are used than are available
@mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1"})
def test_pretrain_tiny_llama(fake_checkpoint_dir, tmp_path, monkeypatch):
    import pretrain.pretrain as module
    from lit_gpt.args import EvalArgs, TrainArgs
    from lit_gpt.config import Config

    model_config = Config(block_size=2, n_layer=2, n_embd=8, n_head=4, padded_vocab_size=8)

    dataset = torch.tensor([[0, 1, 2], [3, 4, 5], [0, 1, 2]])
    dataloader = DataLoader(dataset)
    module.get_dataloaders = Mock(return_value=(dataloader, dataloader))

    tokenizer_mock = Mock()
    monkeypatch.setattr(module, "Tokenizer", tokenizer_mock)

    out_dir = tmp_path / "out"
    stdout = StringIO()
    with redirect_stdout(stdout):
        module.setup(
            devices=2,
            model=model_config,
            out_dir=out_dir,
            checkpoint_dir=fake_checkpoint_dir,
            train=TrainArgs(global_batch_size=2, max_tokens=16, save_interval=1, micro_batch_size=1, max_norm=1.0),
            eval=EvalArgs(interval=1, max_iters=1),
        )

    if torch.distributed.get_rank() == 0:
        # tmp_path is not the same across all ranks, run assert only on rank 0
        out_dir_contents = {p.name for p in out_dir.iterdir()}
        checkpoint_dirs = {"step-00000001", "step-00000002", "step-00000003", "step-00000004"}
        assert checkpoint_dirs.issubset(out_dir_contents)
        assert all((out_dir / p).is_dir() for p in checkpoint_dirs)
        for checkpoint_dir in checkpoint_dirs:
            assert {p.name for p in (out_dir / checkpoint_dir).iterdir()} == {
                "lit_model.pth",
                "lit_config.json",
                "tokenizer_config.json",
                "tokenizer.json",
            }

        # logs only appear on rank 0
        logs = stdout.getvalue()
        assert logs.count("optimizer.step") == 4
        assert logs.count("val loss") == 4
        assert "Total parameters: 1,888" in logs

    torch.distributed.barrier()
