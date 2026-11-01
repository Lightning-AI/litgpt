# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
import importlib
from unittest.mock import Mock

import pytest
import torch
from lightning import Fabric
from torch.utils.data import DataLoader

from litgpt.args import EvalArgs, TrainArgs
from litgpt.data import Alpaca
from litgpt.data.base import get_sft_collate_fn


@pytest.mark.parametrize("method", ["lora", "lora_legacy", "full", "adapter", "adapter_v2"])
@pytest.mark.parametrize(
    ("train_length", "val_length", "limit", "effective_length"),
    [(12, 6, None, 12), (6, 12, None, 12), (12, 6, 10, 10), (12, 6, 8, 8), (6, 5, None, 6), (6, 5, 20, 6)],
)
def test_finetune_sequence_length(
    method, train_length, val_length, limit, effective_length, tmp_path, monkeypatch, capsys
):
    module = importlib.import_module(f"litgpt.finetune.{method}")
    monkeypatch.setattr(module, "Tokenizer", Mock())
    fabric = Fabric(accelerator="cpu", devices=1, precision="32-true")
    model = module.GPT(module.Config(block_size=8, n_layer=1, n_embd=8, n_head=2, padded_vocab_size=16))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    model, optimizer = fabric.setup(model, optimizer)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer.optimizer, lambda _: 1.0)

    def dataloader(length):
        tokens = torch.arange(length) % 16
        return DataLoader(
            [
                {
                    "input_ids": tokens,
                    "labels": tokens,
                    "token_counts": {"raw": length, "raw_plus_prompt_template": length},
                }
            ],
            batch_size=1,
            collate_fn=get_sft_collate_fn(max_seq_length=limit or -1),
        )

    kwargs = dict(
        fabric=fabric,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_dataloader=dataloader(train_length),
        val_dataloader=dataloader(val_length),
        devices=1,
        checkpoint_dir=tmp_path,
        out_dir=tmp_path,
        train=TrainArgs(
            global_batch_size=1,
            micro_batch_size=1,
            epochs=1,
            max_steps=1,
            lr_warmup_steps=0,
            max_seq_length=limit,
            save_interval=None,
        ),
        eval=EvalArgs(max_iters=1, interval=10),
        data=Alpaca(),
    )
    if method in ("full", "adapter_v2"):
        kwargs["resume"] = False
    if method == "full":
        kwargs["state"] = {key: kwargs.pop(key) for key in ("model", "optimizer", "scheduler")}
        kwargs["state"].update(iter_num=0, step_count=0)

    if effective_length > 8:
        with pytest.raises(ValueError, match="--train.max_seq_length=8"):
            module.fit(**kwargs)
        output = capsys.readouterr().out
        assert f"longest sequence length in the train/val data is {max(train_length, val_length)}" in output
        assert f"maximum sequence length is {effective_length}" in output
        assert "context length is 8" in output
        assert model.max_seq_length == 8
    else:
        before = model.transformer.wte.weight.detach().clone()
        module.fit(**kwargs)
        assert model.max_seq_length == effective_length
        assert torch.isfinite(model.transformer.wte.weight).all()
        assert not torch.equal(before, model.transformer.wte.weight)
