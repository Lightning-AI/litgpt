# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import os
import shutil
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

import pytest
import torch
import yaml

from litgpt.lora import GPT as LoRAGPT
from litgpt.lora import lora_filter
from litgpt.model import GPT
from litgpt.scripts.merge_lora import load_lora_metadata, merge_lora


@mock.patch.dict(os.environ, {"LT_ACCELERATOR": "cpu"})
@pytest.mark.parametrize(
    ("pretrained_dtype", "lora_dtype"),
    [(None, None), (torch.float16, torch.float32), (torch.float16, torch.bfloat16)],
)
def test_merge_lora(tmp_path, fake_checkpoint_dir, pretrained_dtype, lora_dtype):
    pretrained_checkpoint_dir = tmp_path / "pretrained"
    lora_checkpoint_dir = tmp_path / "lora"
    shutil.copytree(fake_checkpoint_dir, pretrained_checkpoint_dir)
    shutil.copytree(fake_checkpoint_dir, lora_checkpoint_dir)
    (lora_checkpoint_dir / "lit_model.pth").unlink()  # should not already exist
    shutil.rmtree(tmp_path / "checkpoints")

    # Create a fake pretrained checkpoint
    config = dict(block_size=128, padded_vocab_size=256, n_layer=3, n_head=8, n_embd=16)
    with open(
        pretrained_checkpoint_dir / "model_config.yaml", "w", encoding="utf-8"
    ) as fp:
        yaml.dump(config, fp)
    base_model = GPT.from_name("pythia-14m", **config).to(dtype=pretrained_dtype)
    state_dict = base_model.state_dict()
    assert len(state_dict) == 40
    torch.save(state_dict, pretrained_checkpoint_dir / "lit_model.pth")

    # Create a fake LoRA checkpoint
    lora_kwargs = dict(
        lora_r=8, lora_alpha=16, lora_dropout=0.05, lora_query=True, lora_value=True
    )
    lora_model = LoRAGPT.from_name("pythia-14m", **config, **lora_kwargs).to(
        dtype=lora_dtype
    )
    state_dict = {k: v for k, v in lora_model.state_dict().items() if lora_filter(k, v)}
    assert len(state_dict) == 6
    torch.save(state_dict, lora_checkpoint_dir / "lit_model.pth.lora")
    hparams = dict(checkpoint_dir=str(pretrained_checkpoint_dir), **lora_kwargs)
    with open(
        lora_checkpoint_dir / "hyperparameters.yaml", "w", encoding="utf-8"
    ) as file:
        yaml.dump(hparams, file)
    shutil.copyfile(
        pretrained_checkpoint_dir / "model_config.yaml",
        lora_checkpoint_dir / "model_config.yaml",
    )

    assert set(os.listdir(tmp_path)) == {"lora", "pretrained"}
    merge_lora(lora_checkpoint_dir)
    assert set(os.listdir(tmp_path)) == {"lora", "pretrained"}
    assert set(os.listdir(lora_checkpoint_dir)) == {
        "model_config.yaml",
        "lit_model.pth",
        "lit_model.pth.lora",
        "tokenizer.json",
        "tokenizer_config.json",
        "hyperparameters.yaml",
    }

    # Assert that the merged weights can be loaded back into the base model
    merged = torch.load(lora_checkpoint_dir / "lit_model.pth")
    keys = base_model.load_state_dict(merged, strict=True)
    assert not keys.missing_keys
    assert not keys.unexpected_keys

    # Attempt to merge again
    stdout = StringIO()
    with redirect_stdout(stdout):
        merge_lora(lora_checkpoint_dir)
    assert "LoRA weights have already been merged" in stdout.getvalue()


def test_load_lora_metadata(fake_checkpoint_dir):
    assert not (fake_checkpoint_dir / "hyperparameters.yaml").is_file()
    with pytest.raises(
        FileNotFoundError, match="missing a `hyperparameters.yaml` file"
    ):
        load_lora_metadata(fake_checkpoint_dir)

    hparams = dict(
        precision="bf16-mixed",
        checkpoint_dir="checkpoints/meta-llama/Llama-2-7b",
        lora_r=8,
        lora_alpha=16,
    )
    with open(
        fake_checkpoint_dir / "hyperparameters.yaml", "w", encoding="utf-8"
    ) as file:
        yaml.dump(hparams, file)

    lora_args, pretrained_dir, precision = load_lora_metadata(fake_checkpoint_dir)
    assert lora_args == dict(lora_r=8, lora_alpha=16)
    assert pretrained_dir == Path("checkpoints/meta-llama/Llama-2-7b")
    assert precision == "bf16-mixed"
