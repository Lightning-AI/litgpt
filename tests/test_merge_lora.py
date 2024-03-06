# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import json
import os
import shutil
from unittest import mock

import torch
import yaml


@mock.patch.dict(os.environ, {"LT_ACCELERATOR": "cpu"})
def test_merge_lora(tmp_path, fake_checkpoint_dir):
    from lit_gpt.lora import GPT as LoRAGPT
    from lit_gpt.lora import lora_filter
    from lit_gpt.model import GPT
    from scripts.merge_lora import merge_lora

    pretrained_checkpoint_dir = tmp_path / "pretrained"
    lora_checkpoint_dir = tmp_path / "lora"
    shutil.copytree(fake_checkpoint_dir, pretrained_checkpoint_dir)
    shutil.copytree(fake_checkpoint_dir, lora_checkpoint_dir)
    shutil.rmtree(tmp_path / "checkpoints")

    # Create a fake pretrained checkpoint
    config = dict(block_size=128, padded_vocab_size=256, n_layer=3, n_head=8, n_embd=16)
    with open(pretrained_checkpoint_dir / "lit_config.json", "w") as fp:
        json.dump(config, fp)
    base_model = GPT.from_name("pythia-14m", **config)
    state_dict = base_model.state_dict()
    assert len(state_dict) == 40
    torch.save(state_dict, pretrained_checkpoint_dir / "lit_model.pth")

    # Create a fake LoRA checkpoint
    lora_kwargs = dict(lora_r=8, lora_alpha=16, lora_dropout=0.05, lora_query=True, lora_value=True)
    lora_model = LoRAGPT.from_name("pythia-14m", **config, **lora_kwargs)
    state_dict = {k: v for k, v in lora_model.state_dict().items() if lora_filter(k, v)}
    assert len(state_dict) == 6
    torch.save(state_dict, lora_checkpoint_dir / "lit_model.pth")
    hparams = dict(checkpoint_dir=str(pretrained_checkpoint_dir), **lora_kwargs)
    with open(lora_checkpoint_dir / "hyperparameters.yaml", "w") as file:
        yaml.dump(hparams, file)
    shutil.copyfile(pretrained_checkpoint_dir / "lit_config.json", lora_checkpoint_dir / "lit_config.json")

    assert set(os.listdir(tmp_path)) == {"lora", "pretrained"}
    merge_lora(lora_checkpoint_dir)
    assert set(os.listdir(tmp_path)) == {"lora", "pretrained"}
    assert set(os.listdir(lora_checkpoint_dir)) == {
        "lit_config.json",
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
