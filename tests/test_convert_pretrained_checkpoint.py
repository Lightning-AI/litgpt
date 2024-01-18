# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import os

import torch


def test_convert_pretrained_checkpoint(tmp_path):
    from scripts.convert_pretrained_checkpoint import convert_checkpoint

    # Pretend we made a checkpoint from pretraining
    pretrained_checkpoint = {
        "model": {"some.module.weight": torch.rand(2, 2), "_orig_mod.some.other.module.weight": torch.rand(2, 2)},
        "the_optimizer": "optimizer_state",
        "other": 1,
    }
    torch.save(pretrained_checkpoint, tmp_path / "pretrained.pth")

    # Make a fake tokenizer config file
    llama_checkpoint_folder = tmp_path / "checkpoints" / "meta-llama" / "Llama-2-7b-hf"
    llama_checkpoint_folder.mkdir(parents=True)
    (llama_checkpoint_folder / "tokenizer_config.json").touch()

    convert_checkpoint(
        checkpoint_file=(tmp_path / "pretrained.pth"),
        tokenizer_dir=llama_checkpoint_folder,
        config_name="tiny-llama-1.1b",
        output_dir=(tmp_path / "converted"),
    )

    assert set(os.listdir(tmp_path / "converted")) == {"lit_model.pth", "lit_config.json", "tokenizer_config.json"}
    converted_checkpoint = torch.load(tmp_path / "converted" / "lit_model.pth")
    assert list(converted_checkpoint.keys()) == ["some.module.weight", "some.other.module.weight"]
