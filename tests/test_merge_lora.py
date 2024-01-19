# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import json
import os

import torch


def test_merge_lora(tmp_path, fake_checkpoint_dir):
    from lit_gpt.lora import GPT as LoRAGPT
    from lit_gpt.lora import lora_filter
    from lit_gpt.model import GPT
    from scripts.merge_lora import merge_lora

    # create fake data
    config = dict(block_size=128, padded_vocab_size=256, n_layer=3, n_head=8, n_embd=16)
    with open(fake_checkpoint_dir / "lit_config.json", "w") as fp:
        json.dump(config, fp)
    base_model = GPT.from_name("pythia-14m", **config)
    state_dict = base_model.state_dict()
    assert len(state_dict) == 40
    torch.save(state_dict, fake_checkpoint_dir / "lit_model.pth")
    lora_model = LoRAGPT.from_name("pythia-14m", **config, r=8, alpha=16, dropout=0.05, to_query=True, to_value=True)
    state_dict = {k: v for k, v in lora_model.state_dict().items() if lora_filter(k, v)}
    assert len(state_dict) == 6
    lora_path = tmp_path / "lora"
    torch.save(state_dict, lora_path)

    assert set(os.listdir(tmp_path)) == {"lora", "checkpoints"}
    merge_lora(lora_path, fake_checkpoint_dir, tmp_path)
    assert set(os.listdir(tmp_path)) == {"lora", "checkpoints", "lit_model.pth"}

    # assert that the merged weights can be loaded back into the base model
    merged = torch.load(tmp_path / "lit_model.pth")
    keys = base_model.load_state_dict(merged, strict=True)
    assert not keys.missing_keys
    assert not keys.unexpected_keys
