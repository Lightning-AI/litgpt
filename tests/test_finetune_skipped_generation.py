# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import json
from copy import deepcopy

import lightning as L
import pytest
import torch
from tokenizers import Tokenizer as HFTokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from litgpt.args import EvalArgs
from litgpt.data import JSON
from litgpt.finetune import adapter, adapter_v2, full, lora, lora_legacy
from litgpt.prompts import Default
from litgpt.tokenizer import Tokenizer

ENTRY_POINTS = [full, adapter, adapter_v2, lora, lora_legacy]


@pytest.fixture(autouse=True)
def preserve_torch_state():
    num_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng():
        torch.manual_seed(7)
        yield
    torch.set_num_threads(num_threads)


@pytest.fixture()
def generation_inputs(tmp_path):
    raw = HFTokenizer(WordLevel({"[UNK]": 0, "[EOS]": 1, "train": 2, "sample": 3, "answer": 4}, unk_token="[UNK]"))
    raw.pre_tokenizer = Whitespace()
    raw.save(str(tmp_path / "tokenizer.json"))
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"eos_token": "[EOS]", "add_bos_token": False}))
    tokenizer = Tokenizer(tmp_path)
    for split in ("train", "val"):
        (tmp_path / f"{split}.json").write_text(json.dumps([{"instruction": "train sample", "output": "answer"}]))
    data = JSON(tmp_path, prompt_style=Default(), num_workers=0)
    data.connect(tokenizer, batch_size=1, max_seq_length=8)
    data.setup()
    fabric = L.Fabric(accelerator="cpu", devices=1)
    fabric.launch()
    return fabric, tokenizer, data


def make_model(entry):
    options = {
        "block_size": 8,
        "vocab_size": 8,
        "padding_multiple": 1,
        "n_layer": 1,
        "n_head": 2,
        "n_embd": 8,
        "rotary_percentage": 1.0,
    }
    if entry in (lora, lora_legacy):
        options.update(lora_r=2, lora_alpha=2, lora_dropout=0.5, lora_query=True, lora_value=True)
    elif entry in (adapter, adapter_v2):
        options.update(adapter_prompt_length=2, adapter_start_layer=0)
    return entry.GPT(entry.Config(**options))


@pytest.mark.parametrize("entry", ENTRY_POINTS, ids=lambda entry: entry.__name__.rsplit(".", 1)[-1])
@pytest.mark.parametrize("training", [True, False])
@pytest.mark.parametrize("max_new_tokens", [6, 8], ids=["at-context-limit", "over-context-limit"])
def test_skipped_generation_keeps_model_mode_and_cache(entry, training, max_new_tokens, generation_inputs):
    fabric, tokenizer, data = generation_inputs
    model = make_model(entry)
    model.train(training)
    weights = deepcopy(model.state_dict())

    entry.generate_example(fabric, model, tokenizer, EvalArgs(max_new_tokens=max_new_tokens), data)

    assert model.training is training
    assert all(module.training is training for module in model.modules())
    assert model.mask_cache is None
    assert all(block.attn.kv_cache is None for block in model.transformer.h)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, weights[name], rtol=0, atol=0)


@pytest.mark.parametrize("entry", ENTRY_POINTS, ids=lambda entry: entry.__name__.rsplit(".", 1)[-1])
def test_real_generation_still_returns_to_training_and_clears_cache(entry, generation_inputs):
    fabric, tokenizer, data = generation_inputs
    model = make_model(entry)
    model.train()

    entry.generate_example(fabric, model, tokenizer, EvalArgs(max_new_tokens=2), data)

    assert model.training
    assert model.mask_cache is None
    assert all(block.attn.kv_cache is None for block in model.transformer.h)


@pytest.mark.parametrize("entry", [lora, lora_legacy], ids=lambda entry: entry.__name__.rsplit(".", 1)[-1])
def test_skip_preserves_the_next_lora_dropout_training_update(entry, generation_inputs):
    fabric, tokenizer, data = generation_inputs
    model = make_model(entry)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith("lora_B"):
                parameter.fill_(0.2)
    reference = deepcopy(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
    tokens = torch.tensor([[2, 3, 4, 2]])

    entry.generate_example(fabric, model, tokenizer, EvalArgs(max_new_tokens=8), data)
    torch.manual_seed(23)
    loss = model(tokens).float().square().mean()
    loss.backward()
    optimizer.step()
    torch.manual_seed(23)
    expected_loss = reference(tokens).float().square().mean()
    expected_loss.backward()
    reference_optimizer.step()

    torch.testing.assert_close(loss, expected_loss, rtol=0, atol=0)
    for actual, expected in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
