# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import json
from copy import deepcopy

import pytest
import torch
from tokenizers import Tokenizer as HFTokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from torch.utils.data import DataLoader

from litgpt import Config, GPT, Tokenizer
from litgpt.data.base import SFTDataset, get_sft_collate_fn
from litgpt.prompts import PromptStyle


def _sample(length, dtype=torch.int64, strided=False):
    values = torch.arange(length * (2 if strided else 1), dtype=dtype) % 8
    ids = values[::2] if strided else values
    labels = torch.stack((ids, ids), dim=-1)[:, 0] if strided else ids.clone()
    return {
        "input_ids": ids,
        "labels": labels,
        "token_counts": {"raw": length, "raw_plus_prompt_template": length + 3},
    }


def _reference(samples, limit, pad_id=0, ignore_index=-100):
    output = {}
    for key, padding in (("input_ids", pad_id), ("labels", ignore_index)):
        width = max(sample[key].numel() for sample in samples)
        width = min(width, limit) if limit > 0 else width
        output[key] = samples[0][key].new_full((len(samples), width), padding)
        for index, sample in enumerate(samples):
            length = min(width, sample[key].numel())
            output[key][index, :length] = sample[key][:length]
    output["token_counts"] = {
        key: torch.tensor([[sample["token_counts"][key]] for sample in samples], dtype=torch.int64)
        for key in ("raw", "raw_plus_prompt_template")
    }
    return output


@pytest.mark.parametrize("limit", [-1, 0, 1, 3, 5, 20])
@pytest.mark.parametrize("strided", [False, True])
def test_collation_preserves_values_metadata_and_source_samples(limit, strided):
    samples = [_sample(length, strided=strided) for length in (0, 3, 7)]
    snapshots = deepcopy(samples)
    original_ids = [sample["input_ids"] for sample in samples]
    original_labels = [sample["labels"] for sample in samples]
    batch = get_sft_collate_fn(max_seq_length=limit, pad_id=9, ignore_index=-7)(samples)
    torch.testing.assert_close(batch, _reference(samples, limit, 9, -7), rtol=0, atol=0)
    torch.testing.assert_close(samples, snapshots, rtol=0, atol=0)
    assert all(sample["input_ids"] is original for sample, original in zip(samples, original_ids))
    assert all(sample["labels"] is original for sample, original in zip(samples, original_labels))


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("limit", [1, 128])
@pytest.mark.parametrize("strided", [False, True])
def test_capped_batch_does_not_keep_untruncated_padding_storage(dtype, limit, strided):
    samples = [_sample(length, dtype=dtype, strided=strided) for length in (4096, 1001, 32)]
    batch = get_sft_collate_fn(max_seq_length=limit)(samples)
    torch.testing.assert_close(batch, _reference(samples, limit), rtol=0, atol=0)
    for key in ("input_ids", "labels"):
        assert batch[key].untyped_storage().nbytes() <= len(samples) * limit * batch[key].element_size()


@pytest.mark.parametrize("mask_prompt", [False, True])
def test_native_sft_dataset_and_gpt_updates_match_capped_reference(tmp_path, mask_prompt):
    vocab = {"[UNK]": 0, "<s>": 1, "</s>": 2, "a": 3, "b": 4, "c": 5}
    processor = HFTokenizer(WordLevel(vocab, unk_token="[UNK]"))
    processor.pre_tokenizer = Whitespace()
    processor.save(str(tmp_path / "tokenizer.json"))
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"add_bos_token": False, "bos_token": "<s>", "eos_token": "</s>"}), encoding="utf-8"
    )
    tokenizer = Tokenizer(tmp_path)

    class PlainPrompt(PromptStyle):
        def apply(self, prompt, **kwargs):
            return prompt

    dataset = SFTDataset(
        [
            {"instruction": "a", "output": " ".join(["b", "c"] * 2048)},
            {"instruction": "b", "output": "c a b"},
        ],
        tokenizer=tokenizer,
        prompt_style=PlainPrompt(),
        mask_prompt=mask_prompt,
    )
    loader = DataLoader(dataset, batch_size=2, collate_fn=get_sft_collate_fn(max_seq_length=8))
    torch.manual_seed(17)
    model = GPT(Config(n_layer=2, n_head=2, n_embd=16, block_size=8, vocab_size=8, padding_multiple=1))
    reference = deepcopy(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    expected_optimizer = torch.optim.AdamW(reference.parameters(), lr=0.003)
    for _ in range(2):
        batch = next(iter(loader))
        expected_batch = _reference([dataset[index] for index in range(len(dataset))], 8)
        torch.testing.assert_close(batch, expected_batch, rtol=0, atol=0)
        for key in ("input_ids", "labels"):
            assert batch[key].untyped_storage().nbytes() == batch[key].numel() * batch[key].element_size()
        losses = []
        for current, current_optimizer, current_batch in (
            (model, optimizer, batch),
            (reference, expected_optimizer, expected_batch),
        ):
            current_optimizer.zero_grad()
            logits = current(current_batch["input_ids"])
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)), current_batch["labels"][:, 1:].reshape(-1)
            )
            loss.backward()
            losses.append(loss.detach())
            current_optimizer.step()
        torch.testing.assert_close(losses[0], losses[1], rtol=0, atol=0)
        for parameter, expected in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(parameter.grad, expected.grad, rtol=0, atol=0)
        torch.testing.assert_close(model.state_dict(), reference.state_dict(), rtol=0, atol=0)
        torch.testing.assert_close(optimizer.state_dict(), expected_optimizer.state_dict(), rtol=0, atol=0)
