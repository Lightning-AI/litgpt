# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import copy
import json

import pytest
import torch
from tokenizers import Tokenizer as HFTokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from torch.utils.data import ConcatDataset, Subset

from litgpt import GPT, Config
from litgpt.data import FLAN, LongForm, SFTDataset
from litgpt.data.flan import _transform as flan_transform
from litgpt.data.longform import _transform as longform_transform
from litgpt.finetune.lora import get_longest_seq_length
from litgpt.tokenizer import Tokenizer
from litgpt.utils import chunked_cross_entropy


@pytest.fixture
def tokenizer(tmp_path):
    directory = tmp_path / "tokenizer"
    directory.mkdir()
    vocab = {token: index for index, token in enumerate(["<unk>", "<eos>", "hello", "world", "answer", "good"])}
    processor = HFTokenizer(WordLevel(vocab, unk_token="<unk>"))
    processor.pre_tokenizer = Whitespace()
    processor.save(str(directory / "tokenizer.json"))
    (directory / "tokenizer_config.json").write_text(json.dumps({"eos_token": "<eos>"}), encoding="utf-8")
    return Tokenizer(directory)


def _rows(kind):
    if kind == "longform":
        return [{"input": "hello world", "output": "answer good"}, {"input": "world", "output": "good"}]
    return [{"inputs": "hello world", "targets": "answer good"}, {"inputs": "world", "targets": "good"}]


def _canonical_rows(rows, kind):
    if kind == "longform":
        return [{"instruction": row["input"], "output": row["output"]} for row in rows]
    return [{"instruction": row["inputs"], "output": row["targets"]} for row in rows]


def _assert_same_sample(actual, expected):
    for key in ("input_ids", "labels"):
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    assert actual["token_counts"] == expected["token_counts"]


@pytest.mark.parametrize("kind", ["longform", "flan"])
@pytest.mark.parametrize("use_subset", [False, True])
@pytest.mark.parametrize("mask_prompt", [False, True])
@pytest.mark.parametrize("max_seq_length", [-1, 5])
def test_builtin_transforms_preserve_samples_on_repeated_reads(
    tokenizer, kind, use_subset, mask_prompt, max_seq_length
):
    rows = _rows(kind)
    original = copy.deepcopy(rows)
    expected_rows = _canonical_rows(rows, kind)
    if use_subset:
        data = Subset(rows, [1, 0])
        expected_data = Subset(expected_rows, [1, 0])
    else:
        data, expected_data = rows, expected_rows
    kwargs = {
        "tokenizer": tokenizer,
        "prompt_style": kind,
        "mask_prompt": mask_prompt,
        "max_seq_length": max_seq_length,
    }
    dataset = SFTDataset(data, transform=longform_transform if kind == "longform" else flan_transform, **kwargs)
    reference = SFTDataset(expected_data, **kwargs)

    for _ in range(3):
        for index in range(len(dataset)):
            _assert_same_sample(dataset[index], reference[index])
            assert rows == original


@pytest.mark.parametrize("mask_prompt", [False, True])
@pytest.mark.parametrize("max_seq_length", [-1, 5])
def test_samples_without_transforms_are_unchanged(tokenizer, mask_prompt, max_seq_length):
    rows = _canonical_rows(_rows("longform"), "longform")
    original = copy.deepcopy(rows)
    dataset = SFTDataset(rows, tokenizer, "longform", mask_prompt=mask_prompt, max_seq_length=max_seq_length)
    first = dataset[0]
    _assert_same_sample(dataset[0], first)
    assert rows == original


@pytest.mark.parametrize("kind", ["longform", "flan"])
@pytest.mark.parametrize("mask_prompt", [False, True])
def test_builtin_datamodules_train_after_length_scan_and_across_epochs(tmp_path, tokenizer, kind, mask_prompt):
    directory = tmp_path / kind
    directory.mkdir()
    rows = _rows(kind)
    if kind == "longform":
        for split in ("train", "val"):
            (directory / f"{split}.json").write_text(json.dumps(rows), encoding="utf-8")
        data = LongForm(download_dir=directory, num_workers=0, mask_prompt=mask_prompt)
    else:
        for split in ("train", "test"):
            (directory / f"aeslc_10templates_{split}.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
        data = FLAN(download_dir=directory, subsets="aeslc_10templates", num_workers=0, mask_prompt=mask_prompt)
    data.connect(tokenizer, batch_size=2, max_seq_length=64)
    train_loader, val_loader = data.train_dataloader(), data.val_dataloader()
    longest, _ = get_longest_seq_length(ConcatDataset([train_loader.dataset, val_loader.dataset]))

    torch.manual_seed(123)
    model = GPT(Config(block_size=64, vocab_size=6, padding_multiple=1, n_layer=1, n_head=2, n_embd=8))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    batches = []
    for _ in range(2):
        for batch in train_loader:
            assert batch["input_ids"].size(1) <= longest
            optimizer.zero_grad()
            logits = model(batch["input_ids"])
            loss = chunked_cross_entropy(logits[:, :-1], batch["labels"][:, 1:])
            assert torch.isfinite(loss)
            loss.backward()
            assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())
            optimizer.step()
            batches.append(sorted(tuple(row) for row in batch["input_ids"].tolist()))
        assert len(list(val_loader)) == 1

    assert len(batches) == 2
    assert batches[0] == batches[1]
    assert any(not torch.equal(before[name], parameter) for name, parameter in model.named_parameters())
    assert train_loader.dataset.data == rows
    assert val_loader.dataset.data == rows
