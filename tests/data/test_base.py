# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
from unittest.mock import Mock

import pytest
import torch


class MockTokenizer:
    """A dummy tokenizer that encodes each character as its ASCII code."""
    def encode(self, text, eos=False, **kwargs):
        output = [ord(c) for c in text]
        if eos:
            output.append(1)
        return torch.tensor(output)


@pytest.mark.parametrize("mask_prompt", [True, False])
def test_sft_dataset(mask_prompt):
    from lit_gpt.data import SFTDataset

    prompt_template = "In: {instruction} Out:"
    data = [
        {"instruction": "Foo", "output": "Bar"},
        {"instruction": "Boo", "output": "Ahh"},
    ]

    dataset = SFTDataset(data=data, tokenizer=MockTokenizer(), prompt_template=prompt_template, mask_prompt=mask_prompt)
    expected = {
        "input_ids": torch.tensor([73, 110, 58, 32, 70, 111, 111, 32, 79, 117, 116, 58, 66, 97, 114, 1]),
        "labels": torch.tensor([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 66, 97, 114, 1]),
    }

    assert torch.equal(dataset[0]["input_ids"], expected["input_ids"])
    # If prompt is not masked, labels == input_ids
    assert torch.equal(dataset[0]["labels"], expected["labels"] if mask_prompt else expected["input_ids"])


@pytest.mark.parametrize("ignore_index", [-1, -100])
@pytest.mark.parametrize("pad_id", [0, 100])
def test_sft_collate_fn_padding(pad_id, ignore_index):
    from lit_gpt.data import get_sft_collate_fn

    collate = get_sft_collate_fn(pad_id=pad_id, ignore_index=ignore_index)
    samples = [
        {"input_ids": torch.tensor([1, 2, 3]), "labels": torch.tensor([10, 20, 30])},
        {"input_ids": torch.tensor([4, 5, 6, 7, 8]), "labels": torch.tensor([40, 50, 60, 70, 80])},
    ]
    expected = {
        "input_ids": torch.tensor([[1, 2, 3, pad_id, pad_id], [4, 5, 6, 7, 8]]),
        "labels": torch.tensor([[10, 20, 30, ignore_index, ignore_index], [40, 50, 60, 70, 80]])
    }
    batch = collate(samples)
    assert all(torch.equal(batch[k], expected[k]) for k in ("input_ids", "labels"))


def test_sft_collate_fn_truncation():
    from lit_gpt.data import get_sft_collate_fn

    collate = get_sft_collate_fn(max_seq_length=2)
    samples = [
        {"input_ids": torch.tensor([1, 2, 3]), "labels": torch.tensor([10, 20, 30])},
        {"input_ids": torch.tensor([4, 5, 6, 7, 8]), "labels": torch.tensor([40, 50, 60, 70, 80])},
    ]
    expected = {
        "input_ids": torch.tensor([[1, 2], [4, 5]]),
        "labels": torch.tensor([[10, 20], [40, 50]])
    }
    batch = collate(samples)
    assert all(torch.equal(batch[k], expected[k]) for k in ("input_ids", "labels"))


def test_apply_prompt_template():
    from lit_gpt.data import apply_prompt_template

    # As a format-string
    template = "Human: {instruction} {smile} Assistant:"
    example = {"instruction": "Is a coconut a nut?", "smile": ":)"}
    expected = "Human: Is a coconut a nut? :) Assistant:"
    assert apply_prompt_template(template, example) == expected

    # As a callable
    template = lambda x: f"Human: {x['instruction']} {x.get('smile', '')}Assistant:"
    example = {"instruction": "Is a coconut a nut?"}
    expected = "Human: Is a coconut a nut? Assistant:"
    assert apply_prompt_template(template, example) == expected
