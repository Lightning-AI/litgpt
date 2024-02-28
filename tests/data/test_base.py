# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import pytest
import torch


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
