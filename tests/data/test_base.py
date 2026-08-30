# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.


import pytest
import torch

from litgpt.data.base import MultiturnSFTDataset, SFTDataset, get_sft_collate_fn
from litgpt.prompts import ChatML, PromptStyle


@pytest.mark.parametrize("mask_prompt", [True, False])
@pytest.mark.parametrize("ignore_index", [-1, -100])
@pytest.mark.parametrize("max_seq_length", [1000, 5, -1])
def test_sft_dataset(max_seq_length, ignore_index, mask_prompt, mock_tokenizer):
    class Style(PromptStyle):
        def apply(self, prompt: str, *, sys_prompt: str | None = None, **kwargs) -> str:
            return f"In: {prompt} Out:"

    i = ignore_index
    data = [{"instruction": "Foo", "output": "Bar"}, {"instruction": "Boo", "output": "Ahh"}]

    dataset = SFTDataset(
        data=data,
        tokenizer=mock_tokenizer,
        prompt_style=Style(),
        mask_prompt=mask_prompt,
        ignore_index=ignore_index,
        max_seq_length=max_seq_length,
    )
    assert len(dataset) == len(data)

    expected_input_ids = torch.tensor([73, 110, 58, 32, 70, 111, 111, 32, 79, 117, 116, 58, 66, 97, 114, 1])
    # If prompt is not masked, labels == input_ids
    expected_labels = (
        torch.tensor([i, i, i, i, i, i, i, i, i, i, i, i, 66, 97, 114, 1]) if mask_prompt else expected_input_ids
    )

    if max_seq_length == -1:
        assert torch.equal(dataset[0]["input_ids"], expected_input_ids)
        assert torch.equal(dataset[0]["labels"], expected_labels)
    else:
        assert torch.equal(dataset[0]["input_ids"], expected_input_ids[:max_seq_length])
        assert torch.equal(dataset[0]["labels"], expected_labels[:max_seq_length])


@pytest.mark.parametrize("mask_prompt", [True, False])
@pytest.mark.parametrize("ignore_index", [-1, -100])
@pytest.mark.parametrize("max_seq_length", [1000, 5, -1])
def test_multiturn_sft_dataset(max_seq_length, ignore_index, mask_prompt, mock_tokenizer):
    class Style(PromptStyle):
        supports_multiturn = True

        def apply(self, prompt, *, sys_prompt: str | None = None, add_generation_prompt: bool = True, **kwargs) -> str:
            text = "".join(f"[{m['role']}]{m['content']}" for m in prompt)
            if add_generation_prompt:
                text += "[assistant]"
            return text

    messages = [{"role": "user", "content": "Foo"}, {"role": "assistant", "content": "Bar"}]

    dataset = MultiturnSFTDataset(
        data=[messages],
        tokenizer=mock_tokenizer,
        prompt_style=Style(),
        mask_prompt=mask_prompt,
        ignore_index=ignore_index,
        max_seq_length=max_seq_length,
    )
    assert len(dataset) == 1

    expected_input_ids = mock_tokenizer.encode("[user]Foo[assistant]Bar")
    expected_labels = expected_input_ids.clone()
    if mask_prompt:
        prompt_len = len(mock_tokenizer.encode("[user]Foo"))
        expected_labels[:prompt_len] = ignore_index

    if max_seq_length == -1:
        assert torch.equal(dataset[0]["input_ids"], expected_input_ids)
        assert torch.equal(dataset[0]["labels"], expected_labels)
    else:
        assert torch.equal(dataset[0]["input_ids"], expected_input_ids[:max_seq_length])
        assert torch.equal(dataset[0]["labels"], expected_labels[:max_seq_length])

    expected_raw = len(mock_tokenizer.encode("Foo")) + len(mock_tokenizer.encode("Bar"))
    assert dataset[0]["token_counts"]["raw"] == expected_raw
    assert dataset[0]["token_counts"]["raw_plus_prompt_template"] == len(dataset[0]["input_ids"])


def test_multiturn_sft_dataset_masks_only_assistant_turns(mock_tokenizer):
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "bye"},
        {"role": "assistant", "content": "later"},
    ]
    dataset = MultiturnSFTDataset(data=[messages], tokenizer=mock_tokenizer, prompt_style=ChatML())
    item = dataset[0]

    full_text = ChatML().apply(messages, add_generation_prompt=False)
    assert mock_tokenizer.decode(item["input_ids"]) == full_text

    decoded_labels = "".join(chr(int(t)) if t.item() != -100 else "." for t in item["labels"])
    expected_labels = (
        "." * len("<|im_start|>user\nhi<|im_end|>\n")
        + "<|im_start|>assistant\nhello<|im_end|>\n"
        + "." * len("<|im_start|>user\nbye<|im_end|>\n")
        + "<|im_start|>assistant\nlater<|im_end|>\n"
    )
    assert decoded_labels == expected_labels


def test_multiturn_sft_dataset_requires_multiturn_style(mock_tokenizer):
    class SingleTurnStyle(PromptStyle):
        def apply(self, prompt: str, *, sys_prompt: str | None = None, **kwargs) -> str:
            return prompt

    with pytest.raises(ValueError, match="does not support multiturn"):
        MultiturnSFTDataset(data=[[]], tokenizer=mock_tokenizer, prompt_style=SingleTurnStyle())


def test_multiturn_sft_dataset_requires_trailing_assistant_turn(mock_tokenizer):
    messages = [{"role": "user", "content": "hi"}]
    dataset = MultiturnSFTDataset(data=[messages], tokenizer=mock_tokenizer, prompt_style=ChatML())
    with pytest.raises(ValueError, match="must end with an 'assistant' message"):
        dataset[0]


@pytest.mark.parametrize("ignore_index", [-1, -100])
@pytest.mark.parametrize("pad_id", [0, 100])
def test_sft_collate_fn_padding(pad_id, ignore_index):
    collate = get_sft_collate_fn(pad_id=pad_id, ignore_index=ignore_index)
    samples = [
        {
            "input_ids": torch.tensor([1, 2, 3]),
            "labels": torch.tensor([10, 20, 30]),
            "token_counts": {"raw": 3, "raw_plus_prompt_template": 25},
        },
        {
            "input_ids": torch.tensor([4, 5, 6, 7, 8]),
            "labels": torch.tensor([40, 50, 60, 70, 80]),
            "token_counts": {"raw": 5, "raw_plus_prompt_template": 27},
        },
    ]
    expected = {
        "input_ids": torch.tensor([[1, 2, 3, pad_id, pad_id], [4, 5, 6, 7, 8]]),
        "labels": torch.tensor([[10, 20, 30, ignore_index, ignore_index], [40, 50, 60, 70, 80]]),
        "token_counts": {"raw": torch.tensor([[3], [5]]), "raw_plus_prompt_template": torch.tensor([[25], [27]])},
    }
    batch = collate(samples)
    assert all(torch.equal(batch[k], expected[k]) for k in ("input_ids", "labels"))
    for key in ("raw", "raw_plus_prompt_template"):
        assert torch.equal(batch["token_counts"][key], expected["token_counts"][key]), f"Token count mismatch for {key}"


def test_sft_collate_fn_truncation():
    collate = get_sft_collate_fn(max_seq_length=2)
    samples = [
        {
            "input_ids": torch.tensor([1, 2, 3]),
            "labels": torch.tensor([10, 20, 30]),
            "token_counts": {"raw": 3, "raw_plus_prompt_template": 25},
        },
        {
            "input_ids": torch.tensor([4, 5, 6, 7, 8]),
            "labels": torch.tensor([40, 50, 60, 70, 80]),
            "token_counts": {"raw": 5, "raw_plus_prompt_template": 27},
        },
    ]
    expected = {
        "input_ids": torch.tensor([[1, 2], [4, 5]]),
        "labels": torch.tensor([[10, 20], [40, 50]]),
        "token_counts": {"raw": torch.tensor([[3], [5]]), "raw_plus_prompt_template": torch.tensor([[25], [27]])},
    }
    batch = collate(samples)
    assert all(torch.equal(batch[k], expected[k]) for k in ("input_ids", "labels"))
    for key in ("raw", "raw_plus_prompt_template"):
        assert torch.equal(batch["token_counts"][key], expected["token_counts"][key]), f"Token count mismatch for {key}"
