import pytest
import torch

from litgpt.data.base import SFTDataset, get_sft_collate_fn
from litgpt.prompts import Llama3, PromptStyle


class Encoding:
    def __init__(self, ids, offsets):
        self.ids = ids
        self.offsets = offsets


class OffsetTokenizer:
    backend = "huggingface"
    bos_id = 0
    eos_id = 1

    class Processor:
        @staticmethod
        def encode(text):
            return Encoding(list(range(2, len(text) + 2)), [(i, i + 1) for i in range(len(text))])

    processor = Processor()

    def encode(self, text, device=None, bos=None, eos=False, max_length=-1):
        tokens = self.processor.encode(text).ids
        if max_length > 0:
            tokens = tokens[:max_length]
        return torch.tensor(tokens, dtype=torch.int, device=device)


class DuplicateBosTokenizer(OffsetTokenizer):
    bos_id = 0

    class Processor:
        @staticmethod
        def encode(text):
            base = OffsetTokenizer.Processor.encode(text)
            return Encoding([0, *base.ids], [(0, 0), *base.offsets])

    processor = Processor()

    def encode(self, text, device=None, bos=None, eos=False, max_length=-1):
        tokens = self.processor.encode(text).ids[1:]
        if max_length > 0:
            tokens = tokens[:max_length]
        return torch.tensor(tokens, dtype=torch.int, device=device)


def test_assistant_masking_single_and_multiple_turns():
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
        {"role": "user", "content": "Bye"},
        {"role": "assistant", "content": "See you"},
    ]
    dataset = SFTDataset(
        [{"messages": messages}], OffsetTokenizer(), Llama3(), mask_prompt=False, mask_strategy="assistant"
    )
    item = dataset[0]
    serialized, spans = Llama3().apply_with_assistant_spans(messages)
    assert serialized == Llama3().apply(messages)
    expected = torch.full_like(item["labels"], -100)
    for i, (start, end) in enumerate([(j, j + 1) for j in range(len(serialized))]):
        if any(start < b and end > a for a, b in spans):
            expected[i] = item["input_ids"][i]
    assert torch.equal(item["labels"], expected)
    assert sum(label != -100 for label in item["labels"]) == sum(b - a for a, b in spans)


def test_assistant_masking_truncation_and_padding():
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
    ]
    dataset = SFTDataset(
        [{"messages": messages}],
        OffsetTokenizer(),
        Llama3(),
        mask_prompt=False,
        mask_strategy="assistant",
    )
    item = dataset[0]
    shorter = {
        "input_ids": item["input_ids"][:-2],
        "labels": item["labels"][:-2],
        "token_counts": item["token_counts"],
    }
    batch = get_sft_collate_fn()([item, shorter])
    assert batch["input_ids"].shape == batch["labels"].shape
    assert torch.all(batch["labels"][1, len(item["labels"]) - 2 :] == -100)


def test_assistant_masking_aligns_backend_duplicate_bos():
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
    ]
    item = SFTDataset(
        [{"messages": messages}],
        DuplicateBosTokenizer(),
        Llama3(),
        mask_prompt=False,
        mask_strategy="assistant",
    )[0]
    assert int((item["labels"] != -100).sum()) == len("Hi") + len("<|eot_id|>")


def test_existing_instruction_output_path_unchanged(mock_tokenizer):
    class Style(PromptStyle):
        def apply(self, prompt, *, sys_prompt=None, **kwargs):
            return f"In: {prompt} Out:"

    dataset = SFTDataset(
        [{"instruction": "Foo", "output": "Bar"}],
        mock_tokenizer,
        Style(),
        mask_prompt=False,
    )
    assert dataset.mask_strategy is None
    assert dataset[0]["labels"].equal(dataset[0]["input_ids"])


@pytest.mark.parametrize("bad_messages", [[], [{"role": "user", "content": "no answer"}]])
def test_assistant_masking_requires_assistant_message(bad_messages):
    dataset = SFTDataset(
        [{"messages": bad_messages}], OffsetTokenizer(), Llama3(), mask_prompt=False, mask_strategy="assistant"
    )
    with pytest.raises(ValueError, match="assistant"):
        dataset[0]
