# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
import json

import pytest

from litgpt.data import MultiturnJSON
from litgpt.data.multiturn_json_data import sharegpt_to_messages, to_messages
from litgpt.prompts import PromptStyle


class MultiturnStyle(PromptStyle):
    supports_multiturn = True

    def apply(self, prompt, *, sys_prompt: str | None = None, add_generation_prompt: bool = True, **kwargs) -> str:
        text = "".join(f"[{m['role']}]{m['content']}" for m in prompt)
        if add_generation_prompt:
            text += "[assistant]"
        return text


def _decode_stripped(mock_tokenizer, row) -> str:
    # Right-padded rows get pad_id=0 appended, which MockTokenizer decodes as "\x00".
    return mock_tokenizer.decode(row).rstrip("\x00")


@pytest.mark.parametrize("as_jsonl", [False, True])
def test_multiturn_json(as_jsonl, tmp_path, mock_tokenizer):
    json_path = tmp_path / ("data.jsonl" if as_jsonl else "data.json")
    mock_data = [
        {"messages": [{"role": "user", "content": "Add"}, {"role": "assistant", "content": "4"}]},
        {"messages": [{"role": "user", "content": "Subtract"}, {"role": "assistant", "content": "2"}]},
        {"messages": [{"role": "user", "content": "Multiply"}, {"role": "assistant", "content": "24"}]},
        {"messages": [{"role": "user", "content": "Divide"}, {"role": "assistant", "content": "5"}]},
    ]

    with open(json_path, "w", encoding="utf-8") as fp:
        if as_jsonl:
            for line in mock_data:
                json.dump(line, fp)
                fp.write("\n")
        else:
            json.dump(mock_data, fp)

    data = MultiturnJSON(json_path, val_split_fraction=0.5, prompt_style=MultiturnStyle(), num_workers=0)
    data.connect(tokenizer=mock_tokenizer, batch_size=2)
    data.prepare_data()  # does nothing
    data.setup()

    train_dataloader = data.train_dataloader()
    val_dataloader = data.val_dataloader()

    assert len(train_dataloader) == 1
    assert len(val_dataloader) == 1

    train_data = list(train_dataloader)
    val_data = list(val_dataloader)

    assert train_data[0]["input_ids"].size(0) == 2
    assert val_data[0]["input_ids"].size(0) == 2

    decoded = {
        _decode_stripped(mock_tokenizer, row) for batch in (*train_data, *val_data) for row in batch["input_ids"]
    }
    assert decoded == {
        "[user]Add[assistant]4",
        "[user]Subtract[assistant]2",
        "[user]Multiply[assistant]24",
        "[user]Divide[assistant]5",
    }

    assert isinstance(train_dataloader.dataset.prompt_style, MultiturnStyle)
    assert isinstance(val_dataloader.dataset.prompt_style, MultiturnStyle)

    # has attributes from super class `LightningDataModule`
    assert data.prepare_data_per_node


def test_multiturn_json_sharegpt_format(tmp_path, mock_tokenizer):
    json_path = tmp_path / "data.json"
    mock_data = [
        {"conversations": [{"from": "human", "value": "Add"}, {"from": "gpt", "value": "4"}]},
        {"conversations": [{"from": "human", "value": "Subtract"}, {"from": "gpt", "value": "2"}]},
    ]
    with open(json_path, "w", encoding="utf-8") as fp:
        json.dump(mock_data, fp)

    data = MultiturnJSON(json_path, val_split_fraction=0.5, prompt_style=MultiturnStyle(), num_workers=0)
    data.connect(tokenizer=mock_tokenizer, batch_size=2)
    data.setup()

    decoded = {
        _decode_stripped(mock_tokenizer, row)
        for batch in (*data.train_dataloader(), *data.val_dataloader())
        for row in batch["input_ids"]
    }
    assert decoded == {"[user]Add[assistant]4", "[user]Subtract[assistant]2"}


def test_multiturn_json_mixed_formats(tmp_path, mock_tokenizer):
    # OpenAI-style and ShareGPT-style records in the same file should both work, since format
    # detection happens per example.
    json_path = tmp_path / "data.json"
    mock_data = [
        {"messages": [{"role": "user", "content": "Add"}, {"role": "assistant", "content": "4"}]},
        {"conversations": [{"from": "human", "value": "Subtract"}, {"from": "gpt", "value": "2"}]},
    ]
    with open(json_path, "w", encoding="utf-8") as fp:
        json.dump(mock_data, fp)

    data = MultiturnJSON(json_path, val_split_fraction=0.5, prompt_style=MultiturnStyle(), num_workers=0)
    data.connect(tokenizer=mock_tokenizer, batch_size=2)
    data.setup()

    decoded = {
        _decode_stripped(mock_tokenizer, row)
        for batch in (*data.train_dataloader(), *data.val_dataloader())
        for row in batch["input_ids"]
    }
    assert decoded == {"[user]Add[assistant]4", "[user]Subtract[assistant]2"}


def test_multiturn_json_input_validation(tmp_path):
    with pytest.raises(FileNotFoundError, match="The `json_path` must be a file or a directory"):
        MultiturnJSON(tmp_path / "not exist", prompt_style=MultiturnStyle())

    with pytest.raises(ValueError, match="`val_split_fraction` should not be set"):
        MultiturnJSON(tmp_path, val_split_fraction=0.5, prompt_style=MultiturnStyle())

    data = MultiturnJSON(tmp_path, prompt_style=MultiturnStyle())
    data.prepare_data()  # does nothing

    # Empty directory
    with pytest.raises(FileNotFoundError, match="must be a file or a directory containing"):
        data.setup()

    # Only train.json exists
    (tmp_path / "train.json").touch()
    with pytest.raises(FileNotFoundError, match="must be a file or a directory containing"):
        data.setup()

    # When a single file is passed without val_split_fraction, it defaults to 0.05 and warns.
    with pytest.warns(UserWarning, match="Defaulting to `val_split_fraction=0.05`"):
        data = MultiturnJSON(tmp_path / "train.json", val_split_fraction=None, prompt_style=MultiturnStyle())
    assert data.val_split_fraction == 0.05


def test_multiturn_json_requires_multiturn_style(tmp_path):
    with pytest.raises(ValueError, match="does not support multi-turn conversations"):
        MultiturnJSON(tmp_path, prompt_style="alpaca")


@pytest.mark.parametrize("as_jsonl", [False, True])
def test_multiturn_json_with_splits(as_jsonl, tmp_path, mock_tokenizer):
    mock_train_data = [
        {"messages": [{"role": "user", "content": "Add"}, {"role": "assistant", "content": "4"}]},
        {"messages": [{"role": "user", "content": "Subtract"}, {"role": "assistant", "content": "2"}]},
        {"messages": [{"role": "user", "content": "Multiply"}, {"role": "assistant", "content": "24"}]},
    ]
    mock_test_data = [
        {"messages": [{"role": "user", "content": "Divide"}, {"role": "assistant", "content": "5"}]},
    ]

    train_file = tmp_path / ("train.jsonl" if as_jsonl else "train.json")
    val_file = tmp_path / ("val.jsonl" if as_jsonl else "val.json")

    with open(train_file, "w", encoding="utf-8") as fp:
        if as_jsonl:
            for line in mock_train_data:
                json.dump(line, fp)
                fp.write("\n")
        else:
            json.dump(mock_train_data, fp)
    with open(val_file, "w", encoding="utf-8") as fp:
        if as_jsonl:
            for line in mock_test_data:
                json.dump(line, fp)
                fp.write("\n")
        else:
            json.dump(mock_test_data, fp)

    data = MultiturnJSON(tmp_path, prompt_style=MultiturnStyle(), num_workers=0)
    data.connect(tokenizer=mock_tokenizer, batch_size=2)
    data.prepare_data()  # does nothing
    data.setup()

    train_dataloader = data.train_dataloader()
    val_dataloader = data.val_dataloader()

    assert len(train_dataloader) == 2
    assert len(val_dataloader) == 1


def test_to_messages_openai_style():
    example = {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]}
    assert to_messages(example) == example["messages"]


def test_to_messages_sharegpt_style():
    example = {
        "conversations": [
            {"from": "system", "value": "sys"},
            {"from": "human", "value": "hi"},
            {"from": "gpt", "value": "hello"},
        ]
    }
    assert to_messages(example) == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    assert to_messages(example) == sharegpt_to_messages(example)


def test_to_messages_unknown_format_raises():
    with pytest.raises(ValueError, match="Could not determine the conversation format"):
        to_messages({"foo": "bar"})
