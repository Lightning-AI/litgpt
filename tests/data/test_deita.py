# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
from unittest import mock

from litgpt.data import Deita, SFTDataset
from litgpt.data.deita import format_dataset
from litgpt.prompts import Alpaca as AlpacaPromptStyle


def test_format_dataset():
    data = [
        {
            "prompt": "prompt1",
            "prompt_id": "1",
            "messages": [
                {"content": "question1", "role": "user"},
                {"content": "response1", "role": "assistant"},
                {"content": "question2", "role": "user"},
                {"content": "response2", "role": "assistant"},
            ],
        },
        {
            "prompt": "prompt2",
            "prompt_id": "2",
            "messages": [
                {"content": "question3", "role": "user"},
                {"content": "response3", "role": "assistant"},
                {"content": "question4", "role": "user"},
                {"content": "response4", "role": "assistant"},
            ],
        },
    ]

    assert format_dataset(data, include_multi_turn_conversations=False) == [
        {"instruction": "question1", "output": "response1", "input": ""},
        {"instruction": "question3", "output": "response3", "input": ""},
    ]
    assert format_dataset(data, include_multi_turn_conversations=True) == [
        {"instruction": "question1", "output": "response1", "input": ""},
        {"instruction": "question2", "output": "response2", "input": ""},
        {"instruction": "question3", "output": "response3", "input": ""},
        {"instruction": "question4", "output": "response4", "input": ""},
    ]


@mock.patch("litgpt.data.deita.format_dataset")
@mock.patch("datasets.load_dataset")
def test_deita(_, format_dataset_mock, mock_tokenizer, tmp_path):
    format_dataset_mock.return_value = [
        {"instruction": "inst1", "output": "out1"},
        {"instruction": "inst2", "output": "out2"},
        {"instruction": "inst3", "output": "out3"},
    ]

    deita = Deita(num_workers=0, download_dir=tmp_path)
    assert isinstance(deita.prompt_style, AlpacaPromptStyle)
    deita.connect(mock_tokenizer, batch_size=2, max_seq_length=10)
    deita.prepare_data()
    deita.setup()

    train_dataloader = deita.train_dataloader()
    assert isinstance(train_dataloader.dataset, SFTDataset)
    assert len(train_dataloader) == 2

    val_dataloader = deita.val_dataloader()
    assert isinstance(val_dataloader.dataset, SFTDataset)
    assert len(val_dataloader) == 2

    assert isinstance(train_dataloader.dataset.prompt_style, AlpacaPromptStyle)
    assert isinstance(val_dataloader.dataset.prompt_style, AlpacaPromptStyle)
