# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

from litgpt.data import Dolly
from litgpt.prompts import Alpaca as AlpacaPromptStyle


def test_dolly(mock_tokenizer, dolly_path):
    dolly = Dolly(
        val_split_fraction=0.5,
        download_dir=dolly_path.parent,
        file_name=dolly_path.name,
        num_workers=0,
    )
    assert isinstance(dolly.prompt_style, AlpacaPromptStyle)
    dolly.connect(mock_tokenizer, batch_size=2, max_seq_length=10)
    dolly.prepare_data()
    dolly.setup()

    train_dataloader = dolly.train_dataloader()
    val_dataloader = dolly.val_dataloader()

    assert len(train_dataloader) == 3
    assert len(val_dataloader) == 3

    train_batch = next(iter(train_dataloader))
    val_batch = next(iter(val_dataloader))

    assert train_batch.keys() == val_batch.keys() == {"input_ids", "labels", "token_counts"}
    for key in ["input_ids", "labels"]:
        assert train_batch[key].shape == (2, 10), f"Unexpected shape for train_batch[{key}]"
        assert val_batch[key].shape == (2, 10), f"Unexpected shape for val_batch[{key}]"

    assert isinstance(train_dataloader.dataset.prompt_style, AlpacaPromptStyle)
    assert isinstance(val_dataloader.dataset.prompt_style, AlpacaPromptStyle)

    # has attributes from super class `LightningDataModule`
    assert dolly.prepare_data_per_node


def test_dolly_missing_keys(mock_tokenizer, dolly_path):
    """
    Notes
    -----
    - Added only for the dolly dataset.

    References
    ----------
    - Reference issue: https://github.com/Lightning-AI/litgpt/issues/1760

    Methodology
    -----------
    - Simulate the original behavior by popping `context` key.
    - Run dataloader which will apply `transform`.
        - Previously it would have thrown missing `context` key error because we `popped` the key.
        - Now we are using `get` method to not remove they key(s).
    """

    dolly = Dolly(
        val_split_fraction=0.5,
        download_dir=dolly_path.parent,
        file_name=dolly_path.name,
        num_workers=0,
    )
    dolly.connect(mock_tokenizer, batch_size=2, max_seq_length=10)
    dolly.prepare_data()
    dolly.setup()

    # check if the dataset was created without errors
    assert dolly.train_dataset is not None
    assert dolly.test_dataset is not None

    # Verify that the transform function handled missing keys correctly
    for dataset in [dolly.train_dataset, dolly.test_dataset]:
        for item in dataset.data:
            assert "context" in item
            assert "response" in item
            assert isinstance(item["context"], str)
            assert isinstance(item["response"], str)
            # Drop `context` and `response` keys
            # This is to simulate the behavior of original issue with `item.pop`
            item.pop("context")
            item.pop("response")

    # Check if we can iterate through the dataloader without errors
    # Previous approach would through key error here since we already popped the keys
    train_dataloader = dolly.train_dataloader()
    train_batch = next(iter(train_dataloader))
    assert "input_ids" in train_batch
    assert "labels" in train_batch
