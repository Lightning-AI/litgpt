# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

from litgpt.data import Dolly
from litgpt.prompts import Alpaca as AlpacaPromptStyle


def test_dolly(mock_tokenizer, dolly_path):
    dolly = Dolly(val_split_fraction=0.5, download_dir=dolly_path.parent, file_name=dolly_path.name, num_workers=0)
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

    assert train_batch.keys() == val_batch.keys() == {"input_ids", "labels"}
    assert all(seq.shape == (2, 10) for seq in train_batch.values())
    assert all(seq.shape == (2, 10) for seq in val_batch.values())

    assert isinstance(train_dataloader.dataset.prompt_style, AlpacaPromptStyle)
    assert isinstance(val_dataloader.dataset.prompt_style, AlpacaPromptStyle)

    # has attributes from super class `LightningDataModule`
    assert dolly.prepare_data_per_node
