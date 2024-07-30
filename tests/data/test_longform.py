# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
from litgpt.data import LongForm
from litgpt.prompts import Longform as LongFormPromptStyle


def test_longform(mock_tokenizer, longform_path):
    longform = LongForm(download_dir=longform_path, num_workers=0)
    assert isinstance(longform.prompt_style, LongFormPromptStyle)
    longform.connect(mock_tokenizer, batch_size=2, max_seq_length=10)
    longform.prepare_data()
    longform.setup()

    train_dataloader = longform.train_dataloader()
    val_dataloader = longform.val_dataloader()

    assert len(train_dataloader) == 9
    assert len(val_dataloader) == 5

    train_batch = next(iter(train_dataloader))
    val_batch = next(iter(val_dataloader))

    assert train_batch.keys() == val_batch.keys() == {"input_ids", "labels"}
    assert all(seq.shape == (2, 10) for seq in train_batch.values())
    assert all(seq.shape == (2, 10) for seq in val_batch.values())

    assert isinstance(train_dataloader.dataset.prompt_style, LongFormPromptStyle)
    assert isinstance(val_dataloader.dataset.prompt_style, LongFormPromptStyle)

    # has attributes from super class `LightningDataModule`
    assert longform.prepare_data_per_node
