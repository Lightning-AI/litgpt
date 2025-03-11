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

    assert train_batch.keys() == val_batch.keys() == {"input_ids", "labels", "token_counts"}
    for key in ["input_ids", "labels"]:
        assert train_batch[key].shape == (2, 10), f"Unexpected shape for train_batch[{key}]"
        assert val_batch[key].shape == (2, 10), f"Unexpected shape for val_batch[{key}]"

    assert isinstance(train_dataloader.dataset.prompt_style, LongFormPromptStyle)
    assert isinstance(val_dataloader.dataset.prompt_style, LongFormPromptStyle)

    # has attributes from super class `LightningDataModule`
    assert longform.prepare_data_per_node
