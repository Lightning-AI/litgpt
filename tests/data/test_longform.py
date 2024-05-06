# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
from litgpt.data import LongForm
from litgpt.prompts import Longform as LongFormPromptStyle


def test_longform(mock_tokenizer, longform_path):
    alpaca = LongForm(download_dir=longform_path, num_workers=0)
    assert isinstance(alpaca.prompt_style, LongFormPromptStyle)
    alpaca.connect(mock_tokenizer, batch_size=2, max_seq_length=10)
    alpaca.prepare_data()
    alpaca.setup()

    train_dataloader = alpaca.train_dataloader()
    val_dataloader = alpaca.val_dataloader()

    assert len(train_dataloader) == 9
    assert len(val_dataloader) == 5

    train_batch = next(iter(train_dataloader))
    val_batch = next(iter(val_dataloader))

    assert train_batch.keys() == val_batch.keys() == {"input_ids", "labels"}
    assert all(seq.shape == (2, 10) for seq in train_batch.values())
    assert all(seq.shape == (2, 10) for seq in val_batch.values())

    assert isinstance(train_dataloader.dataset.prompt_style, LongFormPromptStyle)
    assert isinstance(val_dataloader.dataset.prompt_style, LongFormPromptStyle)
