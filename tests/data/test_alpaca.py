# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
from litgpt.data import Alpaca
from litgpt.prompts import Alpaca as AlpacaPromptStyle


def test_alpaca(mock_tokenizer, alpaca_path):
    alpaca = Alpaca(val_split_fraction=0.5, download_dir=alpaca_path.parent, file_name=alpaca_path.name, num_workers=0)
    assert isinstance(alpaca.prompt_style, AlpacaPromptStyle)
    alpaca.connect(mock_tokenizer, batch_size=2, max_seq_length=10)
    alpaca.prepare_data()
    alpaca.setup()

    train_dataloader = alpaca.train_dataloader()
    val_dataloader = alpaca.val_dataloader()

    assert len(train_dataloader) == 6
    assert len(val_dataloader) == 6

    train_batch = next(iter(train_dataloader))
    val_batch = next(iter(val_dataloader))

    assert train_batch.keys() == val_batch.keys() == {"input_ids", "labels", "token_counts"}
    for key in ["input_ids", "labels"]:
        assert train_batch[key].shape == (2, 10), f"Unexpected shape for train_batch[{key}]"
        assert val_batch[key].shape == (2, 10), f"Unexpected shape for val_batch[{key}]"

    assert isinstance(train_dataloader.dataset.prompt_style, AlpacaPromptStyle)
    assert isinstance(val_dataloader.dataset.prompt_style, AlpacaPromptStyle)

    # has attributes from super class `LightningDataModule`
    assert alpaca.prepare_data_per_node
