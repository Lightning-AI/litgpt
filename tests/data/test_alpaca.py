# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

def test_alpaca(mock_tockenizer, alpaca_path):
    from lit_gpt.data import Alpaca

    alpaca = Alpaca(
        test_split_fraction=0.5,
        download_dir=alpaca_path.parent,
        data_file_name=alpaca_path.name,
        num_workers=0,
    )
    alpaca.connect(mock_tockenizer, batch_size=2, max_seq_length=10)
    alpaca.prepare_data()
    alpaca.setup()

    train_dataloader = alpaca.train_dataloader()
    val_dataloader = alpaca.val_dataloader()

    assert len(train_dataloader) == 6
    assert len(val_dataloader) == 6

    train_batch = next(iter(train_dataloader))
    val_batch = next(iter(val_dataloader))

    assert train_batch.keys() == val_batch.keys() == {"input_ids", "labels"}
    assert all(seq.shape == (2, 10) for seq in train_batch.values())
    assert all(seq.shape == (2, 10) for seq in val_batch.values())
