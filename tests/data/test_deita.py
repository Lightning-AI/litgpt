# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

def test_deita(mock_tockenizer, deita_path):
    from lit_gpt.data import Deita

    deita = Deita(
        repo_id=deita_path.name + "/",
        num_workers=0,
    )
    deita.connect(mock_tockenizer, batch_size=2, max_seq_length=10)
    deita.prepare_data()
    deita.setup()

    train_dataloader = deita.train_dataloader()
    val_dataloader = deita.val_dataloader()

    assert len(train_dataloader) == 6
    assert len(val_dataloader) == 6

    train_batch = next(iter(train_dataloader))
    val_batch = next(iter(val_dataloader))

    assert train_batch.keys() == val_batch.keys() == {"input_ids", "labels"}
    assert all(seq.shape == (2, 10) for seq in train_batch.values())
    assert all(seq.shape == (2, 10) for seq in val_batch.values())
