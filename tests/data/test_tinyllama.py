# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import pytest
from litdata.streaming import CombinedStreamingDataset, StreamingDataLoader, StreamingDataset
from torch.utils.data import DataLoader

from litgpt.data import TinyLlama


def test_tinyllama(tmp_path):
    data = TinyLlama(data_path=(tmp_path / "data"))
    assert data.seq_length == 2048
    assert data.batch_size == 1

    data.connect(batch_size=2, max_seq_length=1024)
    assert data.seq_length == 1025
    assert data.batch_size == 2

    with pytest.raises(FileNotFoundError, match="The directory .*data/slimpajama/train does not exist"):
        data.prepare_data()

    (tmp_path / "data" / "slimpajama" / "train").mkdir(parents=True)
    (tmp_path / "data" / "slimpajama" / "val").mkdir(parents=True)
    (tmp_path / "data" / "starcoder").mkdir(parents=True)

    data.prepare_data()
    data.setup()

    train_dataloader = data.train_dataloader()
    assert isinstance(train_dataloader, StreamingDataLoader)
    assert isinstance(train_dataloader.dataset, CombinedStreamingDataset)

    val_dataloader = data.val_dataloader()
    assert isinstance(val_dataloader, DataLoader)
    assert isinstance(val_dataloader.dataset, StreamingDataset)
