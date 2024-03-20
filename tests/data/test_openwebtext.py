# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
import sys
from unittest import mock
from unittest.mock import ANY, call

import pytest
from litdata.streaming import StreamingDataLoader, StreamingDataset
from torch.utils.data import DataLoader

from litgpt.data import OpenWebText


@pytest.mark.skipif(sys.platform == "win32", reason="Not in the mood to add Windows support right now.")
@mock.patch("litdata.optimize")
@mock.patch("datasets.load_dataset")
def test_openwebtext(_, optimize_mock, tmp_path, mock_tokenizer):
    data = OpenWebText(data_path=(tmp_path / "openwebtext"))
    assert data.seq_length == 2048
    assert data.batch_size == 1

    data.connect(tokenizer=mock_tokenizer, batch_size=2, max_seq_length=1024)
    assert data.seq_length == 1025
    assert data.batch_size == 2

    # Data does not exist, preprocess it
    data.prepare_data()
    optimize_mock.assert_has_calls(
        [
            call(
                fn=ANY,
                num_workers=ANY,
                inputs=[],
                output_dir=str(tmp_path / "openwebtext" / "train"),
                chunk_bytes="200MB",
            ),
            call(
                fn=ANY,
                num_workers=ANY,
                inputs=[],
                output_dir=str(tmp_path / "openwebtext" / "val"),
                chunk_bytes="200MB",
            ),
        ]
    )
    optimize_mock.reset_mock()

    # Data exists, already preprocessed
    (tmp_path / "openwebtext" / "train").mkdir(parents=True)
    (tmp_path / "openwebtext" / "val").mkdir(parents=True)
    data.prepare_data()
    optimize_mock.assert_not_called()

    data.setup()

    train_dataloader = data.train_dataloader()
    assert isinstance(train_dataloader, StreamingDataLoader)
    assert isinstance(train_dataloader.dataset, StreamingDataset)

    val_dataloader = data.val_dataloader()
    assert isinstance(val_dataloader, DataLoader)
    assert isinstance(val_dataloader.dataset, StreamingDataset)
