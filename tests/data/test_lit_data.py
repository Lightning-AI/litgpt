# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
import sys
from unittest import mock
from unittest.mock import ANY

import pytest

from litgpt.data import LitData


@pytest.mark.skipif(sys.platform == "win32", reason="Needs to implement platform agnostic path/url joining")
@mock.patch("litgpt.data.lit_data.LitData._dataloader")
def test_input_dir_and_splits(dl_mock, tmp_path):
    with pytest.raises(ValueError, match="If provided `split_names` must be a tuple of two strings"):
        LitData(data_path=tmp_path, split_names=("train",))

    # local dir, no splits
    data = LitData(data_path=tmp_path)
    data.train_dataloader()
    dl_mock.assert_called_with(input_dir=str(tmp_path), train=True)
    data.val_dataloader()
    dl_mock.assert_called_with(input_dir=str(tmp_path), train=False)

    # local dir, splits
    data = LitData(data_path=tmp_path, split_names=("train", "val"))
    data.train_dataloader()
    dl_mock.assert_called_with(input_dir=str(tmp_path / "train"), train=True)
    data.val_dataloader()
    dl_mock.assert_called_with(input_dir=str(tmp_path / "val"), train=False)

    # remote dir, splits
    data = LitData(data_path="s3://mydataset/data", split_names=("train", "val"))
    data.train_dataloader()
    dl_mock.assert_called_with(input_dir="s3://mydataset/data/train", train=True)
    data.val_dataloader()
    dl_mock.assert_called_with(input_dir="s3://mydataset/data/val", train=False)


@pytest.mark.skipif(sys.platform == "win32", reason="Needs to implement platform agnostic path/url joining")
@mock.patch("litdata.streaming.StreamingDataset")
@mock.patch("litdata.streaming.StreamingDataLoader")
def test_dataset_args(streaming_dataloader_mock, streaming_dataset_mock, tmp_path):
    data = LitData(data_path=tmp_path, seed=1000)
    data.train_dataloader()
    streaming_dataset_mock.assert_called_with(
        input_dir=str(tmp_path),
        item_loader=ANY,
        shuffle=True,
        seed=1000,
    )
    streaming_dataloader_mock.assert_called_with(
        streaming_dataset_mock(),
        batch_size=1,
        pin_memory=True,
        num_workers=8,
        drop_last=True,
    )
