import json
from contextlib import redirect_stdout
from io import StringIO

import numpy as np
import pytest
import torch
from torch.utils._pytree import tree_map
from torch.utils.data import ConcatDataset

from litgpt.data.tinystories import PretokDataset, TinyStories, process_shard


def fake_bin(tmp_path, data, name):
    all_tokens = np.array(data, dtype=np.uint16)
    data_path = tmp_path / f"{name}.bin"
    with open(data_path, "wb") as f:
        f.write(all_tokens.tobytes())
    return data_path


@pytest.mark.parametrize(
    ("max_seq_len", "expected"),
    [
        (2, [[0, 23, 15], [15, 63, 0], [0, 73, 5], [5, 0, 1], [1, 1999, 0]]),
        (5, [[0, 23, 15, 63, 0, 73], [73, 5, 0, 1, 1999, 0]]),
        (6, [[0, 23, 15, 63, 0, 73, 5]]),
        (7, [[0, 23, 15, 63, 0, 73, 5, 0]]),
    ],
)
def test_pretok_dataset(tmp_path, max_seq_len, expected):
    fake_data = [0, 23, 15, 63, 0, 73, 5, 0, 1, 1999, 0, 13]
    assert len(fake_data) == 12
    bin_path = fake_bin(tmp_path, fake_data, "data")

    dataset = PretokDataset(str(bin_path), max_seq_len)
    actual = tree_map(torch.Tensor.tolist, list(dataset))
    assert actual == expected


def test_process_shard(tmp_path):
    story1, story2 = "foo bar", "    fun    "
    data = [{"story": story1}, {"story": story2}]
    shard_path = tmp_path / "data.json"
    with open(shard_path, "w") as f:
        json.dump(data, f)

    class Tokenizer:
        bos_id = 0

        def encode(self, text, bos, eos):
            assert bos
            assert not eos
            return [self.bos_id] + [ord(c) for c in text]

    out = StringIO()
    with redirect_stdout(out):
        process_shard((0, str(shard_path)), Tokenizer())

    text = out.getvalue()
    assert text.endswith("data.bin, tokens: 12, bos: 2, average seqlen: 6.00\n")
    assert shard_path.with_suffix(".bin").exists()


def test_tinystories_datamodule(tmp_path):
    datamodule = TinyStories(tmp_path, seed=42)
    datamodule.connect(max_seq_length=2)

    data_dir = tmp_path / "TinyStories_all_data"
    data_dir.mkdir()
    fake_bin(data_dir, [12], "0")
    fake_bin(data_dir, [0, 23, 15, 63, 0], "1")
    fake_bin(data_dir, [73, 5, 0, 1, 1999, 0, 13], "2")

    datamodule.setup()

    assert isinstance(datamodule.train_dataset, ConcatDataset)
    assert len(datamodule.train_dataset.datasets) == 2
    assert isinstance(datamodule.train_dataset.datasets[0], PretokDataset)
    # unordered because it shuffled
    assert datamodule.train_dataset.datasets[0].filepath == str(data_dir / "2.bin")
    assert datamodule.train_dataset.datasets[1].filepath == str(data_dir / "1.bin")

    assert isinstance(datamodule.val_dataset, PretokDataset)
    assert datamodule.val_dataset.filepath == str(data_dir / "0.bin")

    tr_dataloader = datamodule.train_dataloader()
    torch.manual_seed(0)
    actual = tree_map(torch.Tensor.tolist, list(tr_dataloader))
    assert actual == [[[0, 1, 1999]], [[15, 63, 0]], [[1999, 0, 13]], [[0, 23, 15]], [[73, 5, 0]]]
