import json

import pytest
import torch
from litdata import optimize
from litdata.streaming import StreamingDataset, TokensLoader
from torch.utils._pytree import tree_map


def fake_chunk(path, data):
    def fn(_):
        for story in data:
            yield torch.tensor(story)

    optimize(fn=fn, inputs=[None] * len(data), output_dir=str(path), num_workers=1, chunk_bytes="200MB")


@pytest.mark.parametrize(
    ("max_seq_len", "expected"),
    [
        (2, [[0, 23, 15], [63, 0, 73], [5, 0, 1], [1999, 0, 13]]),
        (5, [[0, 23, 15, 63, 0, 73], [5, 0, 1, 1999, 0, 13]]),
        (6, [[0, 23, 15, 63, 0, 73, 5]]),
        (7, [[0, 23, 15, 63, 0, 73, 5, 0]]),
    ],
)
def test_pretok_dataset(tmp_path, max_seq_len, expected):
    fake_data = [0, 23, 15, 63, 0, 73, 5, 0, 1, 1999, 0, 13]
    assert len(fake_data) == 12
    fake_chunk(tmp_path, [fake_data])

    dataset = StreamingDataset(
        input_dir=str(tmp_path), item_loader=TokensLoader(block_size=max_seq_len + 1), shuffle=False, drop_last=False
    )
    actual = tree_map(torch.Tensor.tolist, list(dataset))
    assert actual == expected


def test_tokenize(tmp_path, monkeypatch):
    from litgpt.data.tinystories import tokenize

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

    monkeypatch.setenv("DATA_OPTIMIZER_GLOBAL_RANK", "0")
    monkeypatch.setenv("DATA_OPTIMIZER_NUM_WORKERS", "1")
    data = tokenize(str(shard_path), Tokenizer())
    assert list(data) == [[0, 102, 111, 111, 32, 98, 97, 114], [0, 102, 117, 110]]


def test_tinystories_datamodule(tmp_path):
    from litgpt.data.tinystories import TinyStories

    data_dir = tmp_path / "tinystories"

    datamodule = TinyStories(data_dir, seed=42)
    datamodule.connect(max_seq_length=2)

    # simulate `datamodule.prepare_data`
    train_data_dir = data_dir / "train"
    train_data_dir.mkdir(parents=True)
    fake_chunk(train_data_dir, [[12], [0, 23, 15, 63, 0], [73, 5, 0, 1, 1999, 0, 13]])

    datamodule.setup()

    tr_dataloader = datamodule.train_dataloader()
    torch.manual_seed(0)
    actual = tree_map(torch.Tensor.tolist, list(tr_dataloader))
    # there is 1 sample per index in the data (13)
    assert actual == [
        [[1999, 0, 13]],
        [[0, 13, 12]],
        [[1, 1999, 0]],
        [[63, 0, 73]],
        [[5, 0, 1]],
        [[0, 73, 5]],
        [[0, 23, 15]],
        [[0, 1, 1999]],
        [[15, 63, 0]],
        [[73, 5, 0]],
        [[12, 0, 23]],
        [[23, 15, 63]],
        [[13, 12, 0]],
    ]
