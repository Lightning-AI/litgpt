import random
import string
import os

import torch

from litdata import optimize
from torch.utils._pytree import tree_map


class Tokenizer:
    bos_id = 0

    def encode(self, text, bos, eos):
        assert bos
        assert not eos
        return [self.bos_id] + [ord(c) for c in text]


def tokenize(data):
    for story in data:
        yield torch.tensor(story)


def fake_chunk(path, data):
    optimize(fn=tokenize, inputs=[data] * len(data), output_dir=str(path), num_workers=1, chunk_bytes="200MB")


def test_textfiles_datamodule(tmp_path):
    from litgpt.data.text_files import TextFiles

    data_dir = tmp_path / "textfiles"
    datamodule = TextFiles(train_data_path=data_dir)
    datamodule.connect(max_seq_length=2, tokenizer=Tokenizer())

    # simulate `datamodule.prepare_data`
    train_data_dir = data_dir / "train"
    train_data_dir.mkdir(parents=True)
    fake_chunk(train_data_dir, [[12], [0, 23, 15, 63, 0], [73, 5, 0, 1, 1999, 0, 13]])
    datamodule.setup()

    tr_dataloader = datamodule.train_dataloader()
    torch.manual_seed(123)

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
