import json
import random
import string
import os

import pytest
import torch

from litdata import optimize
from torch.utils._pytree import tree_map


def create_random_strings_file(file_path, num_strings=100, string_length=10):
    def generate_random_string(length):
        characters = string.ascii_uppercase + string.digits
        return "".join(random.choice(characters) for _ in range(length))

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, "w") as f:
        for _ in range(num_strings):
            random_string = generate_random_string(string_length)
            f.write(f"{random_string}\n")


def test_textfiles_datamodule(tmp_path):
    from litgpt.data.text_files import TextFiles

    data_dir = tmp_path / "textfiles"

    random.seed(123)
    create_random_strings_file(data_dir/"file_1.txt")
    create_random_strings_file(data_dir/"file_2.txt")
    create_random_strings_file(data_dir/"file_3.txt")

    datamodule = TextFiles(train_data_path=data_dir)
    datamodule.connect(max_seq_length=2)

    train_data_dir = data_dir / "train"
    train_data_dir.mkdir(parents=True)
    datamodule.setup()
    datamodule.prepare_data()

    tr_dataloader = datamodule.train_dataloader()
    torch.manual_seed(123)

    #import pdb; pdb.set_trace();

    #for batch in tr_dataloader:
    #    pass


    """
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
    """