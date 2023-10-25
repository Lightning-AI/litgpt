import os
import sys
from pathlib import Path

import pytest
import torch

wd = Path(__file__).parent.parent.absolute()


@pytest.fixture(autouse=True)
def add_wd_to_path():
    # this adds support for running tests without the package installed
    sys.path.append(str(wd))


@pytest.fixture()
def fake_checkpoint_dir(tmp_path):
    os.chdir(tmp_path)
    checkpoint_dir = tmp_path / "checkpoints" / "tmp"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "lit_model.pth").touch()
    (checkpoint_dir / "lit_config.json").touch()
    (checkpoint_dir / "tokenizer.json").touch()
    (checkpoint_dir / "tokenizer_config.json").touch()
    return checkpoint_dir


class TensorLike:
    def __eq__(self, other):
        return isinstance(other, torch.Tensor)


@pytest.fixture()
def tensor_like():
    return TensorLike()


@pytest.fixture(autouse=True)
def restore_default_dtype():
    # just in case
    torch.set_default_dtype(torch.float32)
