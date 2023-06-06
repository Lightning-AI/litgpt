import os
import sys
from pathlib import Path

import pytest

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
