import shutil
from pathlib import Path

import pytest


@pytest.fixture()
def alpaca_path(tmp_path):
    file = Path(__file__).parent / "fixtures" / "alpaca.json"
    shutil.copyfile(file, tmp_path / "alpaca.json")
    return tmp_path / "alpaca.json"


@pytest.fixture()
def dolly_path(tmp_path):
    file = Path(__file__).parent / "fixtures" / "dolly.json"
    shutil.copyfile(file, tmp_path / "dolly.json")
    return tmp_path / "dolly.json"


@pytest.fixture()
def longform_path(tmp_path):
    path = tmp_path / "longform"
    path.mkdir()
    for split in ("train", "val"):
        file = Path(__file__).parent / "fixtures" / f"longform_{split}.json"
        shutil.copyfile(file, path / f"{split}.json")
    return path
