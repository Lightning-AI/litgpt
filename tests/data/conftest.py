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
