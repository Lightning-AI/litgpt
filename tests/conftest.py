import sys
from pathlib import Path

import pytest

wd = Path(__file__).parent.parent.absolute()


@pytest.fixture(autouse=True)
def add_wd_to_path():
    # this adds support for running tests without the package installed
    sys.path.append(str(wd))
