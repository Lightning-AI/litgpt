import sys
from pathlib import Path

import pytest

wd = Path(__file__).parent.parent.absolute()


@pytest.fixture()
def lit_parrot():
    # this adds support for running tests without the package installed
    sys.path.append(str(wd))

    import lit_parrot

    return lit_parrot
