import sys
from pathlib import Path

import pytest

wd = Path(__file__).parent.parent.absolute()


@pytest.fixture()
def orig_llama():
    sys.path.append(str(wd))

    from scripts.download import download_original

    download_original(wd)

    import original_model

    return original_model


@pytest.fixture()
def orig_llama_adapter():
    sys.path.append(str(wd))

    from scripts.download import download_original

    download_original(wd)

    import original_adapter

    return original_adapter


@pytest.fixture()
def lit_llama():
    # this adds support for running tests without the package installed
    sys.path.append(str(wd))

    import lit_llama

    return lit_llama
