import os
import sys

import pytest


@pytest.fixture()
def orig_llama():
    wd = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.append(wd)

    from scripts.download import download_original

    download_original(wd)

    import original_model

    return original_model
