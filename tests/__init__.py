# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import warnings

import pytest

warnings.filterwarnings("ignore", category=pytest.PytestWarning, message=r".*\(rm_rf\) error removing.*")
