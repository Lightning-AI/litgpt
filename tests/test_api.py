# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

from collections import OrderedDict
import os
import sys
from pathlib import Path

import pytest
import re
import torch
from unittest.mock import MagicMock
from tests.conftest import RunIf

from lightning.fabric.accelerators import CUDAAccelerator

from litgpt.api import (
    LLM,
    calculate_number_of_devices,
    benchmark_dict_to_markdown_table
)

from litgpt.scripts.download import download_from_hub


def test_template():
    llm = LLM("EleutherAI/pythia-14m")
