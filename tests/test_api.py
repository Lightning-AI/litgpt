# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

from collections import OrderedDict
import os
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



@pytest.fixture
def mock_llm():
    llm = MagicMock(spec=LLM)
    llm.model = MagicMock()
    llm.preprocessor = MagicMock()
    llm.prompt_style = MagicMock()
    llm.checkpoint_dir = MagicMock()
    llm.fabric = MagicMock()
    return llm


def test_load_model(mock_llm):
    assert isinstance(mock_llm, LLM)
    assert mock_llm.model is not None
    assert mock_llm.preprocessor is not None
    assert mock_llm.prompt_style is not None
    assert mock_llm.checkpoint_dir is not None
    assert mock_llm.fabric is not None


def test_generate(mock_llm):
    prompt = "What do Llamas eat?"
    mock_llm.generate.return_value = prompt + " Mock output"
    output = mock_llm.generate(prompt, max_new_tokens=10, temperature=0.8, top_k=5)
    assert isinstance(output, str)
    assert len(output) > len(prompt)


def test_stream_generate(mock_llm):
    prompt = "What do Llamas eat?"

    def iterator():
        outputs = (prompt + " Mock output").split()
        for output in outputs:
            yield output

    mock_llm.generate.return_value = iterator()
    output = mock_llm.generate(prompt, max_new_tokens=10, temperature=0.8, top_k=5, stream=True)
    result = "".join([out for out in output])
    assert len(result) > len(prompt)
