# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

from collections import OrderedDict
import os
from pathlib import Path
import sys

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


skip_in_ci_on_macos = pytest.mark.skipif(
     sys.platform == "darwin" and os.getenv("GITHUB_ACTIONS") == "true",
     reason="Skipped on macOS in CI environment because CI machine may not have enough memory to run this test."
)

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


def test_generate_token_ids(mock_llm):
    prompt = "What do Llamas eat?"
    mock_output_ids = MagicMock(spec=torch.Tensor)
    mock_output_ids.shape = [len(prompt) + 10]
    mock_llm.generate.return_value = mock_output_ids
    output_ids = mock_llm.generate(prompt, max_new_tokens=10, return_as_token_ids=True)
    assert isinstance(output_ids, torch.Tensor)
    assert output_ids.shape[0] > len(prompt)


def test_calculate_number_of_devices():
    assert calculate_number_of_devices(1) == 1
    assert calculate_number_of_devices([0, 1, 2]) == 3
    assert calculate_number_of_devices(None) == 0


# This test causes segfaults on the macOS CI machine but works fine locally
@skip_in_ci_on_macos
def test_llm_load_random_init(tmp_path):
    download_from_hub(repo_id="EleutherAI/pythia-14m", tokenizer_only=True, checkpoint_dir=tmp_path)

    torch.manual_seed(123)
    llm = LLM.load(
        model="pythia-160m",
        init="random",
        tokenizer_dir=Path(tmp_path/"EleutherAI/pythia-14m")
    )

    input_text = "some text text"
    output_text = llm.generate(input_text, max_new_tokens=15)
    ln = len(llm.preprocessor.tokenizer.encode(output_text)) - len(llm.preprocessor.tokenizer.encode(input_text))
    assert ln <= 15

    # The following below tests that generate works with different prompt lengths
    # after the kv cache was set

    input_text = "some text"
    output_text = llm.generate(input_text, max_new_tokens=15)
    ln = len(llm.preprocessor.tokenizer.encode(output_text)) - len(llm.preprocessor.tokenizer.encode(input_text))
    assert ln <= 15

    input_text = "some text text text"
    output_text = llm.generate(input_text, max_new_tokens=15)
    ln = len(llm.preprocessor.tokenizer.encode(output_text)) - len(llm.preprocessor.tokenizer.encode(input_text))
    assert ln <= 15


# This test causes segfaults on the macOS CI machine but works fine locally
@skip_in_ci_on_macos
def test_llm_load_hub_init(tmp_path):
    torch.manual_seed(123)
    llm = LLM.load(
        model="EleutherAI/pythia-14m",
        init="pretrained"
    )

    text_1 = llm.generate("text", max_new_tokens=10, top_k=1)
    assert len(text_1) > 0

    text_2 = llm.generate("text", max_new_tokens=10, top_k=1, stream=True)
    text_2 = "".join(list(text_2))
    assert text_1 == text_2, (text1, text_2)


def test_model_not_initialized(tmp_path):
    llm = LLM.load(
        model="EleutherAI/pythia-14m",
        init="pretrained",
        distribute=None
    )
    s = (
        "The model is not initialized yet; use the .distribute() "
        "or .trainer_setup() method to initialize the model."
    )
    with pytest.raises(AttributeError, match=re.escape(s)):
        llm.generate("text")

    llm = LLM.load(
        model="EleutherAI/pythia-14m",
        tokenizer_dir="EleutherAI/pythia-14m",
        init="random",
        distribute=None
    )
    s = (
        "The model is not initialized yet; use the .distribute() "
        "or .trainer_setup() method to initialize the model."
    )
    with pytest.raises(AttributeError, match=re.escape(s)):
        llm.generate("text")


@RunIf(min_cuda_gpus=2)
def test_more_than_1_device_for_sequential_gpu(tmp_path):

    device_count = CUDAAccelerator.auto_device_count()

    if device_count <= 2:
        model_name = "EleutherAI/pythia-14m"
    else:
        model_name = "EleutherAI/pythia-160m"
    llm = LLM.load(
        model=model_name,
    )

    with pytest.raises(NotImplementedError, match=f"Support for multiple devices is currently only implemented for generate_strategy='sequential'|'tensor_parallel'."):
        llm.distribute(devices=2)

    llm.distribute(devices=2, generate_strategy="sequential")
    assert isinstance(llm.generate("What do llamas eat?"), str)
    assert str(llm.model.transformer.h[0].mlp.fc.weight.device) == "cuda:0"
    last_layer_idx = len(llm.model.transformer.h) - 1
    assert str(llm.model.transformer.h[last_layer_idx].mlp.fc.weight.device) == f"cuda:1"

    # Also check with default (devices="auto") setting
    llm.distribute(generate_strategy="sequential")
    assert isinstance(llm.generate("What do llamas eat?"), str)
    assert str(llm.model.transformer.h[0].mlp.fc.weight.device) == "cuda:0"
    assert str(llm.model.transformer.h[last_layer_idx].mlp.fc.weight.device) == f"cuda:{device_count-1}"


@RunIf(min_cuda_gpus=2)
def test_more_than_1_device_for_tensor_parallel_gpu(tmp_path):
    llm = LLM.load(
        model="EleutherAI/pythia-14m",
    )

    if os.getenv("CI") != "true":
        # this crashes the CI, maybe because of process forking; works fine locally though
        llm.distribute(devices=2, generate_strategy="tensor_parallel")
        assert isinstance(llm.generate("What do llamas eat?"), str)


@RunIf(min_cuda_gpus=1)
def test_sequential_tp_incompatibility_with_random_weights(tmp_path):
    llm = LLM.load(
        model="EleutherAI/pythia-14m",
        tokenizer_dir="EleutherAI/pythia-14m",
        init="random"
    )
    for strategy in ("sequential", "tensor_parallel"):
        with pytest.raises(NotImplementedError, match=re.escape("The LLM was initialized with init='random' but .distribute() currently only supports pretrained weights.")):
            llm.distribute(devices=1, generate_strategy=strategy)
