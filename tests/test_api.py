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
from litgpt.api import LLM, calculate_number_of_devices
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


def test_llm_load_hub_init(tmp_path):
    torch.manual_seed(123)
    llm = LLM.load(
        model="EleutherAI/pythia-14m",
        init="pretrained"
    )

    text_1 = llm.generate("text", max_new_tokens=10, top_k=1)
    assert len(text_1) > 0

    text_2 = llm.generate("text", max_new_tokens=10, top_k=1, stream=True)
    assert text_1 == "".join(list(text_2))


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


def test_sequential_tp_cpu(tmp_path):
    llm = LLM.load(
        model="EleutherAI/pythia-14m",
    )
    for strategy in ("sequential", "tensor_parallel"):
        with pytest.raises(NotImplementedError, match=f"generate_strategy='{strategy}' is only supported for accelerator='cuda'|'gpu'."):
            llm.distribute(
                devices=1,
                accelerator="cpu",
                generate_strategy=strategy
            )


def test_initialization_for_trainer(tmp_path):
    llm = LLM.load(
        model="EleutherAI/pythia-14m",
        distribute=None
    )
    s = (
        "The model is not initialized yet; use the .distribute() "
        "or .trainer_setup() method to initialize the model."
    )
    with pytest.raises(AttributeError, match=re.escape(s)):
        llm.generate("hello world")

    llm.trainer_setup()
    llm.model.to(llm.preprocessor.device)
    assert isinstance(llm.generate("hello world"), str)


@RunIf(min_cuda_gpus=1)
def test_quantization_is_applied(tmp_path):
    llm = LLM.load(
        model="EleutherAI/pythia-14m",
    )
    llm.distribute(devices=1, quantize="bnb.nf4", precision="bf16-true")
    assert "NF4Linear" in str(type(llm.model.lm_head))


def test_fixed_kv_cache(tmp_path):
    llm = LLM.load(
        model="EleutherAI/pythia-14m",
    )
    llm.distribute(devices=1, fixed_kv_cache_size=100)

    # Request too many tokens
    with pytest.raises(NotImplementedError, match="max_seq_length 512 needs to be >= 9223372036854775809"):
        output_text = llm.generate("hello world", max_new_tokens=2**63)


def test_invalid_accelerator(tmp_path):
    llm = LLM.load(
        model="EleutherAI/pythia-14m",
    )
    with pytest.raises(ValueError, match="Invalid accelerator"):
        llm.distribute(accelerator="invalid")


def test_returned_benchmark_dir(tmp_path):
    llm = LLM.load(
        model="EleutherAI/pythia-14m",
    )

    text, bench_d = llm.benchmark(prompt="hello world")
    assert isinstance(bench_d["Inference speed in tokens/sec"], float)

    text, bench_d = llm.benchmark(prompt="hello world", stream=True)
    assert isinstance(bench_d["Inference speed in tokens/sec"], float)


def test_state_dict(tmp_path):
    llm = LLM.load(
        model="EleutherAI/pythia-14m",
    )
    assert isinstance(llm.state_dict(), OrderedDict)
    assert llm.state_dict()['lm_head.weight'].shape == torch.Size([50304, 128])


def test_save_method(tmp_path):
    llm = LLM.load(
        model="EleutherAI/pythia-14m",
    )

    target_dir = "saved_model"
    llm.save(target_dir)

    expected_files = [
        "config.json",
        "generation_config.json",
        "lit_model.pth",
        "model_config.yaml",
        "prompt_style.yaml",
        "tokenizer_config.json",
        "tokenizer.json"
    ]

    files_in_directory = os.listdir(target_dir)
    for file_name in expected_files:
        assert file_name in files_in_directory, f"{file_name} is missing from {target_dir}"


def test_forward_method(tmp_path):
    llm = LLM.load(
        model="EleutherAI/pythia-14m",
    )
    inputs = torch.ones(6, 128, dtype=torch.int64).to(next(llm.model.parameters()).device)

    assert llm(inputs).shape == torch.Size([6, 128, 50304])
    logits, loss = llm(inputs, target_ids=inputs)
    assert logits.shape == torch.Size([6, 128, 50304])
    assert isinstance(loss.item(), float)
