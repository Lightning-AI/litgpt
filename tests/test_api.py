# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import pytest
from unittest.mock import Mock, patch
import torch

import lightning as L
from litgpt.model import GPT
from litgpt.tokenizer import Tokenizer
from litgpt.prompts import PromptStyle
from litgpt.api import LLM, Preprocessor, calculate_number_of_devices


@pytest.fixture
def gpt_model():
    return Mock(spec=GPT)


@pytest.fixture
def tokenizer():
    mock = Mock(spec=Tokenizer)
    mock.encode.return_value = torch.tensor([1, 2, 3], dtype=torch.int32)
    mock.decode.return_value = "decoded text"
    return mock


@pytest.fixture
def prompt_style():
    return Mock(spec=PromptStyle)


@pytest.fixture
def fabric():
    fabric_mock = Mock(spec=L.Fabric)
    fabric_mock.device = "cpu"
    return fabric_mock


@pytest.fixture
def llm_instance(gpt_model, tokenizer, prompt_style, fabric):
    return LLM(gpt_model, tokenizer, prompt_style, devices=1, fabric=fabric)


def test_calculate_number_of_devices():
    assert calculate_number_of_devices(5) == 5
    assert calculate_number_of_devices([0, 1, 2]) == 3
    assert calculate_number_of_devices([0]) == 1


class TestPreprocessor:
    def test_encode(self, tokenizer):
        preprocessor = Preprocessor(tokenizer)
        text = "Hello, world!"
        result = preprocessor.encode(text)
        tokenizer.encode.assert_called_once_with(text, device=preprocessor.device)
        assert torch.equal(result, torch.tensor([1, 2, 3], dtype=torch.int32))

    def test_decode(self, tokenizer):
        preprocessor = Preprocessor(tokenizer)
        tokens = torch.tensor([1, 2, 3], dtype=torch.int32)
        result = preprocessor.decode(tokens)
        tokenizer.decode.assert_called_once_with(tokens)
        assert result == "decoded text"


def mock_path_exists(path):
    if str(path) in ["some/path", "checkpoints/some/path"]:
        return True
    return False


def mock_path_is_file(path):
    if str(path) in ["some/path/tokenizer.model", "some/path/tokenizer.json", "checkpoints/some/path/tokenizer_config.json"]:
        return True
    return False


@patch("litgpt.config.Config.from_file")
@patch("lightning.Fabric", autospec=True)
@patch("litgpt.model.GPT", autospec=True)
@patch("litgpt.tokenizer.Tokenizer", autospec=True)
@patch("litgpt.utils.load_checkpoint")
@patch("torch.cuda.is_available", return_value=True)
@patch("pathlib.Path.exists", side_effect=mock_path_exists)
@patch("pathlib.Path.is_file", side_effect=mock_path_is_file)
def test_llm_load(is_file_mock, exists_mock, is_available_mock, load_checkpoint_mock, gpt_mock, tokenizer_mock, fabric_mock, config_mock, fabric, tokenizer, gpt_model):
    model_path = "some/path"
    device_type = "auto"
    devices = 1
    quantize = None
    precision = None

    config_mock.return_value = Mock()
    fabric_mock.return_value = fabric
    tokenizer_mock.return_value = tokenizer
    gpt_mock.return_value = gpt_model

    llm = LLM.load(model_path, device_type, devices, quantize, precision)

    assert isinstance(llm, LLM)
    assert llm.devices == 1
    assert llm.fabric.device == "cpu"

    with pytest.raises(ValueError):
        LLM.load(model_path, "invalid_device", devices)

    with pytest.raises(NotImplementedError):
        LLM.load(model_path, device_type, [0, 1])
