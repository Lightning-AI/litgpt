# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import pytest
import yaml

import litgpt.config as config_module
from litgpt import Config


def test_config():
    config = Config()
    assert config.name == ""
    assert config.block_size == 4096

    config = Config(block_size=2048)
    assert config.block_size == 2048

    config = Config.from_name("pythia-14m")
    assert config.block_size == 512

    config = Config.from_name("pythia-14m", block_size=4096)
    assert config.block_size == 4096

    config = Config(hf_config={"name": "pythia-14m"})
    assert config.name == "pythia-14m"


def test_from_hf_name():
    # by short-hand name
    config0 = Config.from_name("tiny-llama-1.1b")
    # or by huggingface hub repo name
    config1 = Config.from_name("TinyLlama-1.1B-intermediate-step-1431k-3T")
    assert config0 is not None
    assert config1 is not None
    assert config0 == config1


def test_nonexisting_name():
    with pytest.raises(ValueError, match="'invalid-model-name' is not a supported config name"):
        Config.from_name("invalid-model-name")


@pytest.mark.parametrize("config", config_module.configs, ids=[c["name"] for c in config_module.configs])
def test_short_and_hf_names_are_equal_unless_on_purpose(config):
    # by short-hand name
    config0 = Config.from_name(config["name"])
    # or by huggingface hub repo name
    config1 = Config.from_name(config["hf_config"]["name"])
    assert config0.name == config1.name


def test_from_hf_name_with_org_string():
    # Test case 1: valid input
    config0 = Config.from_name("tiny-llama-1.1b")
    config1 = Config.from_name("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")
    assert config0 is not None
    assert config1 is not None
    assert config0 == config1

    # Test case 2: invalid input - org not found
    with pytest.raises(ValueError, match="'UnknownOrg/TinyLlama-1.1B-intermediate-step-1431k-3T' is not a supported config name"):
        Config.from_name("UnknownOrg/TinyLlama-1.1B-intermediate-step-1431k-3T")

    # Test case 3: invalid input - name not found
    with pytest.raises(ValueError, match="'TinyLlama/TinyLlama-XYZ' is not a supported config name"):
        Config.from_name("TinyLlama/TinyLlama-XYZ")


def test_from_checkpoint(tmp_path):
    # 1. Neither `lit_config.py` nor matching config exists.
    with pytest.raises(FileNotFoundError, match="neither 'model_config.yaml' nor matching config exists"):
        Config.from_checkpoint(tmp_path / "non_existing_checkpoint")

    # 2. If `lit_config.py` doesn't exists, but there is a matching config in `litgpt/config.py`.
    config = Config.from_checkpoint(tmp_path / "pythia-14m")
    assert config.name == "pythia-14m"
    assert config.block_size == 512
    assert config.n_layer == 6

    # 3. If only `lit_config.py` exists.
    config_data = {"name": "pythia-14m", "block_size": 24, "n_layer": 2}
    with open(tmp_path / "model_config.yaml", "w", encoding="utf-8") as file:
        yaml.dump(config_data, file)
    config = Config.from_checkpoint(tmp_path)
    assert config.name == "pythia-14m"
    assert config.block_size == 24
    assert config.n_layer == 2

    # 4. Both `lit_config.py` and a matching config exist, but `lit_config.py` supersedes matching config
    (tmp_path / "pythia-14m").mkdir()
    with open(tmp_path / "pythia-14m/model_config.yaml", "w", encoding="utf-8") as file:
        yaml.dump(config_data, file)
    config = Config.from_checkpoint(tmp_path / "pythia-14m")
    assert config.name == "pythia-14m"
    assert config.block_size == 24
    assert config.n_layer == 2


@pytest.mark.parametrize("head_size", [None, 128])
def test_head_size(head_size):
    config = Config(head_size)

    assert config.head_size == head_size or config.n_embd // config.n_head
