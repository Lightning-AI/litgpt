# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import json
import sys
from pathlib import Path

import pytest

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

import lit_gpt.config as config_module


def test_config():
    from lit_gpt import Config

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


def test_legacy_args(tmp_path):
    from lit_gpt import Config

    config = Config.from_name("pythia-14m", condense_ratio=2)
    assert not hasattr(config, "condense_ratio")
    assert config.rope_condense_ratio == 2

    json_path = tmp_path / "config.json"
    with open(json_path, "w") as fp:
        json.dump({"condense_ratio": 3}, fp)

    config = Config.from_json(json_path)
    assert not hasattr(config, "condense_ratio")
    assert config.rope_condense_ratio == 3
    config = Config.from_json(json_path, condense_ratio=2)
    assert not hasattr(config, "condense_ratio")
    assert config.rope_condense_ratio == 2


def test_from_hf_name():
    from lit_gpt import Config

    # by short-hand name
    config0 = Config.from_name("tiny-llama-1.1b")
    # or by huggingface hub repo name
    config1 = Config.from_name("TinyLlama-1.1B-intermediate-step-1431k-3T")
    assert config0 == config1


def test_hf_config_from_json(tmp_path):
    """Test for backward compatibility with older configs that didn't have the `hf_config` field."""
    from lit_gpt import Config

    legacy_config = {"name": "falcon-40b", "org": "tiiuae"}
    with open(tmp_path / "config.json", "w") as file:
        json.dump(legacy_config, file)
    new_config = Config.from_json(tmp_path / "config.json")
    assert new_config.name == "falcon-40b"
    assert not hasattr(new_config, "org")
    assert new_config.hf_config["org"] == "tiiuae"
    assert new_config.hf_config["name"] == "falcon-40b"

    new_config = Config.from_json(tmp_path / "config.json", org="new-org")
    assert new_config.hf_config["org"] == "new-org"


@pytest.mark.parametrize("config", config_module.configs, ids=[c["name"] for c in config_module.configs])
def test_short_and_hf_names_are_equal_unless_on_purpose(config):
    from lit_gpt import Config

    # by short-hand name
    config0 = Config.from_name(config["name"])
    # or by huggingface hub repo name
    config1 = Config.from_name(config["hf_config"]["name"])
    assert config0.name == config1.name


def test_nonexisting_name():
    from lit_gpt import Config

    with pytest.raises(ValueError, match="not a supported"):
        Config.from_name("foobar")


def test_from_checkpoint(tmp_path):
    from lit_gpt import Config

    # 1. Neither `lit_config.py` nor matching config exists.
    with pytest.raises(FileNotFoundError, match="neither 'lit_config.json' nor matching config exists"):
        Config.from_checkpoint(tmp_path / "non_existing_checkpoint")

    # 2. If `lit_config.py` doesn't exists, but there is a matching config in `lit_gpt/config.py`.
    config = Config.from_checkpoint(tmp_path / "pythia-14m")
    assert config.name == "pythia-14m"
    assert config.block_size == 512
    assert config.n_layer == 6

    # 3. If only `lit_config.py` exists.
    config_data = {"name": "pythia-14m", "block_size": 24, "n_layer": 2}
    with open(tmp_path / "lit_config.json", "w") as file:
        json.dump(config_data, file)
    config = Config.from_checkpoint(tmp_path)
    assert config.name == "pythia-14m"
    assert config.block_size == 24
    assert config.n_layer == 2

    # 4. Both `lit_config.py` and a matching config exist, but `lit_config.py` supersedes matching config
    (tmp_path / "pythia-14m").mkdir()
    with open(tmp_path / "pythia-14m/lit_config.json", "w") as file:
        json.dump(config_data, file)
    config = Config.from_checkpoint(tmp_path / "pythia-14m")
    assert config.name == "pythia-14m"
    assert config.block_size == 24
    assert config.n_layer == 2
