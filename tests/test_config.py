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

    config = Config.from_name("pythia-70m")
    assert config.block_size == 2048

    config = Config.from_name("pythia-70m", block_size=4096)
    assert config.block_size == 4096

    config = Config(hf_config={"name": "pythia-70m"})
    assert config.name == "pythia-70m"


def test_legacy_args(tmp_path):
    from lit_gpt import Config

    config = Config.from_name("pythia-70m", condense_ratio=2)
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


@pytest.mark.parametrize("config", config_module.configs, ids=[c["name"] for c in config_module.configs])
def test_from_hf_name(config):
    from lit_gpt import Config

    # by short-hand name
    config0 = Config.from_name(config["name"])
    # or by huggingface hub repo name
    config1 = Config.from_name(config["hf_config"]["name"])
    assert config0 == config1


def test_hf_config_from_json(tmp_path):
    """Test for backward compatibility with older configs that didn't have the `hf_config` field."""
    from lit_gpt import Config

    legacy_config = {"name": "falcon-40b", "org": "tiiuae"}
    with open(tmp_path / "config.json", "w") as file:
        json.dump(legacy_config, file)
    new_config = Config.from_json(tmp_path / "config.json")
    assert new_config.name == "falcon-40b"
    assert "org" not in new_config
    assert new_config.hf_config["org"] == "tiiuae"
    assert new_config.hf_config["name"] == "falcon-40b"

    new_config = Config.from_json(tmp_path / "config.json", org="new-org")
    assert new_config.hf_config["org"] == "new-org"
