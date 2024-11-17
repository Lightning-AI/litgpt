import importlib
import importlib.util
from pathlib import Path
from unittest import mock
from unittest.mock import Mock

import pytest
from lightning.fabric.plugins import Precision

from litgpt import Config
from litgpt.utils import CLI

fixed_pairs = [
    ("litgpt/pretrain.py", "pretrain/debug.yaml"),
    ("litgpt/pretrain.py", "pretrain/tinyllama.yaml"),
    ("litgpt/pretrain.py", "pretrain/tinystories.yaml"),
    (
        "litgpt/pretrain.py",
        "https://raw.githubusercontent.com/Lightning-AI/litgpt/4d55ab6d0aa404f0da0d03a80a8801ed60e07e83/config_hub/pretrain/tinystories.yaml",  # TODO: Update with path from main after merge
    ),
]

config_hub_path = Path(__file__).parent.parent / "config_hub" / "finetune"
model_pairs = []

for model_dir in config_hub_path.iterdir():
    if model_dir.is_dir():
        model_name = model_dir.name
        for yaml_file in model_dir.glob("*.yaml"):
            config_name = yaml_file.stem
            python_file = "litgpt/finetune/full.py" if config_name == "full" else "litgpt/finetune/lora.py"
            relative_yaml_path = yaml_file.relative_to(config_hub_path.parent)
            model_pairs.append((python_file, str(relative_yaml_path)))

all_pairs = fixed_pairs + model_pairs


@pytest.mark.parametrize(("script_file", "config_file"), all_pairs)
def test_config_help(script_file, config_file, monkeypatch):
    """Test that configs validate against the signature in the scripts."""
    script_file = Path(__file__).parent.parent / script_file
    assert script_file.is_file()
    if "http" not in str(config_file):
        config_file = Path(__file__).parent.parent / "config_hub" / config_file
        assert config_file.is_file()

    spec = importlib.util.spec_from_file_location(str(script_file.parent.name), script_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    monkeypatch.setattr(module, "main", Mock())
    monkeypatch.setattr(module, "Tokenizer", Mock())
    monkeypatch.setattr(module, "BitsandbytesPrecision", Mock(return_value=Precision()), raising=False)
    monkeypatch.setattr(module, "Config", Mock(return_value=Config.from_name("pythia-14m")))
    monkeypatch.setattr(module, "check_valid_checkpoint_dir", Mock(), raising=False)

    try:
        with mock.patch("sys.argv", [script_file.name, "--config", str(config_file), "--devices", "1"]):
            CLI(module.setup)
            module.main.assert_called_once()
    except FileNotFoundError:
        pass
        # FileNotFound occurs here because we have not downloaded the model weights referenced in the config files
        # which is ok because here we just want to validate the config file itself.
