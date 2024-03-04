import importlib
import importlib.util
from pathlib import Path
from unittest import mock
from unittest.mock import Mock

import pytest


@pytest.mark.parametrize(["script_file", "config_file"], [
    ("lit_gpt/pretrain.py", "pretrain/debug.yaml"),
    ("lit_gpt/pretrain.py", "pretrain/tinyllama.yaml"),
])
def test_config_help(script_file, config_file, monkeypatch, tmp_path):
    """Test that configs validate against the signature in the scripts."""
    from lit_gpt.utils import CLI

    script_file = Path(__file__).parent.parent / script_file
    config_file = Path(__file__).parent.parent / "config_hub" / config_file

    assert script_file.is_file()
    assert config_file.is_file()

    spec = importlib.util.spec_from_file_location(str(script_file.parent.name), script_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    module.main = Mock()

    with mock.patch("sys.argv", [script_file.name, "--config", str(config_file)]):
        CLI(module.setup)

    module.main.assert_called_once()
