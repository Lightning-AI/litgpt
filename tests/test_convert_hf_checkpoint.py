from unittest import mock

import pytest


def test_convert_hf_checkpoint(tmp_path):
    from scripts.convert_hf_checkpoint import convert_hf_checkpoint

    with pytest.raises(ValueError, match="to contain .bin"):
        convert_hf_checkpoint(checkpoint_dir=tmp_path, model_name="pythia-70m")

    bin_file = tmp_path / "foo.bin"
    bin_file.touch()
    with mock.patch("scripts.convert_hf_checkpoint.lazy_load") as load:
        convert_hf_checkpoint(checkpoint_dir=tmp_path, model_name="pythia-70m")
    load.assert_called_with(bin_file)

    assert {p.name for p in tmp_path.glob("*")} == {"foo.bin", "lit_config.json", "lit_model.pth"}

    # ensure that the config dict can be loaded
    from lit_gpt import Config

    config = Config.from_json(tmp_path / "lit_config.json")
    assert isinstance(config, Config)
