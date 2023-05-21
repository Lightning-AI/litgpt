from unittest import mock

import pytest


def test_convert_hf_checkpoint(tmp_path):
    from scripts.convert_hf_checkpoint import convert_hf_checkpoint

    with pytest.raises(ValueError, match="to contain .bin"):
        convert_hf_checkpoint(checkpoint_dir=tmp_path, model_name="pythia-70m")

    bin_file = tmp_path / "foo.bin"
    bin_file.touch()
    with mock.patch("torch.load") as torch_load:
        convert_hf_checkpoint(checkpoint_dir=tmp_path, model_name="pythia-70m")
    torch_load.assert_called_with(bin_file, map_location="cpu")

    assert {p.name for p in tmp_path.glob("*")} == {"foo.bin", "lit_config.json", "lit_model.pth"}
