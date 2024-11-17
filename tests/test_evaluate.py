# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import pytest
import subprocess
from contextlib import redirect_stdout
from dataclasses import asdict
from io import StringIO
from unittest import mock

import torch
import yaml

import litgpt.eval.evaluate as module
from litgpt import GPT, Config
from litgpt.scripts.download import download_from_hub


def test_evaluate_script(tmp_path):
    ours_config = Config.from_name("pythia-14m")
    download_from_hub(repo_id="EleutherAI/pythia-14m", tokenizer_only=True, checkpoint_dir=tmp_path)
    checkpoint_dir = tmp_path / "EleutherAI" / "pythia-14m"
    ours_model = GPT(ours_config)
    torch.save(ours_model.state_dict(), checkpoint_dir / "lit_model.pth")
    with open(checkpoint_dir / "model_config.yaml", "w", encoding="utf-8") as fp:
        yaml.dump(asdict(ours_config), fp)

    stdout = StringIO()
    with redirect_stdout(stdout), mock.patch("sys.argv", ["eval/evaluate.py"]):
        with pytest.raises(ValueError) as excinfo:
            module.convert_and_evaluate(
                checkpoint_dir,
                out_dir=tmp_path / "out_dir",
                device=None,
                dtype=torch.float32,
                limit=5,
                tasks="logiqa",
                batch_size=0  # Test for non-positive integer
            )
        assert "batch_size must be a positive integer, 'auto', or in the format 'auto:N'." in str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            module.convert_and_evaluate(
                checkpoint_dir,
                out_dir=tmp_path / "out_dir",
                device=None,
                dtype=torch.float32,
                limit=5,
                tasks="logiqa",
                batch_size="invalid"  # Test for invalid string
            )
        assert "batch_size must be a positive integer, 'auto', or in the format 'auto:N'." in str(excinfo.value)

    stdout = StringIO()
    with redirect_stdout(stdout), mock.patch("sys.argv", ["eval/evaluate.py"]):
        module.convert_and_evaluate(
            checkpoint_dir,
            out_dir=tmp_path / "out_dir",
            device=None,
            dtype=torch.float32,
            limit=5,
            tasks="logiqa",
            batch_size=1  # Valid case
        )
    stdout = stdout.getvalue()
    assert (tmp_path / "out_dir" / "results.json").is_file()
    assert "logiqa" in stdout
    assert "Metric" in stdout
    assert "Loading checkpoint shards" not in stdout


def test_cli():
    args = ["litgpt", "evaluate", "-h"]
    output = subprocess.check_output(args)
    output = str(output.decode())
    assert "Evaluate a model with the LM Evaluation Harness" in output
