# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import shutil
import subprocess
import sys
from contextlib import redirect_stdout
from dataclasses import asdict
from io import StringIO
from pathlib import Path
from unittest import mock

import datasets
import pytest
import torch
import yaml

import litgpt.eval.evaluate as module
from litgpt import GPT, Config
from litgpt.scripts.download import download_from_hub


@pytest.mark.xfail(
    raises=(datasets.builder.DatasetGenerationError, NotImplementedError),
    strict=False,
    match="Loading a dataset cached in a LocalFileSystem is not supported",
)
def test_evaluate_script(tmp_path, monkeypatch):
    ours_config = Config.from_name("pythia-14m")
    download_from_hub(repo_id="EleutherAI/pythia-14m", tokenizer_only=True, checkpoint_dir=tmp_path)
    shutil.move(str(tmp_path / "EleutherAI" / "pythia-14m" / "tokenizer.json"), str(tmp_path))
    shutil.move(str(tmp_path / "EleutherAI" / "pythia-14m" / "tokenizer_config.json"), str(tmp_path))
    ours_model = GPT(ours_config)
    checkpoint_path = tmp_path / "lit_model.pth"
    torch.save(ours_model.state_dict(), checkpoint_path)
    config_path = tmp_path / "model_config.yaml"
    with open(config_path, "w", encoding="utf-8") as fp:
        yaml.dump(asdict(ours_config), fp)

    fn_kwargs = dict(
        checkpoint_dir=tmp_path,
        out_dir=tmp_path / "out_dir",
        device="cpu",
        dtype=torch.float32,
        limit=5,
        tasks="mathqa"
    )
    stdout = StringIO()
    with redirect_stdout(stdout), mock.patch("sys.argv", ["eval/evaluate.py"]):
        module.convert_and_evaluate(**fn_kwargs)
    stdout = stdout.getvalue()
    assert "mathqa" in stdout
    assert "Metric" in stdout


@pytest.mark.parametrize("mode", ["file", "entrypoint"])
def test_cli(mode):
    if mode == "file":
        cli_path = Path(__file__).parent.parent / "litgpt/eval/evaluate.py"
        args = [sys.executable, cli_path, "-h"]
    else:
        args = ["litgpt", "evaluate", "-h"]
    output = subprocess.check_output(args)
    output = str(output.decode())
    assert "run the LM Evaluation Harness" in output
