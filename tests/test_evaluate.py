# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import sys
import os
from dataclasses import asdict
from pathlib import Path
from unittest import mock
import litgpt.eval.evaluate as module
from contextlib import redirect_stdout
from io import StringIO
import subprocess

import datasets
import pytest
import yaml
import torch


from litgpt import GPT, Config

from litgpt.scripts.download import download_from_hub
from litgpt.eval.evaluate import safe_safetensors, prepare_results
from litgpt.scripts.convert_lit_checkpoint import convert_lit_checkpoint
from lm_eval import evaluator

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))


@pytest.mark.xfail(
    raises=(datasets.builder.DatasetGenerationError, NotImplementedError),
    strict=False,
    match="Loading a dataset cached in a LocalFileSystem is not supported",
)
@mock.patch.dict(os.environ, {"LT_ACCELERATOR": "cpu"})
def test_evaluate_script(tmp_path, monkeypatch):
    ours_config = Config.from_name("pythia-14m")
    download_from_hub(repo_id="EleutherAI/pythia-14m", tokenizer_only=True, checkpoint_dir=tmp_path)
    ours_model = GPT(ours_config)
    checkpoint_path = tmp_path / "lit_model.pth"
    config_path = tmp_path / "model_config.yaml"
    torch.save(ours_model.state_dict(), checkpoint_path)
    with open(config_path, "w") as fp:
        yaml.dump(asdict(ours_config), fp)
    torch.save({"model": ours_model.state_dict()}, tmp_path)

    fn_kwargs = dict(
        checkpoint_dir=tmp_path,
        out_dir=tmp_path / "out_dir",
        device="cpu",
        tasks="hellaswag"
    )
    stdout = StringIO()
    with redirect_stdout(stdout), mock.patch("sys.argv", ["eval" / "evaluate.py"]):
        module.convert_and_evaluate(**fn_kwargs)

"""
@pytest.mark.xfail(
    raises=(datasets.builder.DatasetGenerationError, NotImplementedError),
    strict=False,
    match="Loading a dataset cached in a LocalFileSystem is not supported",
)
@mock.patch.dict(os.environ, {"LT_ACCELERATOR": "cpu"})
def test_evaluate_script(tmp_path, fake_checkpoint_dir, monkeypatch):
    ours_config = Config.from_name("pythia-14m")
    ours_model = GPT(ours_config)
    checkpoint_path = fake_checkpoint_dir / "lit_model.pth"
    config_path = fake_checkpoint_dir / "model_config.yaml"
    torch.save(ours_model.state_dict(), checkpoint_path)
    with open(config_path, "w") as fp:
        yaml.dump(asdict(ours_config), fp)
    output_dir = fake_checkpoint_dir / "out_dir"

    fn_kwargs = dict(
        checkpoint_dir=fake_checkpoint_dir,
        out_dir=output_dir,
        device="cpu"
    )
    stdout = StringIO()
    with redirect_stdout(stdout), mock.patch("sys.argv", ["evaluate.py"]):
        module.convert_and_evaluate(**fn_kwargs)
"""

def test_cli(fake_checkpoint_dir):
    cli_path = Path(__file__).parent.parent / "litgpt" / "eval" / "evaluate.py"
    output = subprocess.check_output([sys.executable, cli_path, "-h"])
    output = str(output.decode())
    assert "evaluate" in output
