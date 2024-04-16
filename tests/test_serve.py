# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
from dataclasses import asdict
import shutil

from lightning.fabric import seed_everything
from fastapi.testclient import TestClient
from litserve.server import LitServer
import torch
import yaml


from litgpt import GPT, Config
from litgpt.deploy.serve import SimpleLitAPI
from litgpt.scripts.download import download_from_hub


def test_simple(tmp_path):

    # Create model checkpoint
    seed_everything(123)
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

    server = LitServer(SimpleLitAPI(checkpoint_dir=tmp_path), accelerator="cpu", devices=1, timeout=60)

    with TestClient(server.app) as client:
        response = client.post("/predict", json={"prompt": "Hello, World"})
        assert response.json()["output"][:25] == "Hello, World gcc exchange", response.json()["output"][:25]


if __name__ == """__main__""":
    from pathlib import Path
    test_simple(Path("tmp_path"))
