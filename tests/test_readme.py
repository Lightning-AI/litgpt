from pathlib import Path
import pytest
import subprocess
import os

REPO_ID = Path("EleutherAI/pythia-14m")


def run_command(command):
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    return result.stdout


@pytest.mark.dependency()
def test_download_model():
    command = ["litgpt", "download", "--repo_id", REPO_ID]
    output = run_command(command)
    assert "Saving converted checkpoint to checkpoints/EleutherAI/pythia-14m" in output
    assert os.path.exists(f"checkpoints"/REPO_ID)


@pytest.mark.dependency(depends=["test_download_model"])
def test_chat_with_model():
    command = ["litgpt", "generate", "base", "--checkpoint_dir", f"checkpoints"/REPO_ID]
    prompt = "What do Llamas eat?"
    result = subprocess.run(command, input=prompt, text=True, capture_output=True, check=True)
    assert "What food do llamas eat?" in result.stdout
