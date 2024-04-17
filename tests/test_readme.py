# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

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


@pytest.mark.dependency(depends=["test_download_model"])
def test_finetune_model():

    OUT_DIR = Path("out")/"lora"
    DATASET_PATH = Path("custom_finetuning_dataset.json")
    CHECKPOINT_DIR = f"checkpoints"/REPO_ID

    download_command = ["curl", "-L", "https://huggingface.co/datasets/medalpaca/medical_meadow_health_advice/raw/main/medical_meadow_health_advice.json", "-o", str(DATASET_PATH)]
    subprocess.run(download_command, check=True)

    assert DATASET_PATH.exists(), "Dataset file not downloaded"

    finetune_command = [
        "litgpt", "finetune", "lora",
        "--checkpoint_dir", str(CHECKPOINT_DIR),
        "--data", "JSON",
        "--data.json_path", str(DATASET_PATH),
        "--data.val_split_fraction", "0.1",
        "--train.max_steps", "1",
        "--out_dir", str(OUT_DIR)
    ]
    run_command(finetune_command)

    assert (OUT_DIR/"final").exists(), "Finetuning output directory was not created"
    assert (OUT_DIR/"final"/"lit_model.pth").exists(), "Model file was not created"
