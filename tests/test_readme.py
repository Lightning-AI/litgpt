# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

from pathlib import Path
import pytest
import subprocess
import os

REPO_ID = Path("EleutherAI/pythia-14m")
CUSTOM_TEXTS_DIR = Path("custom_texts")


def run_command(command):
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    return result.stdout


@pytest.mark.dependency()
def test_download_model():
    command = ["litgpt", "download", "--repo_id", REPO_ID]
    output = run_command(command)
    assert "Saving converted checkpoint to checkpoints/EleutherAI/pythia-14m" in output
    assert os.path.exists(f"checkpoints"/REPO_ID)


@pytest.mark.dependency()
def test_download_books():
    CUSTOM_TEXTS_DIR.mkdir(parents=True, exist_ok=True)

    books = [
        ("https://www.gutenberg.org/cache/epub/24440/pg24440.txt", "book1.txt"),
        ("https://www.gutenberg.org/cache/epub/26393/pg26393.txt", "book2.txt")
    ]
    for url, filename in books:
        subprocess.run(["curl", url, "--output", str(CUSTOM_TEXTS_DIR / filename)], check=True)
        # Verify each book is downloaded
        assert (CUSTOM_TEXTS_DIR / filename).exists(), f"{filename} not downloaded"


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


@pytest.mark.dependency(depends=["test_download_model", "test_download_books"])
def test_pretrain_model():
    OUT_DIR = Path("out") / "custom_pretrained"
    pretrain_command = [
        "litgpt", "pretrain",
        "--model_name", "pythia-14m",
        "--tokenizer_dir", str("checkpoints"/REPO_ID),
        "--data", "TextFiles",
        "--data.train_data_path", str(CUSTOM_TEXTS_DIR),
        "--train.max_tokens", "100",
        "--out_dir", str(OUT_DIR)
    ]
    run_command(pretrain_command)

    assert (".." / OUT_DIR / "final").exists(), "Pretraining output directory was not created"
    assert (".." / OUT_DIR / "final" / "lit_model.pth").exists(), "Model file was not created"