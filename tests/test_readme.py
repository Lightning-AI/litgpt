# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

from pathlib import Path
import os
from unittest import mock

import pytest
import requests
import subprocess
import sys
import threading
import time


REPO_ID = Path("EleutherAI/pythia-14m")
CUSTOM_TEXTS_DIR = Path("custom_texts")


def run_command(command):
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        error_message = (
            f"Command '{' '.join(command)}' failed with exit status {e.returncode}\n"
            f"Output:\n{e.stdout}\n"
            f"Error:\n{e.stderr}"
        )
        # You can either print the message, log it, or raise an exception with it
        print(error_message)
        raise RuntimeError(error_message) from None


@pytest.mark.dependency()
def test_download_model():
    repo_id = str(REPO_ID).replace("\\", "/")  # fix for Windows CI
    command = ["litgpt", "download", str(repo_id)]
    output = run_command(command)

    s = Path("checkpoints") / repo_id
    assert f"Saving converted checkpoint to {str(s)}" in output
    assert ("checkpoints" / REPO_ID).exists()


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


@mock.patch.dict(os.environ, {"LT_ACCELERATOR": "cpu"})
@pytest.mark.dependency(depends=["test_download_model"])
def test_chat_with_model():
    command = ["litgpt", "generate", f"checkpoints" / REPO_ID]
    prompt = "What do Llamas eat?"
    result = subprocess.run(command, input=prompt, text=True, capture_output=True, check=True)
    assert "What food do llamas eat?" in result.stdout


@mock.patch.dict(os.environ, {"LT_ACCELERATOR": "cpu"})
@pytest.mark.dependency(depends=["test_download_model"])
@pytest.mark.timeout(300)
def test_finetune_model():

    OUT_DIR = Path("out") / "lora"
    DATASET_PATH = Path("custom_finetuning_dataset.json")
    CHECKPOINT_DIR = "checkpoints" / REPO_ID

    download_command = ["curl", "-L", "https://huggingface.co/datasets/medalpaca/medical_meadow_health_advice/raw/main/medical_meadow_health_advice.json", "-o", str(DATASET_PATH)]
    subprocess.run(download_command, check=True)

    assert DATASET_PATH.exists(), "Dataset file not downloaded"

    finetune_command = [
        "litgpt", "finetune_lora",
        str(CHECKPOINT_DIR),
        "--lora_r", "1",
        "--data", "JSON",
        "--data.json_path", str(DATASET_PATH),
        "--data.val_split_fraction", "0.00001",  # Keep small because new final validation is expensive
        "--train.max_steps", "1",
        "--out_dir", str(OUT_DIR)
    ]
    run_command(finetune_command)

    assert (OUT_DIR/"final").exists(), "Finetuning output directory was not created"
    assert (OUT_DIR/"final"/"lit_model.pth").exists(), "Model file was not created"


@mock.patch.dict(os.environ, {"LT_ACCELERATOR": "cpu"})
@pytest.mark.dependency(depends=["test_download_model", "test_download_books"])
def test_pretrain_model():
    OUT_DIR = Path("out") / "custom_pretrained"
    pretrain_command = [
        "litgpt", "pretrain",
        "pythia-14m",
        "--tokenizer_dir", str("checkpoints" / REPO_ID),
        "--data", "TextFiles",
        "--data.train_data_path", str(CUSTOM_TEXTS_DIR),
        "--train.max_tokens", "100",     # to accelerate things for CI
        "--eval.max_iters", "1",         # to accelerate things for CI
        "--out_dir", str(OUT_DIR)
    ]
    run_command(pretrain_command)

    assert (OUT_DIR / "final").exists(), "Pretraining output directory was not created"
    assert (OUT_DIR / "final" / "lit_model.pth").exists(), "Model file was not created"


@mock.patch.dict(os.environ, {"LT_ACCELERATOR": "cpu"})
@pytest.mark.dependency(depends=["test_download_model", "test_download_books"])
def test_continue_pretrain_model():
    OUT_DIR = Path("out") / "custom_continue_pretrained"
    pretrain_command = [
        "litgpt", "pretrain",
        "pythia-14m",
        "--initial_checkpoint", str("checkpoints" / REPO_ID),
        "--tokenizer_dir", str("checkpoints" / REPO_ID),
        "--data", "TextFiles",
        "--data.train_data_path", str(CUSTOM_TEXTS_DIR),
        "--train.max_tokens", "100",     # to accelerate things for CI
        "--eval.max_iters", "1",         # to accelerate things for CI
        "--out_dir", str(OUT_DIR)
    ]
    run_command(pretrain_command)

    assert (OUT_DIR / "final").exists(), "Continued pretraining output directory was not created"
    assert (OUT_DIR / "final" / "lit_model.pth").exists(), "Model file was not created"


@pytest.mark.dependency(depends=["test_download_model"])
def test_serve():
    CHECKPOINT_DIR = str("checkpoints" / REPO_ID)
    run_command = [
        "litgpt", "serve", str(CHECKPOINT_DIR)
    ]

    process = None

    def run_server():
        nonlocal process
        try:
            process = subprocess.Popen(run_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout, stderr = process.communicate(timeout=60)
        except subprocess.TimeoutExpired:
            print('Server start-up timeout expired')

    server_thread = threading.Thread(target=run_server)
    server_thread.start()

    # Allow time to initialize and start serving
    time.sleep(30)

    try:
        response = requests.get("http://127.0.0.1:8000")
        print(response.status_code)
        assert response.status_code == 200, "Server did not respond as expected."
    finally:
        if process:
            process.kill()
        server_thread.join()
