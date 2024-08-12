# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
from dataclasses import asdict
import shutil

from lightning.fabric import seed_everything
import torch
import requests
import subprocess
from tests.conftest import RunIf
import threading
import time
import yaml

from litgpt import GPT, Config
from litgpt.scripts.download import download_from_hub


def test_simple(tmp_path):
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

    run_command = [
        "litgpt", "serve", tmp_path
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

    time.sleep(30)

    try:
        response = requests.get("http://127.0.0.1:8000")
        print(response.status_code)
        assert response.status_code == 200, "Server did not respond as expected."
    finally:
        if process:
            process.kill()
        server_thread.join()


@RunIf(min_cuda_gpus=1)
def test_quantize(tmp_path):
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

    run_command = [
        "litgpt", "serve", tmp_path, "--quantize", "bnb.nf4"
    ]

    process = None

    def run_server():
        nonlocal process
        try:
            process = subprocess.Popen(run_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout, stderr = process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            print('Server start-up timeout expired')

    server_thread = threading.Thread(target=run_server)
    server_thread.start()

    time.sleep(10)

    try:
        response = requests.get("http://127.0.0.1:8000")
        print(response.status_code)
        assert response.status_code == 200, "Server did not respond as expected."
    finally:
        if process:
            process.kill()
        server_thread.join()


@RunIf(min_cuda_gpus=2)
def test_multi_gpu_serve(tmp_path):
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

    run_command = [
        "litgpt", "serve", tmp_path, "--devices", "2"
    ]

    process = None

    def run_server():
        nonlocal process
        try:
            process = subprocess.Popen(run_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout, stderr = process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            print('Server start-up timeout expired')

    server_thread = threading.Thread(target=run_server)
    server_thread.start()

    time.sleep(10)

    try:
        response = requests.get("http://127.0.0.1:8000")
        print(response.status_code)
        assert response.status_code == 200, "Server did not respond as expected."
    finally:
        if process:
            process.kill()
        server_thread.join()
