# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import asdict

import psutil
import requests
import torch
import yaml
from lightning.fabric import seed_everything
from urllib3.exceptions import MaxRetryError

from litgpt import GPT, Config
from litgpt.scripts.download import download_from_hub
from litgpt.utils import _RunIf


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

    run_command = ["litgpt", "serve", tmp_path]

    process = None

    def run_server():
        nonlocal process
        try:
            process = subprocess.Popen(run_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout, stderr = process.communicate(timeout=60)
            print("Server output:", stdout, file=sys.stderr)
            print("Server error:", stderr, file=sys.stderr)
        except subprocess.TimeoutExpired:
            print("Server start-up timeout expired")

    def kill_process_on_port(port):
        for proc in psutil.process_iter(["pid", "name"]):
            try:
                for conn in proc.connections(kind="inet"):
                    if conn.laddr.port == port:
                        print(f"Killing process {proc.info['name']} (PID: {proc.info['pid']}) using port {port}")
                        proc.terminate()
                        proc.wait()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

    # Kill any process using port 8000
    kill_process_on_port(8000)

    server_thread = threading.Thread(target=run_server)
    server_thread.start()

    for _ in range(30):
        try:
            response = requests.get("http://127.0.0.1:8000", timeout=1)
            response_status_code = response.status_code
        except (MaxRetryError, requests.exceptions.ConnectionError):
            response_status_code = -1
        if response_status_code == 200:
            break
        time.sleep(1)
    assert response_status_code == 200, "Server did not respond as expected."

    if process:
        process.kill()
    server_thread.join()


@_RunIf(min_cuda_gpus=1)
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

    run_command = ["litgpt", "serve", tmp_path, "--quantize", "bnb.nf4"]

    process = None

    def run_server():
        nonlocal process
        try:
            process = subprocess.Popen(run_command, stdout=None, stderr=None, text=True)
        except subprocess.TimeoutExpired:
            print("Server start-up timeout expired")

    def kill_process_on_port(port):
        for proc in psutil.process_iter(["pid", "name"]):
            try:
                for conn in proc.connections(kind="inet"):
                    if conn.laddr.port == port:
                        print(f"Killing process {proc.info['name']} (PID: {proc.info['pid']}) using port {port}")
                        proc.terminate()
                        proc.wait()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

    # Kill any process using port 8000
    kill_process_on_port(8000)

    server_thread = threading.Thread(target=run_server)
    server_thread.start()

    for _ in range(30):
        try:
            response = requests.get("http://127.0.0.1:8000", timeout=1)
            response_status_code = response.status_code
        except (MaxRetryError, requests.exceptions.ConnectionError):
            response_status_code = -1
        if response_status_code == 200:
            break
        time.sleep(1)
    assert response_status_code == 200, "Server did not respond as expected."

    if process:
        process.kill()
    server_thread.join()


@_RunIf(min_cuda_gpus=2)
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

    run_command = ["litgpt", "serve", tmp_path, "--devices", "2"]

    process = None

    def run_server():
        nonlocal process
        try:
            process = subprocess.Popen(run_command, stdout=None, stderr=None, text=True)
        except subprocess.TimeoutExpired:
            print("Server start-up timeout expired")

    def kill_process_on_port(port):
        for proc in psutil.process_iter(["pid", "name"]):
            try:
                for conn in proc.connections(kind="inet"):
                    if conn.laddr.port == port:
                        print(f"Killing process {proc.info['name']} (PID: {proc.info['pid']}) using port {port}")
                        proc.terminate()
                        proc.wait()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

    # Kill any process using port 8000
    kill_process_on_port(8000)

    server_thread = threading.Thread(target=run_server)
    server_thread.start()

    for _ in range(30):
        try:
            response = requests.get("http://127.0.0.1:8000", timeout=1)
            response_status_code = response.status_code
        except (MaxRetryError, requests.exceptions.ConnectionError):
            response_status_code = -1
        if response_status_code == 200:
            break
        time.sleep(1)
    assert response_status_code == 200, "Server did not respond as expected."

    if process:
        process.kill()
    server_thread.join()
