# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
from dataclasses import asdict
import shutil

from lightning.fabric import seed_everything
import torch
import requests
import subprocess
from litgpt.utils import _RunIf
import threading
import time
import yaml
import json

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

@_RunIf(min_cuda_gpus=1)
def test_serve_with_openai_spec_missing_chat_template(tmp_path):
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
        "litgpt", "serve", tmp_path, "--openai_spec", "true"
    ]

    process = None

    def run_server():
        nonlocal process
        try:
            process = subprocess.Popen(run_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout, stderr = process.communicate(timeout=10)
            return stdout, stderr
        except subprocess.TimeoutExpired:
            print('Server start-up timeout expired')


    server_thread = threading.Thread(target=run_server)
    server_thread.start()

    time.sleep(5)
    try:
        stdout, _ = run_server()

        assert "ValueError: chat_template not found in tokenizer config file." in stdout, \
            "Expected ValueError for missing chat_template not found in tokenizer config file."
    finally:
        if process:
            process.kill()
        server_thread.join()

@_RunIf(min_cuda_gpus=1)
def test_serve_with_openai_spec(tmp_path):
    seed_everything(123)
    ours_config = Config.from_name("SmolLM2-135M-Instruct")
    download_from_hub(repo_id="HuggingFaceTB/SmolLM2-135M-Instruct", tokenizer_only=True, checkpoint_dir=tmp_path)
    shutil.move(str(tmp_path / "HuggingFaceTB" / "SmolLM2-135M-Instruct" / "tokenizer.json"), str(tmp_path))
    shutil.move(str(tmp_path / "HuggingFaceTB" / "SmolLM2-135M-Instruct" / "tokenizer_config.json"), str(tmp_path))
    ours_model = GPT(ours_config)
    checkpoint_path = tmp_path / "lit_model.pth"
    torch.save(ours_model.state_dict(), checkpoint_path)
    config_path = tmp_path / "model_config.yaml"
    with open(config_path, "w", encoding="utf-8") as fp:
        yaml.dump(asdict(ours_config), fp)

    run_command = [
        "litgpt", "serve", tmp_path, "--openai_spec", "true"
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
        # Test server health
        response = requests.get("http://127.0.0.1:8000/health")
        assert response.status_code == 200, f"Server health check failed with status code {response.status_code}"
        assert response.text == "ok", "Server did not respond as expected."

        # Test non-streaming chat completion
        response = requests.post(
            "http://127.0.0.1:8000/v1/chat/completions",
            json={
                "model": "SmolLM2-135M-Instruct",
                "messages": [{"role": "user", "content": "Hello!"}],
            },
        )
        assert response.status_code == 200, f"Non-streaming chat completion failed with status code {response.status_code}"
        response_json = response.json()
        assert "choices" in response_json, "Response JSON does not contain 'choices'."
        assert "message" in response_json["choices"][0], "Response JSON does not contain 'message' in 'choices'."
        assert "content" in response_json["choices"][0]["message"], "Response JSON does not contain 'content' in 'message'."
        assert response_json["choices"][0]["message"]["content"], "Content is empty in the response."

        # Test streaming chat completion
        stream_response = requests.post(
            "http://127.0.0.1:8000/v1/chat/completions",
            json={
                "model": "SmolLM2-135M-Instruct",
                "messages": [{"role": "user", "content": "Hello!"}],
                "stream": True,
            },
        )
        assert stream_response.status_code == 200, f"Streaming chat completion failed with status code {stream_response.status_code}"
        for line in stream_response.iter_lines():
            decoded = line.decode("utf-8").replace("data: ", "").replace("[DONE]", "").strip()
            if decoded:
                data = json.loads(decoded)
                assert "choices" in data, "Response JSON does not contain 'choices'."
                assert "delta" in data["choices"][0], "Response JSON does not contain 'delta' in 'choices'."
                assert "content" in data["choices"][0]["delta"], "Response JSON does not contain 'content' in 'delta'."
    finally:
        if process:
            process.kill()
        server_thread.join()