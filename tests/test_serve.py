# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
import contextlib
import json
import platform
import shutil
import socket
import subprocess
import time
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path

import psutil
import pytest
import requests
import torch
import yaml
from lightning.fabric import seed_everything

from litgpt import GPT, Config
from litgpt.scripts.download import download_from_hub
from litgpt.utils import _RunIf, kill_process_tree

# Generous, because a dead server no longer has to wait this out: `_wait_until_ready` gives up as
# soon as the process exits.
_STARTUP_TIMEOUT = 120
_SHUTDOWN_TIMEOUT = 30


def _find_free_port() -> int:
    """Return a port that is currently unused, so that each test can serve on its own."""
    # Bind on all interfaces, like the server does, otherwise a port taken on another interface
    # would look free here and fail at bind time.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def _prepare_checkpoint(checkpoint_dir: Path, repo_id: str) -> None:
    """Write a randomly initialized checkpoint and the matching tokenizer to ``checkpoint_dir``."""
    seed_everything(123)
    config = Config.from_name(repo_id.split("/")[-1])
    download_from_hub(repo_id=repo_id, tokenizer_only=True, checkpoint_dir=checkpoint_dir)
    for filename in ("tokenizer.json", "tokenizer_config.json"):
        shutil.move(str(checkpoint_dir.joinpath(*repo_id.split("/"), filename)), str(checkpoint_dir))
    torch.save(GPT(config).state_dict(), checkpoint_dir / "lit_model.pth")
    with open(checkpoint_dir / "model_config.yaml", "w", encoding="utf-8") as fp:
        yaml.dump(asdict(config), fp)


def _log_tail(log_path: Path, max_chars: int = 8000) -> str:
    output = log_path.read_text(encoding="utf-8", errors="replace")[-max_chars:]
    return f"\n--- server output (tail) ---\n{output}" if output else ""


def _wait_until_ready(url: str, process: subprocess.Popen, log_path: Path) -> None:
    """Poll ``url`` until the server answers, it exits, or ``_STARTUP_TIMEOUT`` passes."""
    deadline = time.monotonic() + _STARTUP_TIMEOUT
    err = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise AssertionError(
                f"Server exited with code {process.returncode} before it was ready.{_log_tail(log_path)}"
            )
        try:
            status_code = requests.get(url, timeout=10).status_code
            if status_code == 200:
                return
            err = f"the server answered with status {status_code}"
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as ex:
            err = str(ex)
        time.sleep(1)
    raise AssertionError(f"Server was not ready within {_STARTUP_TIMEOUT}s: {err}{_log_tail(log_path)}")


def _terminate(process: subprocess.Popen) -> None:
    """Kill the server process tree and wait for it to be gone.

    ``kill_process_tree`` only sends the signals; the workers keep holding their GPU memory and their
    port until they have actually exited, so the next test has to wait for that here.
    """
    # Snapshot the children before killing the parent: once it is gone they are reparented and can no
    # longer be found through it.
    try:
        children = psutil.Process(process.pid).children(recursive=True)
    except psutil.NoSuchProcess:
        children = []
    kill_process_tree(process.pid)
    psutil.wait_procs(children, timeout=_SHUTDOWN_TIMEOUT)
    with contextlib.suppress(subprocess.TimeoutExpired):
        process.wait(timeout=_SHUTDOWN_TIMEOUT)


@contextlib.contextmanager
def _serve(checkpoint_dir: Path, *extra_args: str) -> Iterator[str]:
    """Run ``litgpt serve`` on a free port and yield its base URL once it is ready to answer."""
    port = _find_free_port()
    url = f"http://127.0.0.1:{port}"
    # Log to a file rather than a pipe: nothing reads the pipe while the server starts up, so a full
    # buffer would block it. Keep the file out of the checkpoint directory that is being served.
    log_path = checkpoint_dir.parent / f"{checkpoint_dir.name}-server.log"
    command = ["litgpt", "serve", str(checkpoint_dir), "--port", str(port), *extra_args]
    with open(log_path, "w", encoding="utf-8") as log_fp:
        process = subprocess.Popen(command, stdout=log_fp, stderr=subprocess.STDOUT)
        try:
            _wait_until_ready(url, process, log_path)
            yield url
        finally:
            _terminate(process)


# todo: try to resolve this issue
@pytest.mark.flaky(reruns=2, reruns_delay=30)
@pytest.mark.xfail(condition=platform.system() == "Darwin", reason="it passes locally but having some issues on CI")
def test_simple(tmp_path):
    _prepare_checkpoint(tmp_path, "EleutherAI/pythia-14m")
    with _serve(tmp_path):
        pass


@_RunIf(min_cuda_gpus=1)
def test_quantize(tmp_path):
    _prepare_checkpoint(tmp_path, "EleutherAI/pythia-14m")
    with _serve(tmp_path, "--quantize", "bnb.nf4"):
        pass


@_RunIf(min_cuda_gpus=2)
def test_multi_gpu_serve(tmp_path):
    _prepare_checkpoint(tmp_path, "EleutherAI/pythia-14m")
    with _serve(tmp_path, "--devices", "2"):
        pass


@_RunIf(min_cuda_gpus=1)
def test_serve_with_openai_spec_missing_chat_template(tmp_path):
    _prepare_checkpoint(tmp_path, "EleutherAI/pythia-14m")
    with _serve(tmp_path, "--openai_spec", "true"):
        pass


@_RunIf(min_cuda_gpus=1)
def test_serve_with_openai_spec(tmp_path):
    _prepare_checkpoint(tmp_path, "HuggingFaceTB/SmolLM2-135M-Instruct")

    with _serve(tmp_path, "--openai_spec", "true") as url:
        # Test server health
        response = requests.get(f"{url}/health")
        assert response.status_code == 200, f"Server health check failed with status code {response.status_code}"
        assert response.text == "ok", "Server did not respond as expected."

        # Test non-streaming chat completion
        response = requests.post(
            f"{url}/v1/chat/completions",
            json={
                "model": "SmolLM2-135M-Instruct",
                "messages": [{"role": "user", "content": "Hello!"}],
            },
        )
        assert response.status_code == 200, (
            f"Non-streaming chat completion failed with status code {response.status_code}"
        )
        response_json = response.json()
        assert "choices" in response_json, "Response JSON does not contain 'choices'."
        assert "message" in response_json["choices"][0], "Response JSON does not contain 'message' in 'choices'."
        assert "content" in response_json["choices"][0]["message"], (
            "Response JSON does not contain 'content' in 'message'."
        )
        assert response_json["choices"][0]["message"]["content"], "Content is empty in the response."

        # Test streaming chat completion
        stream_response = requests.post(
            f"{url}/v1/chat/completions",
            json={
                "model": "SmolLM2-135M-Instruct",
                "messages": [{"role": "user", "content": "Hello!"}],
                "stream": True,
            },
        )
        assert stream_response.status_code == 200, (
            f"Streaming chat completion failed with status code {stream_response.status_code}"
        )
        for line in stream_response.iter_lines():
            decoded = line.decode("utf-8").replace("data: ", "").replace("[DONE]", "").strip()
            if decoded:
                data = json.loads(decoded)
                assert "choices" in data, "Response JSON does not contain 'choices'."
                assert "delta" in data["choices"][0], "Response JSON does not contain 'delta' in 'choices'."
                assert "content" in data["choices"][0]["delta"], "Response JSON does not contain 'content' in 'delta'."


@pytest.mark.parametrize(
    "generate_strategy",
    [
        pytest.param("sequential", marks=_RunIf(min_cuda_gpus=1)),
        pytest.param("tensor_parallel", marks=_RunIf(min_cuda_gpus=2)),
    ],
)
def test_serve_with_generate_strategy(tmp_path, generate_strategy):
    _prepare_checkpoint(tmp_path, "EleutherAI/pythia-14m")

    extra_args = ["--generate_strategy", generate_strategy]
    if generate_strategy == "tensor_parallel":
        extra_args += ["--devices", "2"]

    with _serve(tmp_path, *extra_args):
        pass
