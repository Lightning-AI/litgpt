# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
"""CI-safe resolution of tokenizer/config assets for parity tests.

Downloads from Hugging Face, falling back to a public Lightning Model Registry mirror for
gated repos when `HF_TOKEN` is unavailable (e.g. fork PRs), and skips when neither works.
"""

import os
import shutil
from pathlib import Path

import pytest
from huggingface_hub import snapshot_download
from huggingface_hub.errors import GatedRepoError
from transformers import AutoTokenizer

# Tokenizer/config files mirrored for CI. This must stay a superset of what litgpt's
# `Tokenizer` reads (tokenizer.json/model, tokenizer_config.json, generation_config.json)
# and contain enough for `AutoTokenizer.from_pretrained(local_dir)` to load the mirror.
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "generation_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "config.json",
)

# At least one of these is required to build a tokenizer.
_REQUIRED_TOKENIZER_FILES = ("tokenizer.json", "tokenizer.model")

_FIXTURE_TEAMSPACE = "lightning-ai/oss-litgpt"
_FIXTURE_VERSION = "v1"

# Hugging Face repos that are public but gated behind license acceptance
# (verified against the unauthenticated HF model API on 2026-06-10). Keep this in
# sync by regenerating periodically; gating status changes over time.
GATED_TOKENIZER_REPOS = (
    "stabilityai/stablecode-instruct-alpha-3b",
    "tiiuae/falcon-180B",
    "tiiuae/falcon-180B-chat",
    "meta-llama/Llama-2-7b-hf",
    "meta-llama/Llama-2-7b-chat-hf",
    "meta-llama/Llama-2-13b-hf",
    "meta-llama/Llama-2-13b-chat-hf",
    "meta-llama/Llama-2-70b-hf",
    "meta-llama/Llama-2-70b-chat-hf",
    "meta-llama/Meta-Llama-3-8B",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama/Meta-Llama-3.1-8B",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "meta-llama/Meta-Llama-3-70B",
    "meta-llama/Meta-Llama-3-70B-Instruct",
    "meta-llama/Meta-Llama-3.1-70B",
    "meta-llama/Meta-Llama-3.1-70B-Instruct",
    "meta-llama/Meta-Llama-3.1-405B",
    "meta-llama/Meta-Llama-3.1-405B-Instruct",
    "meta-llama/Llama-3.2-1B",
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B",
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.3-70B-Instruct",
    "google/gemma-2b",
    "google/gemma-7b",
    "google/gemma-2-2b",
    "google/gemma-2-9b",
    "google/gemma-2-27b",
    "google/gemma-2b-it",
    "google/gemma-7b-it",
    "google/gemma-2-2b-it",
    "google/gemma-2-9b-it",
    "google/gemma-2-27b-it",
    "google/gemma-3-1b-it",
    "google/gemma-3-4b-it",
    "google/gemma-3-12b-it",
    "google/gemma-3-27b-it",
    "google/codegemma-7b-it",
    "mistralai/Mistral-Large-Instruct-2407",
)


def fixture_slug(repo_id: str) -> str:
    """Maps a HF repo id to a registry-safe model name component."""
    return repo_id.lower().replace("/", "--").replace(".", "-")


def fixture_name(repo_id: str) -> str:
    """Returns the pinned Lightning Model Registry name for a gated repo's tokenizer mirror."""
    return f"{_FIXTURE_TEAMSPACE}/{fixture_slug(repo_id)}-tokenizer:{_FIXTURE_VERSION}"


# Explicit, version-pinned map from HF repo id to its Lightning Registry mirror. Do not
# use floating/latest versions in CI; bump `_FIXTURE_VERSION` when re-uploading fixtures.
HF_TO_LIGHTNING_TOKENIZER_FIXTURE = {repo: fixture_name(repo) for repo in GATED_TOKENIZER_REPOS}


def _is_hf_auth_error(ex: Exception) -> bool:
    """Returns True when HF refused the download because the repo is gated/unauthorized."""
    if isinstance(ex, GatedRepoError):
        return True
    status = getattr(getattr(ex, "response", None), "status_code", None)
    return status in (401, 403)


def _populate_from_hf(repo_id: str, model_dir: Path) -> None:
    """Downloads available tokenizer/config files from Hugging Face into `model_dir`."""
    # `snapshot_download` raises `GatedRepoError` for gated repos so the caller can fall back
    # to the mirror, unlike transformers' `cached_file` which wraps it in a bare `OSError`.
    snapshot_download(
        repo_id,
        local_dir=model_dir,
        allow_patterns=list(TOKENIZER_FILES),
        token=os.getenv("HF_TOKEN"),
    )
    present = {p.name for p in model_dir.iterdir()}
    if not any(name in present for name in _REQUIRED_TOKENIZER_FILES):
        raise ConnectionError(f"Unable to download any tokenizer files from HF for {repo_id}")
    print(f"[fixtures] {repo_id}: resolved via Hugging Face", flush=True)


def _populate_from_lightning_registry(repo_id: str, model_dir: Path) -> None:
    """Downloads the registry mirror for a gated repo into `model_dir`, or skips if unavailable."""
    fixture = HF_TO_LIGHTNING_TOKENIZER_FIXTURE.get(repo_id)
    if fixture is None:
        pytest.skip(
            f"{repo_id} is gated on Hugging Face and HF_TOKEN is unavailable; "
            "no Lightning Model Registry fixture is mapped for it."
        )
    try:
        from litmodels import download_model
    except ImportError:
        pytest.skip(f"{repo_id} is gated and `litmodels` is not installed for the registry fallback.")

    if model_dir.exists():
        shutil.rmtree(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    try:
        download_model(name=fixture, download_dir=str(model_dir))
        print(f"[fixtures] {repo_id}: resolved via Lightning Model Registry fallback ({fixture})", flush=True)
    except Exception as ex:
        # This path is only reached on runs without HF_TOKEN (e.g. fork PRs). A missing or
        # unreachable mirror should skip gracefully rather than fail the job; internal/main
        # runs have HF_TOKEN and never get here.
        pytest.skip(f"Could not fetch Lightning Model Registry fixture '{fixture}' for {repo_id}: {ex}")


def prepare_reference_tokenizer(repo_id: str, model_dir: Path) -> AutoTokenizer:
    """Populates `model_dir` with tokenizer/config files and returns the reference HF tokenizer.

    Args:
        repo_id: The Hugging Face repo id to resolve, e.g. `EleutherAI/pythia-14m`.
        model_dir: Directory to (re)create and populate with the resolved files.

    Returns:
        The reference `AutoTokenizer` loaded from the repo (or the registry mirror).
    """
    model_dir = Path(model_dir)
    if model_dir.exists():
        shutil.rmtree(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    try:
        _populate_from_hf(repo_id, model_dir)
        return AutoTokenizer.from_pretrained(repo_id, token=os.getenv("HF_TOKEN"))
    except Exception as ex:
        if not _is_hf_auth_error(ex):
            raise

    # Gated repo without a usable HF_TOKEN: use the CI mirror instead.
    _populate_from_lightning_registry(repo_id, model_dir)
    return AutoTokenizer.from_pretrained(model_dir)
