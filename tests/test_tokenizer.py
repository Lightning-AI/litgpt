# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
import os
import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
from huggingface_hub import snapshot_download
from huggingface_hub.errors import GatedRepoError
from tokenizers import Tokenizer as HFTokenizer
from tokenizers.models import BPE
from transformers import AutoTokenizer

import litgpt.config as config_module
from litgpt import PromptStyle, Tokenizer

# Tokenizer/config assets only — never weights, so `*.safetensors`/`*.bin` are skipped.
_TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "generation_config.json",
    "special_tokens_map.json",
)


def _is_hf_skip_error(ex: Exception) -> bool:
    """Returns True for errors that should skip the test rather than fail it.

    Covers gated repos (401/403) and anonymous rate-limits (429) that occur when
    fork PRs have no HF_TOKEN and multiple CI jobs download concurrently.
    """
    if isinstance(ex, GatedRepoError):
        return True
    status = getattr(getattr(ex, "response", None), "status_code", None)
    return status in (401, 403, 429)


# Teamspace holding the tokenizer mirrors. The mirror name is derived from the HF repo id.
_TOKENIZER_REGISTRY_TEAMSPACE = "lightning-ai/oss-litgpt"


def _download_gated_tokenizer_mirror(repo_id: str, dest: Path) -> Path | None:
    """Download a gated repo's tokenizer files from the Lightning Model Registry mirror.

    Returns the download directory, or ``None`` if the fallback is disabled or the mirror is
    unavailable. Enabled with ``LITGPT_TOKENIZER_REGISTRY_FALLBACK=1``.
    """
    if os.getenv("LITGPT_TOKENIZER_REGISTRY_FALLBACK", "0") != "1":
        return None
    try:
        from litmodels import download_model
    except ImportError as ex:
        print(f"[registry-fallback] {repo_id}: litmodels not available: {ex!r}")
        return None
    slug = repo_id.lower().replace("/", "--").replace(".", "-")
    name = f"{_TOKENIZER_REGISTRY_TEAMSPACE}/{slug}-tokenizer"
    print(f"[registry-fallback] {repo_id}: trying mirror {name}")
    try:
        files = download_model(name, download_dir=str(dest), progress_bar=False)
    except Exception as ex:
        print(f"[registry-fallback] {repo_id}: mirror download failed: {ex!r}")
        return None
    print(f"[registry-fallback] {repo_id}: downloaded {files}")
    return dest


# @pytest.mark.flaky(reruns=3, rerun_except=["AssertionError", "assert", "TypeError"])
@pytest.mark.flaky(reruns=3, reruns_delay=120)
@pytest.mark.parametrize("config", config_module.configs, ids=[c["hf_config"]["name"] for c in config_module.configs])
def test_tokenizer_against_hf(config, tmp_path):
    config = config_module.Config(**config)

    repo_id = f"{config.hf_config['org']}/{config.hf_config['name']}"

    try:
        # Download only tokenizer/config files (no weights). `snapshot_download` raises a typed
        # `GatedRepoError`, so we handle gated repos cleanly instead of retrying for minutes on
        # fork PRs that have no HF_TOKEN.
        cache_dir = snapshot_download(repo_id, allow_patterns=list(_TOKENIZER_FILES), token=os.getenv("HF_TOKEN"))
        theirs = AutoTokenizer.from_pretrained(repo_id, token=os.getenv("HF_TOKEN"))
    except Exception as ex:
        if not _is_hf_skip_error(ex):
            raise
        # No HF_TOKEN: fall back to the registry mirror and load the tokenizer from local files.
        # Skip if the fallback is disabled or there is no mirror for this repo.
        cache_dir = _download_gated_tokenizer_mirror(repo_id, tmp_path / "registry")
        if cache_dir is None:
            pytest.skip(f"{repo_id} is gated on Hugging Face and has no registry mirror.")
        theirs = AutoTokenizer.from_pretrained(cache_dir)

    # litgpt's Tokenizer infers BOS from the directory name (e.g. `SmolLM2-*-Instruct`, `Llama-3*`),
    # so copy the files into a model-named dir, not the cache's commit-hash snapshot path.
    # Copy, not symlink: CI's relative HF_HOME makes the cache path relative, so symlinks would dangle.
    model_dir = tmp_path / config.hf_config["name"]
    model_dir.mkdir(parents=True, exist_ok=True)
    for file in Path(cache_dir).iterdir():
        if file.is_file():
            shutil.copy(file, model_dir / file.name)

    ours = Tokenizer(model_dir)

    assert ours.vocab_size == theirs.vocab_size
    if config.name == "Mixtral-8x22B-v0.1":
        pytest.xfail(reason="Mixtral certainly lists 32000 vocab in its config")
    else:
        assert ours.vocab_size == config.vocab_size

    if config.name.startswith(("falcon", "stablecode", "Qwen2.5", "QwQ", "Qwen3")):
        # even though their config defines it, it's set as None in HF
        assert isinstance(ours.bos_id, int)
        assert theirs.bos_token_id is None
    elif config.name.startswith("Falcon3"):
        if isinstance(ours.bos_id, int):
            assert theirs.bos_token_id is None
        else:
            assert ours.bos_id == theirs.bos_token_id is None
    else:
        assert ours.bos_id == theirs.bos_token_id

    if config.name.startswith("stablecode"):
        # even though their config defines it, it's set as None in HF
        assert ours.eos_id == 0
        assert ours.eos_id == theirs.eos_token_id or theirs.eos_token_id is None
    else:
        assert ours.eos_id == theirs.eos_token_id

    prompt = "Hello, readers of this test!"
    prompt = PromptStyle.from_config(config).apply(prompt)
    actual = ours.encode(prompt)
    expected = theirs.encode(prompt)
    assert actual.tolist() == expected
    assert ours.decode(actual) == theirs.decode(expected, skip_special_tokens=True)

    if not config.name.startswith(("Mistral", "Mixtral")):
        decoded_output = "".join([ours.decode(x) for x in actual])
        if ours.apply_decoding_fix and decoded_output[0] == " ":
            decoded_output = decoded_output[1:]  # the "hack" adds an empty space to the beginning
        assert decoded_output == ours.decode(actual), type(theirs)


def test_tokenizer_input_validation():
    with pytest.raises(NotADirectoryError, match="The checkpoint directory does not exist"):
        Tokenizer("cocofruit")


@pytest.mark.parametrize("use_bos_by_default", (True, False))
@pytest.mark.parametrize("encode_use_bos", (None, True, False))
@pytest.mark.parametrize("encode_use_eos", (True, False))
@pytest.mark.parametrize("processor_returns_bos", (True, False))
@pytest.mark.parametrize("fake_return_ids", ([], [34, 8, 17, 2]))
def test_tokenizer_bos_eos(
    tmp_path, use_bos_by_default, encode_use_bos, encode_use_eos, processor_returns_bos, fake_return_ids
):
    # let `Tokenizers` create a proper (albeit empty) vocab in json format
    HFTokenizer(BPE()).save(str(tmp_path / "tokenizer.json"))

    tokenizer = Tokenizer(tmp_path)
    tokenizer.bos_id = 0
    tokenizer.eos_id = 1
    tokenizer.use_bos = use_bos_by_default

    if processor_returns_bos:
        fake_return_ids = [tokenizer.bos_id] + fake_return_ids
    fake_return_ids = SimpleNamespace(**dict(ids=fake_return_ids))

    with mock.patch.object(tokenizer.processor, "encode", return_value=fake_return_ids):
        tokens = tokenizer.encode("Hello world", bos=encode_use_bos, eos=encode_use_eos).tolist()

    if encode_use_bos or (encode_use_bos is None and use_bos_by_default):
        assert tokens[0] == tokenizer.bos_id
    else:
        assert not tokens or tokens[0] != tokenizer.bos_id

    if encode_use_eos:
        assert tokens[-1] == tokenizer.eos_id
    else:
        assert not tokens or tokens[-1] != tokenizer.eos_id

    # both `bos` and `eos` should either not be found or occur only once at the begging (bos)
    # or at the end (eos) of the tokens sequence
    assert max([id for id, token in enumerate(tokens) if token == tokenizer.bos_id], default=0) == 0
    assert max([id for id, token in enumerate(tokens[::-1]) if token == tokenizer.eos_id], default=0) == 0
