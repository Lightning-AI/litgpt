# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
from types import SimpleNamespace
from unittest import mock

import pytest
from _fixtures import prepare_reference_tokenizer
from tokenizers import Tokenizer as HFTokenizer
from tokenizers.models import BPE

import litgpt.config as config_module
from litgpt import PromptStyle, Tokenizer


# @pytest.mark.flaky(reruns=3, rerun_except=["AssertionError", "assert", "TypeError"])
@pytest.mark.flaky(reruns=3, reruns_delay=120)
@pytest.mark.parametrize("config", config_module.configs, ids=[c["hf_config"]["name"] for c in config_module.configs])
def test_tokenizer_against_hf(config, tmp_path):
    config = config_module.Config(**config)

    repo_id = f"{config.hf_config['org']}/{config.hf_config['name']}"

    # Populate a clean, model-specific subdirectory with tokenizer/config files and get the
    # reference HF tokenizer. Falls back to the Lightning Model Registry for gated repos
    # without HF_TOKEN (e.g. fork PRs), and skips when no mirror exists for a gated repo.
    model_dir = tmp_path / config.hf_config["name"]
    theirs = prepare_reference_tokenizer(repo_id, model_dir)

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
