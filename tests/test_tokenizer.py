# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
import os
import shutil
import warnings
from types import SimpleNamespace
from unittest import mock

import pytest
from tokenizers import Tokenizer as HFTokenizer
from tokenizers.models import BPE
from transformers import AutoTokenizer
from transformers.utils import cached_file

import litgpt.config as config_module
from litgpt import PromptStyle, Tokenizer


# @pytest.mark.flaky(reruns=3, rerun_except=["AssertionError", "assert", "TypeError"])
@pytest.mark.parametrize("config", config_module.configs, ids=[c["hf_config"]["name"] for c in config_module.configs])
def test_tokenizer_against_hf(config, tmp_path):
    config = config_module.Config(**config)

    repo_id = f"{config.hf_config['org']}/{config.hf_config['name']}"
    theirs = AutoTokenizer.from_pretrained(repo_id, token=os.getenv("HF_TOKEN"))

    # create a checkpoint directory that points to the HF files
    hf_files = {}
    for filename in ("tokenizer.json", "generation_config.json", "tokenizer.model", "tokenizer_config.json"):
        try:  # download the HF tokenizer config
            hf_file = cached_file(path_or_repo_id=repo_id, filename=filename)
            hf_files[filename] = str(hf_file)
        except Exception as ex:
            warnings.warn(str(ex), RuntimeWarning)
    if "tokenizer.json" not in hf_files and "tokenizer.model" not in hf_files:
        raise ConnectionError("Unable to download any tokenizer files from HF")
    for filename, hf_file in hf_files.items():
        shutil.copy(hf_file, str(tmp_path / filename))

    ours = Tokenizer(tmp_path)

    assert ours.vocab_size == theirs.vocab_size
    if config.name.startswith("CodeLlama-70b-Instruct"):
        # TODO: the HF tokenizer returns 1 less token for this model. why?
        assert ours.vocab_size == config.vocab_size - 1
    elif config.name == "Mixtral-8x22B-v0.1":
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
    if (expected[0] == theirs.bos_token_id and actual[0] != theirs.bos_token_id) or (
        expected[0] == theirs.bos_token_id and expected[1] == theirs.bos_token_id
    ):
        # TODO: check what is going on with the bos_tokens
        del expected[0]
    if config.name.startswith("CodeLlama-70b"):
        # TODO: there's a encoding difference with this model. why? note that the decoding is equal
        # "Hello": 10994, "▁Hello": 15043
        assert [15043 if t == 10994 else t for t in actual.tolist()] == expected
    else:
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
