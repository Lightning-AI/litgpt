import os
import sys
from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer
from transformers.utils import cached_file

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

import lit_gpt.config as config_module


def test_tokenizer_against_hf():
    import lit_gpt

    hf_tokenizer = AutoTokenizer.from_pretrained("StabilityAI/stablelm-base-alpha-3b")
    # hacky way to access the data loaded by the above
    folder = Path(hf_tokenizer.init_kwargs["special_tokens_map_file"]).parent

    tokenizer = lit_gpt.Tokenizer(folder)

    assert tokenizer.vocab_size == hf_tokenizer.vocab_size
    assert tokenizer.eos_id == hf_tokenizer.eos_token_id

    string = "What's your mood today?"
    actual = tokenizer.encode(string)
    assert actual.tolist() == hf_tokenizer(string)["input_ids"]
    assert tokenizer.decode(actual) == hf_tokenizer.decode(actual)
    assert tokenizer.decode(torch.tensor(0)) == ""

    with pytest.raises(ValueError, match="'foobarbaz' not found"):
        tokenizer.token_to_id("foobarbaz")

    actual = tokenizer.encode("a b")
    assert torch.equal(actual, torch.tensor([66, 270])), actual
    actual = tokenizer.encode("a b", eos=True)
    assert torch.equal(actual, torch.tensor([66, 270, 0])), actual


@pytest.mark.parametrize("config", config_module.configs, ids=[c["name"] for c in config_module.configs])
def test_against_hf(config):
    from lit_gpt.tokenizer import Tokenizer

    access_token = os.getenv("HF_TOKEN")

    config = config_module.Config(**config)

    repo_id = f"{config.org}/{config.name}"
    cache_dir = Path("/tmp/tokenizer_test_cache")

    # create a checkpoint directory that points to the HF files
    checkpoint_dir = cache_dir / "ligpt" / config.org / config.name
    if not checkpoint_dir.exists():
        file_to_cache = {}
        for file in ("tokenizer.json", "generation_config.json", "tokenizer.model", "tokenizer_config.json"):
            try:
                # download the HF tokenizer config
                hf_file = cached_file(repo_id, file, cache_dir=cache_dir / "hf", token=access_token)
            except OSError as e:
                if "gated repo" in str(e) and not access_token:
                    pytest.xfail("Gated repo")
                if "does not appear to have" in str(e):
                    continue
                raise e
            file_to_cache[file] = str(hf_file)
        checkpoint_dir.mkdir(parents=True)
        for file, hf_file in file_to_cache.items():
            (checkpoint_dir / file).symlink_to(hf_file)

    try:
        theirs = AutoTokenizer.from_pretrained(repo_id, cache_dir=cache_dir / "hf", local_files_only=True, token=access_token)
    except OSError as e:
        assert "gated repo" in str(e)
        pytest.xfail("Gated repo")
        return
    ours = Tokenizer(checkpoint_dir)

    prompt = "Hello, readers of this test!"
    actual = ours.encode(prompt).tolist()
    expected = theirs.encode(prompt)
    if ours.backend == "sentencepiece" or config.name == "LLaMA-2-7B-32K":  # TODO: fix this, missing BOS
        assert actual != expected
    else:
        assert actual == expected
