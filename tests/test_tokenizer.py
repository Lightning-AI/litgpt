from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer


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
