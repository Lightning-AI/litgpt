from pathlib import Path

from transformers import AutoTokenizer


def test_tokenizer_against_hf(lit_stablelm):
    hf_tokenizer = AutoTokenizer.from_pretrained("StabilityAI/stablelm-base-alpha-3b")
    # hacky way to access the data loaded by the above
    folder = Path(hf_tokenizer.init_kwargs["special_tokens_map_file"]).parent

    tokenizer = lit_stablelm.Tokenizer(folder / "tokenizer.json", folder / "tokenizer_config.json")

    assert tokenizer.vocab_size == hf_tokenizer.vocab_size
    assert tokenizer.bos_id == hf_tokenizer.bos_token_id
    assert tokenizer.eos_id == hf_tokenizer.eos_token_id

    string = "What's your mood today?"
    actual = tokenizer.encode(string)
    assert actual.tolist() == hf_tokenizer(string)["input_ids"]
    assert tokenizer.decode(actual) == hf_tokenizer.decode(actual)
