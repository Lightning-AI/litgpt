import torch
import pytest
import warnings
from pathlib import Path
from litgpt.generate.base import next_token, batched_next_token
from litgpt.api import LLM, GPT
from litgpt.scripts.download import download_from_hub

warnings.filterwarnings("ignore")

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires a GPU.")
def test_batched_equivalence(tmp_path):

    model_name = "microsoft/phi-2"
    download_from_hub(repo_id=model_name, tokenizer_only=True, checkpoint_dir=tmp_path)

    device = "cuda:0"
    batch_size = 2
    sample_kwargs = {"top_k": 1}

    llm: LLM = LLM.load(
        model_name,
        tokenizer_dir=Path(tmp_path / model_name),
        init="random",
    )
    model: GPT = llm.model
    model.set_kv_cache(batch_size=1, max_seq_length=50, device=device)

    input_pos_1 = torch.tensor(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.int64, device=device
    )
    input_pos_2 = torch.tensor([10], dtype=torch.int64, device=device)

    x_1 = torch.tensor(
        [43993, 25, 1867, 466, 32660, 17485, 4483, 30, 198, 26410],
        device=device,
        dtype=torch.int64,
    )

    # Single token generation baseline
    tok_1 = next_token(model, input_pos_1, x_1.unsqueeze(0), **sample_kwargs)
    print("Next Token 1:", tok_1)
    tok_2 = next_token(model, input_pos_2, tok_1.unsqueeze(0), **sample_kwargs)
    print("Next Token 2:", tok_2)

    # Switch to batched generation
    model.clear_kv_cache()
    model.set_kv_cache(batch_size=batch_size, max_seq_length=50, device=device)

    toks_1 = batched_next_token(model, input_pos_1, [x_1] * batch_size, sample_kwargs)
    print("Batched Next Token 1:", toks_1)
    toks_2 = batched_next_token(model, input_pos_2, toks_1, sample_kwargs)
    print("Batched Next Token 2:", toks_2)

    # Assert that single and batched next token generation are equivalent
    assert all(t == tok_1 for t in toks_1), f"{tok_1} != {toks_1}"
    assert all(t == tok_2 for t in toks_2), f"{tok_2} != {toks_2}"
