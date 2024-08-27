import torch
import warnings
warnings.filterwarnings("ignore")

from litgpt.generate.base import next_token, batched_next_token
from litgpt.api import LLM, GPT

def test_batched_equivalence():
    batch_size = 2
    sample_kwargs = {"top_k": 1}

    llm: LLM = LLM.load("microsoft/phi-2")
    model: GPT = llm.model
    model.set_kv_cache(batch_size=1, max_seq_length=50, device="cuda:0")

    input_pos_1 = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.int64, device="cuda:0")
    input_pos_2 = torch.tensor([10], dtype=torch.int64, device="cuda:0")

    x_1 = torch.tensor(
        [43993, 25, 1867, 466, 32660, 17485, 4483, 30, 198, 26410],
        device="cuda:0",
        dtype=torch.int64,
    )

    # Single token generation baseline
    tok_1 = next_token(model, input_pos_1, x_1.unsqueeze(0), **sample_kwargs)
    print("Next Token 1:", tok_1)
    tok_2 = next_token(model, input_pos_2, tok_1.unsqueeze(0), **sample_kwargs)
    print("Next Token 2:", tok_2)

    # Switch to batched generation
    model.clear_kv_cache()
    model.set_kv_cache(batch_size=batch_size, max_seq_length=50, device="cuda:0")

    toks_1 = batched_next_token(model, input_pos_1, [x_1] * batch_size, sample_kwargs)
    print("Batched Next Token 1:", toks_1)
    toks_2 = batched_next_token(model, input_pos_2, toks_1, sample_kwargs)
    print("Batched Next Token 2:", toks_2)

    # Assert that single and batched next token generation are equivalent
    assert all(t == tok_1 for t in toks_1), f"{tok_1} != {toks_1}"
    assert all(t == tok_2 for t in toks_2), f"{tok_2} != {toks_2}"