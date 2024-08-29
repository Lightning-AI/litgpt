import torch
import pytest
import warnings
from pathlib import Path
import litgpt
from litgpt.generate.base import next_token, batched_next_token
from litgpt.api import LLM, GPT
from litgpt.scripts.download import download_from_hub
from tests.conftest import RunIf

warnings.filterwarnings("ignore")

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires a GPU.")
def test_batched_equivalence(tmp_path):

    model_name = "microsoft/phi-2"
    download_from_hub(repo_id=model_name, tokenizer_only=True, checkpoint_dir=tmp_path)

    device = "cuda:0"
    batch_size = 3
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

    x = torch.tensor(
        [43993, 25, 1867, 466, 32660, 17485, 4483, 30, 198, 26410],
        device=device,
        dtype=torch.int64,
    )

    batch_x1 = torch.stack([x] * batch_size, dim=0)

    # Single token generation baseline
    tok_1 = next_token(model, input_pos_1, x.unsqueeze(0), **sample_kwargs)
    tok_2 = next_token(model, input_pos_2, tok_1.unsqueeze(0), **sample_kwargs)

    assert tok_1.ndim == 1
    assert tok_2.ndim == 1
    assert tok_1.size(0) == 1
    assert tok_2.size(0) == 1

    # Switch to batched generation
    model.clear_kv_cache()
    model.set_kv_cache(batch_size=batch_size, max_seq_length=50, device="cuda:0")

    toks_1: torch.Tensor = batched_next_token(model, input_pos_1, batch_x1, sample_kwargs)
    toks_2: torch.Tensor = batched_next_token(model, input_pos_2, toks_1, sample_kwargs)

    assert toks_1.ndim == 2
    assert toks_2.ndim == 2
    assert toks_1.size(0) == batch_size
    assert toks_2.size(0) == batch_size

    # Assert that single and batched next token generation are equivalent
    assert all(t == tok_1 for t in toks_1), f"{tok_1} != {toks_1}"
    assert all(t == tok_2 for t in toks_2), f"{tok_2} != {toks_2}"


@RunIf(min_cuda_gpus=1)
def test_simple_batch():
    old_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    config = litgpt.Config.from_name(
        "Llama-3.1-8B", padded_vocab_size=10000, n_layer=2, n_head=8, n_embd=256
    )
    with torch.device("cuda"):
        m = litgpt.GPT(config).requires_grad_(False).eval()
        x0 = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 7]])
        input_pos0 = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 2]])
        x1 = torch.tensor([[1], [2]])
        input_pos1 = torch.tensor([[4], [3]])

    with torch.device("cuda"):
        m.set_kv_cache(2)
    outs0 = m(x0, input_pos0)
    outs1 = m(x1, input_pos1)

    with torch.device("cuda"):
        m.set_kv_cache(1)

    outs0_ref0 = m(x0[:1], input_pos0[0])
    outs1_ref0 = m(x1[:1], input_pos1[0])

    with torch.device("cuda"):
        m.set_kv_cache(1)

    outs0_ref1 = m(x0[1:], input_pos0[1])
    outs1_ref1 = m(x1[1:], input_pos1[1])

    outs0_ref = torch.cat([outs0_ref0, outs0_ref1])
    outs1_ref = torch.cat([outs1_ref0, outs1_ref1])

    print(outs0_ref - outs0)
    print(outs0.shape)
    torch.testing.assert_close(outs0, outs0_ref)
    torch.testing.assert_close(outs1, outs1_ref)
    torch.backends.cuda.matmul.allow_tf32 = old_allow_tf32
