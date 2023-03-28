import torch

import lit_llama.model as lit_llama


@torch.no_grad()
def test_rmsnorm(orig_llama) -> None:
    block_size = 16
    vocab_size = 16

    sample = torch.rand(size=(2, block_size, vocab_size), dtype=torch.float32)

    eps = 1e-6
    orig_llama_rmsnorm = orig_llama.RMSNorm(vocab_size, eps=eps)(sample)
    llama_rmsnorm = lit_llama.RMSNorm(vocab_size, eps=eps)(sample)

    assert torch.allclose(orig_llama_rmsnorm, llama_rmsnorm)
