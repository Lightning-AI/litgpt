import os
import sys

import torch


def compare_rope():
    x = torch.tensor([[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 10]], dtype=torch.float32)
    x = x[:, None, None, :]

    _, seq_len, n_heads, dim = x.shape
    freqs_cis = orig_llama.precompute_freqs_cis(dim // n_heads, seq_len)
    llama_rope_cache = llama.build_rope_cache(seq_len, dim, dtype=x.dtype, device=x.device, base=10000)

    llama_x_rope = llama.apply_rope(x, llama_rope_cache)
    orig_llama_x_rope, _ = orig_llama.apply_rotary_emb(x, x, freqs_cis)
    apply_rope_matches = torch.allclose(llama_x_rope, orig_llama_x_rope)

    print(f"Comparing apply rope:\t\t{'OK' if apply_rope_matches else 'KO'}")


def compare_rmsnorm():
    block_size = 16
    vocab_size = 16

    sample = torch.rand(size=(2, block_size, vocab_size), dtype=torch.float32)

    eps = 1e-6
    orig_llama_rmsnorm = orig_llama.RMSNorm(vocab_size, eps=eps)(sample)
    llama_rmsnorm = llama.RMSNorm(vocab_size, eps=eps)(sample)

    rmsnorm_matches = torch.allclose(orig_llama_rmsnorm, llama_rmsnorm)

    print(f"Comparing rmsnorm:\t\t{'OK' if rmsnorm_matches else 'KO'}")


@torch.no_grad()
def copy_mlp(llama_mlp, orig_llama_mlp):
    orig_llama_mlp.w1.weight.copy_(llama_mlp.c_fc1.weight)
    orig_llama_mlp.w3.weight.copy_(llama_mlp.c_fc2.weight)
    orig_llama_mlp.w2.weight.copy_(llama_mlp.c_proj.weight)


@torch.no_grad()
def copy_attention(llama_attn, orig_llama_attn):
    n_embd = llama_attn.c_attn.weight.shape[1]
    orig_llama_attn.wq.weight.copy_(llama_attn.c_attn.weight[:n_embd])
    orig_llama_attn.wk.weight.copy_(llama_attn.c_attn.weight[n_embd:-n_embd])
    orig_llama_attn.wv.weight.copy_(llama_attn.c_attn.weight[-n_embd:])
    orig_llama_attn.wo.weight.copy_(llama_attn.c_proj.weight)


@torch.no_grad()
def copy_block(llama_block, orig_llama_block):
    orig_llama_block.attention_norm.weight.copy_(llama_block.rms_1.scale)
    copy_attention(llama_block.attn, orig_llama_block.attention)
    orig_llama_block.ffn_norm.weight.copy_(llama_block.rms_2.scale)
    copy_mlp(llama_block.mlp, orig_llama_block.feed_forward)


@torch.no_grad()
def copy_weights(llama_model, orig_llama_model):
    orig_llama_model.tok_embeddings.weight.copy_(llama_model.transformer.wte.weight)
    for llama_block, orig_llama_block in zip(llama_model.transformer.h, orig_llama_model.layers):
        copy_block(llama_block, orig_llama_block)
    orig_llama_model.norm.weight.copy_(llama_model.transformer.ln_f.scale)
    orig_llama_model.output.weight.copy_(llama_model.lm_head.weight)


def compare_to_orig_llama():
    block_size = 64
    vocab_size = 32000
    n_layer = 16
    n_head = 16
    n_embd = 32

    llama_config = llama.LLaMAConfig(
        block_size=block_size,
        vocab_size=vocab_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd
    )
    orig_llama_config = orig_llama.ModelArgs(
        dim=n_embd,
        n_layers=n_layer,
        n_heads=n_head,
        vocab_size=vocab_size,
        norm_eps=1e-6,
        max_seq_len=block_size
    )

    batch_size = 3

    token_sample = torch.randint(0, orig_llama_config.vocab_size, size=(batch_size, orig_llama_config.max_seq_len), dtype=torch.int64)

    llama_model = llama.LLaMA(llama_config)
    orig_llama_model = orig_llama.Transformer(orig_llama_config)

    copy_weights(llama_model, orig_llama_model)

    orig_llama_embed = orig_llama_model.tok_embeddings(token_sample)
    llama_embed = llama_model.transformer.wte(token_sample)
    embed_matches = torch.allclose(orig_llama_embed, llama_embed)

    print(f"Comparing embed:\t\t{'OK' if embed_matches else 'KO'}")

    seq_len = token_sample.shape[1]
    mask = torch.full((1, 1, seq_len, seq_len), float("-inf"))
    mask = torch.triu(mask, diagonal=1)
    orig_llama_block_out = orig_llama_model.layers[0](orig_llama_embed, 0, orig_llama_model.freqs_cis[: seq_len], mask)
    llama_block_out = llama_model.transformer.h[0](llama_embed)
    block_matches = torch.allclose(orig_llama_block_out, llama_block_out)

    print(f"Comparing block out:\t\t{'OK' if block_matches else 'KO'}")

    expected = orig_llama_model(token_sample, 0)
    out = llama_model(token_sample)

    forward_matches = torch.allclose(out, expected)
    print(f"Comparing forward:\t\t{'OK' if forward_matches else 'KO'}")


if __name__ == "__main__":
    wd = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.append(wd)

    from scripts.download import download_original

    download_original(wd)

    import model as llama
    import original_model as orig_llama

    compare_rope()
    compare_rmsnorm()
    compare_to_orig_llama()
