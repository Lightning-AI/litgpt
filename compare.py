import math

import models.llama.model as llama
import models.nano.model as nano

import torch
import torch.nn as nn


# LLAMA  XQ torch.Size([3, 32, 16, 2])  # B T nh hs
# NANO    Q torch.Size([3, 16, 32, 2])  # B nh T hs

# LLAMA COS torch.Size([1, 32, 1, 2])   # 1 T 1 hs
# NANO  COS torch.Size([32, 1, 1, 2])   # 1 1 T hs

def compare_rope():
    x = torch.tensor([[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 10]], dtype=torch.float32)
    x = x[:, None, None, :]

    llama_rot_x = llama.rotate_half(x)
    nano_rot_x = nano.rotate_neg_half(x)

    rot_x_matches = torch.allclose(llama_rot_x, nano_rot_x)

    print(f"Comparing rot half\t\t{'OK' if rot_x_matches else 'KO'}")

    _, seq_len, _, dim = x.shape
    llama_cos_cached, llama_sin_cached = llama.precompute_cos_sin(seq_len, dim, x.dtype, x.device, base=10000)
    nano_rope_cache = nano.build_rope_cache(seq_len, dim, dtype=x.dtype, device=x.device, base=10000)

    cos_sin_cache_matches = torch.allclose(llama_cos_cached, nano_rope_cache[0]) and torch.allclose(llama_sin_cached, nano_rope_cache[1])

    print(f"Comparing cos sin cache:\t{'OK' if cos_sin_cache_matches else 'KO'}")

    nano_x_rope = nano.apply_rope(x, nano_rope_cache)
    llama_x_rope, _ = llama.apply_rotary_pos_emb(x, x, llama_cos_cached, llama_sin_cached)

    apply_rope_matches = torch.allclose(nano_x_rope, llama_x_rope)

    print(f"Comparing apply rope:\t\t{'OK' if apply_rope_matches else 'KO'}")


def compare_rmsnorm():
    block_size = 16
    vocab_size = 16

    sample = torch.rand(size=(2, block_size, vocab_size), dtype=torch.float32)

    eps = 1e-6
    llama_rmsnorm = llama.RMSNorm(vocab_size, eps=eps)(sample)
    nano_rmsnorm = nano.RMSNorm(vocab_size, eps=eps)(sample)

    rmsnorm_matches = torch.allclose(llama_rmsnorm, nano_rmsnorm)

    print(f"Comparing rmsnorm:\t\t{'OK' if rmsnorm_matches else 'KO'}")


def copy_mlp(nano_mlp, llama_mlp):
    llama_mlp.w1.weight.copy_(nano_mlp.c_fc1.weight)
    llama_mlp.w3.weight.copy_(nano_mlp.c_fc2.weight)
    llama_mlp.w2.weight.copy_(nano_mlp.c_proj.weight)


def copy_attention(nano_attn, llama_attn):
    n_embd = nano_attn.c_attn.weight.shape[1]
    llama_attn.wq.weight.copy_(nano_attn.c_attn.weight[:n_embd])
    llama_attn.wk.weight.copy_(nano_attn.c_attn.weight[n_embd:-n_embd])
    llama_attn.wv.weight.copy_(nano_attn.c_attn.weight[-n_embd:])
    llama_attn.wo.weight.copy_(nano_attn.c_proj.weight)


def copy_block(nano_block, llama_block):
    llama_block.attention_norm.weight.copy_(nano_block.rms_1.scale)
    copy_attention(nano_block.attn, llama_block.attention)
    llama_block.ffn_norm.weight.copy_(nano_block.rms_2.scale)
    copy_mlp(nano_block.mlp, llama_block.feed_forward)


def copy_weights(nano_model, llama_model):
    llama_model.tok_embeddings.weight.copy_(nano_model.transformer.wte.weight)
    for nano_block, llama_block in zip(nano_model.transformer.h, llama_model.layers):
        copy_block(nano_block, llama_block)
    llama_model.norm.weight.copy_(nano_model.transformer.ln_f.scale)
    llama_model.output.weight.copy_(nano_model.lm_head.weight)


def compare_to_llama():
    block_size = 32
    vocab_size = 32000
    n_layer = 16
    n_head = 16
    n_embd = 32

    nano_config = nano.LLaMAConfig(
        block_size=block_size,
        vocab_size=vocab_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd
    )
    llama_config = llama.ModelArgs(
        dim=n_embd,
        n_layers=n_layer,
        n_heads=n_head,
        vocab_size=vocab_size,
        norm_eps=1e-6,
        max_seq_length=block_size
    )

    batch_size = 3

    token_sample = torch.randint(0, llama_config.vocab_size, size=(batch_size, llama_config.dim), dtype=torch.int64)

    nano_model = nano.LLaMA(nano_config)
    llama_model = llama.LLaMA(llama_config)

    def _init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02/math.sqrt(2 * nano_config.n_layer))
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02/math.sqrt(2 * nano_config.n_layer))

    nano_model.apply(_init_weights)

    with torch.no_grad():
        copy_weights(nano_model, llama_model)

    llama_embed = llama_model.tok_embeddings(token_sample)
    nano_embed = nano_model.transformer.wte(token_sample)
    embed_matches = torch.allclose(llama_embed, nano_embed)

    print(f"Comparing embed:\t\t{'OK' if embed_matches else 'KO'}")

    seq_len = token_sample.shape[1]
    mask = torch.full((1, 1, seq_len, seq_len), float("-inf"))
    mask = torch.triu(mask, diagonal=1)
    llama_block_out = llama_model.layers[0](llama_embed, llama_model.cos_cached, llama_model.sin_cached, mask)
    nano_block_out = nano_model.transformer.h[0](nano_embed)
    block_matches = torch.allclose(llama_block_out, nano_block_out)

    print(f"Comparing block out:\t\t{'OK' if block_matches else 'KO'}")

    expected = llama_model(token_sample)
    out, _ = nano_model(token_sample)
    forward_matches = torch.allclose(out, expected)

    print(f"Comparing forward:\t\t{'OK' if forward_matches else 'KO'}")


if __name__ == "__main__":
    compare_rope()
    compare_rmsnorm()
    compare_to_llama()
