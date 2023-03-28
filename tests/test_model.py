import torch

import lit_llama.model as lit_llama


def copy_mlp(llama_mlp, orig_llama_mlp) -> None:
    orig_llama_mlp.w1.weight.copy_(llama_mlp.c_fc1.weight)
    orig_llama_mlp.w3.weight.copy_(llama_mlp.c_fc2.weight)
    orig_llama_mlp.w2.weight.copy_(llama_mlp.c_proj.weight)


def copy_attention(llama_attn, orig_llama_attn) -> None:
    n_embd = llama_attn.c_attn.weight.shape[1]
    orig_llama_attn.wq.weight.copy_(llama_attn.c_attn.weight[:n_embd])
    orig_llama_attn.wk.weight.copy_(llama_attn.c_attn.weight[n_embd:-n_embd])
    orig_llama_attn.wv.weight.copy_(llama_attn.c_attn.weight[-n_embd:])
    orig_llama_attn.wo.weight.copy_(llama_attn.c_proj.weight)


def copy_block(llama_block, orig_llama_block) -> None:
    orig_llama_block.attention_norm.weight.copy_(llama_block.rms_1.scale)
    copy_attention(llama_block.attn, orig_llama_block.attention)
    orig_llama_block.ffn_norm.weight.copy_(llama_block.rms_2.scale)
    copy_mlp(llama_block.mlp, orig_llama_block.feed_forward)


def copy_weights(llama_model, orig_llama_model) -> None:
    orig_llama_model.tok_embeddings.weight.copy_(llama_model.transformer.wte.weight)
    for llama_block, orig_llama_block in zip(llama_model.transformer.h, orig_llama_model.layers):
        copy_block(llama_block, orig_llama_block)
    orig_llama_model.norm.weight.copy_(llama_model.transformer.ln_f.scale)
    orig_llama_model.output.weight.copy_(llama_model.lm_head.weight)


@torch.no_grad()
def test_to_orig_llama(orig_llama) -> None:
    block_size = 64
    vocab_size = 32000
    n_layer = 16
    n_head = 16
    n_embd = 32

    llama_config = lit_llama.LLaMAConfig(
        block_size=block_size, vocab_size=vocab_size, n_layer=n_layer, n_head=n_head, n_embd=n_embd
    )
    orig_llama_config = orig_llama.ModelArgs(
        dim=n_embd, n_layers=n_layer, n_heads=n_head, vocab_size=vocab_size, norm_eps=1e-5, max_seq_len=block_size
    )

    batch_size = 3

    token_sample = torch.randint(
        0, orig_llama_config.vocab_size, size=(batch_size, orig_llama_config.max_seq_len), dtype=torch.int64
    )

    llama_model = lit_llama.LLaMA(llama_config)
    orig_llama_model = orig_llama.Transformer(orig_llama_config)

    copy_weights(llama_model, orig_llama_model)

    orig_llama_embed = orig_llama_model.tok_embeddings(token_sample)
    llama_embed = llama_model.transformer.wte(token_sample)
    assert torch.allclose(orig_llama_embed, llama_embed)

    seq_len = token_sample.shape[1]
    mask = torch.full((1, 1, seq_len, seq_len), float("-inf"))
    mask = torch.triu(mask, diagonal=1)
    orig_llama_block_out = orig_llama_model.layers[0](orig_llama_embed, 0, orig_llama_model.freqs_cis[:seq_len], mask)
    llama_block_out = llama_model.transformer.h[0](llama_embed)
    assert torch.allclose(orig_llama_block_out, llama_block_out)

    expected = orig_llama_model(token_sample, 0)
    out = llama_model(token_sample)
    assert torch.allclose(out, expected)
