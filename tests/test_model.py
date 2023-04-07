import torch
import pytest

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
def test_to_orig_llama(lit_llama, orig_llama) -> None:
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
    llama_model.apply(llama_model._init_weights)
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
@torch.no_grad()
def test_bfloat16_llama_init(lit_llama, orig_llama) -> None:
    from lit_llama.utils import EmptyInitOnDevice
    block_size = 64
    vocab_size = 32000
    n_layer = 16
    n_head = 16
    n_embd = 32

    llama_config = lit_llama.LLaMAConfig(
        block_size=block_size,
        vocab_size=vocab_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
    )
    llama_model = lit_llama.LLaMA(llama_config)
    llama_model.apply(llama_model._init_weights)

    batch_size = 3

    token_sample = torch.randint(
        0, vocab_size, size=(batch_size, block_size), dtype=torch.int64
    )

    expected = llama_model(token_sample)

    with EmptyInitOnDevice(device="cuda", dtype=torch.bfloat16):
        llama_model2 = lit_llama.LLaMA(llama_config)
    llama_model2.load_state_dict(llama_model.state_dict(keep_vars=True))

    out = llama_model2(token_sample.cuda()).float().cpu()
    torch.testing.assert_close(out, expected, atol=5e-3, rtol=1e-3)


def copy_adapter_weights(llama_model, orig_llama_model) -> None:
    # copy the gating parameter
    for llama_block, orig_llama_block in zip(llama_model.transformer.h, orig_llama_model.layers):
        if hasattr(llama_block.attn, "gating_factor"):
            llama_block.attn.gating_factor.copy_(orig_llama_block.attention.gate)

    # In the original model, there is one embedding layer for all blocks combined
    orig_adapter_wte = orig_llama_model.adapter_query.weight.reshape(
        orig_llama_model.params.adapter_layer, orig_llama_model.params.adapter_len, orig_llama_model.params.dim
    )

    # In ours, the embedding layer is split across the individual attention layers
    index = 0
    for llama_block in llama_model.transformer.h:
        if hasattr(llama_block.attn, "adapter_wte"):
            llama_block.attn.adapter_wte.weight.copy_(orig_adapter_wte[index])
            index += 1


def enable_gate(model):
    for name, param in model.named_parameters():
        if "gating_factor" in name or "gate" in name:
            param.fill_(1)


@torch.no_grad()
def test_adapter_parity(orig_llama_adapter):
    """Test parity between our implementation of LLaMA-Adapter and the reference code."""
    import lit_llama.adapter as lit_llama
    orig_llama = orig_llama_adapter
    
    block_size = 32
    vocab_size = 100
    n_layer = 2
    n_head = 4
    n_embd = 16
    adapter_prompt_length: int = 10
    adapter_start_layer: int = 0

    llama_config = lit_llama.LLaMAConfig(
        block_size=block_size, vocab_size=vocab_size, n_layer=n_layer, n_head=n_head, n_embd=n_embd,
        adapter_prompt_length=adapter_prompt_length, adapter_start_layer=adapter_start_layer,
    )
    orig_llama_config = orig_llama.ModelArgs(
        dim=n_embd, n_layers=n_layer, n_heads=n_head, vocab_size=vocab_size, norm_eps=1e-5, max_seq_len=block_size,
        adapter_len=adapter_prompt_length, adapter_layer=(n_layer - adapter_start_layer),
    )

    batch_size = 3
    token_sample = torch.randint(
        0, orig_llama_config.vocab_size, size=(batch_size, orig_llama_config.max_seq_len), dtype=torch.int64
    )

    llama_model = lit_llama.LLaMA(llama_config)
    llama_model.apply(llama_model._init_weights)
    orig_llama_model = orig_llama.Transformer(orig_llama_config)

    copy_weights(llama_model, orig_llama_model)
    copy_adapter_weights(llama_model, orig_llama_model)

    # make the gate non-zero, otherwise the adapter is disabled and the model
    # identical to regular LLaMA
    enable_gate(llama_model)
    enable_gate(orig_llama_model)

    expected = orig_llama_model(token_sample, 0)
    out = llama_model(token_sample)
    assert torch.allclose(out, expected)