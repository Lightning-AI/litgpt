

def test_lora_layer_replacement(lit_llama):
    from lit_llama.lora import lora, CausalSelfAttention as LoRACausalSelfAttention
    from lit_llama.model import LLaMA, LLaMAConfig
    
    config = LLaMAConfig()
    config.n_layer = 2
    config.n_head = 4
    config.n_embd = 8
    config.block_size = 8
    config.vocab_size = 8

    with lora(r=8, alpha=8, dropout=0.1):
        model = LLaMA(config)

    assert isinstance(model.transformer.h[0].attn, LoRACausalSelfAttention)
    assert isinstance(model.transformer.h[1].attn, LoRACausalSelfAttention)
