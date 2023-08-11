import sys
from pathlib import Path
from urllib.request import urlretrieve

import pytest
import torch

wd = Path(__file__).parent.parent.absolute()


@pytest.mark.parametrize("rotary_pct", (0.25, 1))
@pytest.mark.parametrize("batch_size", (1, 3))
@pytest.mark.parametrize("n_embd", (16, 32))
@pytest.mark.parametrize("parallel_residual", (False, True))
@pytest.mark.parametrize("kv_cache", (False, True))
def test_against_hf_model(rotary_pct, batch_size, n_embd, parallel_residual, kv_cache) -> None:
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

    import lit_gpt
    from scripts.convert_hf_checkpoint import copy_weights_gpt_neox

    block_size = 64
    # https://huggingface.co/stabilityai/stablelm-base-alpha-3b/blob/main/config.json#L24
    vocab_size = 100
    n_layer = 4
    n_head = 8
    batch_size = 3

    ours_config = lit_gpt.Config(
        block_size=block_size,
        vocab_size=vocab_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        rotary_percentage=rotary_pct,
        parallel_residual=parallel_residual,
    )
    assert ours_config.padded_vocab_size == 512
    theirs_config = GPTNeoXConfig(
        hidden_act="gelu",
        hidden_size=n_embd,
        num_attention_heads=n_head,
        num_hidden_layers=n_layer,
        initializer_range=0.02,
        intermediate_size=n_embd * 4,
        layer_norm_eps=1e-05,
        max_position_embeddings=block_size,
        rotary_emb_base=10000,
        rotary_pct=rotary_pct,
        vocab_size=ours_config.padded_vocab_size,
        use_parallel_residual=parallel_residual,
        use_cache=kv_cache,
    )

    state_dict = {}
    theirs_model = GPTNeoXForCausalLM(theirs_config)
    # load the hf initialization into our model
    copy_weights_gpt_neox(state_dict, theirs_model.state_dict())
    ours_model = lit_gpt.GPT(ours_config)
    ours_model.load_state_dict(state_dict)

    token_sample = torch.randint(0, ours_config.padded_vocab_size, size=(batch_size, block_size), dtype=torch.int64)

    theirs_embed = theirs_model.gpt_neox.embed_in(token_sample)
    ours_embed = ours_model.transformer.wte(token_sample)
    torch.testing.assert_close(ours_embed, theirs_embed)

    rope = ours_model.build_rope_cache(token_sample)
    mask = ours_model.build_mask_cache(token_sample)
    position_ids = torch.arange(block_size).unsqueeze(0)
    if kv_cache:
        (theirs_block_out, theirs_kv_cache) = theirs_model.gpt_neox.layers[0](
            theirs_embed, use_cache=True, position_ids=position_ids
        )
        head_size = n_embd // n_head
        k_cache_shape = (batch_size, n_head, block_size, rope[0].size(-1) + head_size - int(rotary_pct * head_size))
        v_cache_shape = (batch_size, n_head, block_size, head_size)
        ours_kv_cache = torch.zeros(k_cache_shape), torch.zeros(v_cache_shape)
        (ours_block_out, ours_kv_cache) = ours_model.transformer.h[0](
            ours_embed, rope, block_size, mask, torch.arange(block_size), ours_kv_cache
        )
        for ours_cache, theirs_cache in zip(ours_kv_cache, theirs_kv_cache):
            torch.testing.assert_close(ours_cache, theirs_cache)
    else:
        (theirs_block_out,) = theirs_model.gpt_neox.layers[0](theirs_embed, position_ids=position_ids)
        ours_block_out, _ = ours_model.transformer.h[0](ours_embed, rope, block_size, mask)
    torch.testing.assert_close(ours_block_out, theirs_block_out)

    theirs = theirs_model(token_sample)["logits"]
    ours = ours_model(token_sample)
    torch.testing.assert_close(ours, theirs)


@torch.inference_mode()
def test_against_original_falcon_40b():
    file_path = wd / "tests" / "original_falcon_40b.py"
    url = "https://gist.githubusercontent.com/carmocca/feed39b1bc65a29f73c1cecc58a01167/raw/a9a65f2b93716b3c09ec9f354d535ae5953de08f/original_falcon_40b.py"
    if not file_path.is_file():
        urlretrieve(url=url, filename=file_path)

    from lit_gpt import GPT, Config
    from scripts.convert_hf_checkpoint import copy_weights_falcon
    from tests.original_falcon_40b import RWConfig, RWForCausalLM

    ours_config = Config.from_name("falcon-40b", n_layer=2, n_head=8, n_query_groups=4, n_embd=32)
    theirs_config = RWConfig(
        hidden_size=32, n_head=8, n_head_kv=4, n_layer=2, parallel_attn=True, vocab_size=65024, bias=False
    )

    theirs_model = RWForCausalLM(theirs_config)
    theirs_state_dict = theirs_model.state_dict()
    state_dict = {}
    copy_weights_falcon("40b", state_dict, theirs_state_dict)
    ours_model = GPT(ours_config)
    ours_model.load_state_dict(state_dict)

    # test end to end
    x = torch.tensor([[9856, 23, 491, 1536, 304]], dtype=torch.int32)
    ours_y = ours_model(x)
    theirs_y = theirs_model(x)["logits"]
    torch.testing.assert_close(ours_y, theirs_y)


@torch.inference_mode()
def test_against_original_open_llama_3b():
    from transformers.models.llama.configuration_llama import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaForCausalLM, apply_rotary_pos_emb

    from lit_gpt import GPT, Config
    from lit_gpt.model import apply_rope
    from scripts.convert_hf_checkpoint import copy_weights_hf_llama

    ours_config = Config.from_name("open_llama_3b", n_layer=2, n_head=8, n_embd=32, intermediate_size=86)
    T = 5
    theirs_config = LlamaConfig(
        hidden_size=ours_config.n_embd,
        num_attention_heads=ours_config.n_head,
        num_hidden_layers=ours_config.n_layer,
        intermediate_size=ours_config.intermediate_size,
        max_position_embeddings=T,
    )
    assert ours_config.intermediate_size == theirs_config.intermediate_size

    theirs_model = LlamaForCausalLM(theirs_config)
    theirs_state_dict = theirs_model.state_dict()
    state_dict = {}
    copy_weights_hf_llama(ours_config, {}, state_dict, theirs_state_dict)
    ours_model = GPT(ours_config)
    ours_model.load_state_dict(state_dict)

    # test rope
    x = torch.randn(2, T, ours_config.n_embd)  # B, T, n_embd
    ours_cos, ours_sin = ours_model.build_rope_cache(x)
    ours_cos, ours_sin = ours_cos[:T], ours_sin[:T]  # this is done in our model forward
    theirs_cos, theirs_sin = theirs_model.model.layers[0].self_attn.rotary_emb(x, T)
    torch.testing.assert_close(ours_cos, theirs_cos.squeeze())
    torch.testing.assert_close(ours_sin, theirs_sin.squeeze())
    q = torch.randn(1, ours_config.n_head, T, ours_config.head_size)
    ours_q_roped = apply_rope(q, ours_cos, ours_sin)
    theirs_q_roped, _ = apply_rotary_pos_emb(q, q, theirs_cos, theirs_sin, torch.arange(T).unsqueeze(0))
    torch.testing.assert_close(ours_q_roped, theirs_q_roped)

    # test end to end
    x = torch.tensor([[9856, 23, 491, 1536, 304]], dtype=torch.int32)
    assert x.size(1) == T
    ours_y = ours_model(x)
    theirs_y = theirs_model(x)["logits"]
    torch.testing.assert_close(ours_y, theirs_y)


@torch.inference_mode()
@pytest.mark.parametrize("size", ("7b", "70b"))
def test_against_hf_llama2(size):
    from transformers.models.llama.configuration_llama import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaForCausalLM

    from lit_gpt import GPT, Config
    from scripts.convert_hf_checkpoint import copy_weights_hf_llama

    if size == "7b":
        ours_kwargs = {"name": "Llama-2-7b-hf"}
        theirs_kwargs = {}
    else:
        ours_kwargs = {"name": "Llama-2-70b-chat-hf", "n_query_groups": 2}
        theirs_kwargs = {"num_key_value_heads": 2}

    ours_config = Config.from_name(n_layer=2, n_head=8, n_embd=32, intermediate_size=86, **ours_kwargs)
    T = 5
    theirs_config = LlamaConfig(
        hidden_size=ours_config.n_embd,
        num_attention_heads=ours_config.n_head,
        num_hidden_layers=ours_config.n_layer,
        intermediate_size=ours_config.intermediate_size,
        max_position_embeddings=T,
        rms_norm_eps=1e-5,
        **theirs_kwargs
    )
    assert ours_config.intermediate_size == theirs_config.intermediate_size

    theirs_model = LlamaForCausalLM(theirs_config)
    theirs_state_dict = theirs_model.state_dict()
    state_dict = {}
    copy_weights_hf_llama(ours_config, {}, state_dict, theirs_state_dict)
    ours_model = GPT(ours_config)
    ours_model.load_state_dict(state_dict)

    # test end to end
    x = torch.tensor([[9856, 23, 491, 1536, 304]], dtype=torch.int32)
    assert x.size(1) == T
    ours_y = ours_model(x)
    theirs_y = theirs_model(x)["logits"]
    torch.testing.assert_close(ours_y, theirs_y)


@pytest.mark.skipif(sys.platform in ("win32", "darwin"), reason="torch.compile not supported on this platform")
@torch.inference_mode()
def test_model_compile():
    import lit_gpt

    config = lit_gpt.Config(block_size=8, vocab_size=8, n_layer=2, n_head=2, n_embd=4)
    model = lit_gpt.GPT(config)
    model.apply(model._init_weights)

    model = torch.compile(model)

    sample = torch.randint(model.config.vocab_size, size=(2, model.config.block_size), dtype=torch.int64)
    for _ in range(3):
        _ = model(sample)


@torch.inference_mode()
@pytest.mark.parametrize("set_highest_max_seq_length", (False, True))
@pytest.mark.flaky(reruns=5)
def test_kv_cache(set_highest_max_seq_length):
    from lit_gpt import GPT, Config

    config = Config(block_size=25, padded_vocab_size=5, n_layer=2, n_head=2, n_embd=8)
    model = GPT(config)
    idx = torch.randint(0, model.config.padded_vocab_size, (1, 5))
    max_new_tokens = 20
    max_seq_length = 25 if set_highest_max_seq_length else 10

    def generate(logits):
        logits = logits[:, -1:]
        probs = torch.nn.functional.softmax(logits, dim=-1)
        return torch.argmax(probs).unsqueeze(0).unsqueeze(0)

    x_no_cache = idx
    x_cache = idx
    input_pos = torch.arange(0, 5)
    for _ in range(max_new_tokens):
        logits_no_cache = model(x_no_cache, max_seq_length)
        out_no_cache = generate(logits_no_cache)

        logits_cache = model(x_cache, max_seq_length, input_pos)
        out_cache = generate(logits_cache)

        torch.testing.assert_close(out_no_cache, out_cache, rtol=0, atol=0)

        x_no_cache = torch.cat((x_no_cache, out_no_cache), dim=1)
        x_cache = out_cache
        input_pos = torch.tensor([input_pos[-1] + 1])
