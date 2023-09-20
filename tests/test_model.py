import operator
import sys
from pathlib import Path
from urllib.request import urlretrieve

import pytest
import torch
from lightning_utilities.core.imports import compare_version

wd = Path(__file__).parent.parent.absolute()


@pytest.mark.parametrize("rotary_pct", (0.25, 1))
@pytest.mark.parametrize("batch_size", (1, 3))
@pytest.mark.parametrize("n_embd", (16, 32))
@pytest.mark.parametrize("parallel_residual", (False, True))
@pytest.mark.parametrize("kv_cache", (False, True))
def test_against_gpt_neox_model(rotary_pct, batch_size, n_embd, parallel_residual, kv_cache) -> None:
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

    from lit_gpt import GPT, Config
    from scripts.convert_hf_checkpoint import copy_weights_gpt_neox

    batch_size = 3
    ours_config = Config(block_size=64, vocab_size=100, n_layer=4, n_head=8, n_embd=n_embd)
    assert ours_config.padded_vocab_size == 512
    theirs_config = GPTNeoXConfig(
        hidden_act="gelu",
        hidden_size=ours_config.n_embd,
        num_attention_heads=ours_config.n_head,
        num_hidden_layers=ours_config.n_layer,
        initializer_range=0.02,
        intermediate_size=ours_config.intermediate_size,
        layer_norm_eps=1e-05,
        max_position_embeddings=ours_config.block_size,
        rotary_emb_base=10000,
        rotary_pct=ours_config.rotary_percentage,
        vocab_size=ours_config.padded_vocab_size,
        use_parallel_residual=ours_config.parallel_residual,
        use_cache=kv_cache,
    )

    state_dict = {}
    theirs_model = GPTNeoXForCausalLM(theirs_config)
    # load the hf initialization into our model
    copy_weights_gpt_neox(state_dict, theirs_model.state_dict())
    ours_model = GPT(ours_config)
    ours_model.load_state_dict(state_dict)

    token_sample = torch.randint(
        0, ours_config.padded_vocab_size, size=(batch_size, ours_config.block_size), dtype=torch.int64
    )

    theirs_embed = theirs_model.gpt_neox.embed_in(token_sample)
    ours_embed = ours_model.transformer.wte(token_sample)
    torch.testing.assert_close(ours_embed, theirs_embed)

    cos, sin = ours_model.cos, ours_model.sin
    mask = ours_model.mask_cache
    position_ids = torch.arange(ours_config.block_size).unsqueeze(0)
    theirs_block = theirs_model.gpt_neox.layers[0]
    ours_block = ours_model.transformer.h[0]
    if kv_cache:
        theirs_block_out, (theirs_k, theirs_v) = theirs_block(theirs_embed, use_cache=True, position_ids=position_ids)
        ours_model.set_kv_cache(batch_size)
        ours_kv_cache = ours_block.attn.kv_cache
        ours_block_out = ours_block(ours_embed, cos, sin, mask, torch.arange(ours_config.block_size))
        torch.testing.assert_close(ours_kv_cache.k, theirs_k)
        torch.testing.assert_close(ours_kv_cache.v, theirs_v)
    else:
        (theirs_block_out,) = theirs_block(theirs_embed, position_ids=position_ids)
        ours_block_out = ours_block(ours_embed, cos, sin, mask)
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
        hidden_size=ours_config.n_embd,
        n_head=ours_config.n_head,
        n_head_kv=ours_config.n_query_groups,
        n_layer=ours_config.n_layer,
        parallel_attn=ours_config.parallel_residual,
        vocab_size=ours_config.padded_vocab_size,
        bias=ours_config.bias,
    )

    theirs_model = RWForCausalLM(theirs_config)
    theirs_state_dict = theirs_model.state_dict()
    state_dict = {}
    copy_weights_falcon("falcon-40b", state_dict, theirs_state_dict)
    ours_model = GPT(ours_config)
    ours_model.load_state_dict(state_dict)

    # test end to end
    x = torch.tensor([[9856, 23, 491, 1536, 304]], dtype=torch.int32)
    ours_y = ours_model(x)
    theirs_y = theirs_model(x)["logits"]
    torch.testing.assert_close(ours_y, theirs_y)


@torch.inference_mode()
def test_against_original_falcon_180b():
    from transformers.models.falcon import FalconConfig, FalconForCausalLM

    from lit_gpt import GPT, Config
    from scripts.convert_hf_checkpoint import copy_weights_falcon

    ours_config = Config.from_name("falcon-180B", n_layer=2, n_head=8, n_query_groups=4, n_embd=32)
    theirs_config = FalconConfig(
        hidden_size=ours_config.n_embd,
        num_attention_heads=ours_config.n_head,
        num_kv_heads=ours_config.n_query_groups,
        num_hidden_layers=ours_config.n_layer,
        parallel_attn=ours_config.parallel_residual,
        vocab_size=ours_config.padded_vocab_size,
        bias=ours_config.bias,
        new_decoder_architecture=True,
    )

    theirs_model = FalconForCausalLM(theirs_config)
    theirs_state_dict = theirs_model.state_dict()
    state_dict = {}
    copy_weights_falcon("falcon-180B", state_dict, theirs_state_dict)
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
    ours_cos, ours_sin = ours_model.cos[:T], ours_model.sin[:T]  # this is done in our model forward
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
@pytest.mark.parametrize(
    "ours_kwargs",
    [
        {"name": "Llama-2-7b-hf"},
        pytest.param(
            {"name": "CodeLlama-7b-hf"},
            marks=pytest.mark.skipif(
                compare_version("transformers", operator.lt, "4.33.0", use_base_version=True),
                reason="requires rope_theta",
            ),
        ),
        {"name": "Llama-2-70b-chat-hf"},
    ],
)
def test_against_hf_llama2(ours_kwargs):
    from transformers.models.llama.configuration_llama import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaForCausalLM

    from lit_gpt import GPT, Config
    from scripts.convert_hf_checkpoint import copy_weights_hf_llama

    ours_config = Config.from_name(
        padded_vocab_size=10000, n_layer=2, n_head=8, n_embd=32, intermediate_size=86, **ours_kwargs
    )
    T = 5
    theirs_config = LlamaConfig(
        vocab_size=ours_config.padded_vocab_size,
        hidden_size=ours_config.n_embd,
        num_attention_heads=ours_config.n_head,
        num_hidden_layers=ours_config.n_layer,
        intermediate_size=ours_config.intermediate_size,
        max_position_embeddings=T,
        rms_norm_eps=1e-5,
        num_query_value_heads=ours_config.n_query_groups,
        rope_theta=ours_config.rope_base,
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


@torch.inference_mode()
def test_against_hf_phi():
    file_path = wd / "tests" / "original_phi_1_5.py"
    url = "https://gist.githubusercontent.com/carmocca/8ec003d9e0d2fdb09ea92941cd0985b4/raw/2ba35c28824d4f4d5dce14f9588a80067cb6ae7f/original_phi_1_5.py"
    if not file_path.is_file():
        urlretrieve(url=url, filename=file_path)

    from lit_gpt import GPT, Config
    from scripts.convert_hf_checkpoint import copy_weights_phi
    from tests.original_phi_1_5 import MixFormerSequentialConfig, MixFormerSequentialForCausalLM

    ours_config = Config.from_name(
        "phi-1_5", padded_vocab_size=10000, n_layer=2, n_head=4, n_embd=256, rotary_percentage=0.5
    )
    T = 5
    theirs_config = MixFormerSequentialConfig(
        n_positions=ours_config.block_size,
        n_embd=ours_config.n_embd,
        n_head=ours_config.n_head,
        n_layer=ours_config.n_layer,
        rotary_dim=ours_config.rope_n_elem,
        architecture={"block_cls": "parallel", "mixer": {}, "mlp": {"mlp_cls": "mlp"}},
    )
    theirs_config.vocab_size = ours_config.padded_vocab_size

    theirs_model = MixFormerSequentialForCausalLM(theirs_config)
    theirs_state_dict = theirs_model.state_dict()
    state_dict = {}
    copy_weights_phi(ours_config, state_dict, theirs_state_dict)
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
    from lit_gpt import GPT

    model = GPT.from_name("pythia-70m", n_layer=3)
    x = torch.randint(model.config.vocab_size, size=(2, model.config.block_size), dtype=torch.int64)

    from torch._dynamo.backends import debugging

    explanation = torch._dynamo.explain(model, x)
    assert isinstance(explanation, debugging.ExplainOutput)
    assert explanation.graph_count == 1
    assert explanation.graph_break_count == 0

    model = GPT(model.config)
    model.set_kv_cache(2)
    input_pos = torch.arange(model.config.block_size)
    explanation = torch._dynamo.explain(model, x, input_pos)
    assert isinstance(explanation, debugging.ExplainOutput)
    assert explanation.graph_count == 1
    assert explanation.graph_break_count == 0


@torch.inference_mode()
@pytest.mark.parametrize(
    "max_seq_length", (25, pytest.param(23, marks=pytest.mark.xfail(raises=IndexError, strict=True)))
)
@pytest.mark.flaky(reruns=5)
def test_kv_cache(max_seq_length):
    from lit_gpt import GPT, Config

    config = Config(block_size=25, padded_vocab_size=5, n_layer=2, n_head=2, n_embd=8)
    model = GPT(config)
    idx = torch.randint(0, model.config.padded_vocab_size, (1, 5))
    max_new_tokens = 20
    model.max_seq_length = max_seq_length
    model.set_kv_cache(1)

    def generate(logits):
        logits = logits[:, -1:]
        probs = torch.nn.functional.softmax(logits, dim=-1)
        return torch.argmax(probs).unsqueeze(0).unsqueeze(0)

    x_no_cache = idx
    x_cache = idx
    input_pos = torch.arange(0, 5)
    for _ in range(max_new_tokens):
        logits_no_cache = model(x_no_cache[:, -max_seq_length:])
        out_no_cache = generate(logits_no_cache)

        logits_cache = model(x_cache, input_pos)
        out_cache = generate(logits_cache)

        torch.testing.assert_close(out_no_cache, out_cache, rtol=0, atol=0)

        x_no_cache = torch.cat((x_no_cache, out_no_cache), dim=1)
        x_cache = out_cache
        input_pos = input_pos[-1:] + 1


@torch.inference_mode()
def test_model_kv_cache_amp():
    from lit_gpt.model import GPT, Config

    config = Config.from_name("pythia-70m", n_layer=2)
    model = GPT(config)
    encoded = torch.arange(45)
    model.set_kv_cache(batch_size=1)
    with torch.autocast("cpu", torch.bfloat16):
        output = model(encoded.unsqueeze(0), encoded)
    assert output.dtype is torch.bfloat16
