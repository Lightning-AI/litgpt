import operator
import sys
from pathlib import Path
from urllib.request import urlretrieve

import pytest
import torch
from lightning_utilities.core.imports import compare_version

wd = Path(__file__).parent.parent.absolute()


@pytest.fixture(autouse=True)
def restore_default_dtype():
    # just in case
    torch.set_default_dtype(torch.float32)


@torch.inference_mode()
@pytest.mark.parametrize("rotary_pct", (0.25, 1))
@pytest.mark.parametrize("batch_size", (1, 3))
@pytest.mark.parametrize("n_embd", (16, 32))
@pytest.mark.parametrize("parallel_residual", (False, True))
@pytest.mark.parametrize(
    ("device", "dtype"),
    [
        (torch.device("cpu"), torch.float32),
        pytest.param(
            torch.device("cuda"), torch.float16, marks=[
                # the reference does softmax upscaled to fp32 during attention. additionally, the final layernorm input
                # is slightly different
                pytest.mark.xfail(raises=AssertionError, strict=False),
                pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")
            ]
        ),
    ],
)
def test_against_gpt_neox_model(rotary_pct, batch_size, n_embd, parallel_residual, device, dtype) -> None:
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

    from lit_gpt import GPT, Config
    from scripts.convert_hf_checkpoint import copy_weights_gpt_neox

    torch.set_default_dtype(dtype)

    ours_config = Config(
        block_size=64,
        vocab_size=100,
        n_layer=4,
        n_head=8,
        n_embd=n_embd,
        rotary_percentage=rotary_pct,
        parallel_residual=parallel_residual,
    )
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
    )

    state_dict = {}
    theirs_model = GPTNeoXForCausalLM(theirs_config).to(device)
    # load the hf initialization into our model
    copy_weights_gpt_neox(state_dict, theirs_model.state_dict())
    ours_model = GPT(ours_config).to(device)
    ours_model.load_state_dict(state_dict)

    token_sample = torch.randint(
        0, ours_config.padded_vocab_size, size=(batch_size, ours_config.block_size), dtype=torch.int64, device=device
    )

    theirs = theirs_model(token_sample)["logits"]
    ours = ours_model(token_sample)
    torch.testing.assert_close(ours, theirs)


@torch.inference_mode()
@pytest.mark.parametrize(
    "kwargs",
    [
        dict(name="falcon-180B", n_layer=2, n_head=8, n_query_groups=4, n_embd=32),
        dict(name="falcon-40b", n_layer=2, n_head=8, n_query_groups=4, n_embd=32),
    ],
)
@pytest.mark.parametrize(
    ("device", "dtype"),
    [
        (torch.device("cpu"), torch.float32),
        pytest.param(
            torch.device("cuda"), torch.float16, marks=[
                # the reference does softmax upscaled to fp32 during attention. additionally, the final layernorm input
                # is slightly different
                pytest.mark.xfail(raises=AssertionError, strict=False),
                pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")
            ]
        ),
    ],
)
def test_against_hf_falcon(kwargs, device, dtype):
    from transformers.models.falcon import FalconConfig, FalconForCausalLM

    from lit_gpt import GPT, Config
    from scripts.convert_hf_checkpoint import copy_weights_falcon

    torch.set_default_dtype(dtype)

    ours_config = Config.from_name(**kwargs)
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

    theirs_model = FalconForCausalLM(theirs_config).to(device)
    theirs_state_dict = theirs_model.state_dict()
    state_dict = {}
    copy_weights_falcon(kwargs["name"], state_dict, theirs_state_dict)
    ours_model = GPT(ours_config).to(device)
    ours_model.load_state_dict(state_dict)

    # test end to end
    x = torch.tensor([[9856, 23, 491, 1536, 304]], dtype=torch.int32, device=device)
    ours_y = ours_model(x)
    theirs_y = theirs_model(x)["logits"]
    torch.testing.assert_close(ours_y, theirs_y)


@torch.inference_mode()
@pytest.mark.parametrize(
    ("device", "dtype"),
    [
        (torch.device("cpu"), torch.float32),
        pytest.param(
            torch.device("cuda"), torch.float16, marks=[
                # the reference does softmax upscaled to fp32 during attention. additionally, the final layernorm input
                # is slightly different
                pytest.mark.xfail(raises=AssertionError, strict=False),
                pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")
            ]
        ),
    ],
)
def test_against_original_open_llama_3b(device, dtype):
    from transformers.models.llama.configuration_llama import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaForCausalLM

    from lit_gpt import GPT, Config
    from scripts.convert_hf_checkpoint import copy_weights_hf_llama

    torch.set_default_dtype(dtype)

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

    theirs_model = LlamaForCausalLM(theirs_config).to(device)
    theirs_state_dict = theirs_model.state_dict()
    state_dict = {}
    copy_weights_hf_llama(ours_config, {}, state_dict, theirs_state_dict)
    ours_model = GPT(ours_config).to(device)
    ours_model.load_state_dict(state_dict)

    # test end to end
    x = torch.tensor([[9856, 23, 491, 1536, 304]], dtype=torch.int32, device=device)
    assert x.size(1) == T
    ours_y = ours_model(x)
    theirs_y = theirs_model(x)["logits"].to(dtype)  # HF converts logits to float
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
        {"name": "Llama-2-70b-chat-hf", "n_query_groups": 1},
    ],
)
@pytest.mark.parametrize(
    ("device", "dtype"),
    [
        (torch.device("cpu"), torch.float32),
        pytest.param(
            torch.device("cuda"), torch.float16, marks=[
                # the reference does softmax upscaled to fp32 during attention. additionally, the final layernorm input
                # is slightly different
                pytest.mark.xfail(raises=AssertionError, strict=False),
                pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")
            ]
        ),
    ],
)
def test_against_hf_llama2(ours_kwargs, device, dtype):
    from transformers.models.llama.configuration_llama import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaForCausalLM

    from lit_gpt import GPT, Config
    from scripts.convert_hf_checkpoint import copy_weights_hf_llama

    torch.set_default_dtype(dtype)

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
        num_key_value_heads=ours_config.n_query_groups,
        rope_theta=ours_config.rope_base,
        attention_bias=ours_config.bias,
    )
    assert ours_config.intermediate_size == theirs_config.intermediate_size

    theirs_model = LlamaForCausalLM(theirs_config).to(device)
    theirs_state_dict = theirs_model.state_dict()
    state_dict = {}
    copy_weights_hf_llama(ours_config, {}, state_dict, theirs_state_dict)
    ours_model = GPT(ours_config).to(device)
    ours_model.load_state_dict(state_dict)

    # test end to end
    x = torch.tensor([[9856, 23, 491, 1536, 304]], dtype=torch.int32, device=device)
    assert x.size(1) == T
    ours_y = ours_model(x)
    theirs_y = theirs_model(x)["logits"].to(dtype)  # HF converts logits to float
    torch.testing.assert_close(ours_y, theirs_y)


@torch.inference_mode()
@pytest.mark.parametrize(
    ("device", "dtype"),
    [
        (torch.device("cpu"), torch.float32),
        pytest.param(
            torch.device("cuda"), torch.float16, marks=[
                pytest.mark.xfail(raises=AssertionError, strict=False),
                pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")
            ]
        ),
    ],
)
def test_against_hf_phi(device, dtype):
    file_path = wd / "tests" / "original_phi_1_5.py"
    url = "https://gist.githubusercontent.com/carmocca/8ec003d9e0d2fdb09ea92941cd0985b4/raw/2ba35c28824d4f4d5dce14f9588a80067cb6ae7f/original_phi_1_5.py"
    if not file_path.is_file():
        urlretrieve(url=url, filename=file_path)

    from lit_gpt import GPT, Config
    from scripts.convert_hf_checkpoint import copy_weights_phi
    from tests.original_phi_1_5 import MixFormerSequentialConfig, MixFormerSequentialForCausalLM

    torch.set_default_dtype(dtype)

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
        torch_dtype=dtype,
    )
    theirs_config.vocab_size = ours_config.padded_vocab_size

    theirs_model = MixFormerSequentialForCausalLM(theirs_config).to(device)
    theirs_state_dict = theirs_model.state_dict()
    state_dict = {}
    copy_weights_phi(ours_config, state_dict, theirs_state_dict)
    ours_model = GPT(ours_config).to(device)
    ours_model.load_state_dict(state_dict)

    # test end to end
    x = torch.tensor([[9856, 23, 491, 1536, 304]], dtype=torch.int32, device=device)
    assert x.size(1) == T
    ours_y = ours_model(x)
    theirs_y = theirs_model(x)["logits"].to(dtype)  # HF converts logits to float
    torch.testing.assert_close(ours_y, theirs_y)


@torch.inference_mode()
@pytest.mark.skipif(
    compare_version("transformers", operator.lt, "4.33.4", use_base_version=True), reason="requires mistral"
)
@pytest.mark.parametrize(
    ("device", "dtype"),
    [
        (torch.device("cpu"), torch.float32),
        pytest.param(
            torch.device("cuda"), torch.float16, marks=[
                # the reference does softmax upscaled to fp32 during attention. additionally, the final layernorm input
                # is slightly different
                pytest.mark.xfail(raises=AssertionError, strict=False),
                pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")
            ]
        ),
    ],
)
def test_against_hf_mistral(device, dtype):
    from transformers.models.mistral.configuration_mistral import MistralConfig
    from transformers.models.mistral.modeling_mistral import MistralForCausalLM

    from lit_gpt import GPT, Config
    from scripts.convert_hf_checkpoint import copy_weights_hf_llama

    torch.set_default_dtype(dtype)

    ours_config = Config.from_name(
        "Mistral-7B-Instruct-v0.1",
        padded_vocab_size=10000,
        n_layer=2,
        n_embd=32,
        n_head=8,
        n_query_groups=2,
        intermediate_size=86,
    )
    T = 5
    theirs_config = MistralConfig(
        vocab_size=ours_config.padded_vocab_size,
        hidden_size=ours_config.n_embd,
        num_attention_heads=ours_config.n_head,
        num_hidden_layers=ours_config.n_layer,
        intermediate_size=ours_config.intermediate_size,
        max_position_embeddings=T,
        rms_norm_eps=1e-5,
        num_key_value_heads=ours_config.n_query_groups,
        rope_theta=ours_config.rope_base,
    )
    assert ours_config.intermediate_size == theirs_config.intermediate_size

    theirs_model = MistralForCausalLM(theirs_config).to(device)
    theirs_state_dict = theirs_model.state_dict()
    state_dict = {}
    copy_weights_hf_llama(ours_config, {}, state_dict, theirs_state_dict)
    ours_model = GPT(ours_config).to(device)
    ours_model.load_state_dict(state_dict)

    # test end to end
    x = torch.tensor([[9856, 23, 491, 1536, 304]], dtype=torch.int32, device=device)
    assert x.size(1) == T
    ours_y = ours_model(x)
    theirs_y = theirs_model(x)["logits"].to(dtype)  # HF converts logits to float
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
