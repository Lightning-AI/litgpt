# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
import os
from contextlib import redirect_stdout
from copy import deepcopy
from io import StringIO
from itertools import product
from unittest import mock
from unittest.mock import Mock

import pytest
import torch
import yaml
from lightning import Fabric
from lightning.fabric.plugins.precision.bitsandbytes import _BITSANDBYTES_AVAILABLE, BitsandbytesPrecision
from lightning.fabric.wrappers import _FabricOptimizer
from torch._dynamo.backends import debugging
from torch.nn import functional as F
from transformers.models.gemma import GemmaConfig, GemmaForCausalLM
from transformers.models.gemma2 import Gemma2Config, Gemma2ForCausalLM
from transformers.models.gemma3 import Gemma3ForCausalLM, Gemma3TextConfig
from transformers.models.mixtral import MixtralConfig, MixtralForCausalLM

import litgpt.config as config_module
import litgpt.finetune.lora as module
from litgpt.args import EvalArgs, TrainArgs
from litgpt.data import Alpaca
from litgpt.lora import GPT as LoRAGPT
from litgpt.lora import (
    CausalSelfAttention,
    Config,
    LoRALinear,
    LoRAQKVLinear,
    lora_filter,
    mark_only_lora_as_trainable,
    merge_lora_weights,
)
from litgpt.lora import CausalSelfAttention as LoRACausalSelfAttention
from litgpt.model import GPT as BaseGPT
from litgpt.scripts.convert_hf_checkpoint import copy_weights_gemma_2, copy_weights_gemma_3, copy_weights_hf_llama
from litgpt.scripts.convert_lit_checkpoint import qkv_reassemble as make_qkv_interleaved
from litgpt.utils import _RunIf


def test_lora_layer_replacement():
    config = Config(n_layer=2, n_head=4, n_embd=8, block_size=8, vocab_size=8, lora_r=8, lora_alpha=8, lora_dropout=0.1)
    model = LoRAGPT(config)

    assert isinstance(model.transformer.h[0].attn, LoRACausalSelfAttention)
    assert isinstance(model.transformer.h[1].attn, LoRACausalSelfAttention)
    assert isinstance(model.lm_head, LoRALinear)
    assert isinstance(model.transformer.h[0].mlp.proj, LoRALinear)


def test_lora_merge():
    config = Config(
        n_layer=1,
        n_head=2,
        n_embd=8,
        block_size=8,
        vocab_size=8,
        lora_r=8,
        lora_alpha=8,
        lora_dropout=0.1,
        lora_query=True,
        lora_value=True,
        lora_projection=True,
    )
    model = LoRAGPT(config)
    model.train()
    attn_proj = model.transformer.h[0].attn.proj

    initial_weight = attn_proj.linear.weight.clone()
    assert torch.equal(attn_proj.linear.weight, initial_weight)

    # perform an update to the LoRA weights
    mark_only_lora_as_trainable(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    y = model(torch.randint(0, 8, size=(2, 4), dtype=torch.int64))
    y.sum().backward()
    optimizer.step()
    optimizer.zero_grad()
    # the weight remains unchanged (only lora A and B change)
    assert torch.equal(attn_proj.linear.weight, initial_weight)

    # calling merge() multiple times in a row should not merge multiple times
    merge_lora_weights(model)
    assert attn_proj.merged
    weight_after = attn_proj.linear.weight.clone()
    merge_lora_weights(model)
    merge_lora_weights(model)
    assert torch.equal(attn_proj.linear.weight, weight_after)

    # check that `W_after = W_initial + (A x B)`
    delta_w = attn_proj.get_lora_AB()
    torch.testing.assert_close(weight_after, initial_weight + delta_w)


def test_lora_mqa_gqa():
    # MHA
    config = Config(
        n_layer=1,
        n_head=4,
        n_embd=8,
        block_size=1,
        vocab_size=1,
        lora_r=2,
        lora_alpha=8,
        lora_dropout=0.1,
        lora_query=True,
        lora_value=True,
    )
    assert config.n_query_groups == config.n_head
    model = LoRAGPT(config)
    attn = model.transformer.h[0].attn.qkv
    for p in attn.linear.parameters():
        torch.nn.init.zeros_(p)
    torch.nn.init.ones_(attn.lora_B)
    lora_ind = [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23]
    assert attn.linear.weight.shape == (24, 8)
    assert attn.lora_A.shape == (4, 8)
    assert attn.lora_B.shape == (16, 2)
    assert torch.equal(attn.lora_ind, torch.tensor(lora_ind))
    x = torch.randint(0, 8, size=(3, 5, 16), dtype=torch.int64)
    assert attn.zero_pad(x).shape == (3, 5, 24)
    bsz, ctx_len, in_dim = 2, 30, 8
    x_in = torch.randn(bsz, ctx_len, in_dim)
    out = attn(x_in)
    non_lora_ind = list(set(range(24)).difference(lora_ind))
    assert torch.count_nonzero(out[:, :, lora_ind]) == bsz * ctx_len * len(lora_ind)
    assert torch.count_nonzero(out[:, :, non_lora_ind]) == 0

    # MQA
    config.n_query_groups = 1
    model = LoRAGPT(config)
    attn = model.transformer.h[0].attn.qkv
    for p in attn.linear.parameters():
        torch.nn.init.zeros_(p)
    torch.nn.init.ones_(attn.lora_B)
    lora_ind = [0, 1, 2, 3, 4, 5, 6, 7, 10, 11]
    assert attn.linear.weight.shape == (12, 8)
    assert attn.lora_A.shape == (4, 8)
    assert attn.lora_B.shape == (10, 2)
    assert torch.equal(attn.lora_ind, torch.tensor(lora_ind))
    x = torch.randint(0, 8, size=(3, 5, 10), dtype=torch.int64)
    assert attn.zero_pad(x).shape == (3, 5, 12)
    bsz, ctx_len, in_dim = 2, 30, 8
    x_in = torch.randn(bsz, ctx_len, in_dim)
    out = attn(x_in)
    non_lora_ind = list(set(range(12)).difference(lora_ind))
    assert torch.count_nonzero(out[:, :, lora_ind]) == bsz * ctx_len * len(lora_ind)
    assert torch.count_nonzero(out[:, :, non_lora_ind]) == 0

    # GQA
    config.n_query_groups = 2
    model = LoRAGPT(config)
    attn = model.transformer.h[0].attn.qkv
    for p in attn.linear.parameters():
        torch.nn.init.zeros_(p)
    torch.nn.init.ones_(attn.lora_B)
    lora_ind = [0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 15]
    assert attn.linear.weight.shape == (16, 8)
    assert attn.lora_A.shape == (4, 8)
    assert attn.lora_B.shape == (12, 2)
    assert torch.equal(attn.lora_ind, torch.tensor(lora_ind))
    x = torch.randint(0, 8, size=(3, 5, 12), dtype=torch.int64)
    assert attn.zero_pad(x).shape == (3, 5, 16)
    bsz, ctx_len, in_dim = 2, 30, 8
    x_in = torch.randn(bsz, ctx_len, in_dim)
    out = attn(x_in)
    non_lora_ind = list(set(range(16)).difference(lora_ind))
    assert torch.count_nonzero(out[:, :, lora_ind]) == bsz * ctx_len * len(lora_ind)
    assert torch.count_nonzero(out[:, :, non_lora_ind]) == 0


def test_lora_filter(tmp_path):
    fabric = Fabric(devices=1)
    model = LoRAGPT.from_name("pythia-14m", n_layer=3, lora_r=1, lora_query=True, lora_value=True)
    save_path = tmp_path / "model.pth"
    fabric.save(save_path, {"model": model}, filter={"model": lora_filter})
    saved = torch.load(save_path)["model"]

    expected = {
        "transformer.h.1.attn.qkv.lora_B",
        "transformer.h.2.attn.qkv.lora_B",
        "transformer.h.2.attn.qkv.lora_A",
        "transformer.h.1.attn.qkv.lora_A",
        "transformer.h.0.attn.qkv.lora_A",
        "transformer.h.0.attn.qkv.lora_B",
    }
    assert set(saved) == expected


@mock.patch.dict(os.environ, {"LT_ACCELERATOR": "cpu"})
def test_lora_script(tmp_path, fake_checkpoint_dir, monkeypatch, alpaca_path):
    model_config = dict(block_size=128, n_layer=2, n_embd=8, n_head=4, padded_vocab_size=8)
    (fake_checkpoint_dir / "model_config.yaml").write_text(yaml.dump(model_config))
    monkeypatch.setattr(module, "load_checkpoint", Mock())
    monkeypatch.setattr(module, "merge_lora", Mock())

    tokenizer_mock = Mock()
    tokenizer_mock.return_value = tokenizer_mock
    tokenizer_mock.encode = lambda *_, **__: torch.tensor([3, 2, 1])
    monkeypatch.setattr(module, "Tokenizer", tokenizer_mock)

    out_dir = tmp_path / "out"
    stdout = StringIO()
    with redirect_stdout(stdout), mock.patch("sys.argv", ["lora.py", str(fake_checkpoint_dir)]):
        module.setup(
            fake_checkpoint_dir,
            data=Alpaca(
                download_dir=alpaca_path.parent, file_name=alpaca_path.name, val_split_fraction=0.5, num_workers=0
            ),
            out_dir=out_dir,
            precision="32-true",
            train=TrainArgs(global_batch_size=1, save_interval=2, epochs=1, max_steps=6, micro_batch_size=1),
            eval=EvalArgs(interval=2, max_iters=2, max_new_tokens=1),
        )

    out_dir_contents = set(os.listdir(out_dir))
    checkpoint_dirs = {"step-000002", "step-000004", "step-000006", "final"}
    assert checkpoint_dirs.issubset(out_dir_contents)
    assert all((out_dir / p).is_dir() for p in checkpoint_dirs)
    for checkpoint_dir in checkpoint_dirs:
        assert {p.name for p in (out_dir / checkpoint_dir).iterdir()} == {
            "lit_model.pth.lora",
            "model_config.yaml",
            "tokenizer_config.json",
            "tokenizer.json",
            "hyperparameters.yaml",
            "prompt_style.yaml",
        }
    assert (out_dir / "logs" / "csv" / "version_0" / "metrics.csv").is_file()

    logs = stdout.getvalue()
    assert logs.count("(step)") == 6
    assert logs.count("val loss") == 4  # 3 validations + 1 final validation
    assert logs.count("Final evaluation") == 1
    assert "of trainable parameters: 512" in logs


def test_lora_init_when_linear_overridden():
    class MyLinear(torch.nn.Linear):
        def __init__(self, *args, **kwargs):
            # this needs to be implemented to demonstrate the failure
            super().__init__(*args, **kwargs)

    original_linear = torch.nn.Linear
    # Our bnb does this sort of monkey patching
    torch.nn.Linear = MyLinear
    layer = LoRAQKVLinear(1, 1, 1, 1, 1)
    assert isinstance(layer.linear, original_linear)
    torch.nn.Linear = original_linear


@pytest.mark.parametrize(
    ("apply_to", "target_layer_names", "mlp_class_name"),
    (
        ("lora_projection", "transformer.h.0.attn.proj", "GptNeoxMLP"),
        ("lora_mlp", {"transformer.h.0.mlp.fc", "transformer.h.0.mlp.proj"}, "GptNeoxMLP"),
        ("lora_head", "lm_head", "GptNeoxMLP"),
        ("lora_projection", "transformer.h.0.attn.proj", "LLaMAMLP"),
        ("lora_mlp", {"transformer.h.0.mlp.fc_1", "transformer.h.0.mlp.fc_2", "transformer.h.0.mlp.proj"}, "LLaMAMLP"),
        ("lora_head", "lm_head", "LLaMAMLP"),
    ),
)
def test_lora_linear_utilization(apply_to, target_layer_names, mlp_class_name):
    config = Config(
        n_layer=1,
        n_head=4,
        n_embd=8,
        block_size=1,
        vocab_size=1,
        lora_r=2,
        lora_alpha=8,
        lora_dropout=0.1,
        mlp_class_name=mlp_class_name,
        intermediate_size=8 * 3,
        **{apply_to: True},
    )
    model = LoRAGPT(config)
    state_dict = model.state_dict()

    if isinstance(target_layer_names, str):
        target_layer_names = {target_layer_names}
    lora_sublayers = (".lora_A", ".lora_B")

    # check that all the target layers have LoRA weights
    for layer_name in target_layer_names:
        for lora_sublayer in lora_sublayers:
            assert layer_name + lora_sublayer in state_dict

    # check that only target layers have LoRA weights
    lora_params = [k for k in state_dict if k.endswith(lora_sublayers)]
    lora_params = {k[:-7] for k in lora_params}
    assert lora_params == target_layer_names


@torch.inference_mode()
@pytest.mark.parametrize(
    "apply_to", (None, "lora_query", "lora_key", "lora_value", "lora_projection", "lora_mlp", "lora_head")
)
def test_lora_gpt_apply_lora_forward_no_exception(apply_to):
    config = Config(n_layer=1, n_head=4, n_embd=8, block_size=1, vocab_size=1, lora_r=2, lora_alpha=8, lora_dropout=0.1)
    if apply_to:
        setattr(config, apply_to, True)
    input_ids = torch.tensor([[1]])
    model = LoRAGPT(config)
    model.eval()

    model(input_ids)


@torch.inference_mode()
@pytest.mark.parametrize("n_query_groups", (1, 2, 3, 6))
@pytest.mark.parametrize("apply_to", product((False, True), repeat=3))
def test_lora_gpt_query_groups_merge_and_forward_no_exception(n_query_groups, apply_to):
    keys = ("lora_query", "lora_key", "lora_value")
    values = apply_to
    apply_to = dict(zip(keys, values))

    config = Config(
        n_layer=1,
        n_head=6,
        n_embd=12,
        block_size=1,
        vocab_size=1,
        lora_r=2,
        lora_alpha=8,
        lora_dropout=0.1,
        n_query_groups=n_query_groups,
        **apply_to,
    )
    model = LoRAGPT(config)
    merge_lora_weights(model)
    input_ids = torch.tensor([[1]])
    model(input_ids)


@torch.inference_mode()
@pytest.mark.parametrize("head_size", (1, 2, 4))
@pytest.mark.parametrize("n_head", (1, 2, 3, 6, 12))
@pytest.mark.parametrize(
    "enable_lora",
    [
        (False, False, True),
        (False, True, False),
        (False, True, True),
        (True, False, False),
        (True, False, True),
        (True, True, False),
        (True, True, True),
    ],
)
def test_lora_qkv_linear_compare_conv1d(head_size, n_head, enable_lora):
    C = 12
    layer = LoRAQKVLinear(
        C, 3 * C, head_size=head_size, n_head=n_head, n_query_groups=n_head, r=2, enable_lora=enable_lora
    )
    x = torch.randn((1, 1, C))
    a = F.linear(x, layer.lora_A).transpose(-2, -1)  # after_A
    b = layer.lora_B.data.unsqueeze(-1)

    # original PyTorch conv1d function output
    conv1d_pytorch = F.conv1d(a, b, groups=sum(layer.enable_lora))

    # custom conv1d
    conv1d_custom = layer.conv1d(a, b)

    # custom conv1d forced to split, apply and concat tensors
    layer.n_head = layer.n_query_groups + 1
    conv1d_custom_forced = layer.conv1d(a, b)

    assert torch.allclose(conv1d_pytorch, conv1d_custom)
    assert torch.allclose(conv1d_pytorch, conv1d_custom_forced)


@pytest.mark.parametrize(("rank", "expected_merged"), ((0, False), (1, True)))
def test_lora_linear_weights_merged_status(rank, expected_merged):
    layer = LoRALinear(10, 10, r=rank)
    assert not layer.merged
    layer.merge()
    assert layer.merged == expected_merged


@pytest.mark.parametrize(
    ("rank", "enable_lora", "expected_merged"),
    ((0, True, False), (1, True, True), (0, False, False), (1, False, False)),
)
def test_lora_qkv_linear_weights_merged_status(rank, enable_lora, expected_merged):
    C = 10
    layer = LoRAQKVLinear(C, 3 * C, head_size=5, n_head=2, n_query_groups=2, r=rank, enable_lora=enable_lora)
    assert not layer.merged
    layer.merge()
    assert layer.merged == expected_merged


@_RunIf(min_cuda_gpus=1)
def test_lora_merge_with_bitsandbytes():
    if not _BITSANDBYTES_AVAILABLE:
        pytest.skip("BNB not available")
    import bitsandbytes as bnb

    config = Config(
        n_layer=1,
        n_head=2,
        n_embd=8,
        block_size=8,
        vocab_size=8,
        lora_r=8,
        lora_alpha=8,
        lora_dropout=0.1,
        lora_query=True,
        lora_value=True,
        lora_projection=True,
    )
    fabric = Fabric(devices=1, plugins=BitsandbytesPrecision("nf4", dtype=torch.bfloat16, ignore_modules={"lm_head"}))
    model = LoRAGPT(config)
    mark_only_lora_as_trainable(model)

    from bitsandbytes.optim import PagedAdamW

    optimizer = PagedAdamW(model.parameters(), lr=1.0)
    model, optimizer = fabric.setup(model, optimizer)

    model.train()

    attn_proj = model.transformer.h[0].attn.proj
    initial_weight = attn_proj.linear.weight.clone()
    initial_weight_kwargs = attn_proj.linear.weight.__dict__

    # this was skipped
    assert model.lm_head.linear.weight.dtype is torch.float32
    assert attn_proj.linear.weight.dtype is torch.uint8

    # perform an update to the LoRA weights
    y = model(torch.randint(0, 8, size=(2, 4), dtype=torch.int64, device=fabric.device))
    loss = y.sum()
    fabric.backward(loss)
    optimizer.step()
    optimizer.zero_grad()
    # the weight remains unchanged (only lora A and B change)
    assert torch.equal(attn_proj.linear.weight, initial_weight)

    # calling merge() multiple times in a row should not merge multiple times
    merge_lora_weights(model)
    assert attn_proj.merged
    weight_after = attn_proj.linear.weight.clone()
    merge_lora_weights(model)
    merge_lora_weights(model)
    assert torch.equal(attn_proj.linear.weight, weight_after)

    # check that `W_after = W_initial + (A x B)`
    delta_w = attn_proj.get_lora_AB()
    # dequantize initial weight and sum with delta_w
    initial_weight_data = (
        bnb.functional.dequantize_4bit(initial_weight.data, initial_weight_kwargs["quant_state"]) + delta_w
    )
    # quantize again
    initial_weight_data = bnb.nn.Params4bit(
        initial_weight_data.to("cpu"), requires_grad=False, **initial_weight_kwargs
    ).to(initial_weight.device)
    torch.testing.assert_close(weight_after, initial_weight_data)


def test_lora_gpt_init_weights():
    config = Config(n_layer=1, n_head=6, n_embd=12, block_size=1, vocab_size=1, lora_r=2, lora_alpha=8, lora_head=True)
    model = LoRAGPT(config)
    param = model.lm_head.lora_B.data

    assert (param == 0).all()
    torch.nn.init.constant_(param, 1.23)
    assert (param != 0).any()
    model.apply(model._init_weights)
    assert (param == 0).all()


@pytest.mark.parametrize("name", [c["name"] for c in config_module.configs])
def test_base_model_can_be_lora_loaded(name):
    kwargs = {"n_layer": 2, "n_head": 8, "n_query_groups": 4, "n_embd": 16, "padded_vocab_size": 32}
    base_model = BaseGPT.from_name(name, **kwargs)
    base_model_state_dict = base_model.state_dict()
    lora_model = LoRAGPT.from_name(
        name,
        **kwargs,
        lora_r=1,
        lora_query=True,
        lora_key=True,
        lora_value=True,
        lora_projection=True,
        lora_mlp=True,
        lora_head=True,
    )
    keys = lora_model.load_state_dict(base_model_state_dict, strict=False)
    assert not keys.unexpected_keys
    for k in keys.missing_keys:
        assert lora_filter(k, None)


@_RunIf(dynamo=True)
@torch.inference_mode()
def test_lora_compile():
    model = LoRAGPT.from_name(
        "pythia-14m",
        n_layer=3,
        lora_r=8,
        lora_alpha=8,
        lora_dropout=0.1,
        lora_query=True,
        lora_key=True,
        lora_value=True,
        lora_projection=True,
        lora_mlp=True,
        lora_head=True,
    )
    x = torch.randint(model.config.vocab_size, size=(2, model.config.block_size), dtype=torch.int64)

    explanation = torch._dynamo.explain(model)(x)
    assert isinstance(explanation, debugging.ExplainOutput)
    assert explanation.graph_count == 1
    assert explanation.graph_break_count == 0

    model = LoRAGPT(model.config)
    model.set_kv_cache(2)
    input_pos = torch.arange(model.config.block_size)
    explanation = torch._dynamo.explain(model)(x, input_pos)
    assert isinstance(explanation, debugging.ExplainOutput)
    assert explanation.graph_count == 1
    assert explanation.graph_break_count == 0


@torch.inference_mode()
def test_against_hf_mixtral():
    device = torch.device("cpu")
    dtype = torch.float32
    ours_config = Config.from_name(
        "Mixtral-8x7B-Instruct-v0.1",
        padded_vocab_size=10000,
        n_layer=2,
        n_embd=32,
        n_head=8,
        n_query_groups=2,
        intermediate_size=86,
        n_expert=4,
        lora_r=1,
        lora_key=True,
        lora_value=True,
    )
    T = 5
    theirs_config = MixtralConfig(
        vocab_size=ours_config.padded_vocab_size,
        hidden_size=ours_config.n_embd,
        num_attention_heads=ours_config.n_head,
        num_hidden_layers=ours_config.n_layer,
        intermediate_size=ours_config.intermediate_size,
        max_position_embeddings=T,
        rms_norm_eps=ours_config.norm_eps,
        num_key_value_heads=ours_config.n_query_groups,
        rope_theta=ours_config.rope_base,
        num_local_experts=ours_config.n_expert,
    )
    assert ours_config.intermediate_size == theirs_config.intermediate_size

    theirs_model = MixtralForCausalLM(theirs_config).to(device)
    theirs_state_dict = theirs_model.state_dict()
    state_dict = {}
    copy_weights_hf_llama(ours_config, {}, state_dict, theirs_state_dict)
    ours_model = LoRAGPT(ours_config).to(device)
    keys = ours_model.load_state_dict(state_dict, strict=False)
    assert not keys.unexpected_keys
    for k in keys.missing_keys:
        assert lora_filter(k, None)

    # test end to end
    x = torch.tensor([[9856, 23, 491, 1536, 304], [23, 345, 65, 123, 321]], dtype=torch.int32, device=device)
    assert x.size(1) == T
    ours_y = ours_model(x)
    theirs_y = theirs_model(x)["logits"].to(dtype)  # HF converts logits to float
    torch.testing.assert_close(ours_y, theirs_y)


@torch.inference_mode()
@pytest.mark.parametrize("model_name", ["gemma-2b", "gemma-7b"])
def test_against_hf_gemma(model_name):
    device = torch.device("cpu")
    dtype = torch.float32
    T = 5
    ours_config = Config.from_name(
        model_name,
        n_layer=2,
        n_head=16,
        n_embd=32,
        head_size=4,
        intermediate_size=86,
        lora_r=1,
        lora_query=True,
        lora_key=True,
        lora_value=True,
    )
    theirs_config = GemmaConfig(
        vocab_size=ours_config.padded_vocab_size,
        hidden_size=ours_config.n_embd,
        head_dim=ours_config.head_size,
        num_attention_heads=ours_config.n_head,
        num_hidden_layers=ours_config.n_layer,
        intermediate_size=ours_config.intermediate_size,
        max_position_embeddings=T,
        rms_norm_eps=ours_config.norm_eps,
        num_key_value_heads=ours_config.n_query_groups,
        rope_theta=ours_config.rope_base,
        attention_bias=ours_config.bias,
        tie_word_embeddings=True,
        hidden_act="gelu_pytorch_tanh",
    )
    assert ours_config.intermediate_size == theirs_config.intermediate_size

    theirs_model = GemmaForCausalLM(theirs_config).to(device)
    theirs_state_dict = theirs_model.state_dict()
    # Gemma weights are shipped without `lm_head.weight`
    theirs_state_dict.pop("lm_head.weight")
    state_dict = {}
    copy_weights_hf_llama(ours_config, {}, state_dict, theirs_state_dict)
    ours_model = LoRAGPT(ours_config).to(device)
    keys = ours_model.load_state_dict(state_dict, strict=False)
    assert not keys.unexpected_keys
    for k in keys.missing_keys:
        assert lora_filter(k, None)

    # test end to end
    x = torch.tensor([[9856, 23, 491, 1536, 304]], dtype=torch.int32, device=device)
    assert x.size(1) == T
    ours_y = ours_model(x)
    theirs_y = theirs_model(x)["logits"].to(dtype)  # HF converts logits to float
    torch.testing.assert_close(ours_y, theirs_y)


@torch.inference_mode()
@pytest.mark.parametrize("model_name", ("gemma-2-9b", "gemma-2-27b"))
def test_against_original_gemma_2(model_name):
    device = torch.device("cpu")
    dtype = torch.float32
    T = 20
    ours_config = Config.from_name(
        model_name,
        block_size=T,
        sliding_window_size=T // 2,
        n_layer=2,
        n_head=16,
        n_embd=32,
        intermediate_size=86,
    )
    theirs_config = Gemma2Config(
        vocab_size=ours_config.padded_vocab_size,
        hidden_size=ours_config.n_embd,
        head_dim=ours_config.head_size,
        num_attention_heads=ours_config.n_head,
        num_hidden_layers=ours_config.n_layer,
        intermediate_size=ours_config.intermediate_size,
        max_position_embeddings=ours_config.block_size,
        sliding_window=ours_config.sliding_window_size,
        rms_norm_eps=ours_config.norm_eps,
        num_key_value_heads=ours_config.n_query_groups,
        rope_theta=ours_config.rope_base,
        attention_bias=ours_config.bias,
        tie_word_embeddings=True,
        hidden_act="gelu_pytorch_tanh",
        attn_logit_softcapping=ours_config.attention_logit_softcapping,
        final_logit_softcapping=ours_config.final_logit_softcapping,
        initializer_range=1.0,  # to make the affect of attention_logit_softcapping more prominent
        attn_implementation="eager",
        query_pre_attn_scalar=ours_config.attention_scores_scalar,
    )
    assert ours_config.intermediate_size == theirs_config.intermediate_size

    theirs_model = Gemma2ForCausalLM(theirs_config).to(device)
    theirs_state_dict = theirs_model.state_dict()
    # Gemma weights are shipped without `lm_head.weight`
    theirs_state_dict.pop("lm_head.weight")
    state_dict = {}
    copy_weights_gemma_2({}, state_dict, theirs_state_dict)
    ours_model = LoRAGPT(ours_config).to(device)
    ours_model.load_state_dict(state_dict)

    # test end to end
    x = torch.randint(low=0, high=ours_config.padded_vocab_size, size=(T,), device=device).unsqueeze(0)
    assert x.size(1) == T
    ours_y = ours_model(x)
    theirs_y = theirs_model(x)["logits"].to(dtype)  # HF converts logits to float
    torch.testing.assert_close(ours_y, theirs_y, rtol=3e-5, atol=3e-5)


@torch.inference_mode()
@pytest.mark.parametrize("model_name", ("gemma-3-1b-it", "gemma-3-4b-it", "gemma-3-12b-it", "gemma-3-27b-it"))
def test_against_original_gemma_3(model_name):
    device = torch.device("cpu")
    dtype = torch.float32
    T = 20
    ours_config = Config.from_name(
        model_name,
        block_size=T,
        sliding_window_size=T // 2,
        n_layer=2,
        n_head=16,
        n_embd=32,
        intermediate_size=86,
    )
    theirs_config = Gemma3TextConfig(
        vocab_size=ours_config.padded_vocab_size,
        hidden_size=ours_config.n_embd,
        head_dim=ours_config.head_size,
        num_attention_heads=ours_config.n_head,
        num_hidden_layers=ours_config.n_layer,
        intermediate_size=ours_config.intermediate_size,
        max_position_embeddings=ours_config.block_size,
        sliding_window=ours_config.sliding_window_size,
        rms_norm_eps=ours_config.norm_eps,
        num_key_value_heads=ours_config.n_query_groups,
        rope_theta=ours_config.rope_base,
        attention_bias=ours_config.bias,
        tie_word_embeddings=True,
        hidden_act="gelu_pytorch_tanh",
        attn_logit_softcapping=ours_config.attention_logit_softcapping,
        final_logit_softcapping=ours_config.final_logit_softcapping,
        initializer_range=1.0,  # to make the affect of attention_logit_softcapping more prominent
        attn_implementation="eager",
        query_pre_attn_scalar=ours_config.attention_scores_scalar,
    )
    assert ours_config.intermediate_size == theirs_config.intermediate_size

    theirs_model = Gemma3ForCausalLM(theirs_config).to(device)
    theirs_state_dict = theirs_model.state_dict()
    # Gemma weights are shipped without `lm_head.weight`
    theirs_state_dict.pop("lm_head.weight")
    state_dict = {}
    copy_weights_gemma_3({}, state_dict, theirs_state_dict)
    ours_model = LoRAGPT(ours_config).to(device)
    ours_model.load_state_dict(state_dict)

    # test end to end
    x = torch.randint(low=0, high=ours_config.padded_vocab_size, size=(T,), device=device).unsqueeze(0)
    assert x.size(1) == T
    ours_y = ours_model(x)
    theirs_y = theirs_model(x)["logits"].to(dtype)  # HF converts logits to float
    torch.testing.assert_close(ours_y, theirs_y, rtol=3e-5, atol=3e-5)


@_RunIf(min_cuda_gpus=1)
def test_lora_bitsandbytes(monkeypatch, tmp_path, fake_checkpoint_dir, alpaca_path):
    if not _BITSANDBYTES_AVAILABLE:
        pytest.skip("BNB not available")

    from bitsandbytes.optim import PagedAdamW

    model_config = dict(
        block_size=128,
        n_layer=2,
        n_embd=8,
        n_head=4,
        padded_vocab_size=8,
        bias=True,
        lora_r=8,
        lora_alpha=8,
        lora_dropout=0.1,
        lora_query=True,
        lora_value=True,
        lora_projection=True,
    )
    (fake_checkpoint_dir / "model_config.yaml").write_text(yaml.dump(model_config))

    tokenizer_mock = Mock()
    tokenizer_mock.return_value = tokenizer_mock
    tokenizer_mock.encode = lambda *_, **__: torch.tensor([3, 2, 1])
    monkeypatch.setattr(module, "Tokenizer", tokenizer_mock)

    monkeypatch.setattr(module, "load_checkpoint", Mock())
    monkeypatch.setattr(module, "merge_lora", Mock())
    train_mock = Mock()
    train_mock.return_value = {
        "raw_tokens": 1000,
        "raw_tokens_plus_prompt_template": 1100,
        "raw_tokens_plus_prompt_template_and_padding": 1200,
    }
    monkeypatch.setattr(module, "fit", train_mock)

    stdout = StringIO()
    with redirect_stdout(stdout), mock.patch("sys.argv", ["full.py", str(fake_checkpoint_dir)]):
        module.setup(
            fake_checkpoint_dir,
            data=Alpaca(
                download_dir=alpaca_path.parent, file_name=alpaca_path.name, val_split_fraction=0.5, num_workers=0
            ),
            out_dir=tmp_path,
            precision="16-true",
            quantize="bnb.nf4-dq",
        )

    _, kwargs = train_mock.call_args
    fabric = kwargs["fabric"]
    model = kwargs["model"]
    optimizer = kwargs["optimizer"]
    model.transformer.wte = model.transformer.wte.half()
    assert isinstance(fabric.strategy.precision, BitsandbytesPrecision)
    assert isinstance(optimizer, _FabricOptimizer)
    assert isinstance(optimizer._optimizer, PagedAdamW)

    dtype_to_name = {"torch.uint8": set(), "torch.float16": set()}
    for name, layer in model.named_parameters():
        name = name[len("_forward_module.") :]
        dtype_to_name[str(layer.dtype)].add(name)
    assert dtype_to_name == {
        "torch.uint8": {
            "transformer.h.0.attn.qkv.linear.weight",
            "transformer.h.0.attn.proj.linear.weight",
            "transformer.h.0.mlp.fc.linear.weight",
            "transformer.h.1.mlp.proj.linear.weight",
            "transformer.h.0.mlp.proj.linear.weight",
            "transformer.h.1.attn.qkv.linear.weight",
            "lm_head.linear.weight",
            "transformer.h.1.attn.proj.linear.weight",
            "transformer.h.1.mlp.fc.linear.weight",
        },
        "torch.float16": {
            "transformer.h.0.attn.qkv.lora_B",
            "transformer.h.0.norm_2.weight",
            "transformer.wte.weight",
            "transformer.wte.norm.weight",
            "transformer.wte.norm.bias",
            "transformer.h.1.mlp.fc.linear.bias",
            "transformer.ln_f.bias",
            "transformer.h.1.attn.qkv.lora_B",
            "transformer.h.1.attn.proj.linear.bias",
            "transformer.h.1.norm_1.weight",
            "transformer.h.1.attn.qkv.linear.bias",
            "transformer.h.1.attn.qkv.lora_A",
            "transformer.h.1.norm_1.bias",
            "transformer.h.1.norm_2.bias",
            "transformer.h.0.attn.proj.linear.bias",
            "transformer.h.0.norm_1.bias",
            "transformer.h.0.mlp.proj.linear.bias",
            "transformer.h.0.mlp.fc.linear.bias",
            "transformer.h.0.norm_2.bias",
            "transformer.ln_f.weight",
            "transformer.h.0.attn.qkv.lora_A",
            "transformer.h.1.norm_2.weight",
            "transformer.h.1.mlp.proj.linear.bias",
            "transformer.h.0.norm_1.weight",
            "transformer.h.0.attn.qkv.linear.bias",
        },
    }

    assert {p.name for p in tmp_path.rglob("*.lora")} == {"lit_model.pth.lora"}
    state_dict = torch.load(tmp_path / "final" / "lit_model.pth.lora")
    assert len(state_dict) == 1
    dtype_to_name = {"torch.float16": set()}
    for name, layer in state_dict["model"].items():
        dtype_to_name[str(layer.dtype)].add(name)
    assert dtype_to_name == {
        "torch.float16": {
            "transformer.h.1.attn.qkv.lora_A",
            "transformer.h.0.attn.qkv.lora_A",
            "transformer.h.0.attn.qkv.lora_B",
            "transformer.h.1.attn.qkv.lora_B",
        }
    }

    logs = stdout.getvalue()
    assert "of trainable parameters: 512" in logs
    assert "of non-trainable parameters: 1,888" in logs


@_RunIf(standalone=True, min_cuda_gpus=2)
def test_lora_model_fsdp_init():
    config = Config(
        n_layer=1,
        n_head=2,
        n_embd=8,
        block_size=8,
        vocab_size=8,
        lora_r=8,
        lora_alpha=8,
        lora_dropout=0.1,
        lora_query=True,
        lora_value=False,
        lora_projection=True,
    )
    fabric = Fabric(devices=2, strategy="fsdp", precision="16-true")
    fabric.launch()
    with fabric.init_module(empty_init=True):
        model = LoRAGPT(config)
    x = torch.randint(0, config.padded_vocab_size, size=(2, config.block_size), dtype=torch.int64, device=fabric.device)
    model = fabric.setup(model)
    y = model(x)
    assert y.shape == torch.Size([2, 8, 512])

    # verify that all the parameters, buffers and other attributes aren't on `meta` device
    for m in model.modules():
        for p_name, parameter in m.named_parameters():
            assert not parameter.is_meta, f"Parameter `{p_name}` isn't materialized."
        for b_name, buffer in m._buffers.items():
            assert not buffer.is_meta, f"Buffer `{b_name}` isn't materialized."
        for attr_name, attr_value in m.__dict__.items():
            if isinstance(attr_value, torch.Tensor):
                assert not attr_value.is_meta, f"Attribute `{attr_name}` isn't materialized."


def test_zero_pad_cpu_and_mocked_mps():
    head_size = 64
    n_head = 12
    n_query_groups = 3
    in_features = 128
    kv_embed_dim = in_features // (n_head // n_query_groups)
    out_features = in_features + 2 * kv_embed_dim
    enable_lora = [True, False, True]
    r = 4

    model = LoRAQKVLinear(
        in_features=in_features,
        out_features=out_features,
        head_size=head_size,
        n_head=n_head,
        n_query_groups=n_query_groups,
        r=r,
        enable_lora=enable_lora,
    )

    batch_size = 64
    seq_len = 64
    embed_dim = 160
    x = torch.randn(batch_size, seq_len, embed_dim)

    result_cpu = model.zero_pad(x)

    with mock.patch("torch.backends.mps.is_available", return_value=True):
        with mock.patch("torch.Tensor.device", new_callable=mock.PropertyMock) as mock_device:
            mock_device.return_value = torch.device("mps")

            result_mps = model.zero_pad(x)

            assert result_cpu.shape == result_mps.shape, "Shape mismatch between CPU and MPS"
            assert torch.allclose(result_cpu, result_mps), "Tensor values mismatch between CPU and MPS"


def test_load_legacy_state_dict():
    """Check that a legacy state dict (with an interleaved placement in QKV matrix) can be loaded into a model with CausalSelfAttention layers."""
    config = Config(
        n_embd=32, n_head=4, head_size=8, n_query_groups=4, bias=True, lora_r=8, lora_alpha=16, lora_dropout=0.1
    )

    attention_1 = CausalSelfAttention(config=config, block_idx=0)

    # make weights to be as-like in a legacy checkpoint, with `attn.attn.weight` instead of `attn.qkv.weight`
    # and make them interleaved
    state_dict = deepcopy(attention_1.state_dict())
    state_dict["attn.linear.weight"] = make_qkv_interleaved(state_dict.pop("qkv.linear.weight"), config)
    state_dict["attn.linear.bias"] = make_qkv_interleaved(state_dict.pop("qkv.linear.bias"), config)

    attention_2 = CausalSelfAttention(config=config, block_idx=0)
    attention_2.load_state_dict(state_dict)
