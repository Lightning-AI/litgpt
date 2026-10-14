import copy
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
import yaml
from lightning import Fabric

from litgpt import GPT, Config
from litgpt.generate.tp import _tensor_parallel_qkv, tensor_parallel, tensor_parallel_linear
from litgpt.scripts.download import download_from_hub
from litgpt.utils import _RunIf

from .utils import find_forward_hooks


def test_tensor_parallel_linear():
    fabric = Mock()
    fabric.world_size = 4
    fabric.global_rank = 2

    def get_linear(bias=True):
        linear = torch.nn.Linear(8, 8, bias=bias)
        linear.weight.data = torch.arange(64, dtype=torch.float32).reshape(8, 8)
        if bias:
            linear.bias.data = torch.arange(8, dtype=torch.float32)
        return linear

    linear = get_linear()
    tensor_parallel_linear(fabric, linear, "colwise")
    expected = torch.arange(32, 48, dtype=torch.float32).reshape(2, 8)
    torch.testing.assert_close(linear.weight, expected)
    expected = torch.arange(4, 6, dtype=torch.float32)
    torch.testing.assert_close(linear.bias, expected)

    linear = get_linear(bias=False)
    tensor_parallel_linear(fabric, linear, "rowwise")
    expected = torch.arange(4, 62, 8, dtype=torch.float32).reshape(8, 1)
    expected = torch.cat([expected, expected + 1], dim=1)
    torch.testing.assert_close(linear.weight, expected)
    assert linear.bias is None

    # Row-parallel bias must be kept only on rank 0 (added before sum all-reduce).
    linear = get_linear(bias=True)
    tensor_parallel_linear(fabric, linear, "rowwise")
    assert linear.bias is None

    fabric_rank0 = Mock()
    fabric_rank0.world_size = 4
    fabric_rank0.global_rank = 0
    linear = get_linear(bias=True)
    original_bias = linear.bias.data.clone()
    tensor_parallel_linear(fabric_rank0, linear, "rowwise")
    torch.testing.assert_close(linear.bias, original_bias)


def test_tensor_parallel_qkv_splits_heads_not_qkv_blocks():
    """Fused [Q|K|V] must be sharded per Q/K/V, not as one contiguous out-feature chunk."""
    fabric = Mock()
    fabric.world_size = 2
    fabric.global_rank = 0

    config = Config(
        n_embd=8,
        n_head=4,
        n_query_groups=2,
        head_size=2,
        padded_vocab_size=16,
        vocab_size=16,
        block_size=16,
        n_layer=1,
        bias=True,
        intermediate_size=16,
    )
    q_size = config.n_head * config.head_size  # 8
    kv_size = config.n_query_groups * config.head_size  # 4
    out = q_size + 2 * kv_size  # 16
    qkv = torch.nn.Linear(config.n_embd, out, bias=True)
    # Distinct blocks so we can verify each of Q/K/V stayed intact after sharding.
    with torch.no_grad():
        qkv.weight.zero_()
        qkv.bias.zero_()
        qkv.weight[:q_size].fill_(1.0)
        qkv.weight[q_size : q_size + kv_size].fill_(2.0)
        qkv.weight[q_size + kv_size :].fill_(3.0)
        qkv.bias[:q_size].fill_(10.0)
        qkv.bias[q_size : q_size + kv_size].fill_(20.0)
        qkv.bias[q_size + kv_size :].fill_(30.0)

    _tensor_parallel_qkv(fabric, qkv, config)

    local_q = config.n_head // 2 * config.head_size
    local_kv = config.n_query_groups // 2 * config.head_size
    assert qkv.out_features == local_q + 2 * local_kv
    assert torch.all(qkv.weight[:local_q] == 1.0)
    assert torch.all(qkv.weight[local_q : local_q + local_kv] == 2.0)
    assert torch.all(qkv.weight[local_q + local_kv :] == 3.0)
    assert torch.all(qkv.bias[:local_q] == 10.0)
    assert torch.all(qkv.bias[local_q : local_q + local_kv] == 20.0)
    assert torch.all(qkv.bias[local_q + local_kv :] == 30.0)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        (
            "Llama-2-70b-hf",
            {
                "transformer.h.0.attn": [("forward_hook", "all_reduce_output", (8,), {})],
                "transformer.h.0.mlp": [("forward_hook", "all_reduce_output", (8,), {})],
                "transformer.h.1.attn": [("forward_hook", "all_reduce_output", (8,), {})],
                "transformer.h.1.mlp": [("forward_hook", "all_reduce_output", (8,), {})],
                "transformer.h.2.attn": [("forward_hook", "all_reduce_output", (8,), {})],
                "transformer.h.2.mlp": [("forward_hook", "all_reduce_output", (8,), {})],
            },
        ),
        (
            "falcon-180B",
            {
                "transformer.h.0.attn": [("forward_hook", "all_reduce_output", (8,), {})],
                "transformer.h.0.mlp": [("forward_hook", "all_reduce_output", (8,), {})],
                "transformer.h.1.attn": [("forward_hook", "all_reduce_output", (8,), {})],
                "transformer.h.1.mlp": [("forward_hook", "all_reduce_output", (8,), {})],
                "transformer.h.2.attn": [("forward_hook", "all_reduce_output", (8,), {})],
                "transformer.h.2.mlp": [("forward_hook", "all_reduce_output", (8,), {})],
            },
        ),
        (
            "Mixtral-8x7B-v0.1",
            {
                "transformer.h.0.attn": [("forward_hook", "all_reduce_output", (8,), {})],
                "transformer.h.0.mlp.experts.0": [("forward_hook", "all_reduce_output", (8,), {})],
                "transformer.h.0.mlp.experts.1": [("forward_hook", "all_reduce_output", (8,), {})],
                "transformer.h.1.attn": [("forward_hook", "all_reduce_output", (8,), {})],
                "transformer.h.1.mlp.experts.0": [("forward_hook", "all_reduce_output", (8,), {})],
                "transformer.h.1.mlp.experts.1": [("forward_hook", "all_reduce_output", (8,), {})],
                "transformer.h.2.attn": [("forward_hook", "all_reduce_output", (8,), {})],
                "transformer.h.2.mlp.experts.0": [("forward_hook", "all_reduce_output", (8,), {})],
                "transformer.h.2.mlp.experts.1": [("forward_hook", "all_reduce_output", (8,), {})],
            },
        ),
        (
            "Gemma-7b",
            {
                "transformer.h.0.attn": [("forward_hook", "all_reduce_output", (8,), {})],
                "transformer.h.0.mlp": [("forward_hook", "all_reduce_output", (8,), {})],
                "transformer.h.1.attn": [("forward_hook", "all_reduce_output", (8,), {})],
                "transformer.h.1.mlp": [("forward_hook", "all_reduce_output", (8,), {})],
                "transformer.h.2.attn": [("forward_hook", "all_reduce_output", (8,), {})],
                "transformer.h.2.mlp": [("forward_hook", "all_reduce_output", (8,), {})],
            },
        ),
    ],
)
def test_tensor_parallel_hooks_and_config(name, expected):
    fabric = Mock()
    fabric.world_size = 8
    fabric.global_rank = 1

    with torch.device("meta"):
        model = GPT.from_name(name, n_layer=3, n_expert=2)
    config = replace(model.config)  # make a copy

    model = tensor_parallel(fabric, model)

    hooks = find_forward_hooks(model)
    assert hooks == expected

    # n_embd stays full-width (needed for embeddings / Gemma scale_embeddings); heads are sharded.
    assert model.config.n_embd == config.n_embd
    assert model.config.n_head * 8 == config.n_head
    assert model.config.n_query_groups * 8 == config.n_query_groups


def _tp_parity_worker(mlp_class_name: str, scale_embeddings: bool, bias: bool):
    """Compare TP (world_size=2) logits to a non-TP reference on CPU gloo."""
    import torch.distributed as dist

    fabric = Fabric(accelerator="cpu", devices=2, strategy="ddp")
    fabric.launch()

    cfg = Config(
        n_embd=64,
        intermediate_size=128,
        bias=bias,
        padded_vocab_size=128,
        vocab_size=128,
        n_layer=2,
        n_head=4,
        n_query_groups=4,
        block_size=32,
        mlp_class_name=mlp_class_name,
        parallel_residual=(mlp_class_name == "GptNeoxMLP"),
        norm_class_name="RMSNorm" if mlp_class_name != "GptNeoxMLP" else "LayerNorm",
        scale_embeddings=scale_embeddings,
        gelu_approximate="tanh" if mlp_class_name == "GemmaMLP" else "none",
    )

    torch.manual_seed(0)
    ref_model = GPT(cfg)
    ref_model.apply(ref_model._init_weights)
    ref_model.eval()

    state = ref_model.state_dict()
    for tensor in state.values():
        dist.broadcast(tensor, src=0)
    ref_model.load_state_dict(state)

    x = torch.randint(0, cfg.padded_vocab_size, (1, 8), device=fabric.device)
    dist.broadcast(x, src=0)

    with torch.no_grad():
        ref_out = ref_model(x)

    tp_model = copy.deepcopy(ref_model)
    tp_model = tensor_parallel(fabric, tp_model)
    tp_model.eval()
    with torch.no_grad():
        tp_out = tp_model(x)

    if fabric.global_rank == 0:
        torch.testing.assert_close(tp_out, ref_out, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize(
    ("mlp_class_name", "scale_embeddings", "bias"),
    [
        ("GptNeoxMLP", False, True),  # issue #1663: bias duplicated under row-parallel + all-reduce
        ("LLaMAMLP", False, False),
        ("GemmaMLP", True, False),  # issue #1663: n_embd sharding broke scale_embeddings
    ],
)
def test_tensor_parallel_logit_parity(mlp_class_name, scale_embeddings, bias):
    _tp_parity_worker(mlp_class_name, scale_embeddings, bias)


root = Path(__file__).parent.parent.resolve()


@_RunIf(min_cuda_gpus=2)
def test_tp(tmp_path):
    # download the tokenizer
    download_from_hub(repo_id="EleutherAI/pythia-14m", tokenizer_only=True, checkpoint_dir=tmp_path)
    checkpoint_dir = tmp_path / "EleutherAI/pythia-14m"
    # save the config
    config = Config.from_name("pythia-14m")
    (checkpoint_dir / "model_config.yaml").write_text(yaml.dump(asdict(config)))
    # create a state dict to load from
    torch.save(GPT(config).state_dict(), checkpoint_dir / "lit_model.pth")

    args = [
        str(checkpoint_dir),
        "--num_samples=1",
        "--max_new_tokens=10",
        "--precision=16-true",
        "--temperature=0.0",
    ]
    env = {"CUDA_VISIBLE_DEVICES": "0,1"}
    tp_stdout = subprocess.check_output(
        [sys.executable, "-m", "litgpt", "generate_tp", *args], env=env, cwd=root
    ).decode()

    # there is some unaccounted randomness so cannot compare the output with that of `generate/base.py`
    assert "What food do llamas eat?" in tp_stdout


def test_cli():
    args = ["litgpt", "generate_tp", "-h"]
    output = subprocess.check_output(args)
    output = str(output.decode())
    assert "Generation script that uses tensor parallelism" in output
