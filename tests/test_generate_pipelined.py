import itertools
import json
import subprocess
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import pytest
import torch
from conftest import RunIf
from lightning import Fabric


@pytest.mark.parametrize(
    ("n_layer", "devices", "expected"),
    [
        (6, 2, {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1}),
        (6, 3, {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2}),
        (6, 1, {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}),
    ],
)
def test_layer_to_device(n_layer, devices, expected):
    from generate.pipelined import layer_to_device
    from lit_gpt.model import GPT, Block

    with torch.device("meta"):
        model = GPT.from_name("pythia-14m", n_layer=n_layer)

    actual = layer_to_device(model, Block, chunk_size=n_layer // devices)
    assert actual == expected


def path_to_device(model):
    return {k: str(v.device) for k, v in itertools.chain(model.named_parameters(), model.named_buffers())}


def test_materialize_meta_tensors():
    from generate.pipelined import materialize_meta_tensors
    from lit_gpt.model import GPT

    with torch.device("meta"):
        model = GPT.from_name("pythia-14m", n_layer=2)

    materialize_meta_tensors(model.transformer.h[1], torch.device("cpu"))
    assert path_to_device(model) == {
        "cos": "meta",
        "lm_head.weight": "meta",
        "sin": "meta",
        "transformer.h.0.attn.attn.bias": "meta",
        "transformer.h.0.attn.attn.weight": "meta",
        "transformer.h.0.attn.proj.bias": "meta",
        "transformer.h.0.attn.proj.weight": "meta",
        "transformer.h.0.mlp.fc.bias": "meta",
        "transformer.h.0.mlp.fc.weight": "meta",
        "transformer.h.0.mlp.proj.bias": "meta",
        "transformer.h.0.mlp.proj.weight": "meta",
        "transformer.h.0.norm_1.bias": "meta",
        "transformer.h.0.norm_1.weight": "meta",
        "transformer.h.0.norm_2.bias": "meta",
        "transformer.h.0.norm_2.weight": "meta",
        "transformer.h.1.attn.attn.bias": "cpu",
        "transformer.h.1.attn.attn.weight": "cpu",
        "transformer.h.1.attn.proj.bias": "cpu",
        "transformer.h.1.attn.proj.weight": "cpu",
        "transformer.h.1.mlp.fc.bias": "cpu",
        "transformer.h.1.mlp.fc.weight": "cpu",
        "transformer.h.1.mlp.proj.bias": "cpu",
        "transformer.h.1.mlp.proj.weight": "cpu",
        "transformer.h.1.norm_1.bias": "cpu",
        "transformer.h.1.norm_1.weight": "cpu",
        "transformer.h.1.norm_2.bias": "cpu",
        "transformer.h.1.norm_2.weight": "cpu",
        "transformer.ln_f.bias": "meta",
        "transformer.ln_f.weight": "meta",
        "transformer.wte.weight": "meta",
    }

    materialize_meta_tensors(model, torch.device("cpu"))
    assert path_to_device(model) == {
        "cos": "cpu",
        "lm_head.weight": "cpu",
        "sin": "cpu",
        "transformer.h.0.attn.attn.bias": "cpu",
        "transformer.h.0.attn.attn.weight": "cpu",
        "transformer.h.0.attn.proj.bias": "cpu",
        "transformer.h.0.attn.proj.weight": "cpu",
        "transformer.h.0.mlp.fc.bias": "cpu",
        "transformer.h.0.mlp.fc.weight": "cpu",
        "transformer.h.0.mlp.proj.bias": "cpu",
        "transformer.h.0.mlp.proj.weight": "cpu",
        "transformer.h.0.norm_1.bias": "cpu",
        "transformer.h.0.norm_1.weight": "cpu",
        "transformer.h.0.norm_2.bias": "cpu",
        "transformer.h.0.norm_2.weight": "cpu",
        "transformer.h.1.attn.attn.bias": "cpu",
        "transformer.h.1.attn.attn.weight": "cpu",
        "transformer.h.1.attn.proj.bias": "cpu",
        "transformer.h.1.attn.proj.weight": "cpu",
        "transformer.h.1.mlp.fc.bias": "cpu",
        "transformer.h.1.mlp.fc.weight": "cpu",
        "transformer.h.1.mlp.proj.bias": "cpu",
        "transformer.h.1.mlp.proj.weight": "cpu",
        "transformer.h.1.norm_1.bias": "cpu",
        "transformer.h.1.norm_1.weight": "cpu",
        "transformer.h.1.norm_2.bias": "cpu",
        "transformer.h.1.norm_2.weight": "cpu",
        "transformer.ln_f.bias": "cpu",
        "transformer.ln_f.weight": "cpu",
        "transformer.wte.weight": "cpu",
    }


def _test_model_1device(accelerator):
    from generate.pipelined import get_model
    from lit_gpt.config import Config

    fabric = Fabric(accelerator=accelerator, devices=1)
    config = Config.from_name("pythia-14m", n_layer=2)
    model = get_model(fabric, config, 15, 1)

    device_str = str(fabric.device)
    assert path_to_device(model) == {
        "cos": device_str,
        "sin": device_str,
        "lm_head.weight": device_str,
        "transformer.h.0.attn.attn.bias": device_str,
        "transformer.h.0.attn.attn.weight": device_str,
        "transformer.h.0.attn.proj.bias": device_str,
        "transformer.h.0.attn.proj.weight": device_str,
        "transformer.h.0.mlp.fc.bias": device_str,
        "transformer.h.0.mlp.fc.weight": device_str,
        "transformer.h.0.mlp.proj.bias": device_str,
        "transformer.h.0.mlp.proj.weight": device_str,
        "transformer.h.0.norm_1.bias": device_str,
        "transformer.h.0.norm_1.weight": device_str,
        "transformer.h.0.norm_2.bias": device_str,
        "transformer.h.0.norm_2.weight": device_str,
        "transformer.h.0.attn.kv_cache.k": device_str,
        "transformer.h.0.attn.kv_cache.v": device_str,
        "transformer.h.1.attn.attn.bias": device_str,
        "transformer.h.1.attn.attn.weight": device_str,
        "transformer.h.1.attn.proj.bias": device_str,
        "transformer.h.1.attn.proj.weight": device_str,
        "transformer.h.1.mlp.fc.bias": device_str,
        "transformer.h.1.mlp.fc.weight": device_str,
        "transformer.h.1.mlp.proj.bias": device_str,
        "transformer.h.1.mlp.proj.weight": device_str,
        "transformer.h.1.norm_1.bias": device_str,
        "transformer.h.1.norm_1.weight": device_str,
        "transformer.h.1.norm_2.bias": device_str,
        "transformer.h.1.norm_2.weight": device_str,
        "transformer.h.1.attn.kv_cache.k": device_str,
        "transformer.h.1.attn.kv_cache.v": device_str,
        "transformer.ln_f.bias": device_str,
        "transformer.ln_f.weight": device_str,
        "transformer.wte.weight": device_str,
    }
    assert model.max_seq_length == 15


@RunIf(min_cuda_gpus=1)
def test_model_1device_cuda():
    _test_model_1device("cuda")


def test_model_1device_cpu():
    _test_model_1device("cpu")


def find_forward_hooks(module):
    mapping = defaultdict(list)
    for name, submodule in module.named_modules():
        for hook in submodule._forward_pre_hooks.values():
            hook_data = ("forward_pre_hook", hook.func.__name__, hook.args, hook.keywords)
            mapping[name].append(hook_data)
        for hook in submodule._forward_hooks.values():
            hook_data = ("forward_hook", hook.func.__name__, hook.args, hook.keywords)
            mapping[name].append(hook_data)
    return dict(mapping)


@RunIf(min_cuda_gpus=2)
def test_model_forward_hooks(monkeypatch):
    from generate.pipelined import get_model
    from lit_gpt.config import Config

    fabric = Fabric(accelerator="cuda", devices=1)
    config = Config.from_name("pythia-14m")  # 6 layers
    model = get_model(fabric, config, max_seq_length=15, devices=2)

    hooks = find_forward_hooks(model)
    actual = path_to_device(model)
    assert actual == {
        "lm_head.weight": "cuda:0",
        "transformer.wte.weight": "cuda:0",
        "transformer.h.0.norm_1.weight": "cuda:0",
        "transformer.h.0.norm_1.bias": "cuda:0",
        "transformer.h.0.attn.attn.weight": "cuda:0",
        "transformer.h.0.attn.attn.bias": "cuda:0",
        "transformer.h.0.attn.proj.weight": "cuda:0",
        "transformer.h.0.attn.proj.bias": "cuda:0",
        "transformer.h.0.norm_2.weight": "cuda:0",
        "transformer.h.0.norm_2.bias": "cuda:0",
        "transformer.h.0.mlp.fc.weight": "cuda:0",
        "transformer.h.0.mlp.fc.bias": "cuda:0",
        "transformer.h.0.mlp.proj.weight": "cuda:0",
        "transformer.h.0.mlp.proj.bias": "cuda:0",
        "transformer.h.1.norm_1.weight": "cuda:0",
        "transformer.h.1.norm_1.bias": "cuda:0",
        "transformer.h.1.attn.attn.weight": "cuda:0",
        "transformer.h.1.attn.attn.bias": "cuda:0",
        "transformer.h.1.attn.proj.weight": "cuda:0",
        "transformer.h.1.attn.proj.bias": "cuda:0",
        "transformer.h.1.norm_2.weight": "cuda:0",
        "transformer.h.1.norm_2.bias": "cuda:0",
        "transformer.h.1.mlp.fc.weight": "cuda:0",
        "transformer.h.1.mlp.fc.bias": "cuda:0",
        "transformer.h.1.mlp.proj.weight": "cuda:0",
        "transformer.h.1.mlp.proj.bias": "cuda:0",
        "transformer.h.2.norm_1.weight": "cuda:0",
        "transformer.h.2.norm_1.bias": "cuda:0",
        "transformer.h.2.attn.attn.weight": "cuda:0",
        "transformer.h.2.attn.attn.bias": "cuda:0",
        "transformer.h.2.attn.proj.weight": "cuda:0",
        "transformer.h.2.attn.proj.bias": "cuda:0",
        "transformer.h.2.norm_2.weight": "cuda:0",
        "transformer.h.2.norm_2.bias": "cuda:0",
        "transformer.h.2.mlp.fc.weight": "cuda:0",
        "transformer.h.2.mlp.fc.bias": "cuda:0",
        "transformer.h.2.mlp.proj.weight": "cuda:0",
        "transformer.h.2.mlp.proj.bias": "cuda:0",
        "transformer.h.3.norm_1.weight": "cuda:1",
        "transformer.h.3.norm_1.bias": "cuda:1",
        "transformer.h.3.attn.attn.weight": "cuda:1",
        "transformer.h.3.attn.attn.bias": "cuda:1",
        "transformer.h.3.attn.proj.weight": "cuda:1",
        "transformer.h.3.attn.proj.bias": "cuda:1",
        "transformer.h.3.norm_2.weight": "cuda:1",
        "transformer.h.3.norm_2.bias": "cuda:1",
        "transformer.h.3.mlp.fc.weight": "cuda:1",
        "transformer.h.3.mlp.fc.bias": "cuda:1",
        "transformer.h.3.mlp.proj.weight": "cuda:1",
        "transformer.h.3.mlp.proj.bias": "cuda:1",
        "transformer.h.4.norm_1.weight": "cuda:1",
        "transformer.h.4.norm_1.bias": "cuda:1",
        "transformer.h.4.attn.attn.weight": "cuda:1",
        "transformer.h.4.attn.attn.bias": "cuda:1",
        "transformer.h.4.attn.proj.weight": "cuda:1",
        "transformer.h.4.attn.proj.bias": "cuda:1",
        "transformer.h.4.norm_2.weight": "cuda:1",
        "transformer.h.4.norm_2.bias": "cuda:1",
        "transformer.h.4.mlp.fc.weight": "cuda:1",
        "transformer.h.4.mlp.fc.bias": "cuda:1",
        "transformer.h.4.mlp.proj.weight": "cuda:1",
        "transformer.h.4.mlp.proj.bias": "cuda:1",
        "transformer.h.5.norm_1.weight": "cuda:1",
        "transformer.h.5.norm_1.bias": "cuda:1",
        "transformer.h.5.attn.attn.weight": "cuda:1",
        "transformer.h.5.attn.attn.bias": "cuda:1",
        "transformer.h.5.attn.proj.weight": "cuda:1",
        "transformer.h.5.attn.proj.bias": "cuda:1",
        "transformer.h.5.norm_2.weight": "cuda:1",
        "transformer.h.5.norm_2.bias": "cuda:1",
        "transformer.h.5.mlp.fc.weight": "cuda:1",
        "transformer.h.5.mlp.fc.bias": "cuda:1",
        "transformer.h.5.mlp.proj.weight": "cuda:1",
        "transformer.h.5.mlp.proj.bias": "cuda:1",
        "transformer.ln_f.weight": "cuda:0",
        "transformer.ln_f.bias": "cuda:0",
        "cos": "cuda:0",
        "sin": "cuda:0",
        "transformer.h.0.attn.kv_cache.k": "cuda:0",
        "transformer.h.0.attn.kv_cache.v": "cuda:0",
        "transformer.h.1.attn.kv_cache.k": "cuda:0",
        "transformer.h.1.attn.kv_cache.v": "cuda:0",
        "transformer.h.2.attn.kv_cache.k": "cuda:0",
        "transformer.h.2.attn.kv_cache.v": "cuda:0",
        "transformer.h.3.attn.kv_cache.k": "cuda:1",
        "transformer.h.3.attn.kv_cache.v": "cuda:1",
        "transformer.h.4.attn.kv_cache.k": "cuda:1",
        "transformer.h.4.attn.kv_cache.v": "cuda:1",
        "transformer.h.5.attn.kv_cache.k": "cuda:1",
        "transformer.h.5.attn.kv_cache.v": "cuda:1",
    }
    assert hooks == {
        "transformer.h.3": [("forward_pre_hook", "move_block_input", (torch.device(type="cuda", index=1),), {})],
        "transformer.h.4": [("forward_pre_hook", "move_block_input", (torch.device(type="cuda", index=1),), {})],
        "transformer.h.5": [
            ("forward_pre_hook", "move_block_input", (torch.device(type="cuda", index=1),), {}),
            ("forward_hook", "move_block_output", (torch.device(type="cuda", index=0),), {}),
        ],
    }


@RunIf(min_cuda_gpus=2)
def test_base_with_pipelined(tmp_path):
    from lit_gpt import GPT, Config
    from scripts.download import download_from_hub

    # download the tokenizer
    download_from_hub(repo_id="EleutherAI/pythia-14m", tokenizer_only=True, checkpoint_dir=tmp_path)
    checkpoint_dir = tmp_path / "EleutherAI/pythia-14m"
    # save the config
    config = Config.from_name("pythia-14m")
    (checkpoint_dir / "lit_config.json").write_text(json.dumps(asdict(config)))
    # create a state dict to load from
    torch.save(GPT(config).state_dict(), checkpoint_dir / "lit_model.pth")

    args = [
        "--num_samples=1",
        "--max_new_tokens=10",
        "--precision=16-true",
        "--temperature=0.0",
        f"--checkpoint_dir={str(checkpoint_dir)}",
    ]
    base_stdout = subprocess.check_output([sys.executable, "generate/base.py", *args]).decode()
    pipelined_stdout = subprocess.check_output([sys.executable, "generate/pipelined.py", "--devices=2", *args]).decode()

    assert base_stdout.startswith("What food do llamas eat?")
    assert base_stdout == pipelined_stdout


def test_cli():
    cli_path = Path(__file__).parent.parent / "generate" / "pipelined.py"
    output = subprocess.check_output([sys.executable, cli_path, "-h"])
    output = str(output.decode())
    assert "Generates text samples" in output
