# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

from unittest import mock

import pytest
import torch

from litgpt import Config
from litgpt.scripts.convert_hf_checkpoint import convert_hf_checkpoint, copy_weights_hf_llama, qkv_reassemble


def test_llama2_70b_conversion():
    shapes = {
        "model.embed_tokens.weight": (32000, 8192),
        "model.layers.0.input_layernorm.weight": (8192,),
        "model.layers.0.mlp.down_proj.weight": (8192, 28672),
        "model.layers.0.mlp.gate_proj.weight": (28672, 8192),
        "model.layers.0.mlp.up_proj.weight": (28672, 8192),
        "model.layers.0.post_attention_layernorm.weight": (8192,),
        "model.layers.0.self_attn.q_proj.weight": (8192, 8192),
        "model.layers.0.self_attn.k_proj.weight": (1024, 8192),
        "model.layers.0.self_attn.v_proj.weight": (1024, 8192),
        "model.layers.0.self_attn.o_proj.weight": (8192, 8192),
        "model.layers.1.input_layernorm.weight": (8192,),
        "model.layers.1.mlp.down_proj.weight": (8192, 28672),
        "model.layers.1.mlp.gate_proj.weight": (28672, 8192),
        "model.layers.1.mlp.up_proj.weight": (28672, 8192),
        "model.layers.1.post_attention_layernorm.weight": (8192,),
        "model.layers.1.self_attn.o_proj.weight": (8192, 8192),
        "model.layers.2.input_layernorm.weight": (8192,),
        "model.layers.2.mlp.down_proj.weight": (8192, 28672),
        "model.layers.2.mlp.gate_proj.weight": (28672, 8192),
        "model.layers.2.mlp.up_proj.weight": (28672, 8192),
        "model.layers.2.post_attention_layernorm.weight": (8192,),
        "model.layers.2.self_attn.o_proj.weight": (8192, 8192),
        "model.layers.3.input_layernorm.weight": (8192,),
        "model.layers.3.mlp.down_proj.weight": (8192, 28672),
        "model.layers.3.mlp.gate_proj.weight": (28672, 8192),
        "model.layers.3.mlp.up_proj.weight": (28672, 8192),
        "model.layers.3.post_attention_layernorm.weight": (8192,),
        "model.layers.3.self_attn.o_proj.weight": (8192, 8192),
        "model.layers.4.input_layernorm.weight": (8192,),
        "model.layers.4.mlp.down_proj.weight": (8192, 28672),
        "model.layers.4.mlp.gate_proj.weight": (28672, 8192),
        "model.layers.4.mlp.up_proj.weight": (28672, 8192),
        "model.layers.4.post_attention_layernorm.weight": (8192,),
        "model.layers.4.self_attn.o_proj.weight": (8192, 8192),
        "model.layers.5.mlp.gate_proj.weight": (28672, 8192),
        "model.layers.5.self_attn.o_proj.weight": (8192, 8192),
    }

    config = Config.from_name("Llama-2-70b-hf")
    holder = {}
    qkv_weights = {}
    with torch.device("meta"):
        weight_map = {k: torch.empty(s) for k, s in shapes.items()}
    copy_weights_hf_llama(config, qkv_weights, holder, weight_map)

    # NOTE: there are 5 layers, but only in the first layer we have `q`, `k` and `v`
    assert len(qkv_weights) == 1
    # there are no loaded qkv weights
    assert all(v is None for qkv in qkv_weights.values() for v in qkv)
    # the shapes are correct
    holder = {k: tuple(t.shape) for k, t in holder.items()}
    assert holder == {
        "transformer.h.0.attn.qkv.weight": (10240, 8192),
        "transformer.h.0.attn.proj.weight": (8192, 8192),
        "transformer.h.0.mlp.fc_1.weight": (28672, 8192),
        "transformer.h.0.mlp.fc_2.weight": (28672, 8192),
        "transformer.h.0.mlp.proj.weight": (8192, 28672),
        "transformer.h.0.norm_1.weight": (8192,),
        "transformer.h.0.norm_2.weight": (8192,),
        "transformer.h.1.attn.proj.weight": (8192, 8192),
        "transformer.h.1.mlp.fc_1.weight": (28672, 8192),
        "transformer.h.1.mlp.fc_2.weight": (28672, 8192),
        "transformer.h.1.mlp.proj.weight": (8192, 28672),
        "transformer.h.1.norm_1.weight": (8192,),
        "transformer.h.1.norm_2.weight": (8192,),
        "transformer.h.2.attn.proj.weight": (8192, 8192),
        "transformer.h.2.mlp.fc_1.weight": (28672, 8192),
        "transformer.h.2.mlp.fc_2.weight": (28672, 8192),
        "transformer.h.2.mlp.proj.weight": (8192, 28672),
        "transformer.h.2.norm_1.weight": (8192,),
        "transformer.h.2.norm_2.weight": (8192,),
        "transformer.h.3.attn.proj.weight": (8192, 8192),
        "transformer.h.3.mlp.fc_1.weight": (28672, 8192),
        "transformer.h.3.mlp.fc_2.weight": (28672, 8192),
        "transformer.h.3.mlp.proj.weight": (8192, 28672),
        "transformer.h.3.norm_1.weight": (8192,),
        "transformer.h.3.norm_2.weight": (8192,),
        "transformer.h.4.attn.proj.weight": (8192, 8192),
        "transformer.h.4.mlp.fc_1.weight": (28672, 8192),
        "transformer.h.4.mlp.fc_2.weight": (28672, 8192),
        "transformer.h.4.mlp.proj.weight": (8192, 28672),
        "transformer.h.4.norm_1.weight": (8192,),
        "transformer.h.4.norm_2.weight": (8192,),
        "transformer.h.5.attn.proj.weight": (8192, 8192),
        "transformer.h.5.mlp.fc_1.weight": (28672, 8192),
        "transformer.wte.weight": (32000, 8192),
        "lm_head.weight": (32000, 8192),  # due to weight tying lm_head is in the converted weights
    }


@pytest.mark.parametrize("model_name", ("pythia-14m", "falcon-7b", "Llama-2-7b-hf", "phi-2"))
def test_convert_hf_checkpoint(tmp_path, model_name):
    with pytest.raises(ValueError, match="to contain .bin"):
        convert_hf_checkpoint(checkpoint_dir=tmp_path, model_name=model_name)

    bin_file = tmp_path / "foo.bin"
    bin_file.touch()
    with mock.patch("litgpt.scripts.convert_hf_checkpoint.lazy_load") as load:
        # bypass if-statement for weight tying
        if model_name == "Llama-2-7b-hf":
            load.return_value = {"model.embed_tokens.weight": torch.rand((10, 10))}
        convert_hf_checkpoint(checkpoint_dir=tmp_path, model_name=model_name)
    load.assert_called_with(bin_file)

    assert {p.name for p in tmp_path.glob("*")} == {"foo.bin", "model_config.yaml", "lit_model.pth"}

    # ensure that the config dict can be loaded
    config = Config.from_file(tmp_path / "model_config.yaml")
    assert isinstance(config, Config)


def test_qkv_reassemble():
    # MHA
    config = Config(n_embd=4, n_head=4)
    qkv_interleaved = torch.tensor(
        [
            [0, 1, 2, 3],  # query
            [16, 17, 18, 19],  # key
            [32, 33, 34, 35],  # value
            [4, 5, 6, 7],  # query
            [20, 21, 22, 23],  # key
            [36, 37, 38, 39],  # value
            [8, 9, 10, 11],  # query
            [24, 25, 26, 27],  # key
            [40, 41, 42, 43],  # value
            [12, 13, 14, 15],  # query
            [28, 29, 30, 31],  # key
            [44, 45, 46, 47],  # value
        ]
    )
    qkv = qkv_reassemble(qkv_interleaved, config)
    torch.testing.assert_close(
        qkv,
        torch.tensor(
            [
                [0, 1, 2, 3],  # query
                [4, 5, 6, 7],  # query
                [8, 9, 10, 11],  # query
                [12, 13, 14, 15],  # query
                [16, 17, 18, 19],  # key
                [20, 21, 22, 23],  # key
                [24, 25, 26, 27],  # key
                [28, 29, 30, 31],  # key
                [32, 33, 34, 35],  # value
                [36, 37, 38, 39],  # value
                [40, 41, 42, 43],  # value
                [44, 45, 46, 47],  # value
            ]
        ),
    )

    # GQA
    config = Config(n_embd=4, n_head=4, n_query_groups=2)
    qkv_interleaved = torch.tensor(
        [
            [0, 1, 2, 3],  # query
            [4, 5, 6, 7],  # query
            [16, 17, 18, 19],  # key
            [24, 25, 26, 27],  # value
            [8, 9, 10, 11],  # query
            [12, 13, 14, 15],  # query
            [20, 21, 22, 23],  # key
            [28, 29, 30, 31],  # value
        ]
    )
    qkv = qkv_reassemble(qkv_interleaved, config)
    torch.testing.assert_close(
        qkv,
        torch.tensor(
            [
                [0, 1, 2, 3],  # query
                [4, 5, 6, 7],  # query
                [8, 9, 10, 11],  # query
                [12, 13, 14, 15],  # query
                [16, 17, 18, 19],  # key
                [20, 21, 22, 23],  # key
                [24, 25, 26, 27],  # value
                [28, 29, 30, 31],  # value
            ]
        ),
    )

    # MQA
    config = Config(n_embd=4, n_head=4, n_query_groups=1)
    qkv_interleaved = torch.tensor(
        [
            [0, 1, 2, 3],  # query
            [4, 5, 6, 7],  # query
            [8, 9, 10, 11],  # query
            [12, 13, 14, 15],  # query
            [16, 17, 18, 19],  # key
            [20, 21, 22, 23],  # value
        ]
    )
    qkv = qkv_reassemble(qkv_interleaved, config)
    torch.testing.assert_close(
        qkv,
        torch.tensor(
            [
                [0, 1, 2, 3],  # query
                [4, 5, 6, 7],  # query
                [8, 9, 10, 11],  # query
                [12, 13, 14, 15],  # query
                [16, 17, 18, 19],  # key
                [20, 21, 22, 23],  # value
            ]
        ),
    )
