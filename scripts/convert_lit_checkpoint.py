# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import gc
import sys
from functools import partial
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import torch
from lightning.fabric.utilities.load import _NotYetLoadedTensor as NotYetLoadedTensor

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_gpt import Config
from lit_gpt.utils import CLI, incremental_save, lazy_load
from scripts.convert_hf_checkpoint import layer_template, load_param


def copy_weights_falcon(
    config: Config,
    state_dict: Dict[str, torch.Tensor],
    lit_weights: Dict[str, Union[torch.Tensor, NotYetLoadedTensor]],
    saver: Optional[incremental_save] = None,
) -> None:
    weight_map = {
        "transformer.wte.weight": "transformer.word_embeddings.weight",
        "transformer.h.{}.attn.attn.weight": "transformer.h.{}.self_attention.query_key_value.weight",
        "transformer.h.{}.attn.proj.weight": "transformer.h.{}.self_attention.dense.weight",
        "transformer.h.{}.mlp.fc.weight": "transformer.h.{}.mlp.dense_h_to_4h.weight",
        "transformer.h.{}.mlp.proj.weight": "transformer.h.{}.mlp.dense_4h_to_h.weight",
        "transformer.ln_f.bias": "transformer.ln_f.bias",
        "transformer.ln_f.weight": "transformer.ln_f.weight",
        "lm_head.weight": "lm_head.weight",
    }
    # the original model definition is different for each size
    if "7b" in config.name:
        weight_map.update(
            {
                "transformer.h.{}.norm_1.bias": "transformer.h.{}.input_layernorm.bias",
                "transformer.h.{}.norm_1.weight": "transformer.h.{}.input_layernorm.weight",
            }
        )
    elif "40b" in config.name or "180B" in config.name:
        weight_map.update(
            {
                "transformer.h.{}.norm_1.bias": "transformer.h.{}.ln_attn.bias",
                "transformer.h.{}.norm_1.weight": "transformer.h.{}.ln_attn.weight",
                "transformer.h.{}.norm_2.bias": "transformer.h.{}.ln_mlp.bias",
                "transformer.h.{}.norm_2.weight": "transformer.h.{}.ln_mlp.weight",
            }
        )
    else:
        raise NotImplementedError

    for from_name, param in lit_weights.items():
        name_template, layer_idx = layer_template(from_name)
        to_name = weight_map[name_template].format(layer_idx)
        param = load_param(param, from_name, None)
        if from_name.endswith((".attn.attn.weight", ".attn.attn.bias")):
            # Reassemble [q, q, ..., k, k, ..., v, v, ...] --> [q, k, v, q, k, v, ...]
            qs, ks, vs = qkv_split(param, config)
            cycled = [t for group in zip(qs, ks, vs) for t in group]
            param = torch.cat(cycled)
        if saver is not None:
            param = saver.store_early(param)
        state_dict[to_name] = param


def copy_weights_gpt_neox(
    config: Config,
    state_dict: Dict[str, torch.Tensor],
    lit_weights: Dict[str, Union[torch.Tensor, NotYetLoadedTensor]],
    saver: Optional[incremental_save] = None,
) -> None:
    weight_map = {
        "transformer.wte.weight": "gpt_neox.embed_in.weight",
        "transformer.h.{}.norm_1.bias": "gpt_neox.layers.{}.input_layernorm.bias",
        "transformer.h.{}.norm_1.weight": "gpt_neox.layers.{}.input_layernorm.weight",
        "transformer.h.{}.attn.attn.bias": "gpt_neox.layers.{}.attention.query_key_value.bias",
        "transformer.h.{}.attn.attn.weight": "gpt_neox.layers.{}.attention.query_key_value.weight",
        "transformer.h.{}.attn.proj.bias": "gpt_neox.layers.{}.attention.dense.bias",
        "transformer.h.{}.attn.proj.weight": "gpt_neox.layers.{}.attention.dense.weight",
        "transformer.h.{}.norm_2.bias": "gpt_neox.layers.{}.post_attention_layernorm.bias",
        "transformer.h.{}.norm_2.weight": "gpt_neox.layers.{}.post_attention_layernorm.weight",
        "transformer.h.{}.mlp.fc.bias": "gpt_neox.layers.{}.mlp.dense_h_to_4h.bias",
        "transformer.h.{}.mlp.fc.weight": "gpt_neox.layers.{}.mlp.dense_h_to_4h.weight",
        "transformer.h.{}.mlp.proj.bias": "gpt_neox.layers.{}.mlp.dense_4h_to_h.bias",
        "transformer.h.{}.mlp.proj.weight": "gpt_neox.layers.{}.mlp.dense_4h_to_h.weight",
        "transformer.ln_f.bias": "gpt_neox.final_layer_norm.bias",
        "transformer.ln_f.weight": "gpt_neox.final_layer_norm.weight",
        "lm_head.weight": "embed_out.weight",
    }

    for from_name, param in lit_weights.items():
        name_template, layer_idx = layer_template(from_name)
        to_name = weight_map[name_template].format(layer_idx)
        param = load_param(param, from_name, None)
        if from_name.endswith((".attn.attn.weight", ".attn.attn.bias")):
            # Reassemble [q, q, ..., k, k, ..., v, v, ...] --> [q, k, v, q, k, v, ...]
            qs, ks, vs = qkv_split(param, config)
            cycled = [t for group in zip(qs, ks, vs) for t in group]
            param = torch.cat(cycled)
        if saver is not None:
            param = saver.store_early(param)
        state_dict[to_name] = param


def copy_weights_llama(
    config: Config,
    state_dict: Dict[str, torch.Tensor],
    lit_weights: Dict[str, Union[torch.Tensor, NotYetLoadedTensor]],
    untie_weights: bool = False,
    saver: Optional[incremental_save] = None,
) -> None:
    weight_map = {
        "transformer.wte.weight": "model.embed_tokens.weight",
        "transformer.h.{}.norm_1.weight": "model.layers.{}.input_layernorm.weight",
        "transformer.h.{}.norm_1.bias": "model.layers.{}.input_layernorm.bias",
        "transformer.h.{}.attn.proj.weight": "model.layers.{}.self_attn.o_proj.weight",
        "transformer.h.{}.norm_2.weight": "model.layers.{}.post_attention_layernorm.weight",
        "transformer.h.{}.norm_2.bias": "model.layers.{}.post_attention_layernorm.bias",
        "transformer.ln_f.weight": "model.norm.weight",
        "transformer.ln_f.bias": "model.norm.bias",
        "lm_head.weight": "lm_head.weight",
    }
    if config._mlp_class == "LLaMAMoE":
        weight_map.update(
            {
                "transformer.h.{}.mlp.gate.weight": "model.layers.{}.block_sparse_moe.gate.weight",
                "transformer.h.{}.mlp.experts.{}.fc_1.weight": "model.layers.{}.block_sparse_moe.experts.{}.w1.weight",
                "transformer.h.{}.mlp.experts.{}.fc_2.weight": "model.layers.{}.block_sparse_moe.experts.{}.w3.weight",
                "transformer.h.{}.mlp.experts.{}.proj.weight": "model.layers.{}.block_sparse_moe.experts.{}.w2.weight",
            }
        )
    elif config._mlp_class in ("LLaMAMLP", "GemmaMLP"):
        weight_map.update(
            {
                "transformer.h.{}.mlp.fc_1.weight": "model.layers.{}.mlp.gate_proj.weight",
                "transformer.h.{}.mlp.fc_2.weight": "model.layers.{}.mlp.up_proj.weight",
                "transformer.h.{}.mlp.proj.weight": "model.layers.{}.mlp.down_proj.weight",
            }
        )
    else:
        raise NotImplementedError

    for from_name, param in lit_weights.items():
        if from_name == "lm_head.weight" and untie_weights:
            continue
        name_template, *ids = layer_template(from_name, num_matches=2)
        param = load_param(param, from_name, None)
        if from_name.endswith(".attn.attn.weight"):
            to_names = (
                "model.layers.{}.self_attn.q_proj.weight".format(*ids),
                "model.layers.{}.self_attn.k_proj.weight".format(*ids),
                "model.layers.{}.self_attn.v_proj.weight".format(*ids),
            )
            params = [torch.cat(w) for w in qkv_split(param, config)]
        else:
            to_names = (weight_map[name_template].format(*ids),)
            params = (param,)

        for to_name, param in zip(to_names, params):
            if saver is not None:
                param = saver.store_early(param)
            state_dict[to_name] = param


def copy_weights_phi(
    config: Config,
    state_dict: Dict[str, torch.Tensor],
    lit_weights: Dict[str, Union[torch.Tensor, NotYetLoadedTensor]],
    saver: Optional[incremental_save] = None,
) -> None:
    weight_map = {
        "transformer.wte.weight": "model.embed_tokens.weight",
        "transformer.h.{}.norm_1.weight": "model.layers.{}.input_layernorm.weight",
        "transformer.h.{}.norm_1.bias": "model.layers.{}.input_layernorm.bias",
        "transformer.h.{}.attn.proj.weight": "model.layers.{}.self_attn.dense.weight",
        "transformer.h.{}.attn.proj.bias": "model.layers.{}.self_attn.dense.bias",
        "transformer.h.{}.mlp.fc.weight": "model.layers.{}.mlp.fc1.weight",
        "transformer.h.{}.mlp.fc.bias": "model.layers.{}.mlp.fc1.bias",
        "transformer.h.{}.mlp.proj.weight": "model.layers.{}.mlp.fc2.weight",
        "transformer.h.{}.mlp.proj.bias": "model.layers.{}.mlp.fc2.bias",
        "transformer.ln_f.weight": "model.final_layernorm.weight",
        "transformer.ln_f.bias": "model.final_layernorm.bias",
        "lm_head.weight": "lm_head.weight",
        "lm_head.bias": "lm_head.bias",
    }
    for from_name, param in lit_weights.items():
        name_template, layer_idx = layer_template(from_name)
        param = load_param(param, from_name, None)
        if from_name.endswith((".attn.attn.weight", ".attn.attn.bias")):
            weight_type = from_name.split(".")[-1]  # weight or bias
            to_names = (
                f"model.layers.{{}}.self_attn.q_proj.{weight_type}".format(layer_idx),
                f"model.layers.{{}}.self_attn.k_proj.{weight_type}".format(layer_idx),
                f"model.layers.{{}}.self_attn.v_proj.{weight_type}".format(layer_idx),
            )
            params = [torch.cat(w) for w in qkv_split(param, config)]
        else:
            to_names = (weight_map[name_template].format(layer_idx),)
            params = (param,)

        for to_name, param in zip(to_names, params):
            if saver is not None:
                param = saver.store_early(param)
            state_dict[to_name] = param


def qkv_split(
    param: Union[torch.Tensor, NotYetLoadedTensor], config: Config
) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor], Tuple[torch.Tensor]]:
    q, k, v = param.split(
        (
            config.n_head * config.head_size,
            config.n_query_groups * config.head_size,
            config.n_query_groups * config.head_size,
        )
    )
    qs = q.split(config.n_head // config.n_query_groups * config.head_size)
    ks = k.split(config.head_size)
    vs = v.split(config.head_size)
    return qs, ks, vs


def check_conversion_supported(lit_weights: Dict[str, torch.Tensor]) -> None:
    if any("lora" in wn for wn in lit_weights):
        raise ValueError("Checkpoints with LoRA weights cannot be converted. Call `scripts/merge_lora.py` first.")
    if any("adapter" in wn or "gating_factor" in wn for wn in lit_weights):
        raise NotImplementedError("Converting adapter models is supported.")


@torch.inference_mode()
def convert_lit_checkpoint(checkpoint_path: Path, output_path: Path, config_path: Path) -> None:
    config = Config.from_json(config_path)

    if "falcon" in config.name:
        copy_fn = partial(copy_weights_falcon, config)
    elif config._mlp_class in ("LLaMAMLP", "GemmaMLP", "LLaMAMoE"):
        untie_weights = "Gemma" in config.name
        copy_fn = partial(copy_weights_llama, config, untie_weights=untie_weights)
    elif "phi" in config.name:
        copy_fn = partial(copy_weights_phi, config)
    else:
        copy_fn = partial(copy_weights_gpt_neox, config)

    # initialize a new empty state dict to hold our new weights
    sd = {}
    with incremental_save(output_path) as saver:
        lit_weights = lazy_load(checkpoint_path)
        lit_weights = lit_weights.get("model", lit_weights)
        check_conversion_supported(lit_weights)
        copy_fn(sd, lit_weights, saver=saver)
        gc.collect()
        saver.save(sd)


if __name__ == "__main__":
    CLI(convert_lit_checkpoint)
