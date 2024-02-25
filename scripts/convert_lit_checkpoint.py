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
from lit_gpt.utils import incremental_save, lazy_load
from scripts.convert_hf_checkpoint import layer_template, load_param


def copy_weights_falcon(
    model_name: str,
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
    if "7b" in model_name:
        weight_map.update({
            "transformer.h.{}.norm_1.bias": "transformer.h.{}.input_layernorm.bias",
            "transformer.h.{}.norm_1.weight": "transformer.h.{}.input_layernorm.weight",
        })
    elif "40b" in model_name or "180B" in model_name:
        weight_map.update({
            "transformer.h.{}.norm_1.bias": "transformer.h.{}.ln_attn.bias",
            "transformer.h.{}.norm_1.weight": "transformer.h.{}.ln_attn.weight",
            "transformer.h.{}.norm_2.bias": "transformer.h.{}.ln_mlp.bias",
            "transformer.h.{}.norm_2.weight": "transformer.h.{}.ln_mlp.weight",
        })
    else:
        raise NotImplementedError

    for name, param in lit_weights.items():
        if "transformer.h" in name:
            from_name, number = layer_template(name, 2)
            to_name = weight_map[from_name].format(number)
        else:
            to_name = weight_map[name]
        param = load_param(param, name, None)
        if saver is not None:
            param = saver.store_early(param)
        state_dict[to_name] = param


def copy_weights_gpt_neox(
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

    for name, param in lit_weights.items():
        if "transformer.h" in name:
            from_name, number = layer_template(name, 2)
            to_name = weight_map[from_name].format(number)
        else:
            to_name = weight_map[name]
        param = load_param(param, name, None)
        if saver is not None:
            param = saver.store_early(param)
        state_dict[to_name] = param


def copy_weights_llama(
    config: Config,
    state_dict: Dict[str, torch.Tensor],
    lit_weights: Dict[str, Union[torch.Tensor, NotYetLoadedTensor]],
    saver: Optional[incremental_save] = None,
) -> None:
    weight_map = {
        "transformer.wte.weight": "model.embed_tokens.weight",
        "transformer.h.{}.norm_1.weight": "model.layers.{l}.input_layernorm.weight",
        "transformer.h.{}.norm_1.bias": "model.layers.{l}.input_layernorm.bias",
        "transformer.h.{}.attn.proj.weight": "model.layers.{l}.self_attn.o_proj.weight",
        "transformer.h.{}.norm_2.weight": "model.layers.{l}.post_attention_layernorm.weight",
        "transformer.h.{}.norm_2.bias": "model.layers.{l}.post_attention_layernorm.bias",
        "transformer.ln_f.weight": "model.norm.weight",
        "transformer.ln_f.bias": "model.norm.bias",
        "lm_head.weight": "lm_head.weight",
    }
    if config._mlp_class == "LLaMAMoE":
        weight_map.update({
            "transformer.h.{}.mlp.gate.weight": "model.layers.{l}.block_sparse_moe.gate.weight",
            "transformer.h.{}.mlp.experts.{}.fc_1.weight": "model.layers.{l}.block_sparse_moe.experts.{e}.w1.weight",
            "transformer.h.{}.mlp.experts.{}.fc_2.weight": "model.layers.{l}.block_sparse_moe.experts.{e}.w3.weight",
            "transformer.h.{}.mlp.experts.{}.proj.weight": "model.layers.{l}.block_sparse_moe.experts.{e}.w2.weight",
        })
    elif config._mlp_class == "LLaMAMLP":
        weight_map.update({
            "transformer.h.{}.mlp.fc_1.weight": "model.layers.{l}.mlp.gate_proj.weight",
            "transformer.h.{}.mlp.fc_2.weight": "model.layers.{l}.mlp.up_proj.weight",
            "transformer.h.{}.mlp.proj.weight": "model.layers.{l}.mlp.down_proj.weight",
        })
    else:
        raise NotImplementedError

    for name, param in lit_weights.items():
        if name.endswith(".attn.attn.weight"):
            from_name, l = layer_template(name, 2)
            q = "model.layers.{}.self_attn.q_proj.weight".format(l)
            k = "model.layers.{}.self_attn.k_proj.weight".format(l)
            v = "model.layers.{}.self_attn.v_proj.weight".format(l)
            qkv = load_param(param, name, None)
            qp, kp, vp = qkv_split(qkv, config)
            for to_name, param in zip((q, k, v), (qp, kp, vp)):
                if saver is not None:
                    param = saver.store_early(param)
                state_dict[to_name] = param
        else:
            if "transformer.h" in name:
                from_name, l = layer_template(name, 2)
                e = None
                if "mlp.experts" in name:
                    from_name, e = layer_template(from_name, 5)
                to_name = weight_map[from_name]
                to_name = to_name.format(l=l, e=e)
            else:
                to_name = weight_map[name]
            param = load_param(param, name, None)
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

    for name, param in lit_weights.items():
        if name.endswith((".attn.attn.weight", ".attn.attn.bias")):
            from_name, l = layer_template(name, 2)
            weight_type = name.split(".")[-1]  # weight or bias
            q = f"model.layers.{l}.self_attn.q_proj.{weight_type}"
            k = f"model.layers.{l}.self_attn.k_proj.{weight_type}"
            v = f"model.layers.{l}.self_attn.v_proj.{weight_type}"
            qkv = load_param(param, name, None)
            qp, kp, vp = qkv_split(qkv, config)
            for to_name, param in zip((q, k, v), (qp, kp, vp)):
                if saver is not None:
                    param = saver.store_early(param)
                state_dict[to_name] = param
        else:
            if "transformer.h" in name:
                from_name, l = layer_template(name, 2)
                to_name = weight_map[from_name]
                to_name = to_name.format(l)
            else:
                to_name = weight_map[name]
            param = load_param(param, name, None)
            if saver is not None:
                param = saver.store_early(param)
            state_dict[to_name] = param


def qkv_split(
    param: Union[torch.Tensor, NotYetLoadedTensor], config: Config
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q_per_kv = config.n_head // config.n_query_groups
    qs = []
    ks = []
    vs = []
    for chunk in torch.chunk(param, config.n_query_groups):
        split = torch.split(chunk, [config.head_size * q_per_kv, config.head_size, config.head_size])
        qs.append(split[0])
        ks.append(split[1])
        vs.append(split[2])
    q = torch.cat(qs)
    k = torch.cat(ks)
    v = torch.cat(vs)
    return q, k, v


def check_conversion_supported(lit_weights: Dict[str, torch.Tensor]) -> None:
    if any("lora" in wn for wn in lit_weights):
        raise ValueError("Checkpoints with LoRA weights cannot be converted. Call `scripts/merge_lora.py` first.")
    if any("adapter" in wn or "gating_factor" in wn for wn in lit_weights):
        raise NotImplementedError("Converting adapter models is supported.")


@torch.inference_mode()
def convert_lit_checkpoint(checkpoint_path: Path, output_path: Path, config_path: Path) -> None:
    config = Config.from_json(config_path)

    if "falcon" in config.name:
        copy_fn = partial(copy_weights_falcon, config.name)
    elif config._mlp_class in ("LLaMAMLP", "LLaMAMoE"):
        copy_fn = partial(copy_weights_llama, config)
    elif "phi" in config.name:
        copy_fn = partial(copy_weights_phi, config)
    else:
        copy_fn = copy_weights_gpt_neox

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
    from jsonargparse import CLI

    CLI(convert_lit_checkpoint, as_positional=False)
