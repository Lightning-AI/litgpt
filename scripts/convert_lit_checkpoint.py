import contextlib
import gc
import sys
from functools import partial
from pathlib import Path
from typing import Optional, Literal, Dict, List, Union

import torch

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_gpt import Config
from lit_gpt.utils import lazy_load, incremental_save, NotYetLoadedTensor
from scripts.convert_hf_checkpoint import load_param, layer_template


def copy_weights_gpt_neox(
    state_dict: Dict[str, torch.Tensor],
    lit_weights: Dict[str, Union[torch.Tensor, NotYetLoadedTensor]],
    saver: Optional[incremental_save] = None,
):
    weight_map = {
        "gpt_neox.embed_in.weight": "transformer.wte.weight",
        "gpt_neox.layers.{}.input_layernorm.bias": "transformer.h.{}.norm_1.bias",
        "gpt_neox.layers.{}.input_layernorm.weight": "transformer.h.{}.norm_1.weight",
        "gpt_neox.layers.{}.attention.query_key_value.bias": "transformer.h.{}.attn.attn.bias",
        "gpt_neox.layers.{}.attention.query_key_value.weight": "transformer.h.{}.attn.attn.weight",
        "gpt_neox.layers.{}.attention.dense.bias": "transformer.h.{}.attn.proj.bias",
        "gpt_neox.layers.{}.attention.dense.weight": "transformer.h.{}.attn.proj.weight",
        "gpt_neox.layers.{}.attention.rotary_emb.inv_freq": None,
        "gpt_neox.layers.{}.attention.bias": None,
        "gpt_neox.layers.{}.attention.masked_bias": None,
        "gpt_neox.layers.{}.post_attention_layernorm.bias": "transformer.h.{}.norm_2.bias",
        "gpt_neox.layers.{}.post_attention_layernorm.weight": "transformer.h.{}.norm_2.weight",
        "gpt_neox.layers.{}.mlp.dense_h_to_4h.bias": "transformer.h.{}.mlp.fc.bias",
        "gpt_neox.layers.{}.mlp.dense_h_to_4h.weight": "transformer.h.{}.mlp.fc.weight",
        "gpt_neox.layers.{}.mlp.dense_4h_to_h.bias": "transformer.h.{}.mlp.proj.bias",
        "gpt_neox.layers.{}.mlp.dense_4h_to_h.weight": "transformer.h.{}.mlp.proj.weight",
        "gpt_neox.final_layer_norm.bias": "transformer.ln_f.bias",
        "gpt_neox.final_layer_norm.weight": "transformer.ln_f.weight",
        "embed_out.weight": "lm_head.weight",
    }

    for name, param in lit_weights.items():
        if "transformer.h" in name:
            from_name, number = layer_template(name, 2)
            to_name = get_to_name(from_name, weight_map).format(number)
            if to_name is None:
                continue
            to_name = to_name.format(number)
        else:
            to_name = get_to_name(name, weight_map)
        param = load_param(param)
        if saver is not None:
            param = saver.store_early(param)
        state_dict[to_name] = param


def copy_weights_falcon(
    size: Literal["7b", "40b"],
    state_dict: Dict[str, torch.Tensor],
    lit_weights: Dict[str, Union[torch.Tensor, NotYetLoadedTensor]],
    saver: Optional[incremental_save] = None,
):
    weight_map = {
        "transformer.word_embeddings.weight": "transformer.wte.weight",
        "transformer.h.{}.self_attention.query_key_value.weight": "transformer.h.{}.attn.attn.weight",
        "transformer.h.{}.self_attention.dense.weight": "transformer.h.{}.attn.proj.weight",
        "transformer.h.{}.mlp.dense_h_to_4h.weight": "transformer.h.{}.mlp.fc.weight",
        "transformer.h.{}.mlp.dense_4h_to_h.weight": "transformer.h.{}.mlp.proj.weight",
        "transformer.ln_f.bias": "transformer.ln_f.bias",
        "transformer.ln_f.weight": "transformer.ln_f.weight",
        "lm_head.weight": "lm_head.weight",
    }
    # the original model definition is different for each size
    if size == "7b":
        weight_map.update(
            {
                "transformer.h.{}.input_layernorm.bias": "transformer.h.{}.norm_1.bias",
                "transformer.h.{}.input_layernorm.weight": "transformer.h.{}.norm_1.weight",
            }
        )
    elif size == "40b":
        weight_map.update(
            {
                "transformer.h.{}.ln_attn.bias": "transformer.h.{}.norm_1.bias",
                "transformer.h.{}.ln_attn.weight": "transformer.h.{}.norm_1.weight",
                "transformer.h.{}.ln_mlp.bias": "transformer.h.{}.norm_2.bias",
                "transformer.h.{}.ln_mlp.weight": "transformer.h.{}.norm_2.weight",
            }
        )
    else:
        raise NotImplementedError

    for name, param in lit_weights.items():
        if "transformer.h" in name:
            from_name, number = layer_template(name, 2)
            to_name = get_to_name(from_name, weight_map).format(number)
        else:
            to_name = get_to_name(name, weight_map)
        param = load_param(param)
        if saver is not None:
            param = saver.store_early(param)
        state_dict[to_name] = param


def copy_weights_hf_llama(
    config: Config,
    state_dict: Dict[str, torch.Tensor],
    lit_weights: Dict[str, Union[torch.Tensor, NotYetLoadedTensor]],
    saver: Optional[incremental_save] = None,
):
    weight_map = {
        "model.embed_tokens.weight": "transformer.wte.weight",
        "model.layers.{}.input_layernorm.weight": "transformer.h.{}.norm_1.weight",
        "model.layers.{}.self_attn.q_proj.weight": None,
        "model.layers.{}.self_attn.k_proj.weight": None,
        "model.layers.{}.self_attn.v_proj.weight": None,
        "model.layers.{}.self_attn.o_proj.weight": "transformer.h.{}.attn.proj.weight",
        "model.layers.{}.self_attn.rotary_emb.inv_freq": None,
        "model.layers.{}.post_attention_layernorm.weight": "transformer.h.{}.norm_2.weight",
        "model.layers.{}.mlp.gate_proj.weight": "transformer.h.{}.mlp.fc_1.weight",
        "model.layers.{}.mlp.up_proj.weight": "transformer.h.{}.mlp.fc_2.weight",
        "model.layers.{}.mlp.down_proj.weight": "transformer.h.{}.mlp.proj.weight",
        "model.norm.weight": "transformer.ln_f.weight",
        "lm_head.weight": "lm_head.weight",
    }

    for name, param in lit_weights.items():
        # handle name
        if "transformer.h" in name and not name.endswith(".attn.attn.weight"):
            from_name, number = layer_template(name, 2)
            to_name = get_to_name(from_name, weight_map)
            if to_name is None:
                continue
            to_name = to_name.format(number)
        elif name.endswith(".attn.attn.weight"):
            from_name, number = layer_template(name, 2)
            q = "model.layers.{}.self_attn.q_proj.weight".format(number)
            k = "model.layers.{}.self_attn.k_proj.weight".format(number)
            v = "model.layers.{}.self_attn.v_proj.weight".format(number)
        else:
            to_name = get_to_name(name, weight_map)

        # handle param
        if name.endswith(".attn.attn.weight"):
            qkv = load_param(param)
            qp, kp, vp = tensor_split(qkv, config, "llama")
            for to_name, _param in zip((q, k, v), (qp, kp, vp)):
                if saver is not None:
                    param = saver.store_early(_param)
                state_dict[to_name] = param
        else:
            param = load_param(param)
            if saver is not None:
                param = saver.store_early(param)
            state_dict[to_name] = param


def tensor_split(param: Union[torch.Tensor, NotYetLoadedTensor], config: Config, model_name: str) -> torch.Tensor:
    if model_name != "llama":
        raise NotImplementedError(f"{model_name}")
    else:
        splits = [
            (start, start + config.head_size, start + int(config.head_size * 2))
            for start in range(100, param.shape[0] + 1, config.head_size * len(("q", "k", "v")))
        ]

    qc = ()
    kc = ()
    vc = ()

    for split in splits:
        qs, ks, vs = split
        qc += (param[qs - config.head_size : qs, :],)
        kc += (param[qs:ks, :],)
        vc += (param[ks:vs, :],)

    q = torch.cat(qc)
    k = torch.cat(kc)
    v = torch.cat(vc)

    return q, k, v


def get_to_name(lit_key_name: str, weight_map: Dict[str, str]) -> str:
    for k, v in weight_map.items():
        if lit_key_name == v:
            return k


@torch.inference_mode()
def convert_lit_checkpoint(
    *, checkpoint_dir: Path = Path("checkpoints/tiiuae/falcon-7b"), model_name: Optional[str] = None
) -> None:
    if model_name is None:
        model_name = checkpoint_dir.name
    config = Config.from_name(model_name)

    if "falcon" in model_name:
        copy_fn = partial(copy_weights_falcon, "40b" if config.n_embd == 8192 else "7b")
    elif config._mlp_class == "LLaMAMLP":
        # holder to reconstitute the split q, k, v
        copy_fn = partial(copy_weights_hf_llama, config)
    else:
        copy_fn = copy_weights_gpt_neox

    # initialize a new empty state dict to hold our new weights
    sd = {}

    pth_file = checkpoint_dir / "lit_model.pth"

    with incremental_save(checkpoint_dir / "lit_model_finetuned.bin") as saver:
        with contextlib.ExitStack() as stack:
            lit_weights = stack.enter_context(lazy_load(pth_file))
            copy_fn(sd, lit_weights, saver=saver)
            gc.collect()
        saver.save(sd)


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(convert_lit_checkpoint)
