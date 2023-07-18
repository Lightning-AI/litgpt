import contextlib
import gc
import json
import sys
from functools import partial
from pathlib import Path
from typing import Optional, Literal, Tuple, Dict, List, Union

import torch

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_gpt import Config
from lit_gpt.utils import lazy_load, incremental_save, NotYetLoadedTensor


def copy_weights_gpt_neox(
    state_dict: Dict[str, torch.Tensor],
    hf_weights: Dict[str, Union[torch.Tensor, NotYetLoadedTensor]],
    saver: Optional[incremental_save] = None,
) -> None:
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

    for name, param in hf_weights.items():
        if "gpt_neox.layers" in name:
            from_name, number = layer_template(name, 2)
            to_name = weight_map[from_name]
            if to_name is None:
                continue
            to_name = to_name.format(number)
        else:
            to_name = weight_map[name]
        param = load_param(param)
        if saver is not None:
            param = saver.store_early(param)
        state_dict[to_name] = param


def copy_weights_falcon(
    size: Literal["7b", "40b"],
    state_dict: Dict[str, torch.Tensor],
    hf_weights: Dict[str, Union[torch.Tensor, NotYetLoadedTensor]],
    saver: Optional[incremental_save] = None,
) -> None:
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

    for name, param in hf_weights.items():
        if "transformer.h" in name:
            from_name, number = layer_template(name, 2)
            to_name = weight_map[from_name].format(number)
        else:
            to_name = weight_map[name]
        param = load_param(param)
        if saver is not None:
            param = saver.store_early(param)
        state_dict[to_name] = param


def copy_weights_open_llama(
    config: Config,
    qkv_weights: Dict[int, List[Optional[NotYetLoadedTensor]]],
    state_dict: Dict[str, torch.Tensor],
    hf_weights: Dict[str, Union[torch.Tensor, NotYetLoadedTensor]],
    saver: Optional[incremental_save] = None,
) -> None:
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

    for name, param in hf_weights.items():
        if "model.layers" in name:
            from_name, number = layer_template(name, 2)
            qkv = qkv_weights.setdefault(number, [None, None, None])
            if "q_proj" in name:
                qkv[0] = param
            elif "k_proj" in name:
                qkv[1] = param
            elif "v_proj" in name:
                qkv[2] = param
            to_name = weight_map[from_name]
            if to_name is None:
                continue
            to_name = to_name.format(number)
        else:
            to_name = weight_map[name]
        param = load_param(param)
        if saver is not None:
            param = saver.store_early(param)
        state_dict[to_name] = param

    for i, (q, k, v) in list(qkv_weights.items()):
        if q is None or k is None or v is None:
            # split across different .bin files
            continue
        q = load_param(q)
        k = load_param(k)
        v = load_param(v)
        # this assumes MHA which is true for the supported HF checkpoints
        q = q.transpose(0, 1).reshape(-1, config.head_size)
        k = k.transpose(0, 1).reshape(-1, config.head_size)
        v = v.transpose(0, 1).reshape(-1, config.head_size)
        qkv = torch.cat((q, k, v), dim=1).reshape(-1, config.n_embd * 3).transpose(0, 1)
        state_dict[f"transformer.h.{i}.attn.attn.weight"] = qkv
        del qkv_weights[i]


def layer_template(layer_name: str, idx: int) -> Tuple[str, int]:
    split = layer_name.split(".")
    number = int(split[idx])
    split[idx] = "{}"
    from_name = ".".join(split)
    return from_name, number


def load_param(param: Union[torch.Tensor, NotYetLoadedTensor]) -> torch.Tensor:
    if hasattr(param, "_load_tensor"):
        # support tensors loaded via `lazy_load()`
        return param._load_tensor()
    return param


@torch.inference_mode()
def convert_hf_checkpoint(
    *, checkpoint_dir: Path = Path("checkpoints/stabilityai/stablelm-base-alpha-3b"), model_name: Optional[str] = None
) -> None:
    if model_name is None:
        model_name = checkpoint_dir.name
    config = Config.from_name(model_name)
    print(f"Model config {config.__dict__}")
    with open(checkpoint_dir / "lit_config.json", "w") as json_config:
        json.dump(config.__dict__, json_config)

    if "falcon" in model_name:
        copy_fn = partial(copy_weights_falcon, "40b" if config.n_embd == 8192 else "7b")
    elif config._mlp_class == "LLaMAMLP":
        # holder to reconstitute the split q, k, v
        qkv_weights = {}
        copy_fn = partial(copy_weights_open_llama, config, qkv_weights)
    else:
        copy_fn = copy_weights_gpt_neox

    # initialize a new empty state dict to hold our new weights
    sd = {}

    # Load the json file containing weight mapping
    pytorch_bin_map_json_path = checkpoint_dir / "pytorch_model.bin.index.json"
    if pytorch_bin_map_json_path.is_file():  # not all checkpoints have this file
        with open(pytorch_bin_map_json_path) as json_map:
            bin_index = json.load(json_map)
        bin_files = {checkpoint_dir / bin for bin in bin_index["weight_map"].values()}
    else:
        bin_files = set(checkpoint_dir.glob("*.bin"))
    if not bin_files:
        raise ValueError(f"Expected {str(checkpoint_dir)!r} to contain .bin files")

    with incremental_save(checkpoint_dir / "lit_model.pth") as saver:
        # for checkpoints that split the QKV across several files, we need to keep all the bin files
        # open, so we use `ExitStack` to close them all together at the end
        with contextlib.ExitStack() as stack:
            for bin_file in sorted(bin_files):
                print("Processing", bin_file)
                hf_weights = stack.enter_context(lazy_load(bin_file))
                copy_fn(sd, hf_weights, saver=saver)
            gc.collect()
        saver.save(sd)


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(convert_hf_checkpoint)
