# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import gc
import json
from collections import defaultdict
from functools import partial
import os
from pathlib import Path
from pprint import pprint
from typing import Dict, List, Optional, Tuple, Union

from tqdm import tqdm
import torch
from lightning.fabric.utilities.load import _NotYetLoadedTensor as NotYetLoadedTensor

from litgpt.config import Config
from litgpt.utils import extend_checkpoint_dir, incremental_save, lazy_load, save_config


def copy_weights_gpt_neox(
    state_dict: Dict[str, torch.Tensor],
    hf_weights: Dict[str, Union[torch.Tensor, NotYetLoadedTensor]],
    saver: Optional[incremental_save] = None,
    dtype: Optional[torch.dtype] = None,
    pbar: Optional[tqdm] = None,
    progress_per_file: Optional[float] = None,
    debug_mode: Optional[bool] = False

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

    if progress_per_file is not None:
        progress_per_file = progress_per_file / max(1, len(hf_weights))

    for name, param in hf_weights.items():
        if "gpt_neox.layers" in name:
            from_name, number = layer_template(name, 2)
            to_name = weight_map[from_name]
            if to_name is None:
                continue
            to_name = to_name.format(number)
        else:
            to_name = weight_map[name]
        param = load_param(param, name, dtype, verbose=debug_mode)
        if saver is not None:
            param = saver.store_early(param)
        state_dict[to_name] = param

        if progress_per_file is not None:
            pbar.update(progress_per_file)


def copy_weights_falcon(
    model_name: str,
    state_dict: Dict[str, torch.Tensor],
    hf_weights: Dict[str, Union[torch.Tensor, NotYetLoadedTensor]],
    saver: Optional[incremental_save] = None,
    dtype: Optional[torch.dtype] = None,
    pbar: Optional[tqdm] = None,
    progress_per_file: Optional[float] = None,
    debug_mode: Optional[bool] = False
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
    if "7b" in model_name:
        weight_map.update(
            {
                "transformer.h.{}.input_layernorm.bias": "transformer.h.{}.norm_1.bias",
                "transformer.h.{}.input_layernorm.weight": "transformer.h.{}.norm_1.weight",
            }
        )
    elif "40b" in model_name or "180B" in model_name:
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

    if progress_per_file is not None:
        progress_per_file = progress_per_file / max(1, len(hf_weights))

    for name, param in hf_weights.items():
        if "transformer.h" in name:
            from_name, number = layer_template(name, 2)
            to_name = weight_map[from_name].format(number)
        else:
            to_name = weight_map[name]
        param = load_param(param, name, dtype, verbose=debug_mode)
        if saver is not None:
            param = saver.store_early(param)
        state_dict[to_name] = param
        if progress_per_file is not None:
            pbar.update(progress_per_file)


def copy_weights_hf_llama(
    config: Config,
    qkv_weights: Dict[int, List[Optional[NotYetLoadedTensor]]],
    state_dict: Dict[str, torch.Tensor],
    hf_weights: Dict[str, Union[torch.Tensor, NotYetLoadedTensor]],
    saver: Optional[incremental_save] = None,
    dtype: Optional[torch.dtype] = None,
    pbar: Optional[tqdm] = None,
    progress_per_file: Optional[float] = None,
    debug_mode: Optional[bool] = False
) -> None:
    weight_map = {
        "model.embed_tokens.weight": "transformer.wte.weight",
        "model.layers.{}.input_layernorm.weight": "transformer.h.{l}.norm_1.weight",
        "model.layers.{}.input_layernorm.bias": "transformer.h.{l}.norm_1.bias",
        "model.layers.{}.self_attn.q_proj.weight": None,
        "model.layers.{}.self_attn.k_proj.weight": None,
        "model.layers.{}.self_attn.v_proj.weight": None,
        "model.layers.{}.self_attn.o_proj.weight": "transformer.h.{l}.attn.proj.weight",
        "model.layers.{}.self_attn.rotary_emb.inv_freq": None,
        "model.layers.{}.post_attention_layernorm.weight": "transformer.h.{l}.norm_2.weight",
        "model.layers.{}.post_attention_layernorm.bias": "transformer.h.{l}.norm_2.bias",
        "model.norm.weight": "transformer.ln_f.weight",
        "model.norm.bias": "transformer.ln_f.bias",
        "lm_head.weight": "lm_head.weight",
    }
    if config.mlp_class_name == "LLaMAMoE":
        weight_map.update(
            {
                "model.layers.{}.block_sparse_moe.gate.weight": "transformer.h.{l}.mlp.gate.weight",
                "model.layers.{}.block_sparse_moe.experts.{}.w1.weight": "transformer.h.{l}.mlp.experts.{e}.fc_1.weight",
                "model.layers.{}.block_sparse_moe.experts.{}.w3.weight": "transformer.h.{l}.mlp.experts.{e}.fc_2.weight",
                "model.layers.{}.block_sparse_moe.experts.{}.w2.weight": "transformer.h.{l}.mlp.experts.{e}.proj.weight",
            }
        )
    elif config.mlp_class_name in ("LLaMAMLP", "GemmaMLP"):
        weight_map.update(
            {
                "model.layers.{}.mlp.gate_proj.weight": "transformer.h.{l}.mlp.fc_1.weight",
                "model.layers.{}.mlp.up_proj.weight": "transformer.h.{l}.mlp.fc_2.weight",
                "model.layers.{}.mlp.down_proj.weight": "transformer.h.{l}.mlp.proj.weight",
            }
        )
    else:
        raise NotImplementedError

    if progress_per_file is not None:
        progress_per_file = progress_per_file / max(1, len(hf_weights) + len(qkv_weights))

    for name, param in hf_weights.items():
        if "model.layers" in name:
            from_name, l = layer_template(name, 2)
            e = None
            if "block_sparse_moe.experts" in name:
                from_name, e = layer_template(from_name, 5)
            qkv = qkv_weights.setdefault(l, [None, None, None])
            if "q_proj" in name:
                qkv[0] = param
            elif "k_proj" in name:
                qkv[1] = param
            elif "v_proj" in name:
                qkv[2] = param
            to_name = weight_map[from_name]
            if to_name is None:
                continue
            to_name = to_name.format(l=l, e=e)
        else:
            to_name = weight_map[name]
        param = load_param(param, name, dtype, verbose=debug_mode)
        if saver is not None:
            param = saver.store_early(param)
        state_dict[to_name] = param

        if progress_per_file is not None:
            pbar.update(progress_per_file)

    if "lm_head.weight" not in state_dict:
        state_dict["lm_head.weight"] = state_dict["transformer.wte.weight"]

    # convert separate q, k, v matrices into an interleaved qkv
    for i, (q, k, v) in list(qkv_weights.items()):
        if q is None or k is None or v is None:
            # split across different .bin files
            continue
        q = load_param(q, f"layer {i} q", dtype, verbose=debug_mode)
        k = load_param(k, f"layer {i} k", dtype, verbose=debug_mode)
        v = load_param(v, f"layer {i} v", dtype, verbose=debug_mode)
        q_per_kv = config.n_head // config.n_query_groups
        qs = torch.split(q, config.head_size * q_per_kv)
        ks = torch.split(k, config.head_size)
        vs = torch.split(v, config.head_size)
        cycled = [t for group in zip(qs, ks, vs) for t in group]
        qkv = torch.cat(cycled)
        state_dict[f"transformer.h.{i}.attn.attn.weight"] = qkv
        del qkv_weights[i]
        if progress_per_file is not None:
            pbar.update(progress_per_file)


def copy_weights_gemma_2(
    config: Config,
    qkv_weights: Dict[int, List[Optional[NotYetLoadedTensor]]],
    state_dict: Dict[str, torch.Tensor],
    hf_weights: Dict[str, Union[torch.Tensor, NotYetLoadedTensor]],
    saver: Optional[incremental_save] = None,
    dtype: Optional[torch.dtype] = None,
    pbar: Optional[tqdm] = None,
    progress_per_file: Optional[float] = None,
    debug_mode: Optional[bool] = False
) -> None:
    weight_map = {
        "model.embed_tokens.weight": "transformer.wte.weight",
        "model.layers.{}.self_attn.q_proj.weight": None,
        "model.layers.{}.self_attn.k_proj.weight": None,
        "model.layers.{}.self_attn.v_proj.weight": None,
        "model.layers.{}.self_attn.o_proj.weight": "transformer.h.{}.attn.proj.weight",
        "model.layers.{}.mlp.gate_proj.weight": "transformer.h.{}.mlp.fc_1.weight",
        "model.layers.{}.mlp.up_proj.weight": "transformer.h.{}.mlp.fc_2.weight",
        "model.layers.{}.mlp.down_proj.weight": "transformer.h.{}.mlp.proj.weight",
        "model.layers.{}.input_layernorm.weight": "transformer.h.{}.norm_1.weight",
        "model.layers.{}.post_attention_layernorm.weight": "transformer.h.{}.post_attention_norm.weight",
        "model.layers.{}.pre_feedforward_layernorm.weight": "transformer.h.{}.norm_2.weight",
        "model.layers.{}.post_feedforward_layernorm.weight": "transformer.h.{}.post_mlp_norm.weight",
        "model.norm.weight": "transformer.ln_f.weight",
        "lm_head.weight": "lm_head.weight",
    }

    if progress_per_file is not None:
        progress_per_file = progress_per_file / max(1, len(hf_weights) + len(qkv_weights))

    for name, param in hf_weights.items():
        if "model.layers" in name:
            from_name, l_idx = layer_template(name, 2)
            qkv = qkv_weights.setdefault(l_idx, defaultdict(dict))
            if any(w in from_name for w in ("q_proj", "k_proj", "v_proj")):
                weight_name, weight_type = from_name.split(".")[-2:]
                qkv[weight_type][weight_name] = param
            to_name = weight_map[from_name]
            if to_name is None:
                continue
            to_name = to_name.format(l_idx)
        else:
            to_name = weight_map[name]
        param = load_param(param, name, dtype)
        if saver is not None:
            param = saver.store_early(param)
        state_dict[to_name] = param

        if progress_per_file is not None:
            pbar.update(progress_per_file)

    if "lm_head.weight" not in state_dict:
        state_dict["lm_head.weight"] = state_dict["transformer.wte.weight"]

    # convert separate q, k, v matrices into an interleaved qkv
    for i in list(qkv_weights):
        for weight_type in list(qkv_weights[i]):
            qkv = qkv_weights[i][weight_type]
            if len(qkv) != 3:
                # split across different .bin files
                continue
            q = load_param(qkv["q_proj"], f"layer {i} q {weight_type}", dtype)
            k = load_param(qkv["k_proj"], f"layer {i} k {weight_type}", dtype)
            v = load_param(qkv["v_proj"], f"layer {i} v {weight_type}", dtype)
            q_per_kv = config.n_head // config.n_query_groups
            qs = torch.split(q, config.head_size * q_per_kv)
            ks = torch.split(k, config.head_size)
            vs = torch.split(v, config.head_size)
            cycled = [t for group in zip(qs, ks, vs) for t in group]
            qkv = torch.cat(cycled)
            state_dict[f"transformer.h.{i}.attn.attn.{weight_type}"] = qkv
            del qkv_weights[i][weight_type]
            if progress_per_file is not None:
                pbar.update(progress_per_file)


def copy_weights_phi(
    config: Config,
    qkv_weights: dict,
    state_dict: Dict[str, torch.Tensor],
    hf_weights: Dict[str, Union[torch.Tensor, NotYetLoadedTensor]],
    saver: Optional[incremental_save] = None,
    dtype: Optional[torch.dtype] = None,
    pbar: Optional[tqdm] = None,
    progress_per_file: Optional[float] = None,
    debug_mode: Optional[bool] = False
) -> None:
    if any(layer_name.startswith(("layers.", "transformer.")) for layer_name in hf_weights):
        raise ValueError(
            "You are using an outdated Phi checkpoint. Please reload it as described in 'tutorials/download_phi.md'"
        )

    weight_map = {
        "model.embed_tokens.weight": "transformer.wte.weight",
        "model.layers.{}.input_layernorm.weight": "transformer.h.{}.norm_1.weight",
        "model.layers.{}.input_layernorm.bias": "transformer.h.{}.norm_1.bias",
        "model.layers.{}.self_attn.q_proj.weight": None,
        "model.layers.{}.self_attn.q_proj.bias": None,
        "model.layers.{}.self_attn.k_proj.weight": None,
        "model.layers.{}.self_attn.k_proj.bias": None,
        "model.layers.{}.self_attn.v_proj.weight": None,
        "model.layers.{}.self_attn.v_proj.bias": None,
        "model.layers.{}.self_attn.dense.weight": "transformer.h.{}.attn.proj.weight",
        "model.layers.{}.self_attn.dense.bias": "transformer.h.{}.attn.proj.bias",
        "model.layers.{}.mlp.fc1.weight": "transformer.h.{}.mlp.fc.weight",
        "model.layers.{}.mlp.fc1.bias": "transformer.h.{}.mlp.fc.bias",
        "model.layers.{}.mlp.fc2.weight": "transformer.h.{}.mlp.proj.weight",
        "model.layers.{}.mlp.fc2.bias": "transformer.h.{}.mlp.proj.bias",
        "model.final_layernorm.weight": "transformer.ln_f.weight",
        "model.final_layernorm.bias": "transformer.ln_f.bias",
        "lm_head.weight": "lm_head.weight",
        "lm_head.bias": "lm_head.bias",
    }

    if config.name.startswith("Phi-3"):
        weight_map.update(
            {
                "model.layers.{}.self_attn.qkv_proj.weight": "transformer.h.{}.attn.attn.weight",
                "model.layers.{}.self_attn.o_proj.weight": "transformer.h.{}.attn.proj.weight",
                "model.layers.{}.post_attention_layernorm.weight": "transformer.h.{}.norm_2.weight",
                "model.layers.{}.mlp.down_proj.weight": "transformer.h.{}.mlp.proj.weight",
                "model.norm.weight": "transformer.ln_f.weight",
            }
        )

    if progress_per_file is not None:
        progress_per_file = progress_per_file / max(1, len(hf_weights) + len(qkv_weights))

    for name, param in hf_weights.items():
        if name.startswith("model.layers."):
            from_name, l = layer_template(name, 2)
            qkv = qkv_weights.setdefault(l, defaultdict(dict))
            if "qkv_proj" in from_name:
                weight = load_param(param, f"layer {l} qkv", dtype)
                weight = qkv_reassemble(weight, config)
                to_name = weight_map[from_name].format(l)
                state_dict[to_name] = weight
                continue
            if any(w in from_name for w in ("q_proj", "k_proj", "v_proj")):
                weight_name, weight_type = from_name.split(".")[-2:]
                qkv[weight_type][weight_name] = param
            elif from_name.endswith("gate_up_proj.weight"):
                weight = load_param(param, f"layer {l} gate_up_proj", dtype)
                fc_1, fc_2 = weight.chunk(2, dim=0)
                state_dict[f"transformer.h.{l}.mlp.fc_1.weight"] = fc_1
                state_dict[f"transformer.h.{l}.mlp.fc_2.weight"] = fc_2
                continue
            to_name = weight_map[from_name]
            if to_name is None:
                continue
            to_name = to_name.format(l)
        else:
            to_name = weight_map[name]
        param = load_param(param, name, dtype, verbose=debug_mode)
        if saver is not None:
            param = saver.store_early(param)
        state_dict[to_name] = param
        if progress_per_file is not None:
            pbar.update(progress_per_file)

    for i in list(qkv_weights):
        for weight_type in list(qkv_weights[i]):
            qkv = qkv_weights[i][weight_type]
            if len(qkv) != 3:
                # split across different .bin files
                continue
            q = load_param(qkv["q_proj"], f"layer {i} q {weight_type}", dtype, verbose=debug_mode)
            k = load_param(qkv["k_proj"], f"layer {i} k {weight_type}", dtype, verbose=debug_mode)
            v = load_param(qkv["v_proj"], f"layer {i} v {weight_type}", dtype, verbose=debug_mode)
            q_per_kv = config.n_head // config.n_query_groups
            qs = torch.split(q, config.head_size * q_per_kv)
            ks = torch.split(k, config.head_size)
            vs = torch.split(v, config.head_size)
            cycled = [t for group in zip(qs, ks, vs) for t in group]
            qkv = torch.cat(cycled)
            state_dict[f"transformer.h.{i}.attn.attn.{weight_type}"] = qkv
            del qkv_weights[i][weight_type]
            if progress_per_file is not None:
                pbar.update(progress_per_file)


def qkv_reassemble(param: Union[torch.Tensor, NotYetLoadedTensor], config: Config) -> torch.Tensor:
    """Reassemble from a normal to an interleaved placement in a QKV matrix.
    [Q, Q, ..., K, K, ..., V, V, ...] --> [Q, K, V, Q, K, V, ...]
    """
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
    interleaved = [t for group in zip(qs, ks, vs) for t in group]
    return torch.cat(interleaved)


def layer_template(layer_name: str, idx: int) -> Tuple[str, int]:
    split = layer_name.split(".")
    number = int(split[idx])
    split[idx] = "{}"
    from_name = ".".join(split)
    return from_name, number


def load_param(param: Union[torch.Tensor, NotYetLoadedTensor], name: str, dtype: Optional[torch.dtype], verbose=False) -> torch.Tensor:
    if hasattr(param, "_load_tensor"):
        # support tensors loaded via `lazy_load()`
        if verbose:
            print(f"Loading {name!r} into RAM")
        param = param._load_tensor()
    if dtype is not None and type(dtype) is not NotYetLoadedTensor and dtype != param.dtype:
        if verbose:
            print(f"Converting {name!r} from {param.dtype} to {dtype}")
        param = param.to(dtype)
    return param


@torch.inference_mode()
def convert_hf_checkpoint(
    checkpoint_dir: Path,
    *,
    model_name: Optional[str] = None,
    dtype: Optional[str] = None,
    debug_mode: Optional[bool] = False
) -> None:
    """
    Convert a Hugging Face Transformers checkpoint into a LitGPT compatible checkpoint.

    Arguments:
        checkpoint_dir: Where to save the downloaded files.
        model_name: The existing config name to load. This is useful to download alternative weights of existing
            architectures.
        dtype: The data type to convert the checkpoint files to. If not specified, the weights will remain in the
            dtype they are downloaded in.
        debug_mode: Prints the individual layers being loaded instead of a progress bar, which can be useful when
            developing and adding new models to LitGPT.
    """
    checkpoint_dir = extend_checkpoint_dir(checkpoint_dir)
    pprint(locals())

    if model_name is None:
        model_name = checkpoint_dir.name
    if dtype is not None:
        dtype = getattr(torch, dtype)

    config = Config.from_name(model_name)
    save_config(config, checkpoint_dir)

    if "falcon" in model_name:
        copy_fn = partial(copy_weights_falcon, model_name)
    elif model_name.lower().startswith("gemma-2"):
        qkv_weights = {}
        copy_fn = partial(copy_weights_gemma_2, config, qkv_weights)
    elif model_name.lower().startswith("phi"):
        # holder to reconstitute the split q, k, v
        qkv_weights = {}
        copy_fn = partial(copy_weights_phi, config, qkv_weights)
    elif config.mlp_class_name in ("LLaMAMLP", "GemmaMLP", "LLaMAMoE"):
        # holder to reconstitute the split q, k, v
        qkv_weights = {}
        copy_fn = partial(copy_weights_hf_llama, config, qkv_weights)
    else:
        copy_fn = copy_weights_gpt_neox

    # initialize a new empty state dict to hold our new weights
    sd = {}

    # Load the json file containing weight mapping
    pytorch_bin_map_json_path = checkpoint_dir / "pytorch_model.bin.index.json"
    model_safetensor_map_json_path = checkpoint_dir / "model.safetensors.index.json"
    if pytorch_bin_map_json_path.is_file():  # not all checkpoints have this file
        with open(pytorch_bin_map_json_path, encoding="utf-8") as json_map:
            bin_index = json.load(json_map)
        bin_files = {checkpoint_dir / bin for bin in bin_index["weight_map"].values()}
    elif model_safetensor_map_json_path.is_file():
        with open(model_safetensor_map_json_path, encoding="utf-8") as json_map:
            bin_index = json.load(json_map)
        bin_files = {checkpoint_dir / Path(bin).with_suffix(".bin") for bin in bin_index["weight_map"].values()}
    else:
        bin_files = set(checkpoint_dir.glob("*.bin"))
        # some checkpoints serialize the training arguments
        bin_files = {f for f in bin_files if f.name != "training_args.bin"}
    if not bin_files:
        raise ValueError(f"Expected {str(checkpoint_dir)!r} to contain .bin files")

    with incremental_save(checkpoint_dir / "lit_model.pth") as saver:
        # for checkpoints that split the QKV across several files, we need to keep all the bin files
        # open, so we use `ExitStack` to close them all together at the end

        if not debug_mode:
            # Using tqdm progress bar when not in debug mode

            total_size = max(1, sum(os.path.getsize(bin_file) for bin_file in bin_files))
            total_progress = 100

            with tqdm(total=total_progress, desc="Initializing", bar_format="{desc}{percentage:3.0f}%|{bar}| {elapsed}<{remaining}, {rate_fmt}") as pbar:
                for bin_file in sorted(bin_files):
                    pbar.set_description(f"Loading weights: {bin_file.name}")
                    current_file_size = os.path.getsize(bin_file)
                    progress_per_file = (current_file_size / total_size) * total_progress

                    hf_weights = lazy_load(bin_file)
                    copy_fn(sd, hf_weights, saver=saver, dtype=dtype, pbar=pbar, progress_per_file=progress_per_file, debug_mode=debug_mode)
                gc.collect()

                if pbar.n < total_progress:
                    pbar.update(total_progress - pbar.n)
                pbar.close()
        else:
            # Handling files without progress bar in debug mode
            for bin_file in sorted(bin_files):
                hf_weights = lazy_load(bin_file)
                copy_fn(sd, hf_weights, saver=saver, dtype=dtype, debug_mode=debug_mode)

        print(f"Saving converted checkpoint to {checkpoint_dir}")
        saver.save(sd)
