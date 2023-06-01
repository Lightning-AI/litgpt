import contextlib
import gc
import json
import sys
from pathlib import Path
from typing import Optional

import torch

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_parrot import Config
from lit_parrot.utils import lazy_load, incremental_save


def copy_weights(state_dict, hf_weights, saver=None, dtype=torch.float32):
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
        if hasattr(param, "_load_tensor"):
            # support tensors loaded via `lazy_load()`
            param = param._load_tensor()
        param = param.to(dtype=dtype)
        if "gpt_neox.layers" in name:
            split = name.split(".")
            block_id = int(split[2])
            split[2] = "{}"
            from_name = ".".join(split)
            to_name = weight_map[from_name]
            if to_name is None:
                continue
            to_name = to_name.format(block_id)
        else:
            to_name = weight_map[name]
        if saver is not None:
            param = saver.store_early(param)
        state_dict[to_name] = param


@torch.inference_mode()
def convert_hf_checkpoint(
    *,
    checkpoint_dir: Path = Path("checkpoints/stabilityai/stablelm-base-alpha-3b"),
    model_name: Optional[str] = None,
    dtype: str = "float32",
) -> None:
    dt = getattr(torch, dtype, None)
    if not isinstance(dt, torch.dtype):
        raise ValueError(f"{dtype} is not a valid dtype.")
    dtype = dt

    if model_name is None:
        model_name = checkpoint_dir.name
    config = Config.from_name(model_name)
    print(f"Model config {config.__dict__}")

    with open(checkpoint_dir / "lit_config.json", "w") as json_config:
        json.dump(config.__dict__, json_config)

    # initialize a new empty state dict to hold our new weights
    sd = {}

    # Load the json file containing weight mapping
    pytorch_bin_map_json_path = checkpoint_dir / "pytorch_model.bin.index.json"
    if pytorch_bin_map_json_path.is_file():  # not all checkpoints have this file
        with open(pytorch_bin_map_json_path) as json_map:
            bin_index = json.load(json_map)
        bin_files = set(checkpoint_dir / bin for bin in bin_index["weight_map"].values())
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
                copy_weights(sd, hf_weights, saver=saver, dtype=dtype)
            gc.collect()
        saver.save(sd)


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(convert_hf_checkpoint)
