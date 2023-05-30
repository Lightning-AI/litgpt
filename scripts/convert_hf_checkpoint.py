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

from lit_parrot.model import Parrot
from lit_parrot.utils import EmptyInitOnDevice, lazy_load, incremental_save


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
    print(f"Initializing model {model_name!r}")
    with EmptyInitOnDevice(device="meta", dtype=dtype):
        model = Parrot.from_name(model_name)
    print(f"Model config {model.config.__dict__}")
    with open(checkpoint_dir / "lit_config.json", "w") as json_config:
        json.dump(model.config.__dict__, json_config)

    # initialize a new empty state dict to hold our new weights
    sd = {}

    bin_files = list(checkpoint_dir.glob("*.bin"))
    if not bin_files:
        raise ValueError(f"Expected {str(checkpoint_dir)!r} to contain .bin files")
    model_path = checkpoint_dir / "lit_model.pth"
    with contextlib.ExitStack() as stack:
        saver = stack.enter_context(incremental_save(model_path))
        for bin_file in sorted(bin_files):
            print("Processing", bin_file)
            hf_weights = stack.enter_context(lazy_load(bin_file))
            copy_weights(sd, saver, hf_weights, dtype=dtype)
            gc.collect()
        saver.save(sd)


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(convert_hf_checkpoint)
