import gc
import json
import shutil
import sys
from pathlib import Path
from typing import Optional

import torch

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_parrot.model import Parrot
from lit_parrot.utils import EmptyInitOnDevice, check_valid_checkpoint_dir


def copy_weights(state_dict, hf_weights, dtype=torch.float32):
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
        print(f"{name} {tuple(param.shape)} ⟶ {to_name} {tuple(state_dict[to_name].shape)}")
        state_dict[to_name].copy_(param)


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
    with EmptyInitOnDevice(device="cpu", dtype=dtype):
        model = Parrot.from_name(model_name)
    print(f"Model config {model.config.__dict__}")
    with open(checkpoint_dir / "lit_config.json", "w") as json_config:
        json.dump(model.config.__dict__, json_config)

    # initialize a new empty state dict to hold our new weights
    sd = model.state_dict()

    bin_files = list(checkpoint_dir.glob("*.bin"))
    if not bin_files:
        raise ValueError(f"Expected {str(checkpoint_dir)!r} to contain .bin files")
    for bin_file in sorted(bin_files):
        print("Processing", bin_file)
        hf_weights = torch.load(bin_file, map_location="cpu")
        copy_weights(sd, hf_weights, dtype=dtype)
        del hf_weights
        gc.collect()

    model_path = checkpoint_dir / "lit_model.pth"
    print(f"Saving to disk at {str(model_path)!r}")
    torch.save(model.state_dict(), model_path)


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(convert_hf_checkpoint)
