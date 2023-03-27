import os
import shutil
from pathlib import Path

import torch
from tqdm import tqdm

"""
Sample usage:

```bash
python -m scripts.convert_checkpoint -h

python -m scripts.convert_checkpoint converted
```
"""


def convert_state_dict(state_dict):
    converted = {}
    converted["transformer.wte.weight"] = state_dict["tok_embeddings.weight"]
    converted["lm_head.weight"] = state_dict["output.weight"]
    converted["transformer.ln_f.scale"] = state_dict["norm.weight"]

    for key in [k for k in state_dict if k.startswith("layers")]:
        layer_idx = key.split(".")[1]

        # attention
        # the wq, wk, wv from the FB model are stacked in our model as c_attn
        converted[f"transformer.h.{layer_idx}.attn.c_attn.weight"] = torch.cat(
            (
                state_dict[f"layers.{layer_idx}.attention.wq.weight"],
                state_dict[f"layers.{layer_idx}.attention.wk.weight"],
                state_dict[f"layers.{layer_idx}.attention.wv.weight"],
            )
        )
        converted[f"transformer.h.{layer_idx}.attn.c_proj.weight"] = state_dict[
            f"layers.{layer_idx}.attention.wo.weight"
        ]
        # mlp
        converted[f"transformer.h.{layer_idx}.mlp.c_fc1.weight"] = state_dict[
            f"layers.{layer_idx}.feed_forward.w1.weight"
        ]
        converted[f"transformer.h.{layer_idx}.mlp.c_proj.weight"] = state_dict[
            f"layers.{layer_idx}.feed_forward.w2.weight"
        ]
        converted[f"transformer.h.{layer_idx}.mlp.c_fc2.weight"] = state_dict[
            f"layers.{layer_idx}.feed_forward.w3.weight"
        ]
        # rms norm
        converted[f"transformer.h.{layer_idx}.rms_1.scale"] = state_dict[f"layers.{layer_idx}.attention_norm.weight"]
        converted[f"transformer.h.{layer_idx}.rms_2.scale"] = state_dict[f"layers.{layer_idx}.ffn_norm.weight"]
    return converted


def meta_weights_for_nano_model(
    *,
    output_dir: Path = Path("checkpoints/lit-llama"),
    ckpt_dir: Path = Path("checkpoints/llama/"),
    tokenizer_path: Path = Path("checkpoints/llama/tokenizer.model"),
    model_size: str = "7B",
):
    output_dir = output_dir / model_size
    ckpt_dir = ckpt_dir / model_size
    os.makedirs(output_dir, exist_ok=True)

    # the tokenizer is the same for all model sizes, so we store it in the parent dir
    if "tokenizer.model" not in os.listdir(output_dir.parent):
        shutil.copy(tokenizer_path, output_dir.parent)

    checkpoint_files = sorted(ckpt_dir.glob("*.pth"))

    # for the bigger models, there are multiple model-parallel checkpoints
    # and we combine them into one single file
    combined = {}
    for file in tqdm(checkpoint_files, total=len(checkpoint_files)):
        checkpoint = torch.load(file, map_location="cpu")
        converted = convert_state_dict(checkpoint)
        combined.update(converted)

    torch.save(combined, Path(output_dir, "state_dict.pth"))


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(meta_weights_for_nano_model)
