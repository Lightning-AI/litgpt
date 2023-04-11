import gc
import os
import shutil
from pathlib import Path
from typing import Dict

import torch
from tqdm import tqdm

"""
Sample usage:

```bash
python -m scripts.convert_checkpoint -h

python -m scripts.convert_checkpoint converted
```
"""


def convert_state_dict(state_dict: Dict[str, torch.Tensor], dtype: torch.dtype = torch.float32) -> Dict[str, torch.Tensor]:
    converted = {}
    converted["transformer.wte.weight"] = state_dict["tok_embeddings.weight"].to(dtype)
    converted["lm_head.weight"] = state_dict["output.weight"].to(dtype)
    converted["transformer.ln_f.scale"] = state_dict["norm.weight"].to(dtype)

    for key in [k for k in state_dict if k.startswith("layers")]:
        layer_idx = key.split(".")[1]

        # attention
        # the wq, wk, wv from the FB model are stacked in our model as c_attn
        converted[f"transformer.h.{layer_idx}.attn.c_attn.weight"] = torch.cat(
            (
                state_dict[f"layers.{layer_idx}.attention.wq.weight"].to(dtype),
                state_dict[f"layers.{layer_idx}.attention.wk.weight"].to(dtype),
                state_dict[f"layers.{layer_idx}.attention.wv.weight"].to(dtype),
            )
        )
        converted[f"transformer.h.{layer_idx}.attn.c_proj.weight"] = state_dict[
            f"layers.{layer_idx}.attention.wo.weight"
        ].to(dtype)
        # mlp
        converted[f"transformer.h.{layer_idx}.mlp.c_fc1.weight"] = state_dict[
            f"layers.{layer_idx}.feed_forward.w1.weight"
        ].to(dtype)
        converted[f"transformer.h.{layer_idx}.mlp.c_proj.weight"] = state_dict[
            f"layers.{layer_idx}.feed_forward.w2.weight"
        ].to(dtype)
        converted[f"transformer.h.{layer_idx}.mlp.c_fc2.weight"] = state_dict[
            f"layers.{layer_idx}.feed_forward.w3.weight"
        ].to(dtype)
        # rms norm
        converted[f"transformer.h.{layer_idx}.rms_1.scale"] = state_dict[f"layers.{layer_idx}.attention_norm.weight"].to(dtype)
        converted[f"transformer.h.{layer_idx}.rms_2.scale"] = state_dict[f"layers.{layer_idx}.ffn_norm.weight"].to(dtype)
    return converted


shard_dims = {
    "lm_head.weight": 0,
    "wte.weight": 1,
    "attn.c_attn.weight": 0,
    "attn.c_proj.weight": 1,
    "mlp.c_fc1.weight": 0,
    "mlp.c_fc2.weight": 0,
    "mlp.c_proj.weight": 1
}


def meta_weights_for_nano_model(
    *,
    output_dir: Path = Path("checkpoints/lit-llama"),
    ckpt_dir: Path = Path("checkpoints/llama/"),
    tokenizer_path: Path = Path("checkpoints/llama/tokenizer.model"),
    model_size: str = "7B",
    dtype: str = None,
) -> None:
    output_dir = output_dir / model_size
    ckpt_dir = ckpt_dir / model_size
    os.makedirs(output_dir, exist_ok=True)

    if dtype is not None:
        dt = getattr(torch, dtype, None)
        if not isinstance(dt, torch.dtype):
            raise ValueError(f"{dtype} is not a valid dtype.")
        dtype = dt

    # the tokenizer is the same for all model sizes, so we store it in the parent dir
    if "tokenizer.model" not in os.listdir(output_dir.parent):
        shutil.copy(tokenizer_path, output_dir.parent)

    checkpoint_files = sorted(ckpt_dir.glob("*.pth"))
    checkpoint_files.sort()
    n_checkpoints = len(checkpoint_files)

    # for the bigger models, there are multiple model-parallel checkpoints
    # and we combine them into one single file
    combined = None
    for file in tqdm(checkpoint_files, total=n_checkpoints):
        checkpoint = torch.load(file, map_location="cpu")
        converted = convert_state_dict(checkpoint, dtype=dtype)
        if combined is None:
            combined = converted
            continue
        for name, param in converted.items():
            dim = None
            for k, d in shard_dims.items():
                if k in name:
                    dim = d
                    break
            if dim is None:
                # Extra check: assert that tensors are the same if not sharded
                # assert torch.allclose(combined[name], param)
                continue
            combined[name] = torch.cat((combined[name], param), dim=dim)

        del checkpoint
        del converted
        gc.collect()

    for name, param in combined.items():
        if "c_attn" not in name:
            continue

        # Turn [Q1, K1, V1, Q2, K2, V2, ...] into [Q1, Q2, ..., K1, K2, .., V1, V2, ...]

        src_chunk_len = param.shape[0] // n_checkpoints
        mat_len = src_chunk_len // 3
        dst_chunk_len = mat_len * n_checkpoints
        attn = torch.clone(param)
        for i in range(n_checkpoints):
            for j in range(3):
                param[j * dst_chunk_len + i * mat_len: j * dst_chunk_len + (i+1) * mat_len] = \
                    attn[i * src_chunk_len + j * mat_len: i * src_chunk_len + (j+1) * mat_len]

        del attn
        gc.collect()

    torch.save(combined, Path(output_dir, "state_dict.pth"))


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(meta_weights_for_nano_model)
