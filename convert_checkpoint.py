from contextlib import contextmanager
from pathlib import Path

import torch

"""
Sample usage:

```bash
python -m scripts.convert_checkpoint -h

python -m scripts.convert_checkpoint meta_weights_for_meta_model converted
```
"""


@contextmanager
def on_dtype(dtype):
    original = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(original)


def meta_weights_for_nano_model(
    *,
    output_dir: Path,
    ckpt_dir: Path = Path("/srv/data/checkpoints/llama/raw/7B"),
    tokenizer_path: Path = Path("/srv/data/checkpoints/llama/raw/tokenizer.model"),
    model_size: str = "7B",
):

    from pathlib import Path
    from tqdm import tqdm
    import os 
    import shutil

    output_dir = output_dir / model_size
    os.makedirs(output_dir, exist_ok=True)

    if "tokenizer.model" not in os.listdir(output_dir):
        shutil.copy(tokenizer_path, output_dir)

    tokenizer_path = output_dir / "tokenizer.model"
    checkpoint_files = sorted(ckpt_dir.glob("*.pth"))
   
    for file in tqdm(checkpoint_files, total=len(checkpoint_files)):
        converted = {}
        checkpoint = torch.load(file, map_location="cpu")
        
        converted["transformer.wte.weight"] = checkpoint["tok_embeddings.weight"]
        converted["lm_head.weight"] = checkpoint["output.weight"]
        converted["transformer.ln_f.scale"] = checkpoint["norm.weight"]  # todo: correct?

        for key in [k for k in checkpoint if k.startswith("layers")]:
            layer_idx = key.split(".")[1]

            # attention
            # the wq, wk, wv from the FB model are stacked in our model as c_attn
            converted[f"transformer.h.{layer_idx}.attn.c_attn.weight"] = torch.cat((
                # TODO: is order correct?
                checkpoint[f"layers.{layer_idx}.attention.wq.weight"],
                checkpoint[f"layers.{layer_idx}.attention.wk.weight"],
                checkpoint[f"layers.{layer_idx}.attention.wv.weight"],
            ))
            converted[f"transformer.h.{layer_idx}.attn.c_proj.weight"] = checkpoint[f"layers.{layer_idx}.attention.wo.weight"]
            # mlp
            converted[f"transformer.h.{layer_idx}.mlp.c_fc1.weight"] = checkpoint[f"layers.{layer_idx}.feed_forward.w1.weight"]
            converted[f"transformer.h.{layer_idx}.mlp.c_proj.weight"] = checkpoint[f"layers.{layer_idx}.feed_forward.w2.weight"]
            converted[f"transformer.h.{layer_idx}.mlp.c_fc2.weight"] = checkpoint[f"layers.{layer_idx}.feed_forward.w3.weight"]
            # rms norm
            converted[f"transformer.h.{layer_idx}.rms_1.scale"] = checkpoint[f"layers.{layer_idx}.attention_norm.weight"]
            converted[f"transformer.h.{layer_idx}.rms_2.scale"] = checkpoint[f"layers.{layer_idx}.ffn_norm.weight"]

        torch.save(converted, Path(output_dir, file.name))


def lightning_weights_for_nano_model():
    ...


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI([meta_weights_for_nano_model, lightning_weights_for_nano_model])
