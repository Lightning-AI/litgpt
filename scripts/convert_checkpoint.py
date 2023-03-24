import json
import shutil
from contextlib import contextmanager
from pathlib import Path

import torch
from tqdm import tqdm

import models.llama as llama

"""
Sample usage:

```bash
python -m models.llama.convert_checkpoint -h

python -m models.llama.convert_checkpoint meta_weights_for_meta_model converted
```
"""

META_KEY_TO_DIM = {
    "w1": 0,
    "w2": -1,
    "w3": 0,
    "wo": -1,
    "wq": 0,
    "wk": 0,
    "wv": 0,
    "output": 0,
    "tok_embeddings": -1,
    "ffn_norm": None,
    "attention_norm": None,
    "norm": None,
    "rope": None,
}


@contextmanager
def on_dtype(dtype):
    original = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(original)


def meta_weights_for_meta_model(
    *,
    output_dir: Path,
    ckpt_dir: Path = Path("/srv/data/checkpoints/llama/raw"),
    tokenizer_path: Path = Path("/srv/data/checkpoints/llama/raw/tokenizer.model"),
    model_size: str = "7B",
):
    """
    Convert the Meta weights for the Meta model.

    Adapted from https://github.com/juncongmoo/pyllama/blob/321d475f01c88e179c8a30d68b5281e2caca5b07/llama/convert_llama.py#L297-L336
    Licensed under the Apache License, Version 2.0
    """
    model_dir = Path(output_dir) / model_size
    model_dir.mkdir(parents=True, exist_ok=True)
    if not (model_dir / "tokenizer.model").exists():
        shutil.copy(tokenizer_path, output_dir)

    tokenizer_path = output_dir / "tokenizer.model"
    tokenizer = llama.Tokenizer(model_path=str(tokenizer_path))

    ckpt_dir = ckpt_dir / model_size
    with open(ckpt_dir / "params.json") as f:
        params = json.load(f)
    params["vocab_size"] = tokenizer.vocab_size
    params.pop("multiple_of", None)  # unused parameter
    print("Model size:", model_size, "Params:", params)
    model_args = llama.ModelArgs(**params)

    print(f"⌛️ Loading model...Thank you for your patience...")
    with on_dtype(torch.half):
        model = llama.LLaMA(model_args)

    dt = {}
    print(f"⌛️ Converting model...Thank you for your patience...")
    cks = sorted(ckpt_dir.glob("*.pth"))
    for i, ckpt in tqdm(enumerate(cks), total=len(cks)):
        ck = torch.load(ckpt, map_location="cpu")
        for nm, pm in model.named_parameters():
            if nm not in dt:
                dt[nm] = torch.zeros_like(pm, device="cpu")
            short_name = nm.split(".")[-2]
            if META_KEY_TO_DIM[short_name] is None and i == 0:
                dt[nm] = ck[nm]
            elif META_KEY_TO_DIM[short_name] == 0:
                size = ck[nm].size(0)
                dt[nm][size * i : size * (i + 1), :] = ck[nm]
            elif META_KEY_TO_DIM[short_name] == -1:
                size = ck[nm].size(-1)
                dt[nm][:, size * i : size * (i + 1)] = ck[nm]

    with open(output_dir / "params.json", "w") as f:
        json.dump(params, f, indent=4)
    torch.save(dt, output_dir / "state_dict.pt")


def meta_weights_for_lightning_model():
    ...


def lightning_weights_for_lightning_model():
    ...


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI([meta_weights_for_meta_model, meta_weights_for_lightning_model, lightning_weights_for_lightning_model])
