from contextlib import contextmanager
from pathlib import Path

import torch

"""
Sample usage:

```bash
python -m models.llama.convert_checkpoint -h

python -m models.llama.convert_checkpoint meta_weights_for_meta_model converted
```
"""


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
    ...


def meta_weights_for_nano_model():
    ...


def lightning_weights_for_nano_model():
    ...


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI([meta_weights_for_meta_model, meta_weights_for_nano_model, lightning_weights_for_nano_model])
