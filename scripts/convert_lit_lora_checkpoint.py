import contextlib
import gc
import sys
import time
from functools import partial
from pathlib import Path
from typing import Optional

import torch

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

import lightning as L
from lit_gpt.lora import Config
from lit_gpt.utils import incremental_save, lazy_load
from scripts.convert_lit_checkpoint import (
    check_conversion_supported,
    copy_weights_falcon,
    copy_weights_gpt_neox,
    copy_weights_llama,
)
from scripts.merge_lora import merge_lora

lora_r = 8
lora_alpha = 16
lora_dropout = 0.05
lora_query = True
lora_key = False
lora_value = True
lora_projection = False
lora_mlp = False
lora_head = False


def convert_lora_checkpoint(checkpoint_dir: Path, out_dir: Path) -> None:
    fabric = L.Fabric(devices=1)

    config = Config.from_json(
        checkpoint_dir / "lit_config.json",
        r=lora_r,
        alpha=lora_alpha,
        dropout=lora_dropout,
        to_query=lora_query,
        to_key=lora_key,
        to_value=lora_value,
        to_projection=lora_projection,
        to_mlp=lora_mlp,
        to_head=lora_head,
    )

    model_name = config.name
    # set copy_fn according to model_name
    if "falcon" in model_name:
        copy_fn = partial(copy_weights_falcon, "40b" if config.n_embd == 8192 else "7b")
    elif config._mlp_class == "LLaMAMLP":
        copy_fn = partial(copy_weights_llama, config)
    else:
        copy_fn = copy_weights_gpt_neox
    # set bin file path
    pth_file = out_dir / "lit_model.pth"
    bin_file = checkpoint_dir / "converted_merged_lora_model.bin"
    # initialize a new empty state dict to hold our new weights
    sd = {}
    # convert and save
    t0 = time.perf_counter()
    fabric.print("\nConverting Model Format", file=sys.stderr)
    with incremental_save(bin_file) as saver:
        with contextlib.ExitStack() as stack:
            lit_weights = stack.enter_context(lazy_load(pth_file))
            lit_weights = lit_weights.get("model", lit_weights)
            check_conversion_supported(lit_weights)
            copy_fn(sd, lit_weights, saver=saver)
            gc.collect()
        saver.save(sd)
    t = time.perf_counter() - t0
    fabric.print(f"\nTime for Conversion: {t:.02f} sec total", file=sys.stderr)
    fabric.print(
        f"\nSaving converted, merged checkpoint to {bin_file}",
        file=sys.stderr,
    )


@torch.inference_mode()
def convert_lit_lora_checkpoint(
    *,
    checkpoint_dir: Path,
    out_dir: Path,
    lora_path: Optional[Path] = None,
    precision: Optional[str] = None,
    merge: bool = False,
) -> None:
    """Converts a Lit-GPT checkpoint to its original weight format.

    Args:
        model_name: The name of the model.
        checkpoint_path: The path to a Lit-GPT formatted checkpoint.
        lora_path: The finetuned checkpoint found in `out/lora/`.
        lora_config_path: The path to the GPT Config used during finetuning.
        merge_lora: Bool to indicate if merging should be handled before conversion.
    """

    if merge:
        merge_lora(lora_path=lora_path, checkpoint_dir=checkpoint_dir, out_dir=out_dir, precision=precision)

    convert_lora_checkpoint(checkpoint_dir=checkpoint_dir, out_dir=out_dir)


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(convert_lit_lora_checkpoint, as_positional=False)
