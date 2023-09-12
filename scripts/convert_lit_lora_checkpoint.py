import contextlib
import gc
import json
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
from lit_gpt.lora import Config as LoRAConfig
from lit_gpt.utils import incremental_save, lazy_load
from scripts.convert_lit_checkpoint import (
    check_conversion_supported,
    copy_weights_falcon,
    copy_weights_gpt_neox,
    copy_weights_llama,
)
from scripts.merge_lora_weights import merge_lora_weights


def convert_lora_checkpoint(model_name: str, checkpoint_path: Path, lora_config_path: Path) -> None:
    fabric = L.Fabric(devices=1)

    with open(lora_config_path, "r") as lora_config:
        config = json.load(lora_config)["config"]

    config = LoRAConfig(**config)

    # set copy_fn according to model_name
    if "falcon" in model_name:
        copy_fn = partial(copy_weights_falcon, "40b" if config.n_embd == 8192 else "7b")
    elif config._mlp_class == "LLaMAMLP":
        copy_fn = partial(copy_weights_llama, config)
    else:
        copy_fn = copy_weights_gpt_neox
    # set bin file path
    bin_file = checkpoint_path.parent / "converted_merged_lora_model.bin"
    # initialize a new empty state dict to hold our new weights
    sd = {}
    # convert and save
    t0 = time.perf_counter()
    fabric.print(f"\nConverting Model Format", file=sys.stderr)
    with incremental_save(bin_file) as saver:
        with contextlib.ExitStack() as stack:
            lit_weights = stack.enter_context(lazy_load(checkpoint_path))
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
    model_name: str,
    checkpoint_path: Path,
    lora_path: Optional[Path] = None,
    lora_config_path: Optional[Path] = None,
    merge_lora: bool = False,
    save_merge: bool = False,
) -> None:
    """Converts a Lit-GPT checkpoint to its original weight format.

    Args:
        model_name: The name of the model.
        checkpoint_path: The path to a Lit-GPT formatted checkpoint.
        lora_path: The finetuned checkpoint found in `out/lora/`.
        lora_config_path: The path to the GPT Config used during finetuning.
        merge_lora: Bool to indicate if merging should be handled before conversion.
    """

    if merge_lora:
        merge_lora_weights(checkpoint_path=checkpoint_path, lora_path=lora_path, save_merge=save_merge)
        checkpoint_path = checkpoint_path.parent / "lit_model_lora_merged.pth"
        lora_config_path = lora_path.parent / "lit_lora_config.json"

    convert_lora_checkpoint(
        model_name=model_name,
        checkpoint_path=checkpoint_path,
        lora_config_path=lora_config_path,
    )


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(convert_lit_lora_checkpoint, as_positional=False)
