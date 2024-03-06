# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

"""This script merges the LoRA weights with the base model"""
import shutil
import sys
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import yaml

import lightning as L
import torch

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_gpt.lora import GPT, Config, lora_filter, merge_lora_weights
from lit_gpt.utils import CLI, check_valid_checkpoint_dir, get_default_supported_precision, lazy_load


def merge_lora(
    checkpoint_dir: Path = Path("out/finetune/lora/final"),
    pretrained_checkpoint_dir: Optional[Path] = None,
    precision: Optional[str] = None,
) -> None:
    """Merges the LoRA weights with the base model. See `finetune/lora.py`.

    Merging happens in-place in the checkpoint directory that is given as input.

    Args:
        checkpoint_dir: Path to the checkpoint directory with trained LoRA weights, which is the output of
            `finetune/lora.py`.
        pretrained_checkpoint_dir: Optional path to the checkpoint directory with the weights of the base model
            corresponding to the LoRA checkpoint. By default, this will automatically be inferred from the metadata
            in the given `checkpoint_dir` directory. Only set this if the base model checkpoint directory
            has moved or was renamed.
        precision: Indicates the Fabric precision setting to use.
    """
    check_valid_checkpoint_dir(checkpoint_dir)
    if pretrained_checkpoint_dir is not None:
        check_valid_checkpoint_dir(pretrained_checkpoint_dir)

    precision = precision or get_default_supported_precision(training=False)
    fabric = L.Fabric(devices=1, precision=precision)

    lora_params, pretrained_checkpoint_dir = load_lora_metadata(checkpoint_dir)
    config = Config.from_json(checkpoint_dir / "lit_config.json", **lora_params)

    with fabric.init_module(empty_init=True):
        model = GPT(config)

    # Make a backup of the LoRA weights (they are only a few MBs)
    # TODO: Validate it not already exists
    # TODO: The merging below could fail for some reason
    lora_path = checkpoint_dir / "lit_model.pth.lora"
    shutil.move(checkpoint_dir / "lit_model.pth", lora_path)

    pretrained_checkpoint = lazy_load(pretrained_checkpoint_dir / "lit_model.pth")
    lora_checkpoint = lazy_load(lora_path)

    # Merge LoRA weights into the base model
    pretrained_checkpoint.update(lora_checkpoint.get("model", lora_checkpoint))
    model.load_state_dict(pretrained_checkpoint)
    merge_lora_weights(model)

    save_path = checkpoint_dir / "lit_model.pth"
    fabric.print(f"Saving merged weights to {str(save_path)!r}")
    fabric.print(f"A backup of the old LoRA weights is in {str(lora_path)!r}")

    # Remove lora parameters and the lora linear substring
    state_dict = {k.replace("linear.", ""): v for k, v in model.state_dict().items() if not lora_filter(k, v)}
    torch.save(state_dict, save_path)


def load_lora_metadata(checkpoint_dir: Path) -> Tuple[Dict[str, Any], Path]:
    hparams_file = checkpoint_dir / "hyperparameters.yaml"
    if not hparams_file.is_file():
        raise FileNotFoundError()  # TODO

    with open(hparams_file, "r") as file:
        hparams = yaml.safe_load(file)

    lora_params = {k: v for k, v in hparams.items() if k.startswith("lora_")}
    pretrained_checkpoint_dir = Path(hparams["checkpoint_dir"])
    return lora_params, pretrained_checkpoint_dir


if __name__ == "__main__":
    CLI(merge_lora)
