# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

"""This script merges the LoRA weights with the base model"""
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import lightning as L
import torch
import yaml

from litgpt.lora import GPT, Config, lora_filter, merge_lora_weights
from litgpt.utils import CLI, check_valid_checkpoint_dir


def merge_lora(
    checkpoint_dir: Path, pretrained_checkpoint_dir: Optional[Path] = None, precision: Optional[str] = None
) -> None:
    """Merges the LoRA weights with the base model. See ``litgpt finetune lora``.

    Creates a new ``lit_model.pth`` file by merging the LoRA weights (``lit_model.pth.lora``)
    with the original checkpoint weights.

    Args:
        checkpoint_dir: Path to the checkpoint directory with trained LoRA weights, which is the output of
            ``litgpt finetune lora``.
        pretrained_checkpoint_dir: Optional path to the checkpoint directory with the weights of the base model
            corresponding to the LoRA checkpoint. By default, this will automatically be inferred from the metadata
            in the given `checkpoint_dir` directory. Only set this if the base model's checkpoint directory
            has moved or was renamed.
        precision: Optional precision setting to instantiate the model weights in. By default, this will
            automatically be inferred from the metadata in the given ``checkpoint_dir`` directory.
    """
    check_valid_checkpoint_dir(checkpoint_dir, model_filename="lit_model.pth.lora")
    if pretrained_checkpoint_dir is not None:
        check_valid_checkpoint_dir(pretrained_checkpoint_dir)
    if (checkpoint_dir / "lit_model.pth").is_file():
        print("LoRA weights have already been merged in this checkpoint.")
        return
    if (hyperparams_dir := (checkpoint_dir / "hyperparameters.yaml")).is_file():
        with open(hyperparams_dir, "r", encoding="utf-8") as hparams_file:
            hparams = yaml.safe_load(hparams_file)
            remove_last_perc_layers = hparams.get("train", 0.0).get("remove_last_perc_layers", 0.0)
    else:
        remove_last_perc_layers = 0.0

    lora_params, pretrained_checkpoint_dir, lora_precision = load_lora_metadata(checkpoint_dir)
    precision = precision if precision is not None else lora_precision

    fabric = L.Fabric(devices=1, precision=precision, accelerator="cpu")
    config = Config.from_file(checkpoint_dir / "model_config.yaml", **lora_params)

    with fabric.init_module():
        model = GPT(config)

        # Sec. 4.4 of https://arxiv.org/abs/2403.17887
        if remove_last_perc_layers > 0.0:
            layers_num = len(model.transformer.h)
            layers_to_remove = int(config.n_layer * remove_last_perc_layers)
            if layers_to_remove > 0:
                fabric.print(f"Removing last {layers_to_remove} layers")
                model.transformer.h = model.transformer.h[:-layers_to_remove]

    lora_path = checkpoint_dir / "lit_model.pth.lora"
    pretrained_checkpoint = torch.load(str(pretrained_checkpoint_dir / "lit_model.pth"), mmap=True)
    lora_checkpoint = torch.load(str(lora_path), mmap=True)

    # Remove from the pretrained checkpoint layers removed during the finetuning
    if remove_last_perc_layers > 0.0:
        find_number = re.compile(r"\.(\d+)\.")
        pretrained_checkpoint_layers_removed = {}
        for k, v in pretrained_checkpoint.items():
            layer_num = find_number.findall(k)
            if layer_num != []:
                layer_num = int(layer_num[0])
                if layer_num < layers_num - layers_to_remove:
                    pretrained_checkpoint_layers_removed[k] = v
            else:
                pretrained_checkpoint_layers_removed[k] = v
        pretrained_checkpoint = pretrained_checkpoint_layers_removed

    # Merge LoRA weights into the base model
    pretrained_checkpoint.update(lora_checkpoint.get("model", lora_checkpoint))
    model.load_state_dict(pretrained_checkpoint)
    merge_lora_weights(model)

    # Remove LoRA parameters and the LoRA linear substring
    state_dict = {k.replace("linear.", ""): v for k, v in model.state_dict().items() if not lora_filter(k, v)}
    save_path = checkpoint_dir / "lit_model.pth"
    torch.save(state_dict, save_path)

    fabric.print(f"Saved merged weights to {str(checkpoint_dir / 'lit_model.pth')!r}")


def load_lora_metadata(checkpoint_dir: Path) -> Tuple[Dict[str, Any], Path, Optional[str]]:
    hparams_file = checkpoint_dir / "hyperparameters.yaml"
    if not hparams_file.is_file():
        raise FileNotFoundError(
            f"The path {str(hparams_file)!r} is not a valid checkpoint directory. It is missing a"
            f" `hyperparameters.yaml` file. Please point to the checkpoint directory that was produced by"
            f" the `litgpt/finetune/lora.py` script."
        )

    with open(hparams_file, "r") as file:
        hparams = yaml.safe_load(file)

    lora_params = {k: v for k, v in hparams.items() if k.startswith("lora_")}
    pretrained_checkpoint_dir = Path(hparams["checkpoint_dir"])
    precision = hparams.get("precision")
    return lora_params, pretrained_checkpoint_dir, precision


if __name__ == "__main__":
    CLI(merge_lora)
