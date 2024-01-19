# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import json
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

import torch

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_gpt import Config
from lit_gpt.utils import incremental_save


@torch.inference_mode()
def convert_checkpoint(checkpoint_file: Path, tokenizer_dir: Path, config_name: str, output_dir: Path) -> None:
    """Convert a checkpoint after pretraining.

    The pretrained checkpoint contains optimizer states and several other metadata that are not needed after training
    is finished. This script will export the state-dict of the model and place it in the chosen output folder together
    with the tokenizer and model config, which then can be loaded by other scripts for inference, evaluation, etc.

    Args:
        checkpoint_file: Path to a checkpoint file scripts produced by the scripts in ``lit_gpt/pretrain/``.
        tokenizer_dir: A path to the folder that holds the tokenizer configuration files that were used to train
            the model. All files with a name starting with 'tokenizer' will be copied to the output folder.
        config_name: The name of the model loaded with the ``lit_gpt.Config``. The configuration will be saved as a
            JSON file to the output folder.
        output_dir: The output folder where model state-dict file, the tokenizer config file, and the model config
            file will be saved.
    """

    if output_dir.is_dir() and output_dir.glob("*"):
        raise FileExistsError(
            f"The output folder exists and is not empty: {str(output_dir)}."
            " Please delete it first or choose a different name."
        )
    if not tokenizer_dir.is_dir():
        raise FileNotFoundError(f"The tokenizer_dir must be a directory: {str(output_dir)}.")

    output_dir.mkdir(parents=True)
    output_checkpoint_file = output_dir / "lit_model.pth"
    output_config_file = output_dir / "lit_config.json"

    # Save the config to output folder
    config = Config.from_name(config_name)
    with open(output_config_file, "w") as json_config:
        json.dump(asdict(config), json_config)

    # Export the tokenizer configuration to output folder
    for tokenizer_file in tokenizer_dir.glob("tokenizer*"):
        shutil.copyfile(tokenizer_file, output_dir / tokenizer_file.name)

    # Copy config for tokenization if found
    if (tokenizer_dir / "generation_config.json").is_file():
        shutil.copyfile(tokenizer_dir / "generation_config.json", output_dir / "generation_config.json")

    # Extract the model state dict and save to output folder
    with incremental_save(output_checkpoint_file) as saver:
        print("Processing", checkpoint_file)
        full_checkpoint = torch.load(str(checkpoint_file), mmap=True)
        loaded_state_dict = full_checkpoint["model"]
        converted_state_dict = {}
        for param_name, param in loaded_state_dict.items():
            saver.store_early(param)
            # remove prefix for compiled model (if any)
            # this step won't be required with torch 2.2
            # https://github.com/pytorch/pytorch/pull/113423
            param_name = param_name.replace("_orig_mod.", "")
            converted_state_dict[param_name] = param
        print(f"Saving converted checkpoint to {str(output_checkpoint_file)}.")
        saver.save(converted_state_dict)


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(convert_checkpoint, as_positional=False)
