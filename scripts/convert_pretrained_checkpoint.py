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
from lit_gpt.utils import incremental_save, lazy_load


@torch.inference_mode()
def convert_checkpoint(
    checkpoint_file: Path,
    tokenizer_path: Path,
    config_name: str,
    output_folder: Path,
    model_key: str = "model",
) -> None:
    """Convert a checkpoint after pretraining.

    The pretrained checkpoint contains optimizer states and several other metadata that are not needed after training
    is finished. This script will export the state-dict of the model and place it in the chosen output folder together
    with the tokenizer and model config, which then can be loaded by other scripts for inference, evaluation, etc.

    Args:
        checkpoint_file: Path to a checkpoint file scripts produced by the scripts in `lit-gpt/pretrain/`.
        tokenizer_path: A path to the folder that holds the tokenizer configuration files that were used to train
            the model. All files with a name starting with 'tokenizer' will be copied to the output folder.
        config_name: The name of the model loaded with the ``lit_gpt.Config``. The configuration will be saved as a
            JSON file to the output folder.
        output_folder: The output folder where model state-dict file, the tokenizer config file, and the model config
            file will be saved.
        model_key: The key in the checkpoint file under which the model state dict is saved.
    """

    if output_folder.is_dir() and output_folder.glob("*"):
        raise FileExistsError(
            f"The output folder exists and is not empty: {str(output_folder)}."
            " Please delete it first or choose a different name."
        )

    output_folder.mkdir(parents=True)
    output_checkpoint_file = output_folder / "lit_model.pth"
    output_config_file = output_folder / "lit_config.json"

    # Save the config to output folder
    config = Config.from_name(config_name)
    with open(output_config_file, "w") as json_config:
        json.dump(asdict(config), json_config)

    # Export the tokenizer configuration to output folder
    if tokenizer_path.is_file():
        shutil.copyfile(tokenizer_path, output_folder / tokenizer_path.name)
    if tokenizer_path.is_dir():
        for tokenizer_file in tokenizer_path.glob("tokenizer*"):
            shutil.copyfile(tokenizer_file, output_folder / tokenizer_file.name)

    # Extract the model state dict and save to output folder
    with incremental_save(output_checkpoint_file) as saver:
        print("Processing", checkpoint_file)
        full_checkpoint = lazy_load(checkpoint_file)
        loaded_state_dict = full_checkpoint[model_key]
        converted_state_dict = {}
        for param_name, param in loaded_state_dict.items():
            param = param._load_tensor()
            saver.store_early(param)
            # remove prefix for compiled model (if any)
            param_name = param_name.replace("_orig_mod.", "")
            converted_state_dict[param_name] = param
        print(f"Saving converted checkpoint to {str(output_checkpoint_file)}.")
        saver.save(converted_state_dict)


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(convert_checkpoint, as_positional=False)
