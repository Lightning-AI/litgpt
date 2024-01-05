# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

"""Implementation derived from https://github.com/tloen/alpaca-lora"""

import json
import sys
from pathlib import Path
from typing import Optional

import requests
import torch
from tqdm import tqdm

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_gpt.tokenizer import Tokenizer


def prepare(
    destination_path: Path = Path("data/longform"),
    checkpoint_dir: Path = Path("checkpoints/stabilityai/stablelm-base-alpha-3b"),
    mask_inputs: bool = False,  # as in alpaca-lora
    ignore_index: int = -1,
    max_seq_length: Optional[int] = None,
) -> None:
    """Prepare the Alpaca dataset for instruction tuning.

    The output is a training and test dataset saved as `train.pt` and `test.pt`,
    which stores the preprocessed and tokenized prompts and labels.
    """
    if max_seq_length is None:
        with open(checkpoint_dir / "lit_config.json", "r", encoding="utf-8") as file:
            config = json.load(file)
            max_seq_length = config["block_size"]

    destination_path.mkdir(parents=True, exist_ok=True)

    train_file_name = "train.json"
    # val_file_name = "val.json"
    test_file_name = "test.json"

    train_file_url = "https://raw.githubusercontent.com/akoksal/LongForm/main/dataset/train.json"
    # val_file_url = "https://raw.githubusercontent.com/akoksal/LongForm/main/dataset/val.json"
    test_file_url = "https://raw.githubusercontent.com/akoksal/LongForm/main/dataset/test.json"

    train_file_path = destination_path / train_file_name
    print("Loading train data file...")
    download_if_missing(train_file_path, train_file_url)
    with open(train_file_path, "r", encoding="utf-8") as file:
        train_data = json.load(file)

    test_file_path = destination_path / test_file_name
    print("Loading test data file...")
    download_if_missing(test_file_path, test_file_url)
    with open(test_file_path, "r", encoding="utf-8") as file:
        test_data = json.load(file)

    print("Loading tokenizer...")
    tokenizer = Tokenizer(checkpoint_dir)

    print(f"train has {len(train_data):,} samples")
    print(f"test has {len(test_data):,} samples")

    print("Processing train set ...")
    train_data = [
        prepare_sample(
            example=sample,
            tokenizer=tokenizer,
            max_length=max_seq_length,
            mask_inputs=mask_inputs,
            ignore_index=ignore_index,
        )
        for sample in tqdm(train_data)
    ]
    torch.save(train_data, destination_path / "train.pt")

    print("Processing test set ...")
    test_data = [
        prepare_sample(
            example=sample,
            tokenizer=tokenizer,
            max_length=max_seq_length,
            mask_inputs=mask_inputs,
            ignore_index=ignore_index,
        )
        for sample in tqdm(test_data)
    ]
    torch.save(test_data, destination_path / "test.pt")


def download_if_missing(file_path: Path, file_url: str) -> None:
    """Downloads the raw json data file and saves it in the given destination."""
    if file_path.exists() and file_path.stat().st_size > 0:
        return
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(requests.get(file_url).text)


def prepare_sample(example: dict, tokenizer: Tokenizer, max_length: int, mask_inputs: bool, ignore_index: int) -> dict:
    """Processes a single sample.

    Each sample in the dataset consists of:
    - instruction: A string describing the task
    - input: A string holding a special input value for the instruction.
        This only applies to some samples, and in others this is empty.
    - output: The response string

    This function processes this data to produce a prompt text and a label for
    supervised training. The prompt text is formed as a single message including both
    the instruction and the input. The label/target is the same message but with the
    response attached.

    Finally, both the prompt and the label get tokenized. If desired, all tokens
    in the label that correspond to the original input prompt get masked out (default).
    """
    full_prompt = generate_prompt(example)
    full_prompt_and_response = full_prompt + example["output"]
    encoded_full_prompt = tokenizer.encode(full_prompt, max_length=max_length)
    encoded_full_prompt_and_response = tokenizer.encode(full_prompt_and_response, eos=True, max_length=max_length)

    # The labels are the full prompt with response, but with the prompt masked out
    labels = encoded_full_prompt_and_response.clone()
    if mask_inputs:
        labels[: len(encoded_full_prompt)] = ignore_index

    return {**example, "input_ids": encoded_full_prompt_and_response, "labels": labels}


def generate_prompt(example: dict) -> str:
    """Generates a standardized message to prompt the model with an instruction and a
    'response' field."""

    return (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{example['input']}\n\n### Response:"
    )


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(prepare)
