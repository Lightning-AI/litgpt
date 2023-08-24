import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

import torch
from torch.utils.data import random_split

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
logger = logging.getLogger(__name__)
sys.path.append(str(wd))

from lit_gpt.tokenizer import Tokenizer

SEED = 42
MASK_INPUTS = False
IGNORE_INDEX = -1
TEST_SPLIT_FRACTION = 0.1
CHECKPOINT_DIR = Path("checkpoints/stabilityai/stablelm-base-alpha-3b")
CSV_PATH = "instruction.csv"
DESTINATION_DIR = Path("data/custom")


def prepare(
    csv_path: Path = CSV_PATH,
    destination_path: Path = DESTINATION_DIR,
    checkpoint_dir: Path = CHECKPOINT_DIR,
    datafile_name: Optional[str] = None,
    test_split_fraction: float = TEST_SPLIT_FRACTION,
    seed: int = SEED,
    mask_inputs: bool = MASK_INPUTS,
    ignore_index: int = IGNORE_INDEX,
) -> None:
    """Prepare dataset for instruction tuning from csv

    Args:
        llm: str helps to generate dataset with llm specific system prompts. example
        dolly or alpaca etc.

    The output is a training and test dataset saved as `train.pt` and `test.pt`,
    which stores the preprocessed and tokenized prompts and labels.
    """
    # TODO: Provide args inside docstrings

    with open(checkpoint_dir / "lit_config.json", "r") as file:
        config = json.load(file)
        max_seq_length = config["block_size"]

    destination_path.mkdir(parents=True, exist_ok=True)
    datafile_name = "dataset.json" if datafile_name is None else datafile_name

    # before the data file path, get the llm to be used
    logger.info("Loading data file ...")

    # first load the csv using pandas and convert it to a json
    # then fill all the nan with "", assuming all the records are in string
    # then convert the dataframe into json

    dataframe = pd.read_csv(csv_path, index_col=0).fillna("")
    df_json = json.loads(dataframe.to_json(orient="records", indent=4))
    data_file_path = destination_path / datafile_name

    with open(data_file_path, "w") as json_file:
        json.dump(df_json, json_file)

        # REMOVE IT AFTER DONE

    logger.info("Loading tokenizer ...")
    tokenizer = Tokenizer(checkpoint_dir)

    # Partition the dataset into train and test
    train_set, test_set = random_split(
        df_json,
        [1.0 - test_split_fraction, test_split_fraction],
        generator=torch.Generator().manual_seed(seed),
    )

    train_set, test_set = list(train_set), list(test_set)
    logger.info(f"train has {len(train_set):,} samples")
    logger.info(f"test has {len(test_set):,} samples")
    logger.info("Processing train split ...")

    train_set = [
        prepare_sample(
            example=sample,
            tokenizer=tokenizer,
            max_length=max_seq_length,
            mask_inputs=mask_inputs,
            ignore_index=ignore_index,
        )
        for sample in tqdm(train_set)
    ]
    torch.save(train_set, destination_path / "train.pt")

    logger.info("Processing test split ...")
    test_set = [
        prepare_sample(
            example=sample,
            tokenizer=tokenizer,
            max_length=max_seq_length,
            mask_inputs=mask_inputs,
            ignore_index=ignore_index,
        )
        for sample in tqdm(test_set)
    ]
    torch.save(test_set, destination_path / "test.pt")


def generate_prompt(example):
    """Generates a standardized message to prompt the model with an instruction, optional input and a
    'response' field."""

    if example["input"]:
        return (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:"
        )
    return (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{example['instruction']}\n\n### Response:"
    )


def prepare_sample(
    example: dict,
    tokenizer: Tokenizer,
    max_length: int,
    mask_inputs: bool = MASK_INPUTS,
    ignore_index: int = IGNORE_INDEX,
):
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
    encoded_full_prompt_and_response = tokenizer.encode(
        full_prompt_and_response, eos=True, max_length=max_length
    )

    # The labels are the full prompt with response, but with the prompt masked out
    labels = encoded_full_prompt_and_response.clone()
    if mask_inputs:
        labels[: len(encoded_full_prompt)] = ignore_index

    return {
        **example,
        "input_ids": encoded_full_prompt_and_response,
        "input_ids_no_response": encoded_full_prompt,
        "labels": labels,
    }


if __name__ == "__main__":
    from jsonargparse import CLI, ArgumentParser

    parser = ArgumentParser()
    parser.add_function_arguments
    cli = CLI(prepare)
