# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
import tempfile
from functools import partial

from torch import Tensor

import json
import sys
from pathlib import Path
from typing import Optional, Dict, Union, List

import torch
from torch.utils.data import random_split, Dataset, DataLoader
from lightning_utilities.core.imports import RequirementCache
from lightning import LightningDataModule

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_gpt.tokenizer import Tokenizer


class AlpacaDataset(Dataset):
    def __init__(
        self,
        data: List[Dict[str, str]],
        tokenizer: Tokenizer,
        max_seq_length: int = -1,
        mask_prompt: bool = True,
        ignore_index: int = -1,
    ) -> None:
        self.data = data
        self.tokenizer = tokenizer
        self.mask_prompt = mask_prompt
        self.max_seq_length = max_seq_length
        self.ignore_index = ignore_index

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
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
        example = self.data[idx]
        prompt = generate_prompt(example)
        prompt_and_response = prompt + example["output"]
        encoded_prompt = self.tokenizer.encode(prompt, max_length=self.max_seq_length)
        encoded_prompt_and_response = self.tokenizer.encode(
            prompt_and_response, eos=True, max_length=self.max_seq_length
        )

        # The labels are the full prompt with response, but with the prompt masked out
        labels = encoded_prompt_and_response.clone()
        if self.mask_prompt:
            labels[: len(encoded_prompt)] = self.ignore_index

        return {"input_ids": encoded_prompt_and_response, "labels": labels}


class Alpaca(LightningDataModule):
    """Implementation derived from https://github.com/tloen/alpaca-lora"""
    """Prepare the Alpaca dataset for instruction tuning.

        The output is a training and test dataset saved as `train.pt` and `test.pt`,
        which stores the preprocessed and tokenized prompts and labels.
        """

    def __init__(
        self,
        tokenizer_or_path: Union[str, Path, Tokenizer],
        max_seq_length: int = -1,
        mask_prompt: bool = True,
        test_split_fraction: float = 0.03865,  # to get exactly 2000 test samples,
        ignore_index: int = -1,
        seed: int = 42,
        data_file_name: str = "alpaca_data_cleaned_archive.json",
        data_file_url: str = "https://raw.githubusercontent.com/tloen/alpaca-lora/main/alpaca_data_cleaned_archive.json",
        batch_size: int = 1,
        num_workers: int = 4,
    ) -> None:
        super().__init__()

        if isinstance(tokenizer_or_path, (str, Path)):
            self.tokenizer = Tokenizer(Path(tokenizer_or_path))
        else:
            self.tokenizer = tokenizer_or_path

        self.max_seq_length = max_seq_length
        self.mask_prompt = mask_prompt
        self.test_split_fraction = test_split_fraction
        self.ignore_index = ignore_index
        self.seed = seed
        self.batch_size = batch_size
        self.num_workers = num_workers
        # if max_seq_length is None:
        #     with open(checkpoint_dir / "lit_config.json", "r", encoding="utf-8") as file:
        #         config = json.load(file)
        #         max_seq_length = config["block_size"]

        destination_path = Path(tempfile.mkdtemp())
        destination_path.mkdir(parents=True, exist_ok=True)
        self.data_file_path = destination_path / data_file_name
        self.data_file_url = data_file_url

        self.train_dataset: Optional[AlpacaDataset] = None
        self.test_dataset: Optional[AlpacaDataset] = None

    def prepare_data(self) -> None:
        download_if_missing(self.data_file_path, self.data_file_url)

    def setup(self, stage: str = None) -> None:
        with open(self.data_file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        # Partition the dataset into train and test
        train_data, test_data = random_split(
            data,
            [1.0 - self.test_split_fraction, self.test_split_fraction],
            generator=torch.Generator().manual_seed(self.seed)
        )
        train_data, test_data = list(train_data), list(test_data)

        self.train_dataset = AlpacaDataset(
            data=train_data,
            tokenizer=self.tokenizer,
            max_seq_length=self.max_seq_length,
            mask_prompt=self.mask_prompt,
            ignore_index=self.ignore_index,
        )
        self.test_dataset = AlpacaDataset(
            data=test_data,
            tokenizer=self.tokenizer,
            max_seq_length=self.max_seq_length,
            mask_prompt=self.mask_prompt,
            ignore_index=self.ignore_index,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=partial(collate_fn, max_seq_length=self.max_seq_length, ignore_index=self.ignore_index)
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=partial(collate_fn, max_seq_length=self.max_seq_length, ignore_index=self.ignore_index)
        )

    def test_dataloader(self) -> DataLoader:
        return self.val_dataloader()


def download_if_missing(file_path: Path, file_url: str) -> None:
    """Downloads the raw json data file and saves it in the given destination."""
    if file_path.exists() and file_path.stat().st_size > 0:
        return
    requests_available = RequirementCache("requests")
    if not requests_available:
        raise ModuleNotFoundError(str(requests_available))
    import requests

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(requests.get(file_url).text)


def generate_prompt(example: dict) -> str:
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


def collate_fn(samples: List[Dict[str, Tensor]], max_seq_length: int = -1, pad_id: int = 0, ignore_index: int = -1) -> Dict[str, Tensor]:
    longest = max(len(sample["input_ids"]) for sample in samples)
    max_length = max_seq_length if max_seq_length > 0 else longest

    batched = {}
    for key in ("input_ids", "labels"):
        pad_value = pad_id if key == "input_ids" else ignore_index
        batched[key] = torch.stack([
            torch.nn.functional.pad(sample[key], (0, longest - len(sample[key])), value=pad_value)
            for sample in samples
        ])
        batched[key] = batched[key][:, :max_length]
    return batched


if __name__ == "__main__":
    alpaca = Alpaca(tokenizer_or_path="checkpoints/TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    alpaca.prepare_data()
    alpaca.setup()

    train_dataloader = alpaca.train_dataloader()
    for batch in train_dataloader:
        print(batch)
        break



