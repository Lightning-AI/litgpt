# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch.utils.data import random_split
from lit_gpt.data import SFTDataset, Alpaca
from lit_gpt.data.alpaca import prompt_template

_URL: str = "https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl"


@dataclass
class Dolly(Alpaca):
    """Dolly data module for supervised finetuning."""

    mask_prompt: bool = False
    """Whether to mask the prompt section from the label (with ``ignore_index``)."""
    test_split_fraction: float = 0.1
    """The fraction of the dataset to use for the test/validation dataset. The rest is used for training."""
    ignore_index: int = -1
    """The index to use for elements to be ignored in the label."""
    seed: int = 42
    """The random seed for creating the train/val splits and shuffling the dataset."""
    num_workers: int = 4
    """How many DataLoader processes to use for loading."""
    download_dir: Path = Path("./data/dolly")
    """The directory in which the downloaded dataset gets saved."""
    file_url: str = field(repr=False, default=_URL)
    """The URL from where to download the dataset."""
    file_name: str = field(repr=False, default="dolly_data_cleaned.json")
    """The name of the dataset file to download."""

    def setup(self, stage: str = "") -> None:
        with open(self.download_dir / self.data_file_name, "r", encoding="utf-8") as file:
            data = file.readlines()
            data = [json.loads(line) for line in data]
        for item in data:
            item["input"] = item.pop("context")
            item["output"] = item.pop("response")

        # Partition the dataset into train and test
        train_data, test_data = random_split(
            data,
            [1.0 - self.test_split_fraction, self.test_split_fraction],
            generator=torch.Generator().manual_seed(self.seed)
        )
        train_data, test_data = list(train_data), list(test_data)

        self.train_dataset = SFTDataset(
            data=train_data,
            tokenizer=self.tokenizer,
            prompt_template=prompt_template,
            max_seq_length=self.max_seq_length,
            mask_prompt=self.mask_prompt,
            ignore_index=self.ignore_index,
        )
        self.test_dataset = SFTDataset(
            data=test_data,
            tokenizer=self.tokenizer,
            prompt_template=prompt_template,
            max_seq_length=self.max_seq_length,
            mask_prompt=self.mask_prompt,
            ignore_index=self.ignore_index,
        )
