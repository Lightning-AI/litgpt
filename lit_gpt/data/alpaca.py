# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
"""Implementation derived from https://github.com/tloen/alpaca-lora"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import torch
from torch.utils.data import random_split, DataLoader
from lightning_utilities.core.imports import RequirementCache
from lit_gpt.data import SFTDataset, get_sft_collate_fn, LitDataModule
from lit_gpt.prompts import PromptStyle
from lit_gpt.tokenizer import Tokenizer

_URL = "https://raw.githubusercontent.com/tloen/alpaca-lora/main/alpaca_data_cleaned_archive.json"


@dataclass
class Alpaca(LitDataModule):
    """Alpaca data module for supervised finetuning."""

    mask_prompt: bool = False
    """Whether to mask the prompt section from the label (with ``ignore_index``)."""
    test_split_fraction: float = 0.03865  # to get exactly 2000 test samples,
    """The fraction of the dataset to use for the test/validation dataset. The rest is used for training."""
    prompt_style: Union[str, PromptStyle] = "alpaca"
    """The style to apply to instruction prompts. See `lit_gpt.prompts` for a list of available styles."""
    ignore_index: int = -1
    """The index to use for elements to be ignored in the label."""
    seed: int = 42
    """The random seed for creating the train/val splits and shuffling the dataset."""
    num_workers: int = 4
    """How many DataLoader processes to use for loading."""
    download_dir: Path = Path("./data/alpaca")
    """The directory in which the downloaded dataset gets saved."""
    file_url: str = field(repr=False, default=_URL)
    """The URL from where to download the dataset."""
    file_name: str = field(repr=False, default="alpaca_data_cleaned_archive.json")
    """The name of the dataset file to download."""

    tokenizer: Optional[Tokenizer] = field(default=None, init=False, repr=False)
    batch_size: int = field(default=1, init=False, repr=False)
    max_seq_length: int = field(default=-1, init=False, repr=False)
    train_dataset: Optional[SFTDataset] = field(default=None, init=False, repr=False)
    test_dataset: Optional[SFTDataset] = field(default=None, init=False, repr=False)

    def connect(
        self,
        tokenizer: Optional[Tokenizer] = None,
        batch_size: int = 1,
        max_seq_length: Optional[int] = None
    ) -> None:
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_seq_length = -1 if max_seq_length is None else max_seq_length

    def prepare_data(self) -> None:
        self.download_dir.mkdir(parents=True, exist_ok=True)
        download_if_missing(self.download_dir / self.file_name, self.file_url)

    def setup(self, stage: str = "") -> None:
        with open(self.download_dir / self.file_name, "r", encoding="utf-8") as file:
            data = json.load(file)

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
            prompt_style=self.prompt_style,
            max_seq_length=self.max_seq_length,
            mask_prompt=self.mask_prompt,
            ignore_index=self.ignore_index,
        )
        self.test_dataset = SFTDataset(
            data=test_data,
            tokenizer=self.tokenizer,
            prompt_style=self.prompt_style,
            max_seq_length=self.max_seq_length,
            mask_prompt=self.mask_prompt,
            ignore_index=self.ignore_index,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.seed),
            num_workers=self.num_workers,
            collate_fn=get_sft_collate_fn(max_seq_length=self.max_seq_length, ignore_index=self.ignore_index)
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=get_sft_collate_fn(max_seq_length=self.max_seq_length, ignore_index=self.ignore_index)
        )


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
