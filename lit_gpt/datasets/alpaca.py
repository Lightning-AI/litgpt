# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
"""Implementation derived from https://github.com/tloen/alpaca-lora"""

import tempfile
from functools import partial

import json
from pathlib import Path
from typing import Optional, Dict

import torch
from torch.utils.data import random_split, DataLoader
from lightning_utilities.core.imports import RequirementCache
from lit_gpt.datasets.base import SFTDataset, sft_collate_fn, LitDataModule
from lit_gpt.tokenizer import Tokenizer


class Alpaca(LitDataModule):
    """Alpaca data module for supervised finetuning.

    Provides train- and val-dataloaders. The batches return keys "input_ids" and "labels".
    """

    def __init__(
        self,
        max_seq_length: int = -1,
        mask_prompt: bool = True,
        test_split_fraction: float = 0.03865,  # to get exactly 2000 test samples,
        ignore_index: int = -1,
        seed: int = 42,
        data_file_name: str = "alpaca_data_cleaned_archive.json",
        data_file_url: str = "https://raw.githubusercontent.com/tloen/alpaca-lora/main/alpaca_data_cleaned_archive.json",
        num_workers: int = 4,
    ) -> None:
        super().__init__()
        self.max_seq_length = max_seq_length
        self.mask_prompt = mask_prompt
        self.test_split_fraction = test_split_fraction
        self.ignore_index = ignore_index
        self.seed = seed
        self.num_workers = num_workers
        self.batch_size = 1

        destination_path = Path(tempfile.mkdtemp())
        destination_path.mkdir(parents=True, exist_ok=True)
        self.data_file_path = destination_path / data_file_name
        self.data_file_url = data_file_url

        self.tokenizer: Optional[Tokenizer] = None
        self.train_dataset: Optional[SFTDataset] = None
        self.test_dataset: Optional[SFTDataset] = None

    def connect(self, tokenizer: Tokenizer, batch_size: int = 1) -> None:
        self.tokenizer = tokenizer
        self.batch_size = batch_size

    def prepare_data(self) -> None:
        download_if_missing(self.data_file_path, self.data_file_url)

    def setup(self, stage: str = "") -> None:
        with open(self.data_file_path, "r", encoding="utf-8") as file:
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

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=partial(sft_collate_fn, max_seq_length=self.max_seq_length, ignore_index=self.ignore_index)
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=partial(sft_collate_fn, max_seq_length=self.max_seq_length, ignore_index=self.ignore_index)
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


def prompt_template(example: Dict[str, str]) -> str:
    if example.get("input"):
        return (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n"
        )
    return (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{example['instruction']}\n\n### Response:\n"
    )


if __name__ == "__main__":
    alpaca = Alpaca()
    alpaca.connect(tokenizer=Tokenizer("checkpoints/"), batch_size=2)
    alpaca.prepare_data()
    alpaca.setup()

    train_dataloader = alpaca.train_dataloader()
    for batch in train_dataloader:
        print(batch)
