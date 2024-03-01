# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
"""Implementation derived from https://github.com/tloen/alpaca-lora"""

import json
from pathlib import Path
from typing import Optional, Dict

import torch
from torch.utils.data import random_split, DataLoader
from lightning_utilities.core.imports import RequirementCache
from lit_gpt.data import SFTDataset, get_sft_collate_fn, LitDataModule
from lit_gpt.tokenizer import Tokenizer

_URL = "https://raw.githubusercontent.com/tloen/alpaca-lora/main/alpaca_data_cleaned_archive.json"


class Alpaca(LitDataModule):
    """Alpaca data module for supervised finetuning.

    Provides train- and val-dataloaders. The batches return keys "input_ids" and "labels".
    """

    def __init__(
        self,
        mask_prompt: bool = False,
        test_split_fraction: float = 0.03865,  # to get exactly 2000 test samples,
        ignore_index: int = -1,
        seed: int = 42,
        num_workers: int = 4,
        data_file_url: str = _URL,
        data_file_name: str = "alpaca_data_cleaned_archive.json",
        download_dir: Path = Path("./data/alpaca"),
    ) -> None:
        super().__init__()
        self.mask_prompt = mask_prompt
        self.test_split_fraction = test_split_fraction
        self.ignore_index = ignore_index
        self.seed = seed
        self.num_workers = num_workers
        self.data_file_url = data_file_url
        self.data_file_name = data_file_name
        self.download_dir = download_dir

        self.tokenizer: Optional[Tokenizer] = None
        self.batch_size: int = 1
        self.max_seq_length: int = -1
        self.train_dataset: Optional[SFTDataset] = None
        self.test_dataset: Optional[SFTDataset] = None

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
        download_if_missing(self.download_dir / self.data_file_name, self.data_file_url)

    def setup(self, stage: str = "") -> None:
        with open(self.download_dir / self.data_file_name, "r", encoding="utf-8") as file:
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


def prompt_template(example: Dict[str, str]) -> str:
    """The Alpaca prompt template."""
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
