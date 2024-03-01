# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import json
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import random_split, DataLoader
from lit_gpt.data import SFTDataset, get_sft_collate_fn, LitDataModule
from lit_gpt.data.alpaca import prompt_template
from lit_gpt.tokenizer import Tokenizer


class JSON(LitDataModule):
    """Loads JSON data for supervised finetuning.

    Provides train- and val-dataloaders. The batches return keys "input_ids" and "labels".

    Args:
        json_path: A path to a JSON file containing the data. The file should contain a list of samples (dicts).
            Each dict must have the keys 'instruction' and 'output', and can optionally have a key 'input'
            (see Alpaca).
        mask_prompt: Whether to mask the prompt section from the label (with ``ignore_index``).
        test_split_fraction: A number in the range [0, 1] that determines the fraction of the dataset
            to use for testing.
        ignore_index: The index to use for elements to be ignored in the label.
        seed: The random seed for creating the train/val splits and shuffling the dataset.
        num_workers: How many DataLoader processes to use for loading.
    """

    def __init__(
        self,
        json_path: Path,
        mask_prompt: bool = False,
        test_split_fraction: float = 0.1,
        ignore_index: int = -1,
        seed: int = 42,
        num_workers: int = 4,
    ) -> None:
        super().__init__()
        self.json_path = json_path
        self.mask_prompt = mask_prompt
        self.test_split_fraction = test_split_fraction
        self.ignore_index = ignore_index
        self.seed = seed
        self.num_workers = num_workers

        self.tokenizer: Optional[Tokenizer] = None
        self.batch_size: int = 1
        self.max_seq_length: int = -1
        self.train_dataset: Optional[SFTDataset] = None
        self.test_dataset: Optional[SFTDataset] = None

        if not self.json_path.is_file():
            raise FileNotFoundError(f"The file {self.json_path} does not exist.")

    def connect(
        self,
        tokenizer: Optional[Tokenizer] = None,
        batch_size: int = 1,
        max_seq_length: Optional[int] = None
    ) -> None:
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_seq_length = -1 if max_seq_length is None else max_seq_length

    def setup(self, stage: str = "") -> None:
        with open(self.json_path, "r", encoding="utf-8") as file:
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
