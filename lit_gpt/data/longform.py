# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import json
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader
from lit_gpt.data import SFTDataset, get_sft_collate_fn, LitDataModule
from lit_gpt.data.alpaca import download_if_missing
from lit_gpt.tokenizer import Tokenizer


_URL = "https://raw.githubusercontent.com/akoksal/LongForm/main/dataset"


class LongForm(LitDataModule):
    """LongForm data module for supervised finetuning.

    Provides train- and val-dataloaders. The batches return keys "input_ids" and "labels".
    """

    def __init__(
        self,
        mask_prompt: bool = False,
        ignore_index: int = -1,
        seed: int = 42,
        num_workers: int = 4,
        download_dir: Path = Path("./data/longform"),
    ) -> None:
        super().__init__()
        self.mask_prompt = mask_prompt
        self.ignore_index = ignore_index
        self.seed = seed
        self.num_workers = num_workers
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
        download_if_missing(self.download_dir / "train.json", f"{_URL}/train.json")
        download_if_missing(self.download_dir / "val.json", f"{_URL}/val.json")

    def train_dataloader(self):
        return self._dataloader("train")

    def val_dataloader(self):
        return self._dataloader("val")

    def _dataloader(self, split: str) -> DataLoader:
        with open(self.download_dir / f"{split}.json", "r", encoding="utf-8") as file:
            data = json.load(file)

        dataset = SFTDataset(
            data=data,
            tokenizer=self.tokenizer,
            prompt_template=prompt_template,
            max_seq_length=self.max_seq_length,
            mask_prompt=self.mask_prompt,
            ignore_index=self.ignore_index,
        )
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=(split == "train"),
            generator=torch.Generator().manual_seed(self.seed),
            num_workers=self.num_workers,
            collate_fn=get_sft_collate_fn(max_seq_length=self.max_seq_length, ignore_index=self.ignore_index)
        )


def prompt_template(example: dict) -> str:
    """A modified Alpaca prompt template without the 'input'."""
    return (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{example['input']}\n\n### Response:\n"
    )
