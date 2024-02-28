# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import json
from pathlib import Path

import torch
from torch.utils.data import random_split
from lit_gpt.data import SFTDataset, Alpaca
from lit_gpt.data.alpaca import prompt_template

_URL: str = "https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl"


class Dolly(Alpaca):
    """Dolly data module for supervised finetuning.

    Provides train- and val-dataloaders. The batches return keys "input_ids" and "labels".
    """

    def __init__(
        self,
        mask_prompt: bool = False,
        test_split_fraction: float = 0.1,
        ignore_index: int = -1,
        seed: int = 42,
        num_workers: int = 4,
        data_file_url: str = _URL,
        data_file_name: str = "dolly_data_cleaned.json",
        download_dir: Path = Path("./data/dolly"),
    ) -> None:
        super().__init__(
            mask_prompt=mask_prompt,
            test_split_fraction=test_split_fraction,
            ignore_index=ignore_index,
            seed=seed,
            num_workers=num_workers,
            data_file_url=data_file_url,
            data_file_name=data_file_name,
            download_dir=download_dir,
        )

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
