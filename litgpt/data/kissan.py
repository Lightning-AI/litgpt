# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
"""Implementation derived from https://github.com/tloen/alpaca-lora"""

import os
from dataclasses import dataclass, field
from typing import List, Optional, Union
from venv import logger

import torch
from torch.utils.data import DataLoader, random_split

from litgpt.data import DataModule, SFTDataset, get_sft_collate_fn
from litgpt.prompts import PromptStyle
from litgpt.tokenizer import Tokenizer


@dataclass
class KissanAI(DataModule):
    """KissanAI data module for supervised finetuning."""

    mask_prompt: bool = False
    """Whether to mask the prompt section from the label (with ``ignore_index``)."""
    val_split_fraction: float = 0.1
    """The fraction of the dataset to use for the validation dataset. The rest is used for training."""
    prompt_style: Union[str, PromptStyle] = "pragna-1b"
    """The style to apply to instruction prompts. See `litgpt.prompts` for a list of available styles."""
    ignore_index: int = -100
    """The index to use for elements to be ignored in the label."""
    seed: int = 42
    """The random seed for creating the train/val splits and shuffling the dataset."""
    num_workers: int = 4
    """How many DataLoader processes to use for loading."""
    include_multiturn_conversations: bool = True
    """Whether to include multi-turn conversations in the dataset."""
    repo_id: str = "KissanAI/Thinking-climate-100k"
    """The Hugging Face dataset repository ID from where to download the data."""
    access_token: Optional[str] = field(repr=False, default=os.getenv("HF_TOKEN"))
    """The Hugging Face API token to use for authentication."""

    tokenizer: Optional[Tokenizer] = field(default=None, init=False, repr=False)
    batch_size: int = field(default=1, init=False, repr=False)
    max_seq_length: int = field(default=-1, init=False, repr=False)
    train_dataset: Optional[SFTDataset] = field(default=None, init=False, repr=False)
    test_dataset: Optional[SFTDataset] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        super().__init__()
        if isinstance(self.prompt_style, str):
            self.prompt_style = PromptStyle.from_name(self.prompt_style)

    def connect(
        self,
        tokenizer: Optional[Tokenizer] = None,
        batch_size: int = 1,
        max_seq_length: Optional[int] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_seq_length = -1 if max_seq_length is None else max_seq_length

    def prepare_data(self) -> None:
        from datasets import load_dataset
        load_dataset(self.repo_id, token=self.access_token)

    def setup(self, stage: str = "") -> None:
        from datasets import load_dataset

        dataset = load_dataset(self.repo_id, token=self.access_token)
        data = format_dataset(dataset["train"], self.include_multiturn_conversations)

        train_data, test_data = random_split(
            data,
            [1.0 - self.val_split_fraction, self.val_split_fraction],
            generator=torch.Generator().manual_seed(self.seed),
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
            collate_fn=get_sft_collate_fn(
                max_seq_length=self.max_seq_length,
                ignore_index=self.ignore_index,
            ),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=get_sft_collate_fn(
                max_seq_length=self.max_seq_length,
                ignore_index=self.ignore_index,
            ),
        )


def format_dataset(dataset_partition, include_multi_turn_conversations: bool) -> List[dict]:
    formatted_ds = []

    for entry in dataset_partition:
        convo = entry["conversations"] 

        user_turns = [x["value"] for x in convo if x.get("from") == "user"]
        assistant_turns = [x["value"] for x in convo if x.get("from") == "assistant"]

        if not user_turns or not assistant_turns:
            continue

        def replace_think_tokens(text: str) -> str:
            return (
                text
                .replace("<think>", "<unused0>")
                .replace("</think>", "<unused1>")
            )

        if include_multi_turn_conversations:
            for u, a in zip(user_turns, assistant_turns):
                formatted_ds.append({
                    "instruction": u.strip(),
                    "input": "",
                    "output": replace_think_tokens(a.strip()),
                })
        else:
            formatted_ds.append({
                "instruction": user_turns[0].strip(),
                "input": "",
                "output": replace_think_tokens(assistant_turns[0].strip()),
            })
    if formatted_ds:
        logger.info(f"formatted_ds[0]: {formatted_ds[0]}")

    return formatted_ds


def data_collator(self, batch):
    logger.debug(f"Collating batch of size {len(batch)}")
