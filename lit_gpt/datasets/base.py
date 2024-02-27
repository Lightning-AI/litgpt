# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
from abc import abstractmethod
from typing import List, Dict, Union, Optional

import torch
from torch import Tensor
from torch.utils.data import Dataset

from lightning import LightningDataModule
from lit_gpt import Tokenizer


class LitDataModule(LightningDataModule):
    """Base class for all data modules in Lit-GPT."""

    @abstractmethod
    def connect(
        self,
        tokenizer: Optional[Tokenizer] = None,
        batch_size: int = 1,
        max_seq_length: Optional[int] = None
    ) -> None:
        pass

    def setup(self, stage: str = "") -> None:
        pass


class SFTDataset(Dataset):
    """A dataset for supervised finetuning with `input_ids` and `labels`."""
    def __init__(
        self,
        data: List[Dict[str, str]],
        tokenizer: Tokenizer,
        prompt_template: Union[str, callable],
        max_seq_length: int = -1,
        mask_prompt: bool = True,
        ignore_index: int = -1,
    ) -> None:
        self.data = data
        self.tokenizer = tokenizer
        self.prompt_template = prompt_template
        self.max_seq_length = max_seq_length
        self.mask_prompt = mask_prompt
        self.ignore_index = ignore_index

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        """Processes a single sample.

        Returns a dict with two keys:
            input_ids: The encoded prompt + response
            labels: Same as input_ids, unless ``mask_prompt=True`` in which case the 'prompt' part is replaced with
                the ``ignore_index``.
        """
        example = self.data[idx]
        prompt = apply_prompt_template(self.prompt_template, example)
        prompt_and_response = prompt + example["output"]
        encoded_prompt = self.tokenizer.encode(prompt, max_length=self.max_seq_length)
        encoded_prompt_and_response = self.tokenizer.encode(
            prompt_and_response, eos=True, max_length=self.max_seq_length
        )

        # The labels are the full prompt with response, but with the prompt masked out
        labels = encoded_prompt_and_response.clone()
        if self.mask_prompt:
            labels[: len(encoded_prompt)] = self.ignore_index

        return {"input_ids": encoded_prompt_and_response.type(torch.int64), "labels": labels.type(torch.int64)}


def sft_collate_fn(samples: List[Dict[str, Tensor]], max_seq_length: int = -1, pad_id: int = 0, ignore_index: int = -1) -> Dict[str, Tensor]:
    longest = max(len(sample["input_ids"]) for sample in samples)
    max_length = max_seq_length if max_seq_length > 0 else longest

    batched = {}
    for key in ("input_ids", "labels"):
        pad_value = pad_id if key == "input_ids" else ignore_index

        # Pad right based on the longest sequence
        batched[key] = torch.stack([
            torch.nn.functional.pad(sample[key], (0, longest - len(sample[key])), value=pad_value)
            for sample in samples
        ])

        # Truncate if needed
        batched[key] = batched[key][:, :max_length]

    return batched


def apply_prompt_template(template: Union[str, callable], example: Dict[str, str]) -> str:
    if isinstance(template, str):
        prompt = template.format(**example)
    else:
        prompt = template(example)
    return prompt
