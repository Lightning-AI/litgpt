# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
from abc import abstractmethod
from functools import partial
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
        """All settings that can't be determined at the time of instantiation need to be passed through here
        before any dataloaders can be accessed.
        """

    def setup(self, stage: str = "") -> None:
        # Stub is to redefine the default signature, because the concept of 'stage' does not exist in Lit-GPT
        pass


class SFTDataset(Dataset):
    """An in-memory dataset for supervised finetuning with `input_ids` and `labels`.

    Args:
        data: A list of samples (dicts). The target/label must be stored under the key 'output' and the instruction
            or other data can be stored under any key as long as it is compatible with the given prompt template.
        tokenizer: The tokenizer to use. Should match the one that was used to pretrain the model.
        prompt_template: A prompt template (format string or callable).
        max_seq_length: Truncate sequences that are longer than this value. By default, no truncation is applied.
        mask_prompt: Whether to mask the prompt section from the label (with ``ignore_index``).
        ignore_index: The index to use for elements to be ignored in the label.

    Returns a dict with two keys:
        input_ids: The encoded prompt + response
        labels: Same as input_ids, unless ``mask_prompt=True`` in which case the 'prompt' part is replaced with
            the ``ignore_index``.
    """
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
        example = self.data[idx]
        prompt = apply_prompt_template(self.prompt_template, example)
        prompt_and_response = prompt + example["output"]
        encoded_prompt = self.tokenizer.encode(prompt, max_length=self.max_seq_length)
        encoded_prompt_and_response = self.tokenizer.encode(
            prompt_and_response,
            eos=True,
            max_length=self.max_seq_length,
        )

        # The labels are the full prompt with response, but with the prompt masked out
        labels = encoded_prompt_and_response.clone()
        if self.mask_prompt:
            labels[: len(encoded_prompt)] = self.ignore_index

        return {"input_ids": encoded_prompt_and_response.type(torch.int64), "labels": labels.type(torch.int64)}


def apply_prompt_template(template: Union[str, callable], example: Dict[str, str]) -> str:
    if isinstance(template, str):
        prompt = template.format(**example)
    else:
        prompt = template(example)
    return prompt


def get_sft_collate_fn(max_seq_length: int = -1, pad_id: int = 0, ignore_index: int = -1):
    """Returns the collate function for supervised finetuning (needed in the DataLoader).

    The collate function gets a list of dicts with keys `input_ids` and `labels`.
    It returns a dict with batched `input_ids` and `labels`. Also pads short sequences to the longest element in
    the batch. Optionally truncates all sequences to the specified maximum length.
    """
    return partial(_sft_collate_fn, max_seq_length=max_seq_length, pad_id=pad_id, ignore_index=ignore_index)


def _sft_collate_fn(
    samples: List[Dict[str, Tensor]], max_seq_length: int = -1, pad_id: int = 0, ignore_index: int = -1
) -> Dict[str, Tensor]:

    batched = {}
    for key in ("input_ids", "labels"):
        pad_value = pad_id if key == "input_ids" else ignore_index

        # Pad right based on the longest sequence
        batched[key] = torch.nn.utils.rnn.pad_sequence(
            [sample[key] for sample in samples], batch_first=True, padding_value=pad_value
        )

        # Truncate if needed
        if max_seq_length > 0:
            batched[key] = batched[key][:, :max_seq_length]

    return batched
