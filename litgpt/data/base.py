# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
from abc import abstractmethod
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from lightning import LightningDataModule
from torch import Tensor
from torch.utils.data import Dataset

from litgpt.tokenizer import Tokenizer
from litgpt.prompts import PromptStyle


class DataModule(LightningDataModule):
    """Base class for all data modules in LitGPT."""

    @abstractmethod
    def connect(
        self, tokenizer: Optional[Tokenizer] = None, batch_size: int = 1, max_seq_length: Optional[int] = None
    ) -> None:
        """All settings that can't be determined at the time of instantiation need to be passed through here
        before any dataloaders can be accessed.
        """

    def setup(self, stage: str = "") -> None:
        # Stub is to redefine the default signature, because the concept of 'stage' does not exist in LitGPT
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class SFTDataset(Dataset):
    """An in-memory dataset for supervised finetuning with `input_ids` and `labels`.

    Args:
        data: A list of samples (dicts). The target/label must be stored under the key 'output' and the instruction
            or other data can be stored under any key as long as it is compatible with the given prompt template.
        tokenizer: The tokenizer to use. Should match the one that was used to pretrain the model.
        prompt_style: The style to apply to prompts. See `litgpt.prompts` for a list of available styles.
        max_seq_length: Truncate sequences that are longer than this value. By default, no truncation is applied.
        mask_prompt: Whether to mask the prompt section from the label (with ``ignore_index``).
        ignore_index: The index to use for elements to be ignored in the label.
        transform: An optional transform to apply to the sample before it gets tokenized. Use this to rename the
            keys in the dataset to the expected 'instruction' and 'output' keys.

    Returns a dict with two keys:
        input_ids: The encoded prompt + response
        labels: Same as input_ids, unless ``mask_prompt=True`` in which case the 'prompt' part is replaced with
            the ``ignore_index``.
    """

    def __init__(
        self,
        data: List[Dict[str, str]],
        tokenizer: Tokenizer,
        prompt_style: Union[str, PromptStyle],
        max_seq_length: int = -1,
        mask_prompt: bool = True,
        ignore_index: int = -100,
        transform: Optional[Callable[[Any], Any]] = None,
    ) -> None:
        self.data = data
        self.tokenizer = tokenizer
        self.prompt_style = (
            prompt_style if isinstance(prompt_style, PromptStyle) else PromptStyle.from_name(prompt_style)
        )
        self.max_seq_length = max_seq_length
        self.mask_prompt = mask_prompt
        self.ignore_index = ignore_index
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Union[Tensor, Dict[str, int]]]:
        example = self.data[idx]
        if self.transform is not None:
            example = self.transform(example)
        prompt = self.prompt_style.apply(prompt=example["instruction"], **example)
        encoded_prompt = self.tokenizer.encode(prompt, max_length=self.max_seq_length)
        encoded_response = self.tokenizer.encode(example["output"], bos=False, eos=True, max_length=self.max_seq_length)
        encoded_prompt_and_response = torch.cat((encoded_prompt, encoded_response)).type(torch.int64)
        if self.max_seq_length > 0:  # do not slice off last token when self.max_seq_length = -1
            encoded_prompt_and_response = encoded_prompt_and_response[: self.max_seq_length]

        # The labels are the full prompt with response, but with the prompt masked out
        labels = encoded_prompt_and_response.clone()
        if self.mask_prompt:
            labels[: len(encoded_prompt)] = self.ignore_index

        raw_token_count = len(self.tokenizer.encode(example["instruction"], max_length=self.max_seq_length)) + len(encoded_response)

        return {
            "input_ids": encoded_prompt_and_response,
            "labels": labels,
            "token_counts": {
                "raw": raw_token_count,
                "raw_plus_prompt_template": len(encoded_prompt_and_response),
            }
        }


def get_sft_collate_fn(max_seq_length: int = -1, pad_id: int = 0, ignore_index: int = -100):
    """Returns the collate function for supervised finetuning (needed in the DataLoader).

    The collate function gets a list of dicts with keys `input_ids` and `labels`.
    It returns a dict with batched `input_ids` and `labels`. Also pads short sequences to the longest element in
    the batch. Optionally truncates all sequences to the specified maximum length.
    """
    return partial(_sft_collate_fn, max_seq_length=max_seq_length, pad_id=pad_id, ignore_index=ignore_index)


def _sft_collate_fn(
    samples: List[Dict[str, Tensor]], max_seq_length: int = -1, pad_id: int = 0, ignore_index: int = -100
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

    batched["token_counts"] = {}
    batched["token_counts"]["raw"] = torch.tensor(  # Token count without padding and without prompt template
        [sample["token_counts"]["raw"] for sample in samples], dtype=torch.int64
    ).unsqueeze(1)
    batched["token_counts"]["raw_plus_prompt_template"] = torch.tensor(  # Token count without padding but with prompt template
        [sample["token_counts"]["raw_plus_prompt_template"] for sample in samples], dtype=torch.int64
    ).unsqueeze(1)

    return batched
