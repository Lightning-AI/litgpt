# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
from abc import abstractmethod
from collections.abc import Callable
from functools import partial
from typing import Any

import torch
from lightning import LightningDataModule
from torch import Tensor
from torch.utils.data import Dataset

from litgpt.prompts import PromptStyle
from litgpt.tokenizer import Tokenizer


class DataModule(LightningDataModule):
    """Base class for all data modules in LitGPT."""

    @abstractmethod
    def connect(
        self,
        tokenizer: Tokenizer | None = None,
        batch_size: int = 1,
        max_seq_length: int | None = None,
        **kwargs,
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
        data: list[dict[str, str]],
        tokenizer: Tokenizer,
        prompt_style: str | PromptStyle,
        max_seq_length: int = -1,
        mask_prompt: bool = True,
        ignore_index: int = -100,
        transform: Callable[[Any], Any] | None = None,
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

    def __getitem__(self, idx: int) -> dict[str, Tensor | dict[str, int]]:
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

        raw_token_count = len(self.tokenizer.encode(example["instruction"], max_length=self.max_seq_length)) + len(
            encoded_response
        )

        return {
            "input_ids": encoded_prompt_and_response,
            "labels": labels,
            "token_counts": {
                "raw": raw_token_count,
                "raw_plus_prompt_template": len(encoded_prompt_and_response),
            },
        }


class MultiturnSFTDataset(Dataset):
    """An in-memory dataset for supervised finetuning on multi-turn conversations, producing `input_ids` and
    `labels`.

    Args:
        data: A list of conversations, each a list of ``{"role": "system"|"user"|"assistant", "content": str}``
            turns. Each conversation must end on an "assistant" turn (that's the training target).
        tokenizer: The tokenizer to use. Should match the one that was used to pretrain the model.
        prompt_style: The style to apply to prompts. Must have ``supports_multiturn=True`` (e.g. ChatML, Llama3,
            R1Base). See `litgpt.prompts` for a list of available styles.
        max_seq_length: Truncate sequences that are longer than this value. By default, no truncation is applied.
        mask_prompt: Whether to mask non-assistant turns from the label (with ``ignore_index``), so loss is only
            computed on the assistant's own tokens.
        ignore_index: The index to use for elements to be ignored in the label.
        transform: An optional transform to apply to the sample before it gets tokenized. Use this to reshape a
            differently-keyed conversation (e.g. ShareGPT's ``{"from": ..., "value": ...}``) into the expected
            ``{"role": ..., "content": ...}`` turns, with ``role`` one of ``"system"``, ``"user"``, or
            ``"assistant"``.

    Returns a dict with two keys:
        input_ids: The encoded conversation.
        labels: Same as input_ids, unless ``mask_prompt=True`` in which case every non-assistant turn is replaced
            with the ``ignore_index``.
    """

    def __init__(
        self,
        data: list[list[dict[str, str]]],
        tokenizer: Tokenizer,
        prompt_style: str | PromptStyle,
        max_seq_length: int = -1,
        mask_prompt: bool = True,
        ignore_index: int = -100,
        transform: Callable[[Any], Any] | None = None,
    ) -> None:
        self.data = data
        self.tokenizer = tokenizer
        self.prompt_style = (
            prompt_style if isinstance(prompt_style, PromptStyle) else PromptStyle.from_name(prompt_style)
        )
        if not self.prompt_style.supports_multiturn:
            raise ValueError(f"Prompt style {self.prompt_style} does not support multiturn prompts.")
        self.max_seq_length = max_seq_length
        self.mask_prompt = mask_prompt
        self.ignore_index = ignore_index
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, Tensor | dict[str, int]]:
        messages = self.data[idx]
        if self.transform is not None:
            messages = self.transform(messages)
        if not messages or messages[-1]["role"] != "assistant":
            raise ValueError(f"Conversation at index {idx} must end with an 'assistant' message.")

        if not self.mask_prompt:
            # No masking needed, so there's no need to know per-turn boundaries: tokenize the
            # whole rendered conversation in a single pass and use it as both input and label.
            text = self.prompt_style.apply(messages, add_generation_prompt=False)
            input_ids = self.tokenizer.encode(text).type(torch.int64)
            labels = input_ids.clone()
        else:
            # Masking needs to know which tokens belong to which turn. apply() only returns a
            # flat string, so we recover turn boundaries by re-rendering growing prefixes of the
            # conversation and diffing the tokenized lengths.
            chunks: list[tuple[Tensor, bool]] = []
            prev_len = 0
            for k in range(1, len(messages) + 1):
                text = self.prompt_style.apply(messages[:k], add_generation_prompt=False)
                # bos must be applied the same way on every iteration: the offset it adds to the
                # first prefix has to stay constant across all prefixes for the length-diffing
                # below to isolate the right turn boundary, not just the first one.
                ids = self.tokenizer.encode(text)
                turn_ids = ids[prev_len:]
                is_assistant = messages[k - 1]["role"] == "assistant"
                chunks.append((turn_ids, is_assistant))
                prev_len = len(ids)

            input_ids = torch.cat([ids for ids, _ in chunks]).type(torch.int64)
            labels = torch.cat(
                [
                    ids.clone() if is_assistant else torch.full_like(ids, self.ignore_index)
                    for ids, is_assistant in chunks
                ]
            ).type(torch.int64)

        if self.max_seq_length > 0:  # do not slice off last token when self.max_seq_length = -1
            input_ids = input_ids[: self.max_seq_length]
            labels = labels[: self.max_seq_length]

        # Token count with no prompt-style template overhead: just the turns' own content, plain.
        raw_token_count = sum(len(self.tokenizer.encode(m["content"])) for m in messages)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "token_counts": {
                "raw": raw_token_count,
                "raw_plus_prompt_template": len(input_ids),
            },
        }


def get_sft_collate_fn(max_seq_length: int = -1, pad_id: int = 0, ignore_index: int = -100):
    """Returns the collate function for supervised finetuning (needed in the DataLoader).

    The collate function gets a list of dicts with keys `input_ids` and `labels`.
    It returns a dict with batched `input_ids` and `labels`. Also pads short sequences to the longest element in
    the batch. Optionally truncates all sequences to the specified maximum length.
    """
    return partial(_sft_collate_fn, max_seq_length=max_seq_length, pad_id=pad_id, ignore_index=ignore_index)


def _sft_collate_fn(
    samples: list[dict[str, Tensor]], max_seq_length: int = -1, pad_id: int = 0, ignore_index: int = -100
) -> dict[str, Tensor]:
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
    batched["token_counts"]["raw_plus_prompt_template"] = (
        torch.tensor(  # Token count without padding but with prompt template
            [sample["token_counts"]["raw_plus_prompt_template"] for sample in samples], dtype=torch.int64
        ).unsqueeze(1)
    )

    return batched
