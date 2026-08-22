# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, random_split

from litgpt.data import DataModule, SFTDataset, get_sft_collate_fn
from litgpt.prompts import PromptStyle
from litgpt.tokenizer import Tokenizer

_JSONScalar = str | int | float | bool | None
_JSONValue = _JSONScalar | dict[str, Any] | list[Any]


@dataclass
class JSON(DataModule):
    """Loads JSON or JSONL data for supervised finetuning."""

    json_path: Path | None = None
    """A path to a JSON file or a directory with `train.json` and `val.json` containing the data. Mutually exclusive
    with ``json_data``.
    The file(s) should contain a list of samples (dicts). Each dict must have the keys 'instruction' and 'output',
    and can optionally have a key 'input' (see Alpaca)."""
    mask_prompt: bool = False
    """Whether to mask the prompt section from the label (with ``ignore_index``)."""
    val_split_fraction: float | None = None
    """The fraction of the dataset to use for the validation dataset. The rest is used for training.
    Only applies if you passed in a single file to `json_path` or an in-memory JSON string to `json_data`."""
    prompt_style: str | PromptStyle = "alpaca"
    """The style to apply to instruction prompts. See `litgpt.prompts` for a list of available styles."""
    ignore_index: int = -100
    """The index to use for elements to be ignored in the label."""
    seed: int = 42
    """The random seed for creating the train/val splits and shuffling the dataset."""
    num_workers: int = 4
    """How many DataLoader processes to use for loading."""
    json_data: str | list[dict[str, _JSONValue]] | None = field(default=None, repr=False)
    """An in-memory JSON string or decoded list of samples (dicts). Mutually exclusive with ``json_path``.
    When converting a pandas DataFrame, use ``df.to_json(orient="records")``."""

    tokenizer: Tokenizer | None = field(default=None, init=False, repr=False)
    batch_size: int = field(default=1, init=False, repr=False)
    max_seq_length: int = field(default=-1, init=False, repr=False)
    train_dataset: SFTDataset | None = field(default=None, init=False, repr=False)
    val_dataset: SFTDataset | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        super().__init__()
        if (self.json_path is None) == (self.json_data is None):
            raise ValueError("Exactly one of `json_path` or `json_data` must be provided.")
        single_source = None
        if self.json_data is not None:
            single_source = "The `json_data` argument was provided"
        elif self.json_path is not None and self.json_path.is_file():
            single_source = "The `json_path` points to a single file"
        if single_source is not None and self.val_split_fraction is None:
            self.val_split_fraction = 0.05
            warnings.warn(
                f"{single_source} and `val_split_fraction` was not set. "
                "Defaulting to `val_split_fraction=0.05`. Set `val_split_fraction` explicitly "
                "to use a different split percentage.",
                UserWarning,
                stacklevel=2,
            )
        if self.json_path is not None and self.json_path.is_dir() and self.val_split_fraction is not None:
            raise ValueError(
                "If `json_path` is a directory, it must contain 'train.json' and 'val.json' files and"
                f" hence `val_split_fraction` should not be set. Got `{self.val_split_fraction=}`."
            )
        if self.json_path is not None and not self.json_path.exists():
            raise FileNotFoundError(
                "The `json_path` must be a file or a directory containing 'train.json' and 'val.json' files,"
                f" but '{self.json_path!s}' does not exist."
            )
        if isinstance(self.prompt_style, str):
            self.prompt_style = PromptStyle.from_name(self.prompt_style)

    def connect(
        self, tokenizer: Tokenizer | None = None, batch_size: int = 1, max_seq_length: int | None = None
    ) -> None:
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_seq_length = -1 if max_seq_length is None else max_seq_length

    def setup(self, stage: str = "") -> None:
        train_data, test_data = self.get_splits()

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
            collate_fn=get_sft_collate_fn(max_seq_length=self.max_seq_length, ignore_index=self.ignore_index),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=get_sft_collate_fn(max_seq_length=self.max_seq_length, ignore_index=self.ignore_index),
        )

    def get_splits(self) -> tuple:
        if self.json_data is not None:
            if isinstance(self.json_data, str):
                try:
                    data = json.loads(self.json_data)
                except json.JSONDecodeError as ex:
                    raise ValueError("`json_data` must be valid JSON.") from ex
            else:
                data = self.json_data
            data = _validate_json_data(data)
        else:
            json_path = self.json_path
            if json_path is None:
                raise RuntimeError("Expected `json_path` when `json_data` is not provided.")
            if not json_path.is_file():
                # A directory containing train.json and val.json
                if (train_file := self.find_split("train")) and (val_file := self.find_split("val")):
                    train_data = load_split(train_file)
                    test_data = load_split(val_file)
                    return train_data, test_data

                raise FileNotFoundError(
                    "The `json_path` must be a file or a directory containing 'train.json' and 'val.json' files."
                )
            data = load_split(json_path)

        val_split_fraction = self.val_split_fraction
        if val_split_fraction is None:
            raise RuntimeError("Expected `val_split_fraction` for a single JSON data source.")
        return random_split(
            data,
            [1.0 - val_split_fraction, val_split_fraction],
            generator=torch.Generator().manual_seed(self.seed),
        )

    def find_split(self, split_name: str) -> Path | None:
        if self.json_path is None:
            return None
        for suffix in (".json", ".jsonl"):
            if (file := self.json_path / f"{split_name}{suffix}").is_file():
                return file
        return None


def _validate_json_data(data: Any) -> list[dict[str, _JSONValue]]:
    if not isinstance(data, list):
        raise ValueError(
            f"`json_data` must decode to a list of JSON objects, got {type(data).__name__}. "
            'When using pandas, call `DataFrame.to_json(orient="records")`.'
        )

    required_fields = {"instruction", "output"}
    for index, sample in enumerate(data):
        if not isinstance(sample, dict):
            raise ValueError(f"`json_data` sample at index {index} must be a JSON object, got {type(sample).__name__}.")
        if missing_fields := required_fields - sample.keys():
            formatted_fields = ", ".join(f"`{field}`" for field in sorted(missing_fields))
            raise ValueError(f"`json_data` sample at index {index} is missing required field(s): {formatted_fields}.")
    return data


def load_split(json_path: Path) -> Any:
    if json_path.suffix == ".json":
        with open(json_path, encoding="utf-8") as file:
            return json.load(file)
    if json_path.suffix == ".jsonl":
        with open(json_path, encoding="utf-8") as file:
            return [json.loads(line) for line in file]
    else:
        raise ValueError(f"Unsupported file format: {json_path.suffix}. Expected `.json` or `.jsonl`.")
