# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
import glob
import os
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from tqdm import tqdm
from typing import Optional

from torch.utils.data import DataLoader

from litgpt import Tokenizer
from litgpt.data import DataModule


def get_half_workers():
    num_workers = (os.cpu_count() - 1) // 2
    if num_workers < 1:
        return 1
    else:
        return num_workers


@dataclass
class TextFiles(DataModule):
    """The TextFile data module used for pretraining.

    Reads in text data from plaintext files contained in a data folder
    and provides training and validation dataloaders that return batches of tokens.
    Every sample is set to a fixed length.
    """
    train_data_path: Path = Path("data/")
    """The path to the data directory used for training that
    contains .txt files"""
    val_data_path: Optional[str] = None
    """The path to the data directory used for validation that
    contains .txt files. Splits off data for validation from the
    training set if None."""
    seed: int = 42
    """The seed to use for shuffling the dataset."""
    num_workers: Optional[int] = None
    """The number of workers to use for data loading.
       Sets the number of workers equal to the number of avaialable CPUs-1 by default."""

    tokenizer: Optional[Tokenizer] = field(default=None, init=False, repr=False)
    batch_size: int = field(default=1, init=False, repr=False)
    max_seq_length: int = field(default=-1, init=False, repr=False)

    def connect(self, tokenizer: Optional[Tokenizer] = None, batch_size: int = 1, max_seq_length: int = -1) -> None:
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length + 1  # Increase by one because we need the next token as well

    def __post_init__(self) -> None:
        self.data_path_train = self.train_data_path / "train"
        if self.val_data_path is None:
            self.data_path_val = self.train_data_path / "val"
        else:
            self.data_path_val = Path(self.val_data_path) / "val"

    def prepare_data(self) -> None:
        from litdata import optimize

        train_files = sorted(glob.glob(str(self.train_data_path / "*.txt")))
        assert len(train_files) > 0, f"No .txt files found in train data {train_files}"
        assert len(train_files) > 1, f"Expected at least two .txt files in {train_files}"

        if self.val_data_path is not None:
            self.val_data_path = Path(self.val_data_path)
            val_files = sorted(glob.glob(str(self.val_data_path / "*.txt")))
            assert len(val_files) > 0, f"No .txt files found in validation data {val_files}"
        # train/test split. let's use only shard 0 for test split, rest train
        else:
            val_files, *train_files = train_files

        if self.num_workers is None:
            num_workers = os.cpu_count() - 1
        else:
            num_workers = self.num_workers

        if not Path(self.data_path_train).is_dir():
            optimize(
                fn=partial(tokenize, tokenizer=self.tokenizer),
                inputs=train_files,
                output_dir=str(self.data_path_train),
                num_workers=num_workers,
                chunk_bytes="50MB",
            )
        if not Path(self.data_path_val).is_dir():
            optimize(
                fn=partial(tokenize, tokenizer=self.tokenizer),
                inputs=[val_files] if not isinstance(val_files, list) else val_files,
                output_dir=str(self.data_path_val),
                num_workers=1,  # there's only 1 file
                chunk_bytes="50MB",
            )

    def train_dataloader(self) -> DataLoader:
        from litdata.streaming import StreamingDataLoader, StreamingDataset, TokensLoader

        train_dataset = StreamingDataset(
            input_dir=str(self.data_path_train),
            item_loader=TokensLoader(block_size=self.max_seq_length),
            shuffle=True,
            drop_last=True,
        )
        if self.num_workers is None:
            num_workers = get_half_workers()

        train_dataloader = StreamingDataLoader(
            train_dataset, batch_size=self.batch_size, pin_memory=True, num_workers=num_workers, drop_last=True
        )
        return train_dataloader

    def val_dataloader(self) -> DataLoader:
        from litdata.streaming import StreamingDataset, TokensLoader

        if self.num_workers is None:
            num_workers = get_half_workers()

        val_dataset = StreamingDataset(
            input_dir=str(self.data_path_val),
            item_loader=TokensLoader(block_size=self.max_seq_length),
            shuffle=True,
            # Consider setting to False, but we would lose some samples due to truncation when world size > 1
            drop_last=True,
        )
        val_dataloader = DataLoader(
            val_dataset, batch_size=self.batch_size, pin_memory=True, num_workers=num_workers, drop_last=True
        )
        return val_dataloader


def tokenize(filename: str, tokenizer: Tokenizer):
    if tokenizer is None:
        raise ValueError(
            "Tokenizer is None. If you are using this data module via `litgpt pretrain`, "
            "please provide a valid `--tokenizer_dir` path."
        )
    with open(filename, "r", encoding="utf-8") as file:
        text = file.read()
    text = text.strip()

    chunks = []
    total_length = len(text)
    num_chunks = 10
    chunk_size = total_length // num_chunks
    for i in range(num_chunks):
        start_index = i * chunk_size
        end_index = (i + 1) * chunk_size if i < 9 else total_length
        chunks.append(text[start_index:end_index])


    global_rank = int(os.environ["DATA_OPTIMIZER_GLOBAL_RANK"])
    num_workers = int(os.environ["DATA_OPTIMIZER_NUM_WORKERS"])
    local_rank = global_rank % num_workers
    for example in tqdm(chunks, position=local_rank):
        tokens = tokenizer.encode(example.strip(), bos=True, eos=False)  # encode the text, use BOS
        yield tokens

