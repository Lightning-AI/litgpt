"""https://github.com/karpathy/llama2.c/blob/b3c4b6/tinystories.py"""

import glob
import json
import os
import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

from litgpt.data.alpaca import download_if_missing
from litgpt.data.base import DataModule
from litgpt.tokenizer import Tokenizer


@dataclass
class TinyStories(DataModule):
    """The TinyStories data module: https://huggingface.co/datasets/roneneldan/TinyStories

    Provides training and validation dataloaders that return batches of tokens. Every sample is set to a fixed length.
    """

    path: Path = Path("data/")
    """Path to the data directory where data will be downloaded and preprocessed"""
    num_workers: int = 0
    """How many DataLoader processes to use for loading."""
    seed: int = 42
    """The random seed for creating the train/val splits and shuffling the dataset."""

    tokenizer: Optional[Tokenizer] = field(default=None, init=False, repr=False)
    batch_size: int = field(default=1, init=False, repr=False)
    max_seq_length: int = field(default=-1, init=False, repr=False)
    train_dataset: Optional[torch.utils.data.Dataset] = field(default=None, init=False, repr=False)
    test_dataset: Optional[torch.utils.data.Dataset] = field(default=None, init=False, repr=False)

    def connect(self, tokenizer: Optional[Tokenizer] = None, batch_size: int = 1, max_seq_length: int = -1) -> None:
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length

    def prepare_data(self) -> None:
        download(self.path)
        assert self.tokenizer is not None
        pretokenize(self.path, self.tokenizer)

    def setup(self, stage: str = "") -> None:
        # the .bin files are right along the .json files
        bin_dir = self.path / "TinyStories_all_data"
        shard_filenames = sorted(glob.glob(str(bin_dir / "*.bin")))
        assert len(shard_filenames) > 0, f"No bin files found in {bin_dir}"
        assert len(shard_filenames) > 1, f"Expected at least two bins in {bin_dir}"
        # train/test split. let's use only shard 0 for test split, rest train
        va_files, *train_files = shard_filenames
        # shuffle the training files
        random.Random(self.seed).shuffle(train_files)
        self.train_dataset = ConcatDataset([PretokDataset(f, self.max_seq_length) for f in train_files])
        self.val_dataset = PretokDataset(shard_filenames[0], self.max_seq_length)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, pin_memory=True, shuffle=True, num_workers=self.num_workers
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            pin_memory=True,
            shuffle=True,  # llama2.c shuffles validation too
            num_workers=self.num_workers,
        )


_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories_all_data.tar.gz"


def download(data_dir: Path):
    data_dir.mkdir(exist_ok=True)

    # download the TinyStories dataset, unless it's already downloaded
    data_filename = data_dir / "TinyStories_all_data.tar.gz"
    download_if_missing(data_filename, _URL, stream=True, mode="wb")
    print("Download done.")

    # unpack the tar.gz file into all the data shards (json files)
    data_dir = data_dir / "TinyStories_all_data"
    shard_filenames = sorted(glob.glob(str(data_dir / "*.json")))
    if shard_filenames:
        print(f"{data_dir} already exists, skipping unpacking...")
    else:
        data_dir.mkdir(exist_ok=True)
        print(f"Unpacking {data_filename}...")
        os.system(f"tar -xzf {data_filename} -C {data_dir}")
        shard_filenames = sorted(glob.glob(str(data_dir / "*.json")))

    print(f"Number of shards: {len(shard_filenames)}")
    # print a single example just for debugging and such
    # with open(shard_filenames[0], "r") as f:
    #     data = json.load(f)
    # print(f"Example story:\n{data[0]}")


def process_shard(args, tokenizer):
    shard_id, shard = args
    with open(shard, "r") as f:
        data = json.load(f)
    all_tokens = []
    for example in tqdm(data, position=shard_id):
        text = example["story"]
        text = text.strip()  # get rid of leading/trailing whitespace
        tokens = tokenizer.encode(text, bos=True, eos=False)  # encode the text, use BOS
        all_tokens.extend(tokens)
    # convert to uint16 nparray
    all_tokens = np.array(all_tokens, dtype=np.uint16)
    # just save the tokenized file in the same dir
    tokenized_filename = shard.replace(".json", ".bin")
    # write the bytes
    with open(tokenized_filename, "wb") as f:
        f.write(all_tokens.tobytes())
    # calculate the average sequence length (they are separated by BOS=1)
    bos_id = tokenizer.bos_id
    assert bos_id >= 0  # uint16 is unsigned
    bos_tokens = (all_tokens == tokenizer.bos_id).sum()
    assert bos_tokens > 0
    avg_seq_len = all_tokens.size / bos_tokens
    print(
        f"Saved {tokenized_filename}, tokens: {all_tokens.size}, bos: {bos_tokens}, average seqlen: {avg_seq_len:.2f}"
    )


def pretokenize(data_dir: Path, tokenizer: Tokenizer):
    bins_path = str(data_dir / "TinyStories_all_data" / "*.bin")
    shard_filenames = sorted(glob.glob(bins_path))
    if shard_filenames:
        print("Already pretokenized.")
        return
    # iterate the shards and tokenize all of them one by one
    jsons_path = str(data_dir / "TinyStories_all_data" / "*.json")
    shard_filenames = sorted(glob.glob(jsons_path))
    if not shard_filenames:
        raise ValueError(f"No json files found in {jsons_path!r}. Did you run `python tinystories.py download`?")
    # process all the shards in a process pool
    fun = partial(process_shard, tokenizer=tokenizer)
    with ProcessPoolExecutor() as executor:
        executor.map(fun, enumerate(shard_filenames))
    print("Done.")


class PretokDataset(torch.utils.data.Dataset):
    """Loads a pre-tokenized array from disk and returns chunks of `max_seq_length` length."""

    def __init__(self, filepath: str, max_seq_len: int):
        super().__init__()
        self.filepath = filepath
        # open the dataset for reading but keep it on disk with memmap
        self.shard = np.memmap(filepath, dtype=np.uint16, mode="r")
        self.shard_length = len(self.shard)
        self.length = self.shard_length // max_seq_len
        assert max_seq_len > 1
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, ix: int) -> torch.Tensor:
        if ix < 0:
            raise NotImplementedError
        start = ix * self.max_seq_len
        end = start + self.max_seq_len + 1
        if end > self.shard_length:
            raise IndexError
        # calling .astype will copy the data into a new numpy array, now in RAM
        chunk = torch.from_numpy((self.shard[start:end]).astype(np.int64))
        return chunk
