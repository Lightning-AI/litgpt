# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
# Note: this script was borrowed from Zichun. Not sure original source somewhere?
import argparse
import os
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Optional, Union

from litdata.streaming import (
    CombinedStreamingDataset,
    StreamingDataLoader,
    StreamingDataset,
    TokensLoader,
)

from litgpt import Tokenizer
from litgpt.data import DataModule

from torch.utils.data import DataLoader


def tokenize(data, index):
    yield self.tokenizer.encode(data[index]["text"], eos=True)


def safe_tokenize(data, index):
    """Tokenize text and skip invalid cases."""
    try:
        text = data[index].get("text", "")
        if text and isinstance(text, str):
            tokens = self.tokenizer.encode(text.strip(), eos=True)
            if tokens:
                yield tokens
    except Exception as e:
        logging.warning(f"Error processing index {index}: {str(e)}")


@dataclass
class FineWebDataset(DataModule):
    """The FineWeb data module for pretraining."""

    data_path: Union[str, Path] = Path(
        "/data/users/nightingal3/manifold/all_in_one_pretraining/datasets/pretraining/fineweb/"
    )
    val_data_path: Optional[Union[str, Path]] = None  # specify a separate val data path
    """The path to the data directory, containing two folders 'train' and 'val'
    which are the output of the preprocessing step. The path can also be a remote path (e.g., s3://)."""
    val_split_fraction: float = 0.003
    """The fraction of data that should be put aside for validation."""
    seed: int = 42
    """The seed to use for shuffling the training data."""
    num_workers: int = 8
    """The number of workers to use for the dataloaders."""
    data_split: str = "sample-100BT"
    fast_dev_run: bool = False

    tokenizer: Optional[Tokenizer] = field(default=None, repr=False, init=False)
    batch_size: int = field(default=1, repr=False, init=False)
    seq_length: int = field(default=2048, repr=False, init=False)

    def __post_init__(self) -> None:
        # Could be a remote path (s3://) or a local path
        if not self.val_data_path:
            self.data_path_train = os.path.join(self.data_path, "train")
            self.data_path_val = os.path.join(self.data_path, "val")
        else:
            self.data_path_train = self.data_path
            self.data_path_val = self.val_data_path

    def _has_sharded_structure(self, base_dir: Union[str, Path]) -> bool:
        """Check if the directory has numbered subdirectories (0-7) with index.json files."""
        for i in range(10):
            shard_dir = os.path.join(base_dir, str(i))
            if os.path.isdir(shard_dir) and os.path.exists(
                os.path.join(shard_dir, "index.json")
            ):
                return True
        return False

    def connect(
        self,
        tokenizer: Optional[Tokenizer] = None,
        batch_size: int = 1,
        max_seq_length: Optional[int] = 2048,
    ) -> None:
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.seq_length = (
            max_seq_length + 1
        )  # Increase by one because we need the next token as well

    def _create_combined_dataset(
        self, base_dir: Union[str, Path]
    ) -> CombinedStreamingDataset:
        """Create a combined dataset from sharded directories."""
        datasets = []

        for i in range(10):
            shard_dir = os.path.join(base_dir, str(i))
            if os.path.isdir(shard_dir) and os.path.exists(
                os.path.join(shard_dir, "index.json")
            ):
                print(f"Loading shard {i} from {shard_dir}")
                dataset = StreamingDataset(
                    input_dir=shard_dir,
                    item_loader=TokensLoader(block_size=self.seq_length),
                    shuffle=True,
                    drop_last=True,
                )
                datasets.append(dataset)
            else:
                print(
                    f"Warning: Shard {i} at {shard_dir} not found or missing index.json"
                )

        if not datasets:
            raise ValueError(f"No valid shards found in {base_dir}")

        print(f"Created combined dataset from {len(datasets)} shards")
        return CombinedStreamingDataset(
            datasets=datasets, seed=self.seed, iterate_over_all=True
        )

    def prepare_data(self) -> None:
        from datasets import Dataset, load_dataset
        from litdata import optimize

        hf_cache_dir = os.getenv("HF_HOME")

        if str(self.data_path).startswith("s3://"):
            print(
                f"The FineWeb data path points to an S3 location: {self.data_path}. Skipping preprocessing."
            )
            return

        if Path(self.data_path_train).is_dir() and Path(self.data_path_val).is_dir():
            print(
                f"Found FineWeb train and val dir: {self.data_path}. Skipping preprocessing."
            )
            return

        dataset = load_dataset(
            "HuggingFaceFW/fineweb",
            num_proc=8,
            name=self.data_split,  # 149M examples
            cache_dir=hf_cache_dir,
            split="train",
        )
        print("Total examples:", len(dataset))
        # save dataset to manifold
        # print("Saving dataset to disk")
        # dataset.save_to_disk(self.data_path)

        # Split the data in training and validation
        split_dataset = dataset.train_test_split(
            test_size=self.val_split_fraction, seed=self.seed, shuffle=True
        )
        split_dataset["val"] = split_dataset.pop("test")  # rename the test split to val
        breakpoint()
        optimize(
            fn=partial(safe_tokenize, split_dataset["train"], self.tokenizer),
            inputs=list(range(len(split_dataset["train"]))),
            output_dir=self.data_path_train,
            num_workers=8,
            chunk_bytes="200MB",
            fast_dev_run=self.fast_dev_run,
        )

        optimize(
            fn=partial(safe_tokenize, split_dataset["val"], self.tokenizer),
            inputs=list(range(len(split_dataset["val"]))),
            output_dir=self.data_path_val,
            num_workers=8,
            chunk_bytes="200MB",
            fast_dev_run=self.fast_dev_run,
        )
        print(f"Finished preprocessing of {self.data_path}")

    def train_dataloader(self) -> DataLoader:
        if self._has_sharded_structure(self.data_path_train):
            train_dataset = self._create_combined_dataset(self.data_path_train)
        else:
            train_dataset = StreamingDataset(
                input_dir=self.data_path_train,
                item_loader=TokensLoader(block_size=self.seq_length),
                shuffle=True,
                drop_last=True,
            )
        train_dataloader = StreamingDataLoader(
            train_dataset,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=self.num_workers,
            drop_last=True,
        )
        return train_dataloader

    def val_dataloader(self) -> DataLoader:
        from litdata.streaming import StreamingDataset, TokensLoader

        val_dataset = StreamingDataset(
            input_dir=self.data_path_val,
            item_loader=TokensLoader(block_size=self.seq_length),
            shuffle=True,
            # Consider setting to False, but we would lose some samples due to truncation when world size > 1
            drop_last=True,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=self.num_workers,
            drop_last=True,
        )
        return val_dataloader


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Process the pretraining data for this project, or load it as a sanity check. "
    )
    parser.add_argument(
        "--data_path",
        type=str,
        help="Data directory (base dir)",
        default="/data/users/nightingal3/manifold/all_in_one_pretraining/datasets/pretraining/fineweb-350BT",
    )
    parser.add_argument("--data_split", type=str, help="split of fineweb to use")
    parser.add_argument("--fast_dev_run", action="store_true")

    args = parser.parse_args()
    fw = FineWebDataset(
        data_path=args.data_path,
        data_split=args.data_split,
        fast_dev_run=args.fast_dev_run,
    )
    fw.prepare_data()
