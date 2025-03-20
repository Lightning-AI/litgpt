# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Union

from torch.utils.data import DataLoader

from litgpt.tokenizer import Tokenizer
from litgpt.data import DataModule


@dataclass
class LitData(DataModule):
    """Loads data using LitData's StreamingDataset given a path to a folder of preprocessed data (chunks)."""

    data_path: Union[str, Path] = Path("data/")
    """The path to the data directory containing the preprocessed chunks for the streaming dataset
    The path can also be a remote path (e.g., s3://). See also ``split_names`` if this path contains subfolders
    for training- and validation splits."""
    split_names: Optional[Tuple[str, str]] = None
    """Optional tuple for names of subfolders for training and validation under ``data_path``. If not provided,
    all data under data_path will be used for training, and the validation dataloader will be identical to the
    train dataloader."""
    seed: int = 42
    """The random seed for shuffling the dataset."""
    num_workers: int = 8
    """How many DataLoader processes to use for loading."""

    batch_size: int = field(init=False, repr=False, default=1)
    seq_length: int = field(init=False, repr=False, default=2048)

    def __post_init__(self) -> None:
        super().__init__()
        if self.split_names is not None and len(self.split_names) != 2:
            raise ValueError("If provided `split_names` must be a tuple of two strings, for example: ('train', 'val').")

    def connect(
        self, tokenizer: Optional[Tokenizer] = None, batch_size: int = 1, max_seq_length: Optional[int] = None
    ) -> None:
        self.batch_size = batch_size
        self.seq_length = max_seq_length + 1  # Increase by one because we need the next token as well

    def train_dataloader(self) -> DataLoader:
        input_dir = os.path.join(self.data_path, self.split_names[0]) if self.split_names else str(self.data_path)
        return self._dataloader(input_dir=input_dir, train=True)

    def val_dataloader(self) -> DataLoader:
        input_dir = os.path.join(self.data_path, self.split_names[1]) if self.split_names else str(self.data_path)
        return self._dataloader(input_dir=input_dir, train=False)

    def _dataloader(self, input_dir: str, train: bool):
        from litdata.streaming import StreamingDataset, StreamingDataLoader, TokensLoader

        dataset = StreamingDataset(
            input_dir=input_dir,
            item_loader=TokensLoader(block_size=self.seq_length),
            shuffle=train,
            seed=self.seed,
        )
        dataloader = StreamingDataLoader(
            dataset, batch_size=self.batch_size, pin_memory=True, num_workers=self.num_workers, drop_last=True
        )
        return dataloader
