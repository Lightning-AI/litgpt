# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union, Optional

from torch.utils.data import DataLoader

from litgpt import Tokenizer
from litgpt.data import LitDataModule


@dataclass
class Tokens(LitDataModule):
    """Loads data using LitData's StreamingDataset given a path to a folder of preprocessed data (chunks)."""

    data_path: Union[str, Path] = Path("data/")
    """The path to the data directory containing the preprocessed chunks for the streaming dataset
    The path can also be a remote path (e.g., s3://)."""
    seed: int = 42
    """The random seed for shuffling the dataset."""
    num_workers: int = 8
    """How many DataLoader processes to use for loading."""

    batch_size: int = field(init=False, repr=False, default=1)
    seq_length: int = field(init=False, repr=False, default=2048)

    def connect(
        self,
        tokenizer: Optional[Tokenizer] = None,
        batch_size: int = 1,
        max_seq_length: Optional[int] = None
    ) -> None:
        self.batch_size = batch_size
        self.seq_length = max_seq_length + 1  # Increase by one because we need the next token as well

    def train_dataloader(self) -> DataLoader:
        from litdata.streaming import StreamingDataset, TokensLoader

        dataset = StreamingDataset(
            input_dir=str(self.data_path),
            item_loader=TokensLoader(block_size=self.seq_length),
            shuffle=True,
            drop_last=True,
        )

        train_dataloader = DataLoader(
            dataset, batch_size=self.batch_size, pin_memory=True, num_workers=self.num_workers, drop_last=True)
        return train_dataloader

    def val_dataloader(self) -> DataLoader:
        # TODO: Return a validation loader
        return self.train_dataloader()
