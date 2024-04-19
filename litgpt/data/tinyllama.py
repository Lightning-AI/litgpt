# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

from torch.utils.data import DataLoader

from litgpt import Tokenizer
from litgpt.data import DataModule


@dataclass
class TinyLlama(DataModule):
    """The TinyLlama data module is composed of a mix of SlimPajama and Starcoder data.

    Provides training and validation streaming dataloaders that return batches of tokens.
    """

    data_path: Union[str, Path] = Path("/data/adrian")
    """The path to the data directory, containing two folders 'slimpajama' and 'starcoder'
    which are the output of the preprocessing step done in advance. See the `tutorial/pretrain_tinyllama.md`
    for instructions. The path can also be a remote path (e.g., s3://)."""
    seed: int = 42
    """The random seed for shuffling the dataset."""
    num_workers: int = 8
    """How many DataLoader processes to use for loading."""

    batch_size: int = field(init=False, repr=False, default=1)
    seq_length: int = field(init=False, repr=False, default=2048)

    def __post_init__(self):
        # Could be a remote path (s3://) or a local path
        self.slimpajama_train_list = [str(self.data_path).rstrip("/") + f"/refinedweb{i+1}" for i in range(2)]
        self.slimpajama_train_list = [str(self.data_path / "refinedweb-500")]
        self.slimpajama_val = self.slimpajama_train_list[0]  # str(self.data_path).rstrip("/") + "/eval_refinedweb"

    def connect(
        self, tokenizer: Optional[Tokenizer] = None, batch_size: int = 1, max_seq_length: Optional[int] = None
    ) -> None:
        self.batch_size = batch_size
        self.seq_length = max_seq_length + 1  # Increase by one because we need the next token as well

    def prepare_data(self) -> None:
        for path in self.slimpajama_train_list + [self.slimpajama_val]:
            if not path.startswith("s3://") and not Path(path).is_dir():
                raise FileNotFoundError(
                    f"The data path for TinyLlama is expected to be a directory but found {self.slimpajama_train_list} and {self.slimpajama_val}"
                )

    def train_dataloader(self) -> DataLoader:
        from litdata.streaming import CombinedStreamingDataset, StreamingDataLoader, StreamingDataset, TokensLoader

        train_datasets = [
            StreamingDataset(
                input_dir=train_set,
                item_loader=TokensLoader(block_size=self.seq_length),
                shuffle=True,
                drop_last=True,
            )
            for train_set in self.slimpajama_train_list
        ]


        
        weights = [1.0 / len(self.slimpajama_train_list) for _ in self.slimpajama_train_list]
        
        combined_dataset = CombinedStreamingDataset(datasets=train_datasets, seed=self.seed, weights=weights)

        train_dataloader = StreamingDataLoader(
            combined_dataset, batch_size=self.batch_size, pin_memory=True, num_workers=self.num_workers, drop_last=True
        )
        return train_dataloader

    def val_dataloader(self) -> DataLoader:
        from litdata.streaming import StreamingDataset, TokensLoader

        val_dataset = StreamingDataset(
            input_dir=self.slimpajama_val,
            item_loader=TokensLoader(block_size=self.seq_length),
            shuffle=True,
            # Consider setting to False, but we would lose some samples due to truncation when world size > 1
            drop_last=True,
        )
        val_dataloader = DataLoader(
            val_dataset, batch_size=self.batch_size, pin_memory=True, num_workers=self.num_workers, drop_last=True
        )
        return val_dataloader

