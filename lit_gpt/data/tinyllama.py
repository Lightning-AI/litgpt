# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

from pathlib import Path
from typing import Union, Optional

from torch.utils.data import DataLoader

from lit_gpt import Tokenizer
from lit_gpt.data import LitDataModule


class TinyLlama(LitDataModule):
    """The TinyLlama data module is composed of a mix of SlimPajama and Starcoder data.

    Provides training and validation streaming dataloaders that return batches of tokens.

    Args:
        data_path: The path to the data directory, containing two folders 'slimpajama' and 'starcoder'
            which are the output of the preprocessing step done in advance. See the `tutorial/pretrain_tinyllama.md`
            for instructions. The path can also be a remote path (e.g., s3://).
        seed: The seed to use for shuffling the training data.
        num_workers: The number of workers to use for the dataloaders.
    """

    def __init__(
        self,
        data_path: Union[str, Path] = Path("data/"),
        seed: int = 42,
        num_workers: int = 8,
    ) -> None:
        super().__init__()
        self.seed = seed
        self.num_workers = num_workers

        self.batch_size = 1
        self.seq_length = 2048

        # Could be a remote path (s3://) or a local path
        self.slimpajama_train = str(data_path).rstrip("/") + "/slimpajama/train"
        self.slimpajama_val = str(data_path).rstrip("/") + "/slimpajama/val"
        self.starcoder_train = str(data_path).rstrip("/") + "/starcoder"

    def connect(
        self,
        tokenizer: Optional[Tokenizer] = None,
        batch_size: int = 1,
        max_seq_length: Optional[int] = None
    ) -> None:
        self.batch_size = batch_size
        self.seq_length = max_seq_length + 1  # Increase by one because we need the next token as well

    def prepare_data(self) -> None:
        for path in (self.slimpajama_train, self.slimpajama_val, self.starcoder_train):
            if not path.startswith("s3://") and not Path(path).is_dir():
                raise FileNotFoundError(
                    "The data path for TinyLlama is expected to be the directory containing these subdirectories:"
                    f" `slimpajama/train`, `slimpajama/val`, `starcoder`. The directory {path} does not exist."
                )

    def train_dataloader(self) -> DataLoader:
        from lightning.data.streaming import CombinedStreamingDataset, StreamingDataLoader, StreamingDataset, TokensLoader

        train_datasets = [
            StreamingDataset(
                input_dir=self.slimpajama_train,
                item_loader=TokensLoader(block_size=self.seq_length),
                shuffle=True,
                drop_last=True,
            ),
            StreamingDataset(
                input_dir=self.starcoder_train,
                item_loader=TokensLoader(block_size=self.seq_length),
                shuffle=True,
                drop_last=True,
            ),
        ]

        # Mix SlimPajama data and Starcoder data with these proportions:
        weights = (0.693584, 0.306416)
        combined_dataset = CombinedStreamingDataset(datasets=train_datasets, seed=self.seed, weights=weights)
        train_dataloader = StreamingDataLoader(
            combined_dataset, batch_size=self.batch_size, pin_memory=True, num_workers=self.num_workers, drop_last=True
        )
        return train_dataloader

    def val_dataloader(self) -> DataLoader:
        from lightning.data.streaming import StreamingDataset, TokensLoader

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
