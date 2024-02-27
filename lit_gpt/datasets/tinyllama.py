# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

from pathlib import Path
from typing import Union, Optional

from torch.utils.data import DataLoader

from lit_gpt import Tokenizer
from lit_gpt.datasets import LitDataModule


class TinyLlama(LitDataModule):
    """The TinyLlama data module is composed of a mix of SlimPajama and Starcoder data.

    Provides train- and val-dataloaders which return directly the token tensors.
    """

    def __init__(
        self,
        slimpajama_path: Union[str, Path] = Path("data/slimpajama"),
        starcoder_path: Union[str, Path] = Path("data/starcoder"),
        seed: int = 42,
        num_workers: int = 8,
    ) -> None:
        super().__init__()
        self.slimpajama_path = slimpajama_path
        self.starcoder_path = starcoder_path
        self.seed = seed
        self.num_workers = num_workers

        self.batch_size = 1
        self.seq_length = 2048

    def connect(
        self,
        tokenizer: Optional[Tokenizer] = None,
        batch_size: int = 1,
        max_seq_length: Optional[int] = None
    ) -> None:
        self.batch_size = batch_size
        self.seq_length = max_seq_length + 1  # Increase by one because we need the next token as well

    def prepare_data(self) -> None:
        pass

    def train_dataloader(self) -> DataLoader:
        from lightning.data.streaming import CombinedStreamingDataset, StreamingDataLoader, StreamingDataset, TokensLoader

        # Could be a remote path (s3://) or a local path
        slimpajama_train = str(self.slimpajama_path).rstrip("/") + "/train"

        train_datasets = [
            StreamingDataset(
                input_dir=slimpajama_train,
                item_loader=TokensLoader(block_size=self.seq_length),
                shuffle=True,
                drop_last=True,
            ),
            StreamingDataset(
                input_dir=str(self.starcoder_path),
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

        slimpajama_val = str(self.slimpajama_path).rstrip("/") + "/val"

        val_dataset = StreamingDataset(
            input_dir=slimpajama_val,
            item_loader=TokensLoader(block_size=self.seq_length),
            shuffle=True,
            # Consider setting to False, but we would lose some samples due to truncation when world size > 1
            drop_last=True,
        )
        val_dataloader = DataLoader(
            val_dataset, batch_size=self.batch_size, pin_memory=True, num_workers=self.num_workers, drop_last=True
        )
        return val_dataloader

    def test_dataloader(self) -> DataLoader:
        return self.val_dataloader()
