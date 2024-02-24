# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

from pathlib import Path
from typing import Optional, Union

from torch.utils.data import DataLoader
from lightning import LightningDataModule
from lightning.data import CombinedStreamingDataset, StreamingDataLoader, StreamingDataset
from lightning.data.streaming.item_loader import TokensLoader
from lit_gpt.datasets.base import SFTDataset


class TinyLlama(LightningDataModule):
    """The TinyLlama data module is composed of a mix of SlimPajama and Starcoder data.

    Provides train- and val-dataloaders which return directly the token tensors.
    """

    def __init__(
        self,
        data_path: Union[str, Path],  # TODO
        seq_length: int = 2048,
        mask_prompt: bool = True,
        seed: int = 42,
        batch_size: int = 1,
        num_workers: int = 8,
    ) -> None:
        super().__init__()
        self.data_path = data_path  # TODO: check and raise error
        self.seq_length = seq_length + 1  # Increase by one because we need the next token as well
        self.mask_prompt = mask_prompt
        self.seed = seed
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: str = None) -> None:
        pass

    def train_dataloader(self) -> DataLoader:
        train_datasets = [
            StreamingDataset(
                input_dir="data/slimpajama/train",
                item_loader=TokensLoader(block_size=self.seq_length),
                shuffle=True,
                drop_last=True,
            ),
            StreamingDataset(
                input_dir="data/starcoder",
                item_loader=TokensLoader(block_size=self.seq_length),
                shuffle=True,
                drop_last=True,
            ),
        ]

        # Mix SlimPajama data and Starcoder data with these proportions:
        weights = (0.693584, 0.306416)
        combined_dataset = CombinedStreamingDataset(datasets=train_datasets, seed=42, weights=weights)
        train_dataloader = StreamingDataLoader(
            combined_dataset, batch_size=self.batch_size, pin_memory=True, num_workers=self.num_workers, drop_last=True
        )
        return train_dataloader

    def val_dataloader(self) -> DataLoader:
        val_dataset = StreamingDataset(
            input_dir="data/slimpajama/val",
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
