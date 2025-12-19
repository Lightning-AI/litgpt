# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence, Union

from torch.utils.data import DataLoader

from litgpt.data import DataModule
from litgpt.tokenizer import Tokenizer


@dataclass
class Pragna(DataModule):
    """The Pragna data module is composed of a mix of SlimPajama and Starcoder data.

    Provides training and validation streaming dataloaders that return batches of tokens.
    """

    data_path: Union[str, Path] = Path("/raid/")
    """The path to the data directory, containing two folders 'slimpajama' and 'starcoder'
    which are the output of the preprocessing step done in advance. See the `tutorial/pretrain_tinyllama.md`
    for instructions. The path can also be a remote path (e.g., s3://)."""
    seed: int = 42
    """The random seed for shuffling the dataset."""
    num_workers: int = 8
    """Toggle for using Starcoder data."""
    sangraha_versions: Optional[Sequence[str]] = None
    sangraha_train_dirs: list[str] = field(init=False, repr=False, default_factory=list)

    batch_size: int = field(init=False, repr=False, default=1)
    seq_length: int = field(init=False, repr=False, default=2048)

    def __post_init__(self):
        super().__init__()
        # Could be a remote path (s3://) or a local path
        base_path = str(self.data_path).rstrip("/")
        data_prefix = base_path + "/data"
        self.sangraha_val = data_prefix + "/sangraha_processed_eval_data"
        # self.sangraha_val = "/projects/data/datasets/text_data/agri_data/mock_data/sangraha-processed-data-v2/"
        self.sangraha_train_dirs = self._resolve_sangraha_train_dirs(base_path, data_prefix)
        # self.sangraha_train_dirs = ["/projects/data/datasets/text_data/agri_data/mock_data/sangraha-processed-data-v2/"]
        self.sangraha_train = self.sangraha_train_dirs[0]

    def connect(
        self, tokenizer: Optional[Tokenizer] = None, batch_size: int = 1, max_seq_length: Optional[int] = None
    ) -> None:
        self.batch_size = batch_size
        self.seq_length = max_seq_length + 1  # Increase by one because we need the next token as well

    def prepare_data(self) -> None:
        pass

    def train_dataloader(self) -> DataLoader:
        from litdata.streaming import CombinedStreamingDataset, StreamingDataLoader, StreamingDataset, TokensLoader

        print(f"Sangraha train data paths: {', '.join(self.sangraha_train_dirs)}")
        train_datasets = [
            StreamingDataset(
                input_dir=dataset_path,
                # item_loader=TokensLoader(block_size=self.seq_length),
                shuffle=True,
                drop_last=True,
            )
            for dataset_path in self.sangraha_train_dirs
        ]
        train_data = train_datasets[0] if len(train_datasets) == 1 else CombinedStreamingDataset(
            datasets=train_datasets, seed=self.seed
        )

        # if self.use_starcoder:
        #     train_datasets = [
        #         slim_train_data,
        #         StreamingDataset(
        #             input_dir=self.starcoder_train,
        #             item_loader=TokensLoader(block_size=self.seq_length),
        #             shuffle=True,
        #             drop_last=True,
        #         ),
        #     ]

            # Mix SlimPajama data and Starcoder data with these proportions:
            # weights = (0.693584, 0.306416)
            # train_data = CombinedStreamingDataset(
            #     datasets=train_datasets, seed=self.seed, weights=weights, iterate_over_all=False
            # )

        train_dataloader = StreamingDataLoader(
            train_data, 
            batch_size=self.batch_size, 
            pin_memory=True, 
            num_workers=self.num_workers, 
            drop_last=True,
            collate_fn=self.data_collator
        )
        return train_dataloader

    def val_dataloader(self) -> DataLoader:
        from litdata.streaming import StreamingDataLoader, StreamingDataset, TokensLoader

        val_dataset = StreamingDataset(
            input_dir=self.sangraha_val,
            # item_loader=TokensLoader(block_size=self.seq_length),
            shuffle=True,
        )
        val_dataloader = StreamingDataLoader(
            val_dataset, 
            batch_size=self.batch_size, 
            pin_memory=True, 
            num_workers=self.num_workers, 
            drop_last=True,
            collate_fn=self.data_collator
        )
        return val_dataloader

    def data_collator(self, batch):
        from torch.nn.utils.rnn import pad_sequence
        batched = pad_sequence(batch, batch_first=True, padding_value=2)
        return batched

    def _resolve_sangraha_train_dirs(self, base_path: str, data_prefix: str) -> list[str]:
        """Resolve sangraha training directories, combining all discovered or user-specified versions."""
        is_remote = "://" in base_path
        if self.sangraha_versions:
            resolved_dirs: list[str] = []
            for version in self.sangraha_versions:
                cleaned_version = version.strip().strip("/")
                suffix = (
                    cleaned_version
                    if cleaned_version.startswith("sangraha-processed-data-")
                    else f"sangraha-processed-data-{cleaned_version}"
                )
                candidate = f"{data_prefix}/{suffix}"
                if not is_remote and not Path(candidate).is_dir():
                    raise FileNotFoundError(
                        f"Sangraha training directory `{candidate}` does not exist. "
                        "Update `sangraha_versions` or ensure the preprocessing step has completed."
                    )
                resolved_dirs.append(candidate)
            return list(dict.fromkeys(resolved_dirs))

        if not is_remote:
            root_path = Path(data_prefix)
            if root_path.is_dir():
                candidates = sorted(
                    str(path)
                    for path in root_path.glob("sangraha-processed-data-*")
                    if path.is_dir()
                )
                if candidates:
                    return candidates

        # Fallback to the original default directory naming convention
        return [f"{data_prefix}/sangraha-processed-data-v2"]
