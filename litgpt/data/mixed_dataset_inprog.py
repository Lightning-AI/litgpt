# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
### For CombinedLoader
import contextlib
import glob
import os
import time
from collections import defaultdict
from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass, field
from itertools import cycle
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    Union,
)

import numpy as np
import torch
import torch.distributed as dist
from datasets import load_dataset

from lightning.fabric.utilities.data import sized_len
from lightning.fabric.utilities.types import _Stateful
from lightning.pytorch.utilities._pytree import (
    _map_and_unflatten,
    _tree_flatten,
    tree_unflatten,
)

# from litgpt.data.text_files import optimize_data
from litdata import optimize

from litdata.streaming import StreamingDataLoader, StreamingDataset, TokensLoader, CombinedStreamingDataset

from litgpt import PromptStyle
from litgpt.data import DataModule
from litgpt.data.base import (
    _sft_collate_fn,
    get_sft_collate_fn,
    pad_and_stack,
    SFTDataset,
)
from litgpt.data.json_data import find_split, get_splits, load_split
from litgpt.tokenizer import Tokenizer

from torch.distributed import get_rank

from torch.utils.data import BatchSampler, ConcatDataset, DataLoader, IterableDataset, Dataset

from torch.utils.data.dataloader import (
    _BaseDataLoaderIter,
    _InfiniteConstantSampler,
    _MultiProcessingDataLoaderIter,
)
from typing_extensions import override, Self, TypedDict

@dataclass
class MixedDataset(DataModule):
    """
    A dataset that blends together unstructured text (the usual pretraining data) and structured text (SFT data). Can have different proportions of each that evolve over time.
    """

    # A path to a directory containing both pretraining data (files in txt form) as well as SFT data (json files). Should contain two subdirectories "texts" and "sft".
    pretraining_data_path: str = "data/"
    pretraining_val_path: Optional[str] = None # optional val path if in a different dir
    # format of the data: name_1 weight_1 (between 0-1) path_1, name_2 weight_2 path_2, ...
    # if no weights, will combine the sft sets in equal proportions.
    sft_data_paths: str = "sft/"
    pretrain_data_type: str = "streaming"  # "streaming" or "txt"
    sft_val_split_fraction: float = 0.05
    max_seq_length: int = field(init=False, repr=False, default=2048)
    tokenizer: Optional[Tokenizer] = field(default=None, init=False, repr=False)
    prompt_style: Union[str, PromptStyle] = "alpaca"
    mask_prompt: bool = False
    ignore_index: int = -100
    seed: int = 42
    num_workers: int = 8
    cycle_mode: Literal[
        "max_size_cycle", "min_size_cycle", "max_size", "sequential", "max_size_spread"
    ] = "max_size_spread"
    fast_dev_run: bool = False
    # number of times to repeat the sft datasets
    num_repeats: int = 1
    # the max iters through the pretraining dataset. Used to space out the sft datasets accordingly
    max_iters: int = 100000
    use_adaptive_sampling: str = False
    initial_sampling_rates: Optional[List[float]] = None

    def __post_init__(self):
        if not self.pretraining_val_path:
            self.lm_train_path = os.path.join(self.pretraining_data_path, "train")
            self.lm_val_path = os.path.join(self.pretraining_data_path, "val") 
        else:
            self.lm_train_path = self.pretraining_data_path
            self.lm_val_path = self.pretraining_val_path
            
        if not self.lm_val_path:
            raise OSError("No path found for validation dir, either pass a pretraining_data_path with train/ and val/ subdirs, or a separate pretraining_val_path.")
        
        self.out_path_train_lm = self.lm_train_path
        self.out_path_val_lm = self.lm_val_path

        self.sft_datasets_and_sample_rates = self.initialize_sft()
        self.sft_train_datasets = {}
        self.sft_val_datasets = {}

    def _has_sharded_structure(self, base_dir: Union[str, Path]) -> bool:
        """Check if the directory has numbered subdirectories (0-7) with index.json files."""
        for i in range(4):  # Check for shards 0-3
            shard_dir = os.path.join(base_dir, str(i))
            if os.path.isdir(shard_dir) and os.path.exists(os.path.join(shard_dir, "index.json")):
                return True
        return False

    def _create_combined_dataset(self, base_dir: Union[str, Path]) -> CombinedStreamingDataset:
        """Create a combined dataset from sharded directories."""
        datasets = []

        for i in range(4):  # Process shards 0-3
            shard_dir = os.path.join(base_dir, str(i))
            if os.path.isdir(shard_dir) and os.path.exists(os.path.join(shard_dir, "index.json")):
                print(f"Loading shard {i} from {shard_dir}")
                dataset = StreamingDataset(
                    input_dir=shard_dir,
                    item_loader=TokensLoader(block_size=self.max_seq_length_lm),
                    shuffle=True,
                    drop_last=True,
                )
                datasets.append(dataset)
            else:
                print(f"Warning: Shard {i} at {shard_dir} not found or missing index.json")

        if not datasets:
            raise ValueError(f"No valid shards found in {base_dir}")

        print(f"Created combined dataset from {len(datasets)} shards")
        return CombinedStreamingDataset(datasets=datasets, seed=self.seed, iterate_over_all=True)



    def initialize_sft(self) -> dict:
        self.sft_sets = {}
        try:
            for dataset_str in self.sft_data_paths.split(","):
                if " " in dataset_str:
                    name, path, p_dataset = dataset_str.strip().split(" ")
                    self.sft_sets[name] = (float(p_dataset), path)
                else:
                    path = dataset_str.strip()
                    self.sft_sets[path] = (
                        1 / len(self.sft_data_paths.split(",")),
                        path,
                    )
        except:
            print(
                "Error parsing your datasets. Please put them in the format '<path> <init_weight>,...'"
            )

    def setup(self):
        for sft_data_name in self.sft_sets:
            data_weight, sft_data_path = self.sft_sets[sft_data_name]
            sft_train_data, sft_val_data = get_splits(
                Path(sft_data_path),
                self.sft_val_split_fraction,
                num_repeats=self.num_repeats,
            )

            sft_train_dataset = SFTDataset(
                data=sft_train_data,
                tokenizer=self.tokenizer,
                prompt_style=self.prompt_style,
                max_seq_length=self.max_seq_length_sft,
                mask_prompt=self.mask_prompt,
                ignore_index=self.ignore_index,
            )
            sft_val_dataset = SFTDataset(
                data=sft_val_data,
                tokenizer=self.tokenizer,
                prompt_style=self.prompt_style,
                max_seq_length=self.max_seq_length_sft,
                mask_prompt=self.mask_prompt,
                ignore_index=self.ignore_index,
            )

            self.sft_train_datasets[sft_data_name] = sft_train_dataset
            self.sft_val_datasets[sft_data_name] = sft_val_dataset

    def connect(
        self,
        tokenizer: Optional[Tokenizer] = None,
        batch_size: int = 1,
        max_seq_length: int = -1,
        max_iters: int = 1000000,
    ) -> None:
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_seq_length_sft = max_seq_length
        self.max_seq_length_lm = max_seq_length + 1
        self.max_iters = max_iters

    def prepare_data(self) -> None:
        if self.pretrain_data_type == "streaming":
            hf_cache_dir = os.getenv("HF_HOME")

            if str(self.pretraining_data_path).startswith("s3://"):
                print(
                    f"The FineWeb data path points to an S3 location: {self.pretraining_data_path}. Skipping preprocessing."
                )
                return

            print("LM train path: ", self.lm_train_path)
            print("LM val path: ", self.lm_val_path)
            if Path(self.lm_train_path).is_dir() and Path(self.lm_val_path).is_dir():
                print(
                    f"Found FineWeb train and val dir: {self.pretraining_data_path}. Skipping preprocessing."
                )
                return

            dataset = load_dataset(
                "HuggingFaceFW/fineweb-edu",
                num_proc=os.cpu_count() // 8,
                name="train",  # 149M examples
                cache_dir=hf_cache_dir,
                split="train",
            )
            print("Total examples:", len(dataset))
            # save dataset to manifold
            print("Saving dataset to disk")
            dataset.save_to_disk(self.pretraining_data_path)

            # Split the data in training and validation
            split_dataset = dataset.train_test_split(
                test_size=self.val_split_fraction, seed=self.seed, shuffle=True
            )
            split_dataset["val"] = split_dataset.pop(
                "test"
            )  # rename the test split to val

            def tokenize(data: Dataset, index: int):
                yield self.tokenizer.encode(data[index]["text"], eos=True)

            optimize(
                fn=partial(tokenize, split_dataset["train"]),
                inputs=list(range(len(split_dataset["train"]))),
                output_dir=self.out_path_train_lm,
                num_workers=(os.cpu_count() // 8),
                # num_workers=8,
                chunk_bytes="200MB",
                fast_dev_run=self.fast_dev_run,
            )

            optimize(
                fn=partial(tokenize, split_dataset["val"]),
                inputs=list(range(len(split_dataset["val"]))),
                output_dir=self.out_path_val_lm,
                num_workers=(os.cpu_count() // 8),
                chunk_bytes="200MB",
                fast_dev_run=self.fast_dev_run,
            )
            print(f"Finished preprocessing of {self.data_path}")
        else:
            train_files = sorted(glob.glob(str(self.lm_train_path / "*.txt")))
            assert (
                len(train_files) > 0
            ), f"No .txt files found in train data {train_files}"

            if self.lm_val_path is not None:
                self.lm_val_path = Path(self.lm_val_path)
                val_files = sorted(glob.glob(str(self.lm_val_path / "*.txt")))
                assert (
                    len(val_files) > 0
                ), f"No .txt files found in validation data {val_files}"
            # train/test split. let's use only shard 0 for test split, rest train
            else:
                assert (
                    len(train_files) > 1
                ), f"Expected at least two .txt files in {train_files}"
                val_files, *train_files = train_files
                val_files = [val_files]

            # It's ok to use almost all CPUs here because this runs in a single process
            num_workers = os.cpu_count() - 1

            optimize_data(
                num_workers,
                str(self.out_path_train_lm),
                str(self.out_path_val_lm),
                self.tokenizer,
                train_files,
                val_files,
            )

    def train_dataloader(self) -> DataLoader:
        # fixes to prevent deadlocks when running distributed sampling
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        total_datasets = len(self.sft_train_datasets) + 1
        workers_per_dataset = max(1, self.num_workers // total_datasets)
        print(f"Allocating {workers_per_dataset} workers per dataset")


        sampling_batch_size = self.batch_size if not self.use_adaptive_sampling else 1

        # Handle sharded or non-sharded pretraining data
        if self._has_sharded_structure(self.out_path_train_lm):
            self.lm_train_dataset = self._create_combined_dataset(self.out_path_train_lm)
        else:
            self.lm_train_dataset = StreamingDataset(
                input_dir=self.out_path_train_lm,
                item_loader=TokensLoader(block_size=self.max_seq_length_lm),
                shuffle=True,
                drop_last=True,
            )

        lm_train_dataloader = StreamingDataLoader(
            self.lm_train_dataset,
            batch_size=sampling_batch_size,
            pin_memory=True,
            drop_last=True,
        )

        sft_train_dataloaders = {}
        for sft_name in self.sft_train_datasets:
            sft_train_dataset = self.sft_train_datasets[sft_name]
            sft_train_dataloader = DataLoader(
                sft_train_dataset,
                batch_size=1,
                shuffle=True,
                num_workers=workers_per_dataset,
                collate_fn=get_sft_collate_fn(
                    max_seq_length=self.max_seq_length_sft,
                    ignore_index=self.ignore_index,
                ),
            )
            sft_train_dataloaders[sft_name] = sft_train_dataloader

        if self.use_adaptive_sampling:
            return CombinedLoaderWithSamplingRates(
                {"lm": lm_train_dataloader, **sft_train_dataloaders},
                self.initial_sampling_rates,
                batch_size=self.batch_size,
                max_iters=self.max_iters,
                seq_max_len=self.max_seq_length_sft,
                num_workers=1,
            )
        else:
            return CombinedLoader(
                {"lm": lm_train_dataloader, **sft_train_dataloaders},
                self.cycle_mode,
                max_iters=self.max_iters,
            )

    def val_dataloader(self) -> DataLoader:
        # fixes to prevent deadlocks when running distributed sampling
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        total_datasets = len(self.sft_train_datasets) + 1
        workers_per_dataset = max(1, self.num_workers // total_datasets)

        # Handle sharded or non-sharded validation data
        if self._has_sharded_structure(self.out_path_val_lm):
            self.lm_test_dataset = self._create_combined_dataset(self.out_path_val_lm)
        else:
            self.lm_test_dataset = StreamingDataset(
                input_dir=self.out_path_val_lm,
                item_loader=TokensLoader(block_size=self.max_seq_length_lm),
                shuffle=True,
                drop_last=True,
            )

        lm_test_dataloader = StreamingDataLoader(
            self.lm_test_dataset,
            batch_size=self.batch_size,
            pin_memory=True,
            drop_last=True,
        )

        sft_val_dataloaders = {}
        for sft_name in self.sft_val_datasets:
            sft_val_dataset = self.sft_val_datasets[sft_name]
            sft_val_dataloader = DataLoader(
                sft_val_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=workers_per_dataset,
                collate_fn=get_sft_collate_fn(
                    max_seq_length=self.max_seq_length_sft,
                    ignore_index=self.ignore_index,
                ),
            )
            sft_val_dataloaders[sft_name] = sft_val_dataloader

        return CombinedLoader(
            {"lm": lm_test_dataloader, **sft_val_dataloaders}, 
            "max_size"
        )



class CombinedLoaderWithSamplingRates(DataLoader):
    ### A combined loader where we can pass in a list of sampling rates for each dataset.
    ### Note: the final batch may not exactly represent the sampling rates.
    ### Entries will be returned in the same format as the combinedloader: {"data_1": data1_samples, "data_2": data2_samples, ...}
    ### The data samples will be None if there were no samples from that dataset selected.
    def __init__(
        self, loaders, sampling_rates, batch_size, max_iters, seq_max_len, **kwargs
    ):
        self.loaders = loaders
        self.sampling_rates = sampling_rates
        print("The initial sampling rates are: ", self.sampling_rates)
        assert len(self.loaders) == len(
            self.sampling_rates
        ), "The length of sampling rates should be the same as the number of iterables"
        self.max_iters = max_iters
        self.batch_size = batch_size
        self.seq_max_len = seq_max_len
        self.cumulative_rates = [
            sum(sampling_rates[:i]) for i in range(len(sampling_rates))
        ]
        self.multiprocessing_context = None
        self.iterators = {}

        self._flattened, self._spec = _tree_flatten(loaders)
        self.dataset = ConcatIterableDataset(
            [getattr(dl, "dataset", None) for dl in self.flattened]
        )

        # if dist.is_initialized():
        #     rank = dist.get_rank()
        #     world_size = dist.get_world_size()
        #     seed = 42 + rank
        # else:
        #     seed = 42

        self.sharded_loaders = {}
        if dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            for name, loader in self.loaders.items():
                if not isinstance(loader, StreamingDataLoader):
                    # note: we're keeping the same seed, but shouldn't be an issue since each rank has a subset
                    sharded_dataset = torch.utils.data.Subset(
                        loader.dataset, range(rank, len(loader.dataset), world_size)
                    )
                    self.sharded_loaders[name] = DataLoader(
                        sharded_dataset,
                        batch_size=loader.batch_size,
                        collate_fn=loader.collate_fn,
                        num_workers=loader.num_workers,
                    )
                else:
                    # don't need to shard pretraining datasets
                    self.sharded_loaders[name] = loader
        else:
            self.sharded_loaders = self.loaders
        self.rng = np.random.default_rng(seed=42)

        # dataset = SampledCombinedDataset(loaders, sampling_rates, max_iters, batch_size)
        # super().__init__(dataset, batch_size, **kwargs)

    def __len__(self):
        return self.max_iters

    def __iter__(self):
        rank = dist.get_rank() if dist.is_initialized() else 0
        print(f"[Rank {rank}] Starting iterator initialization")

        # initialize the iterators
        for name, loader in self.sharded_loaders.items():
            print(f"[Rank {rank}] Initializing iterator for {name}")
            if isinstance(loader, DataLoader) or hasattr(loader, "__iter__"):
                try:
                    self.iterators[name] = iter(cycle(loader))
                    print(f"[Rank {rank}] Successfully initialized iterator for {name}")
                except Exception as e:
                    print(f"[Rank {rank}] Failed to initialize iterator for {name}: {str(e)}")
                    raise
            else:
                print(f"[Rank {rank}] Failed to initialize iterator for {name}")
                raise ValueError(
                    f"Expected a DataLoader or iterable, but got {type(loader)}"
                )
        # fill the batch by random sampling from each dataset
        iterators = {name: iterable for name, iterable in self.iterators.items()}
        print(f"[Rank {rank}] All iterators initialized")

        for _ in range(self.max_iters):
            batch = defaultdict(list)
            try:
                chosen_datasets = self.rng.choice(
                    list(self.iterators.keys()),
                    size=self.batch_size,
                    p=self.sampling_rates,
                )
            except:
                self.sampling_rates = [
                    x / sum(self.sampling_rates) for x in self.sampling_rates
                ]  # sometimes it doesn't sum to 1 due to floating point
                chosen_datasets = self.rng.choice(
                    list(self.iterators.keys()),
                    size=self.batch_size,
                    p=self.sampling_rates,
                )

            for dataset_name in chosen_datasets:
                if batch[dataset_name] is None:
                    batch[dataset_name] = []
                try:
                    item = next(iterators[dataset_name])
                except StopIteration:
                    # reset
                    iterators[dataset_name] = iter(self.iterators[dataset_name])
                    try:
                        item = next(iterators[dataset_name])
                    except:
                        print(f"[Rank {rank}] Failed to reinitialize iterator: {str(e)}")
                        raise

                batch[dataset_name].append(item)

            # stack to tensor
            for dataset_name in batch:
                if dataset_name == "lm":
                    batch[dataset_name] = torch.cat(batch[dataset_name], dim=0)
                else:  # sft format
                    stacked = {}
                    for key in ["input_ids", "labels"]:
                        stacked[key] = pad_and_stack(
                            [sample[key] for sample in batch[dataset_name]],
                            self.seq_max_len,
                        )
                    batch[dataset_name] = stacked

            yield dict(batch)

    def set_sampling_rates(self, new_sampling_rates):
        assert len(self.loaders.keys()) == len(
            new_sampling_rates
        ), "The length of sampling rates should be equal to the num datasets"
        self.sampling_rates = new_sampling_rates
        self.cumulative_rates = [
            sum(self.sampling_rates[:i]) for i in range(len(self.sampling_rates))
        ]

    @property
    def iterables(self) -> Any:
        """Return the original collection of iterables."""
        return self.loaders

    @property
    def sampler(self) -> Any:
        """Return a collections of samplers extracted from iterables."""
        return UnifiedBatchSampler(
            _map_and_unflatten(
                lambda x: getattr(x, "sampler", None),
                self.flattened,
                self._spec,
            ),
            max_iters=self.max_iters,
        )

    @property
    def batch_sampler(self) -> Any:
        """Return a collections of batch samplers extracted from iterables."""
        return UnifiedBatchSampler(
            _map_and_unflatten(
                lambda x: getattr(x, "batch_sampler", None),
                self.flattened,
                self._spec,
            ),
            max_iters=self.max_iters,
        )

    @property
    def flattened(self) -> List[Any]:
        """Return the flat list of iterables."""
        return self._flattened

    @flattened.setter
    def flattened(self, flattened: List[Any]) -> None:
        """Setter to conveniently update the list of iterables."""
        if len(flattened) != len(self._flattened):
            raise ValueError(
                f"Mismatch in flattened length ({len(flattened)}) and existing length ({len(self._flattened)})"
            )
        # update the iterable collection
        self._iterables = tree_unflatten(flattened, self._spec)
        self._flattened = flattened

    @property
    def limits(self) -> Optional[List[Union[int, float]]]:
        """Optional limits per iterator."""
        return self._limits


class _CombinedLoaderWithSamplingRates(DataLoader):
    ### A combined loader where we can pass in a list of sampling rates for each dataset.
    ### Note: the final batch may not exactly represent the sampling rates.
    ### Entries will be returned in the same format as the combinedloader: {"data_1": data1_samples, "data_2": data2_samples, ...}
    ### The data samples will be None if there were no samples from that dataset selected.
    def __init__(
        self, loaders, sampling_rates, batch_size, max_iters, seq_max_len, num_workers=1, **kwargs
    ):

        self.loaders = loaders
        self.sampling_rates = sampling_rates
        print("The initial sampling rates are: ", self.sampling_rates)
        assert len(self.loaders) == len(
            self.sampling_rates
        ), "The length of sampling rates should be the same as the number of iterables"
        self.max_iters = max_iters
        self.batch_size = batch_size
        self.seq_max_len = seq_max_len
        self.cumulative_rates = [
            sum(sampling_rates[:i]) for i in range(len(sampling_rates))
        ]
        self.multiprocessing_context = None
        self.iterators = {}

        self._flattened, self._spec = _tree_flatten(loaders)
        self.dataset = ConcatIterableDataset(
            [getattr(dl, "dataset", None) for dl in self.flattened]
        )
        self.rng = torch.Generator()
        self.seed = 42
        self.set_seed_and_shard()


    def __len__(self):
        return self.max_iters
    
    def set_seed_and_shard(self):
        if dist.is_initialized():
            rank = dist.get_rank()
            print(f"Rank {rank}: The dataloader is distributed. Setting the seed to {self.seed} + {rank}")
            self.rng.manual_seed(self.seed + rank)
            self.world_size = dist.get_world_size()
            self.rank = rank

            # self.sharded_loaders = {}
            # for name, loader in self.loaders.items():
            #     if not isinstance(loader, StreamingDataLoader):
            #         sharded_dataset = torch.utils.data.Subset(
            #             loader.dataset, range(self.rank, len(loader.dataset), self.world_size)
            #         )
            #         self.sharded_loaders[name] = DataLoader(
            #             sharded_dataset,
            #             batch_size=loader.batch_size,
            #             collate_fn=loader.collate_fn,
            #             num_workers=loader.num_workers,
            #         )
            #     else:
            #         # don't need to shard pretraining datasets
            #         self.sharded_loaders[name] = loader
            
        else:
            print(f"The dataloader is not distributed. Setting the seed to {self.seed}")
            self.rng.manual_seed(self.seed)
            self.sharded_loaders = self.loaders
            self.world_size = 1
            self.rank = 0

    def __iter__(self):
        for name, loader in self.loaders.items():
            if isinstance(loader, DataLoader) or hasattr(loader, "__iter__"):
                self.iterators[name] = iter(cycle(loader))
            else:
                raise ValueError(
                    f"Expected a DataLoader or iterable, but got {type(loader)}"
                )
        # fill the batch by random sampling from each dataset
        iterators = {name: iterable for name, iterable in self.iterators.items()}

        for _ in range(self.max_iters):
            batch = defaultdict(list)
            try:
                # chosen_datasets = self.rng.choice(
                #     list(self.iterators.keys()),
                #     size=self.batch_size,
                #     p=self.sampling_rates,
                # )
                chosen_dataset_inds = torch.multinomial(torch.tensor(self.sampling_rates), self.batch_size, replacement=True, generator=self.rng)
            except:
                self.sampling_rates = [
                    x / sum(self.sampling_rates) for x in self.sampling_rates
                ]  # sometimes it doesn't sum to 1 due to floating point
                # chosen_datasets = self.rng.choice(
                #     list(self.iterators.keys()),
                #     size=self.batch_size,
                #     p=self.sampling_rates,
                # )
                chosen_dataset_inds = torch.multinomial(torch.tensor(self.sampling_rates), self.batch_size, replacement=True, generator=self.rng)
            
            chosen_datasets = [list(self.iterators.keys())[i] for i in chosen_dataset_inds]
            rank = dist.get_rank() if dist.is_initialized() else 0
            for dataset_name in chosen_datasets:
                if batch[dataset_name] is None:
                    batch[dataset_name] = []
                try:
                    item = next(iterators[dataset_name])

                except StopIteration:
                    # reset
                    iterators[dataset_name] = iter(self.iterators[dataset_name])
                    try:
                        item = next(iterators[dataset_name])
                    except:
                        breakpoint()

                batch[dataset_name].append(item)

            # stack to tensor
            for dataset_name in batch:
                if dataset_name == "lm":
                    batch[dataset_name] = torch.cat(batch[dataset_name], dim=0)
                else:  # sft format
                    stacked = {}
                    for key in ["input_ids", "labels"]:
                        stacked[key] = pad_and_stack(
                            [sample[key] for sample in batch[dataset_name]],
                            self.seq_max_len,
                        )
                    batch[dataset_name] = stacked

            yield dict(batch)

    def set_sampling_rates(self, new_sampling_rates):
        assert len(self.loaders.keys()) == len(
            new_sampling_rates
        ), "The length of sampling rates should be equal to the num datasets"
        self.sampling_rates = new_sampling_rates
        self.cumulative_rates = [
            sum(self.sampling_rates[:i]) for i in range(len(self.sampling_rates))
        ]

    @property
    def iterables(self) -> Any:
        """Return the original collection of iterables."""
        return self.loaders

    @property
    def sampler(self) -> Any:
        """Return a collections of samplers extracted from iterables."""
        return UnifiedBatchSampler(
            _map_and_unflatten(
                lambda x: getattr(x, "sampler", None),
                self.flattened,
                self._spec,
            ),
            max_iters=self.max_iters,
        )

    @property
    def batch_sampler(self) -> Any:
        """Return a collections of batch samplers extracted from iterables."""
        return UnifiedBatchSampler(
            _map_and_unflatten(
                lambda x: getattr(x, "batch_sampler", None),
                self.flattened,
                self._spec,
            ),
            max_iters=self.max_iters,
        )

    @property
    def flattened(self) -> List[Any]:
        """Return the flat list of iterables."""
        return self._flattened

    @flattened.setter
    def flattened(self, flattened: List[Any]) -> None:
        """Setter to conveniently update the list of iterables."""
        if len(flattened) != len(self._flattened):
            raise ValueError(
                f"Mismatch in flattened length ({len(flattened)}) and existing length ({len(self._flattened)})"
            )
        # update the iterable collection
        self._iterables = tree_unflatten(flattened, self._spec)
        self._flattened = flattened

    @property
    def limits(self) -> Optional[List[Union[int, float]]]:
        """Optional limits per iterator."""
        return self._limits

### This code was not present in the current version of lightning. Adding it here
# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


_ITERATOR_RETURN = Tuple[Any, int, int]  # batch, batch_idx, dataloader_idx


class _ModeIterator(Iterator[_ITERATOR_RETURN]):
    def __init__(
        self,
        iterables: List[Iterable],
        limits: Optional[List[Union[int, float]]] = None,
    ) -> None:
        if limits is not None and len(limits) != len(iterables):
            raise ValueError(
                f"Mismatch in number of limits ({len(limits)}) and number of iterables ({len(iterables)})"
            )
        self.iterables = iterables
        self.iterators: List[Iterator] = []
        self._idx = 0  # what would be batch_idx
        self.limits = limits

    @override
    def __next__(self) -> _ITERATOR_RETURN:
        raise NotImplementedError

    @override
    def __iter__(self) -> Self:
        self.iterators = [iter(iterable) for iterable in self.iterables]
        self._idx = 0
        return self

    def __len__(self) -> int:
        raise NotImplementedError

    def reset(self) -> None:
        self.iterators = []
        self._idx = 0

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()

        # workaround an inconvenient `NotImplementedError`:
        # https://github.com/pytorch/pytorch/blob/v2.0.0/torch/utils/data/dataloader.py#L652-L658
        state["iterators"] = [
            None if isinstance(iterator, _BaseDataLoaderIter) else iterator_state
            for iterator, iterator_state in zip(self.iterators, state["iterators"])
        ]

        return state


class _MaxSizeCycle(_ModeIterator):
    def __init__(
        self,
        iterables: List[Iterable],
        limits: Optional[List[Union[int, float]]] = None,
    ) -> None:
        super().__init__(iterables, limits)
        self._consumed: List[bool] = []

    @override
    def __next__(self) -> _ITERATOR_RETURN:
        n = len(self.iterators)
        out = [None] * n  # values per iterator
        for i in range(n):
            try:
                out[i] = next(self.iterators[i])
            except StopIteration:
                self._consumed[i] = True
                if all(self._consumed):
                    raise
                # reset the consumed dataloader
                self.iterators[i] = iter(self.iterables[i])
                out[i] = next(self.iterators[i])
        index = self._idx
        self._idx += 1
        return out, index, 0

    @override
    def __iter__(self) -> Self:
        super().__iter__()
        self._consumed = [False] * len(self.iterables)
        return self

    @override
    def __len__(self) -> int:
        lengths = _get_iterables_lengths(self.iterables)
        if self.limits is not None:
            return max(min(length, limit) for length, limit in zip(lengths, self.limits))  # type: ignore[return-value]
        return max(lengths)  # type: ignore[return-value]

    @override
    def reset(self) -> None:
        super().reset()
        self._consumed = []


class _MaxSizeDistributed(_ModeIterator):
    # Largest dataset should always be placed at ind 0. Assume we want to distribute the rest
    max_steps = 100000  # TODO: the global variable hack doesn't work, fix the pass through later.

    def __init__(self, iterables, limits=None):
        super().__init__(iterables, limits)
        self.large_dataset_idx = 0
        self.small_datasets_idx = [0 for _ in range(len(iterables) - 1)]
        self.large_data_len = len(self.iterables[0])
        self.small_data_len = [
            len(self.iterables[i]) for i in range(1, len(self.iterables))
        ]

        self.small_data_per_batch = [
            small_data_len / min(self.large_data_len, self.max_steps)
            for small_data_len in self.small_data_len
        ]
        self.data_intervals = []

        for i, small_data_num in enumerate(self.small_data_per_batch):
            if small_data_num < 1:  # have 1 item oer N batches
                self.data_intervals.append(int(1 / small_data_num))
            else:
                self.data_intervals.append(1)

        print(f"Pretraining dataset batches: {self.large_data_len}")
        print(f"SFT dataset batches: {self.small_data_len}")
        print(f"Adding SFT data every {self.data_intervals} batches")

        self.intervals_passed = deepcopy(self.data_intervals)

        self._consumed: List[bool] = []

    def __next__(self):
        out = [None] * len(self.iterators)
        try:
            large_batch = next(self.iterators[0])
            out[0] = large_batch
            small_batches = []
            for i, small_dataset in enumerate(self.iterators[1:]):
                data_interval = self.data_intervals[i]
                if data_interval == 1:
                    small_batch = next(small_dataset)
                    small_batches.extend(small_batch)
                    out[i + 1] = small_batch
                else:
                    # note: length of an iterable is the # batches, not # samples
                    self.intervals_passed[i] -= 1
                    if self.intervals_passed[i] <= 0:
                        if self.small_data_per_batch[i] > 1:
                            small_batch = []
                            for _ in range(self.small_data_per_batch[i]):
                                small_batch.append(next(small_dataset))
                        else:
                            small_batch = next(small_dataset)
                        small_batches.extend(small_batch)
                        out[i + 1] = small_batch
                        self.intervals_passed[i] = data_interval

            index = self._idx
            self._idx += 1

            return out, index, 0

        except StopIteration:
            raise StopIteration

    @override
    def __len__(self) -> int:
        lengths = _get_iterables_lengths(self.iterables)
        if self.limits is not None:
            return max(min(length, limit) for length, limit in zip(lengths, self.limits))  # type: ignore[return-value]
        return max(lengths)  # type: ignore[return-value]

    @override
    def reset(self) -> None:
        super().reset()
        self._consumed = []


class _MinSize(_ModeIterator):
    @override
    def __next__(self) -> _ITERATOR_RETURN:
        out = [next(it) for it in self.iterators]
        index = self._idx
        self._idx += 1
        return out, index, 0

    @override
    def __len__(self) -> int:
        lengths = _get_iterables_lengths(self.iterables)
        return min(lengths + self.limits) if self.limits is not None else min(lengths)  # type: ignore[return-value]


class _Sequential(_ModeIterator):
    def __init__(
        self,
        iterables: List[Iterable],
        limits: Optional[List[Union[int, float]]] = None,
    ) -> None:
        super().__init__(iterables, limits)
        self._iterator_idx = 0  # what would be dataloader_idx

    @override
    def __next__(self) -> _ITERATOR_RETURN:
        n = len(self.iterables)
        if n == 0 or self._iterator_idx >= n:
            raise StopIteration

        # if limits are set, go to the correct iterator
        if self.limits is not None:
            while self.limits[self._iterator_idx] <= self._idx:
                self._use_next_iterator()
                if self._iterator_idx >= n:
                    raise StopIteration

        try:
            out = next(self.iterators[0])
        except StopIteration:
            # try the next iterator
            self._use_next_iterator()
            return self.__next__()
        index = self._idx
        self._idx += 1
        return out, index, self._iterator_idx

    @override
    def __iter__(self) -> Self:
        self._iterator_idx = 0
        self._idx = 0
        self._load_current_iterator()
        return self

    @override
    def __len__(self) -> int:
        lengths = _get_iterables_lengths(self.iterables)
        if self.limits is not None:
            return sum(min(length, limit) for length, limit in zip(lengths, self.limits))  # type: ignore[misc]
        return sum(lengths)  # type: ignore[arg-type]

    @override
    def reset(self) -> None:
        super().reset()
        self._iterator_idx = 0

    def _load_current_iterator(self) -> None:
        # Load a single DataLoader, prevents multiple sets of workers from starting unnecessarily
        if self._iterator_idx < len(self.iterables):
            self.iterators = [iter(self.iterables[self._iterator_idx])]
        else:
            # No more iterables to step through, return an empty list
            self.iterators = []

    def _use_next_iterator(self) -> None:
        self._iterator_idx += 1
        self._idx = 0
        self._load_current_iterator()


class _MaxSize(_ModeIterator):
    @override
    def __next__(self) -> _ITERATOR_RETURN:
        n = len(self.iterators)
        out = [None] * n
        all_exhausted = True
        for i in range(n):
            with contextlib.suppress(StopIteration):
                out[i] = next(self.iterators[i])
                all_exhausted = False
        if all_exhausted:
            raise StopIteration
        index = self._idx
        self._idx += 1
        return out, index, 0

    @override
    def __len__(self) -> int:
        lengths = _get_iterables_lengths(self.iterables)
        if self.limits is not None:
            return max(min(length, limit) for length, limit in zip(lengths, self.limits))  # type: ignore[return-value]
        return max(lengths)  # type: ignore[return-value]


class _CombinationMode(TypedDict):
    fn: Callable[[List[int]], int]
    iterator: Type[_ModeIterator]


_SUPPORTED_MODES = {
    "min_size": _CombinationMode(fn=min, iterator=_MinSize),
    "max_size_cycle": _CombinationMode(fn=max, iterator=_MaxSizeCycle),
    "max_size": _CombinationMode(fn=max, iterator=_MaxSize),
    "sequential": _CombinationMode(fn=sum, iterator=_Sequential),
    "max_size_spread": _CombinationMode(fn=max, iterator=_MaxSizeDistributed),
}

# max_size_distributed: mimic scattering a small dataset across a large one (no repeats, but evenly spaced data)
_LITERAL_SUPPORTED_MODES = Literal[
    "min_size", "max_size_cycle", "max_size", "sequential", "max_size_spread"
]


class CombinedLoader(DataLoader):
    """Combines different iterables under specific sampling modes.

    Args:
        iterables: the iterable or collection of iterables to sample from.
        mode: the mode to use. The following modes are supported:

            * ``min_size``: stops after the shortest iterable (the one with the lowest number of items) is done.
            * ``max_size_cycle``: stops after the longest iterable (the one with most items) is done, while cycling
              through the rest of the iterables.
            * ``max_size``: stops after the longest iterable (the one with most items) is done, while returning None
              for the exhausted iterables.
            *  ``max_size_spread``: mimics scattering a small dataset across a large one (no repeats, but evenly spaced data)
            * ``sequential``: completely consumes each iterable sequentially, and returns a triplet
              ``(data, idx, iterable_idx)``

    Examples:
        >>> from torch.utils.data import DataLoader
        >>> iterables = {'a': DataLoader(range(6), batch_size=4),
        ...              'b': DataLoader(range(15), batch_size=5)}
        >>> combined_loader = CombinedLoader(iterables, 'max_size_cycle')
        >>> _ = iter(combined_loader)
        >>> len(combined_loader)
        3
        >>> for batch, batch_idx, dataloader_idx in combined_loader:
        ...     print(f"{batch}, {batch_idx=}, {dataloader_idx=}")
        {'a': tensor([0, 1, 2, 3]), 'b': tensor([0, 1, 2, 3, 4])}, batch_idx=0, dataloader_idx=0
        {'a': tensor([4, 5]), 'b': tensor([5, 6, 7, 8, 9])}, batch_idx=1, dataloader_idx=0
        {'a': tensor([0, 1, 2, 3]), 'b': tensor([10, 11, 12, 13, 14])}, batch_idx=2, dataloader_idx=0

        >>> combined_loader = CombinedLoader(iterables, 'max_size')
        >>> _ = iter(combined_loader)
        >>> len(combined_loader)
        3
        >>> for batch, batch_idx, dataloader_idx in combined_loader:
        ...     print(f"{batch}, {batch_idx=}, {dataloader_idx=}")
        {'a': tensor([0, 1, 2, 3]), 'b': tensor([0, 1, 2, 3, 4])}, batch_idx=0, dataloader_idx=0
        {'a': tensor([4, 5]), 'b': tensor([5, 6, 7, 8, 9])}, batch_idx=1, dataloader_idx=0
        {'a': None, 'b': tensor([10, 11, 12, 13, 14])}, batch_idx=2, dataloader_idx=0

        >>> combined_loader = CombinedLoader(iterables, 'min_size')
        >>> _ = iter(combined_loader)
        >>> len(combined_loader)
        2
        >>> for batch, batch_idx, dataloader_idx in combined_loader:
        ...     print(f"{batch}, {batch_idx=}, {dataloader_idx=}")
        {'a': tensor([0, 1, 2, 3]), 'b': tensor([0, 1, 2, 3, 4])}, batch_idx=0, dataloader_idx=0
        {'a': tensor([4, 5]), 'b': tensor([5, 6, 7, 8, 9])}, batch_idx=1, dataloader_idx=0

        >>> combined_loader = CombinedLoader(iterables, 'sequential')
        >>> _ = iter(combined_loader)
        >>> len(combined_loader)
        5
        >>> for batch, batch_idx, dataloader_idx in combined_loader:
        ...     print(f"{batch}, {batch_idx=}, {dataloader_idx=}")
        tensor([0, 1, 2, 3]), batch_idx=0, dataloader_idx=0
        tensor([4, 5]), batch_idx=1, dataloader_idx=0
        tensor([0, 1, 2, 3, 4]), batch_idx=0, dataloader_idx=1
        tensor([5, 6, 7, 8, 9]), batch_idx=1, dataloader_idx=1
        tensor([10, 11, 12, 13, 14]), batch_idx=2, dataloader_idx=1

    """

    def __init__(
        self,
        iterables: Any,
        mode: _LITERAL_SUPPORTED_MODES = "min_size",
        max_iters=None,
    ) -> None:
        if mode not in _SUPPORTED_MODES:
            raise ValueError(
                f"Unsupported mode {mode!r}, please select one of: {list(_SUPPORTED_MODES)}."
            )
        self._iterables = iterables
        self._flattened, self._spec = _tree_flatten(iterables)
        self._mode = mode
        self._iterator: Optional[_ModeIterator] = None
        self._limits: Optional[List[Union[int, float]]] = None
        self.multiprocessing_context = None  # is this a good default?
        self.max_iters = max_iters
        self.dataset = ConcatIterableDataset(
            [getattr(dl, "dataset", None) for dl in self.flattened]
        )

    @property
    def iterables(self) -> Any:
        """Return the original collection of iterables."""
        return self._iterables

    @property
    def sampler(self) -> Any:
        """Return a collections of samplers extracted from iterables."""
        return UnifiedBatchSampler(
            _map_and_unflatten(
                lambda x: getattr(x, "sampler", None),
                self.flattened,
                self._spec,
            ),
            max_iters=self.max_iters,
        )

    @property
    def batch_sampler(self) -> Any:
        """Return a collections of batch samplers extracted from iterables."""
        return UnifiedBatchSampler(
            _map_and_unflatten(
                lambda x: getattr(x, "batch_sampler", None),
                self.flattened,
                self._spec,
            ),
            max_iters=self.max_iters,
        )

    @property
    def flattened(self) -> List[Any]:
        """Return the flat list of iterables."""
        return self._flattened

    @flattened.setter
    def flattened(self, flattened: List[Any]) -> None:
        """Setter to conveniently update the list of iterables."""
        if len(flattened) != len(self._flattened):
            raise ValueError(
                f"Mismatch in flattened length ({len(flattened)}) and existing length ({len(self._flattened)})"
            )
        # update the iterable collection
        self._iterables = tree_unflatten(flattened, self._spec)
        self._flattened = flattened

    @property
    def limits(self) -> Optional[List[Union[int, float]]]:
        """Optional limits per iterator."""
        return self._limits

    @limits.setter
    def limits(
        self, limits: Optional[Union[int, float, List[Union[int, float]]]]
    ) -> None:
        if isinstance(limits, (int, float)):
            limits = [limits] * len(self.flattened)
        elif isinstance(limits, list) and len(limits) != len(self.flattened):
            raise ValueError(
                f"Mismatch in number of limits ({len(limits)}) and number of iterables ({len(self.flattened)})"
            )
        self._limits = limits

    def __next__(self) -> _ITERATOR_RETURN:
        assert self._iterator is not None
        out = next(self._iterator)
        if isinstance(self._iterator, _Sequential):
            return out
        out, batch_idx, dataloader_idx = out
        return tree_unflatten(out, self._spec), batch_idx, dataloader_idx

    @override
    def __iter__(self) -> Self:
        cls = _SUPPORTED_MODES[self._mode]["iterator"]
        iterator = cls(self.flattened, self._limits)
        iter(iterator)
        self._iterator = iterator
        return self

    def __len__(self) -> int:
        """Compute the number of batches."""
        if self._iterator is None:
            raise RuntimeError("Please call `iter(combined_loader)` first.")
        return len(self._iterator)

    def reset(self) -> None:
        """Reset the state and shutdown any workers."""
        if self._iterator is not None:
            self._iterator.reset()
            self._iterator = None
        for iterable in self.flattened:
            _shutdown_workers_and_reset_iterator(iterable)

    def _dataset_length(self) -> int:
        """Compute the total length of the datasets according to the current mode."""
        datasets = [getattr(dl, "dataset", None) for dl in self.flattened]
        lengths = [length for ds in datasets if (length := sized_len(ds)) is not None]
        if not lengths:
            raise NotImplementedError("All datasets are iterable-style datasets.")
        fn = _SUPPORTED_MODES[self._mode]["fn"]
        return fn(lengths)

    def _state_dicts(self) -> List[Dict[str, Any]]:
        """Returns the list of state dicts for iterables in `self.flattened` that are stateful."""
        return [
            loader.state_dict()
            for loader in self.flattened
            if isinstance(loader, _Stateful)
        ]

    def _load_state_dicts(self, states: List[Dict[str, Any]]) -> None:
        """Loads the state dicts for iterables in `self.flattened` that are stateful."""
        if not states:
            return
        stateful_loaders = [
            loader for loader in self.flattened if isinstance(loader, _Stateful)
        ]
        if len(stateful_loaders) != len(states):
            raise RuntimeError(
                f"The CombinedLoader has {len(stateful_loaders)} stateful loaders, but found {len(states)} states"
                " in the checkpoint. Please make sure you define the same dataloaders that were used when saving"
                " the checkpoint."
            )
        for loader, state_dict in zip(stateful_loaders, states):
            loader.load_state_dict(state_dict)


def _shutdown_workers_and_reset_iterator(dataloader: object) -> None:
    if hasattr(dataloader, "_iterator"):
        if isinstance(dataloader._iterator, _MultiProcessingDataLoaderIter):
            dataloader._iterator._shutdown_workers()
        dataloader._iterator = None


def _get_iterables_lengths(iterables: List[Iterable]) -> List[Union[int, float]]:
    return [
        (float("inf") if (length := sized_len(iterable)) is None else length)
        for iterable in iterables
    ]


class UnifiedBatchSampler(BatchSampler):
    def __init__(self, batch_samplers, max_iters=None):
        self.batch_samplers = batch_samplers
        self.samplers_iter = {
            key: iter(sampler) for key, sampler in batch_samplers.items()
        }
        self.max_iters = max_iters

        self._max_size = 0 if self.max_iters is None else self.max_iters

        for sampler in self.batch_samplers.values():
            try:
                self._max_size = max(self._max_size, len(sampler))
            except:
                continue

    def __iter__(self):
        active_samplers = list(self.samplers_iter.keys())
        while active_samplers:
            for key in list(
                active_samplers
            ):  # List to avoid modification during iteration
                try:
                    batch = next(self.samplers_iter[key])
                    yield batch
                except StopIteration:
                    active_samplers.remove(key)
                    del self.samplers_iter[key]

    def __len__(self):
        # This might need to be adjusted based on how you decide to combine the batches
        sum_size = 0
        for sampler in self.batch_samplers.values():
            try:
                sum_size += len(sampler)
            except:
                sum_size += self._max_size  # just set this for infinite datasets
        return sum_size


class ConcatIterableDataset(IterableDataset):
    def __init__(self, datasets):
        super(ConcatIterableDataset, self).__init__()
        self.datasets = datasets

    def __iter__(self):
        for dataset in self.datasets:
            for item in dataset:
                yield item
