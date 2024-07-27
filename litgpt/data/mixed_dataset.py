# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
### For CombinedLoader
import contextlib
import glob
import os
from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass, field
from functools import partial
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

from litdata.streaming import StreamingDataLoader, StreamingDataset, TokensLoader
from litgpt import PromptStyle
from litgpt.data import DataModule
from litgpt.data.base import get_sft_collate_fn, SFTDataset
from litgpt.data.json_data import find_split, get_splits, load_split
from litgpt.tokenizer import Tokenizer

from torch.utils.data import BatchSampler, ConcatDataset, DataLoader, IterableDataset

from torch.utils.data.dataloader import (
    _BaseDataLoaderIter,
    _InfiniteConstantSampler,
    _MultiProcessingDataLoaderIter,
)
from typing_extensions import override, Self, TypedDict


# hacky
MAX_ITERS = None


@dataclass
class MixedDatasetClassic(DataModule):
    """
    A dataset that blends together unstructured text (the usual pretraining data) and structured text (SFT data). Can have different proportions of each that evolve over time.
    """

    # A path to a directory containing both pretraining data (files in txt form) as well as SFT data (json files). Should contain two subdirectories "texts" and "sft".
    pretraining_data_path: str = "data/"
    sft_data_path: str = "sft/"
    pretrain_data_type: str = "streaming"  # "streaming" or "txt"
    sft_val_split_fraction: float = 0.05
    max_seq_length: int = field(init=False, repr=False, default=2048)
    tokenizer: Optional[Tokenizer] = field(default=None, init=False, repr=False)
    prompt_style: Union[str, PromptStyle] = "alpaca"
    mask_prompt: bool = False
    ignore_index: int = -100
    seed: int = 42
    num_workers: int = 4
    cycle_mode: Literal[
        "max_size_cycle", "min_size_cycle", "max_size", "sequential", "max_size_spread"
    ] = "max_size_spread"
    fast_dev_run: bool = False
    # number of times to repeat the sft datasets
    num_repeats: int = 1
    # the max iters through the pretraining dataset. Used to space out the sft datasets accordingly
    max_iters: int = 5000

    def __post_init__(self):
        self.lm_train_path = os.path.join(self.pretraining_data_path, "train")
        self.lm_val_path = os.path.join(self.pretraining_data_path, "val")

        # TODO: for now assume there's only one jsonl/json file in each sft dir. We should iterate over multiple though to make it easier to add datasets
        self.out_path_train_lm = self.lm_train_path
        self.out_path_val_lm = self.lm_val_path

    def setup(self):
        sft_train_data, sft_val_data = get_splits(
            Path(self.sft_data_path),
            self.sft_val_split_fraction,
            num_repeats=self.num_repeats,
        )

        self.sft_train_dataset = SFTDataset(
            data=sft_train_data,
            tokenizer=self.tokenizer,
            prompt_style=self.prompt_style,
            max_seq_length=self.max_seq_length_sft,
            mask_prompt=self.mask_prompt,
            ignore_index=self.ignore_index,
        )
        self.sft_test_dataset = SFTDataset(
            data=sft_val_data,
            tokenizer=self.tokenizer,
            prompt_style=self.prompt_style,
            max_seq_length=self.max_seq_length_sft,
            mask_prompt=self.mask_prompt,
            ignore_index=self.ignore_index,
        )

    def connect(
        self,
        tokenizer: Optional[Tokenizer] = None,
        batch_size: int = 1,
        max_seq_length: int = -1,
        max_iters: int = 5000,
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

            if Path(self.lm_train_path).is_dir() and Path(self.lm_val_path).is_dir():
                print(
                    f"Found FineWeb train and val dir: {self.pretraining_data_path}. Skipping preprocessing."
                )
                return

            dataset = load_dataset(
                "HuggingFaceFW/fineweb-edu",
                num_proc=os.cpu_count() // 8,
                name=self.data_split,  # 149M examples
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
        from litgpt.data import DataModule, get_sft_collate_fn, SFTDataset

        self.lm_train_dataset = StreamingDataset(
            input_dir=self.out_path_train_lm,
            item_loader=TokensLoader(block_size=self.max_seq_length_lm),
            shuffle=True,
            drop_last=True,
        )

        lm_train_dataloader = StreamingDataLoader(
            self.lm_train_dataset,
            batch_size=self.batch_size,
            pin_memory=True,
            drop_last=True,
        )
        print("BATCH SIZE:", self.batch_size)

        if self.cycle_mode == "max_size_spread":
            # sft_batch_size = 1  # to better control sampling rate
            sft_batch_size = self.batch_size  # TODO: should I change this back?
        else:
            sft_batch_size = self.batch_size

        sft_train_dataloader = DataLoader(
            self.sft_train_dataset,
            batch_size=sft_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=get_sft_collate_fn(
                max_seq_length=self.max_seq_length_sft, ignore_index=self.ignore_index
            ),
        )

        return CombinedLoader(
            {"lm": lm_train_dataloader, "sft": sft_train_dataloader},
            self.cycle_mode,
            max_iters=self.max_iters,
        )

    # TODO - should we return two separate validation sets and report metrics on both separately -- probably
    def val_dataloader(self) -> DataLoader:
        from litgpt.data import DataModule, get_sft_collate_fn, SFTDataset

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
        sft_test_dataloader = DataLoader(
            self.sft_test_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=get_sft_collate_fn(
                max_seq_length=self.max_seq_length_sft, ignore_index=self.ignore_index
            ),
        )

        return CombinedLoader(
            {"lm": lm_test_dataloader, "sft": sft_test_dataloader}, "max_size"
        )


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
    max_steps = (
        5000  # TODO: the global variable hack doesn't work, fix the pass through later.
    )

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
        max_iters=5000,
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
        breakpoint()

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
