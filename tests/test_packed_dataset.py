import os
from unittest.mock import MagicMock
import requests

from torch.utils.data import IterableDataset


def train_tokenizer(destination_path):
    destination_path.mkdir(parents=True, exist_ok=True)

    # download the tiny shakespeare dataset
    input_file_path = destination_path / "input.txt"
    if not input_file_path.exists():
        data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        with open(input_file_path, "w") as f:
            f.write(requests.get(data_url).text)

    from lit_llama import Tokenizer
    Tokenizer.train(
        input=input_file_path,
        destination=destination_path,
        vocab_size=100,
    )

    return destination_path / "tokenizer.model"


def test_packed_dataset(tmp_path):
    tokenizer_path = train_tokenizer(tmp_path)

    from lit_llama import Tokenizer
    tokenizer = Tokenizer(tokenizer_path)

    texts = [
      "The moment of truth is upon us.",
      "Time to open the fridge."
    ]

    from lit_llama.packed_dataset import PackedDatasetBuilder, PackedDataset, HDR_SIZE

    block_size = 10
    n_blocks = 2
    chunk_size = block_size * n_blocks

    builder = PackedDatasetBuilder(
        outdir=tmp_path,
        prefix="packed_dataset",
        chunk_size=chunk_size,
        sep_token=tokenizer.bos_id,
        dtype="auto",
        vocab_size=100,
    )

    text_ids = []

    for text in texts:
        text_ids = tokenizer.encode(text)
        assert text_ids[0] == tokenizer.bos_id
        builder.add_array(text_ids)

    filenames = builder.filenames

    assert len(filenames) == 2
    assert os.path.basename(filenames[0]) == "packed_dataset_0000000000.bin"
    assert os.path.basename(filenames[1]) == "packed_dataset_0000000001.bin"

    import numpy as np

    ex_tokenized = [
        tokenizer.encode(text).numpy().astype(builder.dtype)
        for text in texts
    ]
    ex_tokenized = np.concatenate(ex_tokenized)
    ex_tokenized = ex_tokenized[:2 * chunk_size]

    for filename, el in zip(filenames, np.array_split(ex_tokenized, 2)):
        mmap = np.memmap(filename, mode="r", order="C", offset=HDR_SIZE)
        count = len(mmap) // np.dtype(builder.dtype).itemsize
        arr = np.frombuffer(
            mmap, dtype=builder.dtype, count=count, offset=0
        )
        where_bos = np.where(arr == tokenizer.bos_id)
        # we expect two BOS tokens, one per file
        assert len(where_bos) == 1
        assert np.array_equal(arr, el)

    dataset = PackedDataset(filenames=filenames, n_chunks=2, block_size=block_size, shuffle=False)

    ex_split = np.array_split(ex_tokenized, ex_tokenized.shape[0] // block_size)

    for item, el in zip(dataset, ex_split):
        assert np.array_equal(item, el)

    dataset = PackedDataset(filenames=filenames, n_chunks=2, block_size=block_size, seed=12345)

    for i, item in enumerate(dataset):
        block_idxs = iter(dataset)._block_idxs
        assert np.array_equal(item, ex_split[block_idxs[i]])

    dataset = PackedDataset(filenames=filenames, n_chunks=2, block_size=block_size, seed=12345, wrap=True)

    for i, item in enumerate(dataset):
        if i > 24:
            break

    dataset = PackedDataset(filenames=filenames, n_chunks=1, block_size=block_size, seed=12345)

    for i, item in enumerate(dataset):
        block_idxs = iter(dataset)._block_idxs
        chunk_idx = i // n_blocks * n_blocks
        assert np.array_equal(item, ex_split[chunk_idx + block_idxs[i % n_blocks]])

    block_size_ = block_size // 2
    ex_split = np.array_split(ex_tokenized, ex_tokenized.shape[0] // block_size_)
    dataset = PackedDataset(filenames=filenames, n_chunks=2, block_size=block_size_, seed=12345)

    for i, item in enumerate(dataset):
        block_idxs = iter(dataset)._block_idxs
        assert np.array_equal(item, ex_split[block_idxs[i]])

    block_size_ = block_size // 3
    n_chunks = 2
    ex_chunks = np.split(ex_tokenized, n_chunks)
    n_splits = ex_tokenized.shape[0] // n_chunks // block_size_
    ex_splits = [np.split(el[:n_splits * block_size_], n_splits) for el in ex_chunks]
    ex_split = sum(ex_splits, [])

    dataset = PackedDataset(filenames=filenames, n_chunks=n_chunks, block_size=block_size_, seed=12345)

    for i, item in enumerate(dataset):
        block_idxs = iter(dataset)._block_idxs
        assert np.array_equal(item, ex_split[block_idxs[i]])


class SimpleDataset(IterableDataset):
    def __init__(self, start, end):
        super().__init__()
        self._start = start
        self._end = end

    def __iter__(self):
        return iter(range(self._start, self._end))
        

def test_combined_dataset(tmp_path):
    from lit_llama.packed_dataset import CombinedDataset

    dataset1 = SimpleDataset(0, 10)
    dataset2 = SimpleDataset(10, 20)
    dataset = CombinedDataset(datasets=[dataset1, dataset2], weights=[1.0, 0.0], seed=12345)

    res = [el for el in dataset]
    assert res == list(range(0, 10))

    dataset1 = SimpleDataset(0, 10)
    dataset2 = SimpleDataset(10, 20)
    dataset = CombinedDataset(datasets=[dataset1, dataset2], weights=[0.0, 1.0], seed=12345)

    res = [el for el in dataset]
    assert res == list(range(10, 20))

    dataset1 = SimpleDataset(0, 10)
    dataset2 = SimpleDataset(10, 20)
    dataset = CombinedDataset(datasets=[dataset1, dataset2], weights=[0.5, 0.5], seed=12345)

    res = [el for el in dataset]
    assert 9 in res or 19 in res
    if len(res) > 10:
        assert 0 in res and 10 in res


def test_sharded_packed_dataset(monkeypatch):
    import lit_llama.packed_dataset
    from lit_llama.packed_dataset import PackedDataset

    dataset_iterator_mock = MagicMock()
    monkeypatch.setattr(lit_llama.packed_dataset, "PackedDatasetIterator", dataset_iterator_mock)
    filenames = [str(i) for i in range(10)]

    # world_size = 1, rank = 0
    iter(PackedDataset(filenames=filenames, n_chunks=2, block_size=2))
    assert dataset_iterator_mock.call_args[1]["filenames"] == filenames
    dataset_iterator_mock.reset_mock()
    # world_size = 2, rank = 0
    iter(PackedDataset(filenames=filenames, n_chunks=2, block_size=2, num_processes=2, process_rank=0))
    assert dataset_iterator_mock.call_args[1]["filenames"] == ["0", "2", "4", "6", "8"]
    dataset_iterator_mock.reset_mock()
    # world_size = 2, rank = 1
    iter(PackedDataset(filenames=filenames, n_chunks=2, block_size=2, num_processes=2, process_rank=1))
    assert dataset_iterator_mock.call_args[1]["filenames"] == ["1", "3", "5", "7", "9"]
    dataset_iterator_mock.reset_mock()
    
    # world_size = 3, rank = 0 (dataset size not cleanly divisible by world size)
    iter(PackedDataset(filenames=filenames, n_chunks=2, block_size=2, num_processes=3, process_rank=0))
    assert dataset_iterator_mock.call_args[1]["filenames"] == ["0", "3", "6"]
    dataset_iterator_mock.reset_mock()
    # world_size = 3, rank = 1 (dataset size not cleanly divisible by world size)
    iter(PackedDataset(filenames=filenames, n_chunks=2, block_size=2, num_processes=3, process_rank=1))
    assert dataset_iterator_mock.call_args[1]["filenames"] == ["1", "4", "7"]
    dataset_iterator_mock.reset_mock()
    # world_size = 3, rank = 2 (dataset size not cleanly divisible by world size)
    iter(PackedDataset(filenames=filenames, n_chunks=2, block_size=2, num_processes=3, process_rank=2))
    assert dataset_iterator_mock.call_args[1]["filenames"] == ["2", "5", "8"]
