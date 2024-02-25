# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

# saves the openwebtext dataset to a binary file for training. following was helpful:
# https://github.com/HazyResearch/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py
import os
import sys
from pathlib import Path
from typing import Union

import numpy as np
from tqdm import tqdm

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_gpt import Tokenizer


def prepare(
    destination_path: Path = Path("data/openwebtext"),
    checkpoint_dir: Path = Path("checkpoints/stabilityai/stablelm-base-alpha-3b"),
    seed: int = 42,
    test_size: Union[float, int, None] = 0.0005,
) -> None:
    from datasets import load_dataset  # huggingface datasets

    destination_path.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(checkpoint_dir)

    # number of workers in .map() call
    # good number to use is ~order number of cpu cores // 2
    num_proc = os.cpu_count() // 2

    # number of workers in load_dataset() call
    # best number might be different from num_proc above as it also depends on HW speed.
    # it is better than 1 usually though
    num_proc_load_dataset = num_proc

    # takes 54GB in huggingface .cache dir, about 8M documents (8,013,769)
    dataset = load_dataset("openwebtext", num_proc=num_proc_load_dataset)

    # owt by default only contains the 'train' split, so create a test split
    split_dataset = dataset["train"].train_test_split(test_size=test_size, seed=seed, shuffle=True)
    split_dataset["val"] = split_dataset.pop("test")  # rename the test split to val

    def process(example):
        ids = tokenizer.encode(example["text"]).tolist()
        ids.append(tokenizer.eos_id)

        # ids = enc.encode_ordinary(example['text']) # encode_ordinary ignores any special tokens
        # ids.append(enc.eot_token) # add the end of text token, e.g. 50256 for gpt2 bpe
        # note: I think eot should be prepended not appended... hmm. it's called "eot" though...
        return {"ids": ids, "len": len(ids)}

    # tokenize the dataset
    tokenized = split_dataset.map(process, remove_columns=["text"], desc="tokenizing the splits", num_proc=num_proc)

    # concatenate all the ids in each dataset into one large file we can use for training
    for split, dset in tokenized.items():
        arr_len = np.sum(dset["len"], dtype=np.uint64)
        filename = destination_path / f"{split}.bin"
        dtype = np.uint16  # (can do since enc.max_token_value == 50256 is < 2**16)
        arr = np.memmap(str(filename), dtype=dtype, mode="w+", shape=(arr_len,))
        total_batches = 1024

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f"writing {filename}"):
            # Batch together samples for faster write
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format("numpy")
            arr_batch = np.concatenate(batch["ids"])
            # Write into mmap
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(prepare)
