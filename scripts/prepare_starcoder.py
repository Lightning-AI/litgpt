import glob
import json
import os
import sys
import time
from multiprocessing import Process, cpu_count
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
import zstandard as zstd
from lightning.data import DatasetOptimizer, StreamingDataset
from lightning.data.streaming.item_loader import TokensLoader
from tqdm import tqdm

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_gpt import Tokenizer


class StarcoderDataProcessor:
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    def prepare_dataset_structure(self, root, filepaths):
        # TODO: should no longer be necessary to filter:
        return [p for p in filepaths if p.endswith(".parquet")]

    def prepare_item(self, item_metadata):
        filepath = item_metadata
        contents = pd.read_parquet(filepath, engine='pyarrow')['content']
        for text in contents:
            text_ids = self.tokenizer.encode(text)
            yield text_ids


def prepare(
    source_path: Path = Path("data/starcoderdata"),
    tokenizer_path: Path = Path("checkpoints/Llama-2-7b-hf/"),
    name: str = "starcoder",
    chunk_size: int = 2049 * 10000,
    fast_dev_run: bool = False,
) -> None:

    tokenizer = Tokenizer(tokenizer_path)
    optimizer = StarcoderDataProcessor(tokenizer=tokenizer)
    dataset_optimizer = DatasetOptimizer(
        name=name,
        src_dir=str(source_path),
        fast_dev_run=fast_dev_run,
        num_workers=os.cpu_count(),
        chunk_size=chunk_size,
    )

    start_time = time.time()
    dataset_optimizer.run(optimizer)
    elapsed_time = time.time() - start_time
    print(f"Time taken: {elapsed_time:.2f} seconds")

    # Verify we can read the data
    dataset = StreamingDataset(name="starcoder", version="latest", item_loader=TokensLoader(block_size=2048))
    print(f"Number of samples: {len(dataset)}")
    print(dataset[0])
    print(len(dataset[0]))
    print(type(dataset[0]))


if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(prepare)