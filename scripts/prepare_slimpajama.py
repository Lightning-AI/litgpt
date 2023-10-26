import glob
import json
import os
import sys
import time
from multiprocessing import Process, cpu_count
from pathlib import Path
from typing import List

import numpy as np
import torch
import zstandard as zstd
from lightning.data import DatasetOptimizer, StreamingDataset
from lightning.data.streaming.item_loader import TokensLoader
from tqdm import tqdm

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_gpt import Tokenizer

    
class SlimPajamaDataProcessor:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def prepare_dataset_structure(self, root, filepaths):
        return filepaths

    def prepare_item(self, item_metadata):
        filepath = item_metadata
        with zstd.open(open(filepath, "rb"), "rt", encoding="utf-8") as f:
            for row in f:
                text = json.loads(row)["text"]
                if json.loads(row)["meta"]["redpajama_set_name"] == "RedPajamaGithub":
                    continue # we don't want to include the github data
                text_ids = self.tokenizer.encode(text, bos=False, eos=True)
                yield text_ids


def prepare(
    source_path: Path = Path("data/SlimPajama-627B/train"),
    tokenizer_path: Path = Path("checkpoints/Llama-2-7b-hf/"),
    name: str = "slimpajama/train",
    chunk_size: int = (2049 * 8192),
    fast_dev_run: bool = False,
) -> None:

    tokenizer = Tokenizer(tokenizer_path)
    optimizer = SlimPajamaDataProcessor(tokenizer=tokenizer)
    dataset_optimizer = DatasetOptimizer(
        name=name,
        src_dir=str(source_path),
        fast_dev_run=fast_dev_run,
        chunk_size=chunk_size,
        num_workers=os.cpu_count(),
        num_downloaders=1,
    )

    start_time = time.time()
    dataset_optimizer.run(optimizer)
    elapsed_time = time.time() - start_time
    print(f"Time taken: {elapsed_time:.2f} seconds")

    time.sleep(10)

    # Verify we can read the data
    dataset = StreamingDataset(name=name, version="latest", item_loader=TokensLoader(block_size=(2048 + 1)))
    print(f"Number of samples: {len(dataset)}")
    print(dataset[0])
    print(len(dataset[0]))
    print(type(dataset[0]))


if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(prepare)