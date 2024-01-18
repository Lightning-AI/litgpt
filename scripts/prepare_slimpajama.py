# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import json
import os
import sys
import time
from pathlib import Path

import zstandard as zstd
from lightning.data.streaming import DataChunkRecipe, DataProcessor

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_gpt import Tokenizer


class SlimPajamaDataRecipe(DataChunkRecipe):
    def __init__(self, tokenizer: Tokenizer, chunk_size: int):
        super().__init__(chunk_size)
        self.tokenizer = tokenizer

    def prepare_structure(self, input_dir):
        files = Path(input_dir).rglob("*.zst")
        return [str(file) for file in files]

    def prepare_item(self, filepath):
        with zstd.open(open(filepath, "rb"), "rt", encoding="utf-8") as f:
            for row in f:
                text = json.loads(row)["text"]
                if json.loads(row)["meta"]["redpajama_set_name"] == "RedPajamaGithub":
                    continue  # exclude the GitHub data since it overlaps with starcoder
                text_ids = self.tokenizer.encode(text, bos=False, eos=True)
                yield text_ids


def prepare(
    input_dir: Path = Path("data/SlimPajama-627B/train"),
    output_dir: Path = Path("data/slimpajama/train"),
    tokenizer_path: Path = Path("checkpoints/Llama-2-7b-hf/"),
    chunk_size: int = (2049 * 16384),
    fast_dev_run: bool = False,
) -> None:
    tokenizer = Tokenizer(tokenizer_path)
    data_recipe = SlimPajamaDataRecipe(tokenizer=tokenizer, chunk_size=chunk_size)
    data_processor = DataProcessor(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        fast_dev_run=fast_dev_run,
        num_workers=os.cpu_count(),
        num_downloaders=1,
    )

    start_time = time.time()
    data_processor.run(data_recipe)
    elapsed_time = time.time() - start_time
    print(f"Time taken: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(prepare)
