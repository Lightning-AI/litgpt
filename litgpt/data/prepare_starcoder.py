# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import os
import time
import traceback
from pathlib import Path

from lightning_utilities.core.imports import RequirementCache

from litgpt.tokenizer import Tokenizer
from litgpt.utils import CLI, extend_checkpoint_dir

_LITDATA_AVAILABLE = RequirementCache("litdata")
if _LITDATA_AVAILABLE:
    from litdata.processing.data_processor import DataChunkRecipe
else:
    DataChunkRecipe = object


class StarcoderDataRecipe(DataChunkRecipe):
    is_generator = True

    def __init__(self, tokenizer: Tokenizer, chunk_size: int):
        super().__init__(chunk_size)
        self.tokenizer = tokenizer

    def prepare_structure(self, input_dir):
        files = Path(input_dir).rglob("*.parquet")
        return [str(file) for file in files]

    def prepare_item(self, item_metadata):
        import pyarrow.parquet as pq

        filepath = item_metadata
        start = time.time()

        try:
            parquet_file = pq.ParquetFile(filepath)
            # reduce RAM usage
            for batch in parquet_file.iter_batches(batch_size=8192, columns=["content"]):
                for text in batch.to_pandas()["content"]:
                    yield self.tokenizer.encode(text, bos=False, eos=True)

        except Exception:
            print(traceback.format_exc())
            print(f"Error reading {filepath}")
            return

        parquet_file.close()
        end = time.time()
        print(f"Took {end - start:.2f} seconds total", filepath)


def prepare(
    input_dir: Path = Path("data/starcoderdata"),
    output_dir: Path = Path("data/starcoder"),
    tokenizer_path: Path = Path("checkpoints/Llama-2-7b-hf/"),
    chunk_size: int = (2049 * 8192),
    fast_dev_run: bool = False,
) -> None:
    from litdata.processing.data_processor import DataProcessor

    tokenizer_path = extend_checkpoint_dir(tokenizer_path)
    tokenizer = Tokenizer(tokenizer_path)
    data_recipe = StarcoderDataRecipe(tokenizer=tokenizer, chunk_size=chunk_size)
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
    CLI(prepare)
