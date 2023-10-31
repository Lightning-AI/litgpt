import os
import sys
import time
from pathlib import Path
import traceback
from lightning.data import DataProcessor, DataChunkRecipe
import pyarrow.parquet as pq

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_gpt import Tokenizer


class StarCoderDataRecipe(DataChunkRecipe):
    def __init__(self, tokenizer: Tokenizer, chunk_size: int):
        super().__init__(chunk_size)
        self.tokenizer = tokenizer

    def prepare_structure(self, input_dir):
        filepaths = []
        for directory, _, filenames in os.walk(input_dir):
            filepaths.extend([
                os.path.join(directory, filename) for filename in filenames if p.endswith(".parquet")])
        return filepaths

    def prepare_item(self, filepath):
        try:
            parquet_file = pq.ParquetFile(filepath)

            # reduce RAM usage
            for batch in parquet_file.iter_batches(batch_size=8192, columns=["content"]):
                for text in batch.to_pandas()['content']:
                    yield self.tokenizer.encode(text)

            parquet_file.close()
        except Exception:
            print(traceback.format_exc())
            pass


def prepare(
    input_dir: Path = Path("data/starcoderdata"),
    tokenizer_path: Path = Path("checkpoints/Llama-2-7b-hf/"),
    name: str = "starcoder",
    chunk_size: int = 2049 * 8192,
    fast_dev_run: bool = False,
) -> None:

    tokenizer = Tokenizer(tokenizer_path)
    data_recipe = StarCoderDataRecipe(tokenizer=tokenizer, chunk_size=chunk_size)
    data_processor = DataProcessor(
        name=name,
        input_dir=str(input_dir),
        fast_dev_run=fast_dev_run,
        chunk_size=chunk_size,
        num_workers=48,
        num_downloaders=1,
    )

    start_time = time.time()
    data_processor.run(data_recipe)
    elapsed_time = time.time() - start_time
    print(f"Time taken: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(prepare)