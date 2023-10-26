import os
import sys
import time
from pathlib import Path

from lightning.data import DatasetOptimizer, StreamingDataset
from lightning.data.streaming.item_loader import TokensLoader
from tqdm import tqdm
import time
import pyarrow.parquet as pq
import traceback

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
        start = time.time()

        try:
            parquet_file = pq.ParquetFile(filepath)
            # reduce RAM usage
            for batch in parquet_file.iter_batches(batch_size=8192, columns=["content"]):
                for text in batch.to_pandas()['content']:
                    yield self.tokenizer.encode(text, bos=False, eos=True)
        except Exception:
            print(traceback.format_exc())
            print(f"Error reading {filepath}")
            return

        parquet_file.close()
        end = time.time()
        print(f"Took {end - start:.2f} seconds total", filepath)


def prepare(
    source_path: Path = Path("data/starcoderdata"),
    tokenizer_path: Path = Path("checkpoints/Llama-2-7b-hf/"),
    name: str = "starcoder",
    chunk_size: int = (2049 * 8192),
    fast_dev_run: bool = False,
) -> None:

    tokenizer = Tokenizer(tokenizer_path)
    optimizer = StarcoderDataProcessor(tokenizer=tokenizer)
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