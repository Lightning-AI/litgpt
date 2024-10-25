import argparse
import os
import tempfile
from functools import partial
from pathlib import Path

from datasets import Dataset, load_dataset
from litdata import optimize
from litgpt import Tokenizer

tokenizer = Tokenizer("checkpoints/EleutherAI/pythia-1b")


def tokenize(data: Dataset, index: int):
    yield tokenizer.encode(data[index]["text"], eos=True)

def setup_directories(job_id: str):
    """Set up and clean temp and cache directories for a specific job."""
    # Create job-specific paths
    temp_dir = f"/data/users/nightingal3/tmp/tmp_job{job_id}"
    cache_dir = f"data/users/nightingal3/tmp/huggingface_cache_job{job_id}"
    
    # Clean up existing directories if they exist
    for dir_path in [temp_dir, cache_dir]:
        if os.path.exists(dir_path):
            try:
                shutil.rmtree(dir_path)
                print(f"Cleaned up existing directory: {dir_path}")
            except Exception as e:
                print(f"Error cleaning {dir_path}: {e}")
    
    # Create fresh directories
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    
    # Set environment variables
    os.environ["HF_HOME"] = cache_dir
    os.environ["TMPDIR"] = temp_dir
    os.environ["TEMP"] = temp_dir
    os.environ["TMP"] = temp_dir
    tempfile.tempdir = temp_dir
    
    print(f"Set up fresh directories for job {job_id}:")
    print(f"Temp dir: {temp_dir}")
    print(f"Cache dir: {cache_dir}")
    
    return cache_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=".")
    parser.add_argument("--data_split", type=str, default="CC-MAIN-2024-10")
    parser.add_argument("--job_id", type=str, required=True, help="Unique identifier for this job")
    args = parser.parse_args()

    cache_dir = setup_directories(args.job_id)

    #temp_dir = "/data/users/nightingal3/manifold/tmp"
    #os.makedirs(temp_dir, exist_ok=True)

    # Set all temp variables to be safe
    #os.environ["TMPDIR"] = temp_dir
    #os.environ["TEMP"] = temp_dir
    #os.environ["TMP"] = temp_dir

    # Also set Python's tempfile directory explicitly
    #tempfile.tempdir = temp_dir

    print(f"Temporary directory is: {tempfile.gettempdir()}")

    dataset = load_dataset(
        "HuggingFaceFW/fineweb",
        num_proc=(os.cpu_count() - 1),
        name=args.data_split,
        split="train",
        cache_dir=None,
        download_mode="force_redownload"
    )
    print("Total examples:", len(dataset))

    # Split the data in training and validation
    split_dataset = dataset.train_test_split(test_size=0.003, seed=42, shuffle=True)
    split_dataset["val"] = split_dataset.pop("test")  # rename the test split to val
    output_dir = Path(args.data_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    optimize(
        fn=partial(tokenize, split_dataset["train"]),
        inputs=list(range(len(split_dataset["train"]))),
        output_dir=f"{args.data_path}/train",
        num_workers=16,
        chunk_bytes="100MB",
    )
    optimize(
        fn=partial(tokenize, split_dataset["val"]),
        inputs=list(range(len(split_dataset["val"]))),
        output_dir=f"{args.data_path}/val",
        num_workers=16,
        chunk_bytes="100MB",
    )
