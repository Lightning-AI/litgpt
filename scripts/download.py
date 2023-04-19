import os
from typing import Optional
from urllib.request import urlretrieve

files = {
    "original_model.py": "https://gist.githubusercontent.com/lantiga/fd36849fb1c498da949a0af635318a7b/raw/7dd20f51c2a1ff2886387f0e25c1750a485a08e1/llama_model.py",
    "original_adapter.py": "https://gist.githubusercontent.com/awaelchli/546f33fcdb84cc9f1b661ca1ca18418d/raw/e81d8f35fb1fec53af1099349b0c455fc8c9fb01/original_adapter.py",
}


def download_original(wd: str) -> None:
    for file, url in files.items():
        filepath = os.path.join(wd, file)
        if not os.path.isfile(filepath):
            print(f"Downloading original implementation to {filepath!r}")
            urlretrieve(url=url, filename=file)
            print("Done")
        else:
            print("Original implementation found. Skipping download.")


def download_from_hub(repo_id: Optional[str] = None, local_dir: str = "checkpoints/hf-llama/7B") -> None:
    if repo_id is None:
        raise ValueError("Please pass `--repo_id=...`. You can try googling 'huggingface hub llama' for options.")

    from huggingface_hub import snapshot_download

    snapshot_download(repo_id, local_dir=local_dir)


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(download_from_hub)
