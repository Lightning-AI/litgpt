import os
import urllib


def download_original(wd: str):
    filepath = os.path.join(wd, "original_model.py")
    if not os.path.isfile(filepath):
        print(f"Downloading original implementation to {filepath!r}")
        urllib.request.urlretrieve(
            url="https://gist.githubusercontent.com/lantiga/fd36849fb1c498da949a0af635318a7b/raw/c4509e48a53ebb6a195a6f073b5267a69e47b45a/llama_model.py",
            filename="original_model.py",
        )
        print("Done")
    else:
        print("Original implementation found. Skipping download.")
