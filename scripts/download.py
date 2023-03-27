import os
from urllib.request import urlretrieve


def download_original(wd: str) -> None:
    filepath = os.path.join(wd, "original_model.py")
    if not os.path.isfile(filepath):
        print(f"Downloading original implementation to {filepath!r}")
        urlretrieve(
            url="https://gist.githubusercontent.com/lantiga/fd36849fb1c498da949a0af635318a7b/raw/7dd20f51c2a1ff2886387f0e25c1750a485a08e1/llama_model.py",
            filename="original_model.py",
        )
        print("Done")
    else:
        print("Original implementation found. Skipping download.")
