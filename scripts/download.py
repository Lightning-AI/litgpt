import os
import urllib


def download_original(wd: str):
    filepath = os.path.join(wd, "original_model.py")
    if not os.path.isfile(filepath):
        print(f"Downloading original implementation to {filepath!r}")
        urllib.request.urlretrieve(
            url="https://gist.githubusercontent.com/lantiga/fd36849fb1c498da949a0af635318a7b/raw/fe9561d1abd8d2c61c82dd62155fe98d3ac74c43/llama_model.py",
            filename="original_model.py",
        )
        print("Done")
    else:
        print("Original implementation found. Skipping download.")
