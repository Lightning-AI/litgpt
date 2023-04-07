import os
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
