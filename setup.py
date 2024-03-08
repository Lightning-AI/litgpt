# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import os

from setuptools import find_packages, setup

_PATH_ROOT = os.path.dirname(__file__)

with open(os.path.join(_PATH_ROOT, "README.md"), encoding="utf-8") as fo:
    readme = fo.read()

setup(
    name="litgpt",
    version="0.1.0",
    description="Open source large language model implementation",
    author="Lightning AI",
    url="https://github.com/lightning-AI/litgpt",
    install_requires=[
        "torch>=2.2.0",
        "lightning @ git+https://github.com/Lightning-AI/lightning@f23b3b1e7fdab1d325f79f69a28706d33144f27e",
    ],
    extras_require={
        "all": [
            "jsonargparse[signatures]",  # CLI
            "bitsandbytes==0.41.0",  # quantization
            "scipy",  # required by bitsandbytes
            "sentencepiece",  # llama-based models
            "tokenizers",  # pythia, falcon, redpajama
            "datasets",  # eval
            "requests",  # litgpt.data
            "litdata",  # litgpt.data
            "zstandard",  # litgpt.data.prepare_slimpajama.py
            "pandas",  # litgpt.data.prepare_starcoder.py
            "pyarrow",  # litgpt.data.prepare_starcoder.py
            "tensorboard",  # litgpt.pretrain
            "torchmetrics",  # litgpt.pretrain
            # eval
            "git+https://github.com/EleutherAI/lm-evaluation-harness.git@115206dc89dad67b8beaa90051fb52db77f0a529",
        ],
        "test": ["pytest", "pytest-rerunfailures", "pytest-timeout", "transformers>=4.38.0", "einops", "protobuf"],
    },
    packages=find_packages(),
    long_description=readme,
    long_description_content_type="text/markdown",
)
