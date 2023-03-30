import os

from setuptools import setup, find_packages


_PATH_ROOT = os.path.dirname(__file__)

with open(os.path.join(_PATH_ROOT, "README.md"), encoding="utf-8") as fo:
    readme = fo.read()

setup(
    name='lit-llama',
    version='0.1.0',
    description='Implementation of the LLaMA language model',
    author='Lightning AI',
    url='https://github.com/lightning-AI/lit-llama',
    install_requires=[
        "torch>=2.0.0",
        "lightning>=2.0.0",
        "sentencepiece",
        "bitsandbytes",
    ],
    packages=find_packages(),
    long_description=readme,
    long_description_content_type="text/markdown",
)
