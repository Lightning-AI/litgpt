from setuptools import setup, find_packages

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
)
