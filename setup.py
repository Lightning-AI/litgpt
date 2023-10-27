import os

from setuptools import find_packages, setup

_PATH_ROOT = os.path.dirname(__file__)

with open(os.path.join(_PATH_ROOT, "README.md"), encoding="utf-8") as fo:
    readme = fo.read()

setup(
    name="lit-gpt",
    version="0.1.0",
    description="Open source large language model implementation",
    author="Lightning AI",
    url="https://github.com/lightning-AI/lit-gpt",
    install_requires=[
        "torch==2.2.0.dev20231026",
        "lightning @ git+https://github.com/Lightning-AI/lightning@af442055419e07405ee6a8e77c0bb08ad8fefd4f",
    ],
    packages=find_packages(),
    long_description=readme,
    long_description_content_type="text/markdown",
)
