# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

from pathlib import Path

from setuptools import find_packages, setup

root = Path(__file__).parent
readme = (root / "README").read_text()
requirements = (root / "requirements.txt").read_text().split("\n")
requirements_all = (root / "requirements-all.txt").read_text().split("\n")
requirements_all = [r for r in requirements_all if r and not r.strip().startswith("-r")]

setup(
    name="litgpt",
    version="0.1.0",
    description="Open source large language model implementation",
    author="Lightning AI",
    url="https://github.com/lightning-AI/litgpt",
    install_requires=requirements,
    extras_require={
        "all": requirements_all,
        "test": ["pytest", "pytest-rerunfailures", "pytest-timeout", "transformers>=4.38.0", "einops", "protobuf"],
    },
    packages=find_packages(),
    long_description=readme,
    long_description_content_type="text/markdown",
)
