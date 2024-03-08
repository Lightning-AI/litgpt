[project]
name = "litgpt"
version = "0.1.0"
description = "Open source large language model implementation"
authors = [{ name = "Lightning AI" }]
readme = "README.md"
license = { file = "LICENSE" }
dependencies = {file = "requirements.txt"}
optional-dependencies = {
    "all": {file = "requirements-all.txt"},
    "test": [
        "pytest",
        "pytest-rerunfailures",
        "pytest-timeout",
        "transformers>=4.38.0",
        "einops",
        "protobuf",
    ]
}

[project.urls]
homepage = "https://github.com/lightning-AI/litgpt"

[build-system]
requires = ["setuptools", "wheel"]
build-backend = "setuptools.build_meta"

