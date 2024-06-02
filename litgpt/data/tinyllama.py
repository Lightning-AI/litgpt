# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
from dataclasses import dataclass
from pathlib import Path
from typing import Union

from litgpt.data import LlamaDataModule

@dataclass
class TinyLlama(LlamaDataModule):
    """The TinyLlama data module is composed of a mix of SlimPajama and Starcoder data."""

    def __init__(self, data_path: Union[str, Path] = Path("data/"), seed: int = 42, num_workers: int = 8):
        super().__init__(data_path=data_path, seed=seed, num_workers=num_workers, use_starcoder=True)