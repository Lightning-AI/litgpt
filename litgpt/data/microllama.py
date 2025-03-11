# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
from dataclasses import dataclass
from pathlib import Path
from typing import Union

from litgpt.data import TinyLlama


@dataclass
class MicroLlama(TinyLlama):
    """The MicroLlama data module is composed of only SlimPajama data."""

    def __init__(self, data_path: Union[str, Path] = Path("data/"), seed: int = 42, num_workers: int = 8):
        super().__init__(data_path=data_path, seed=seed, num_workers=num_workers, use_starcoder=False)
