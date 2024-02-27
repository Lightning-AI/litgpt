# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
from abc import abstractmethod
from typing import Optional

from lightning import LightningDataModule
from lit_gpt import Tokenizer


class LitDataModule(LightningDataModule):
    """Base class for all data modules in Lit-GPT."""

    @abstractmethod
    def connect(
        self,
        tokenizer: Optional[Tokenizer] = None,
        batch_size: int = 1,
        max_seq_length: Optional[int] = None
    ) -> None:
        pass

    def setup(self, stage: str = "") -> None:
        # Stub is to redefine the default signature, because the concept of 'stage' does not exist in Lit-GPT
        pass
