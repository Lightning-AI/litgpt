import os
import torch
from sentencepiece import SentencePieceProcessor, SentencePieceTrainer


class Tokenizer:
    """Tokenizer for LLaMA."""

    def __init__(self, model_path: str) -> None:
        self.processor = SentencePieceProcessor(model_file=model_path)
        self.bos_id = self.processor.bos_id()
        self.eos_id = self.processor.eos_id()
        self.pad_id = self.processor.pad_id()

    @property
    def vocab_size(self) -> int:
        return self.processor.vocab_size()

    def encode(self, string: str, bos: bool = True, eos: bool = False) -> torch.Tensor:
        tokens = self.processor.encode(string)
        if bos:
            tokens = [self.bos_id] + tokens
        if eos:
            tokens = tokens + [self.eos_id]
        return torch.tensor(tokens, dtype=torch.int)

    def decode(self, tokens: torch.Tensor) -> str:
        return self.processor.decode(tokens.tolist())

    @staticmethod
    def train(input: str, destination: str) -> None:
        model_prefix = os.path.join(destination, "tokenizer")
        SentencePieceTrainer.Train(input=input, model_prefix=model_prefix)
