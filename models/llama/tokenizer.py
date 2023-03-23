import torch
from sentencepiece import SentencePieceProcessor


class Tokenizer:
    """Tokenizer for LLaMA."""

    def __init__(self, model_path: str):
        self.processor = SentencePieceProcessor(model_file=model_path)
        self.bos_id = self.processor.bos_id()
        self.eos_id = self.processor.eos_id()
        self.pad_id = self.processor.pad_id()

    @property
    def vocab_size(self):
        return self.processor.vocab_size()

    def encode(self, string: str, bos: bool, eos: bool) -> torch.Tensor:
        tokens = self.processor.encode(string)
        if bos:
            tokens = [self.bos_id] + tokens
        if eos:
            tokens = tokens + [self.eos_id]
        return torch.tensor(tokens, dtype=torch.int)

    def decode(self, tokens: torch.Tensor) -> str:
        return self.processor.decode(tokens.tolist())
