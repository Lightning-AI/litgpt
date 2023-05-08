import json
from pathlib import Path
from typing import Optional

import torch
from tokenizers import Tokenizer as HFTokenizer


class Tokenizer:
    def __init__(self, vocabulary_path: Path, config_path: Path) -> None:
        # https://github.com/Stability-AI/StableLM/blob/e60081/configs/stablelm-base-alpha-3b.yaml#L108
        self.processor = HFTokenizer.from_file(str(vocabulary_path))
        with open(config_path) as fp:
            config = json.load(fp)
        self.bos_id = self.token_to_id(config["bos_token"])
        self.eos_id = self.token_to_id(config["eos_token"])

    @property
    def vocab_size(self) -> int:
        return self.processor.get_vocab_size(with_added_tokens=False)

    def token_to_id(self, token: str) -> int:
        id_ = self.processor.token_to_id(token)
        if id_ is None:
            raise ValueError(f"token {token!r} not found in the collection.")
        return id_

    def encode(self, string: str, device: Optional[torch.device] = None, bos: bool = True,
        eos: bool = False,
        max_length: int = -1,
        pad: bool = False) -> torch.Tensor:

        tokens = self.processor.encode(string).ids
        if bos:
            tokens = [self.bos_id] + tokens
        if eos:
            tokens = tokens + [self.eos_id]
        if max_length > 0:
            tokens = tokens[:max_length]
        if pad and len(tokens) < max_length:
            tokens += [self.pad_id] * (max_length - len(tokens))

        return torch.tensor(tokens, dtype=torch.int, device=device)

    def decode(self, tensor: torch.Tensor) -> str:
        tokens = [tensor.item()] if tensor.ndim == 0 else tensor.tolist()
        return self.processor.decode(tokens)
