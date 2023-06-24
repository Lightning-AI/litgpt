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
        bos_token = config.get("bos_token")
        self.bos_id = self.token_to_id(bos_token) if bos_token is not None else None
        self.eos_id = self.token_to_id(config["eos_token"])

    @property
    def vocab_size(self) -> int:
        return self.processor.get_vocab_size(with_added_tokens=False)

    def token_to_id(self, token: str) -> int:
        id_ = self.processor.token_to_id(token)
        if id_ is None:
            raise ValueError(f"token {token!r} not found in the collection.")
        return id_

    def encode(
        self,
        string: str,
        device: Optional[torch.device] = None,
        bos: bool = False,
        eos: bool = False,
        max_length: int = -1,
    ) -> torch.Tensor:
        tokens = self.processor.encode(string).ids
        if bos:
            bos_id = self.bos_id
            if bos_id is None:
                raise NotImplementedError("This tokenizer does not defined a bos token")
            tokens = [bos_id] + tokens
        if eos:
            tokens = tokens + [self.eos_id]
        if max_length > 0:
            tokens = tokens[:max_length]
        return torch.tensor(tokens, dtype=torch.int, device=device)

    def decode(self, tensor: torch.Tensor) -> str:
        tokens = [tensor.item()] if tensor.ndim == 0 else tensor.tolist()
        return self.processor.decode(tokens)
