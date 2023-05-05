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
        self.bos_id = self.processor.token_to_id(config["bos_token"])
        self.eos_id = self.processor.token_to_id(config["eos_token"])

    @property
    def vocab_size(self) -> int:
        return self.processor.get_vocab_size(with_added_tokens=False)

    def encode(self, string: str, device: Optional[torch.device] = None) -> torch.Tensor:
        tokens = self.processor.encode(string).ids
        return torch.tensor(tokens, dtype=torch.int, device=device)

    def decode(self, tensor: torch.Tensor) -> str:
        tokens = tensor.tolist()
        return self.processor.decode(tokens)
