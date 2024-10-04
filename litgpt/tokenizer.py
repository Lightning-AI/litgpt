# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import json
from pathlib import Path
from typing import Optional, Union, Iterable, Iterator

from litgpt.utils import fix_and_load_json
import torch


class Tokenizer:
    def __init__(self, checkpoint_dir: Union[Path, str]) -> None:
        checkpoint_dir = Path(checkpoint_dir)
        if not checkpoint_dir.exists():
            raise NotADirectoryError(f"The checkpoint directory does not exist: {str(checkpoint_dir)}")

        self.model_name = checkpoint_dir.stem
        self.use_bos = self.check_if_bos_token_used(checkpoint_dir)
        self.bos_id = None
        self.eos_id = None

        # some checkpoints have both files, `.json` takes precedence
        if (vocabulary_path := checkpoint_dir / "tokenizer.json").is_file():
            from tokenizers import Tokenizer as HFTokenizer

            self.processor = HFTokenizer.from_file(str(vocabulary_path))
            self.backend = "huggingface"

            if (special_tokens_path := checkpoint_dir / "tokenizer_config.json").is_file():
                with open(special_tokens_path, encoding="utf-8") as fp:
                    config = json.load(fp)
                bos_token = config.get("bos_token")
                eos_token = config.get("eos_token")
                if bos_token is not None and isinstance(bos_token, dict):
                    bos_token = bos_token.get("content")
                if eos_token is not None and isinstance(eos_token, dict):
                    eos_token = eos_token.get("content")
                self.bos_id = self.token_to_id(bos_token) if bos_token is not None else None
                self.eos_id = self.token_to_id(eos_token) if eos_token is not None else None
            if (special_tokens_path := checkpoint_dir / "generation_config.json").is_file():
                try:
                    with open(special_tokens_path, encoding="utf-8") as fp:
                        config = json.load(fp)
                except json.JSONDecodeError:  # Some files like the Llama 3.2 one have bugs
                    with open(special_tokens_path, encoding="utf-8") as fp:
                        json_string = fp.read()
                        config = fix_and_load_json(json_string)
                if self.bos_id is None:
                    self.bos_id = config.get("bos_token_id")
                if self.eos_id is None:
                    self.eos_id = config.get("eos_token_id")

        elif (vocabulary_path := checkpoint_dir / "tokenizer.model").is_file():
            from sentencepiece import SentencePieceProcessor

            self.processor = SentencePieceProcessor(model_file=str(vocabulary_path))
            self.backend = "sentencepiece"
            self.bos_id = self.processor.bos_id()
            self.eos_id = self.processor.eos_id()
        else:
            raise NotImplementedError

        # NOTE: A temporary fix until it's resolved on Tokenizers side.
        # LlaMA tokenizer strips leading spaces if to decode a single token at a time.
        # https://github.com/huggingface/transformers/issues/31643
        self.apply_decoding_fix = None
        if (config_path := checkpoint_dir / "tokenizer_config.json").is_file():
            with open(config_path, encoding="utf-8") as fp:
                self.apply_decoding_fix = "LlamaTokenizer" in json.load(fp)["tokenizer_class"]

    @property
    def vocab_size(self) -> int:
        if self.backend == "huggingface":
            return self.processor.get_vocab_size(with_added_tokens=False)
        if self.backend == "sentencepiece":
            return self.processor.vocab_size()
        raise RuntimeError

    def token_to_id(self, token: str) -> int:
        if self.backend == "huggingface":
            id_ = self.processor.token_to_id(token)
        elif self.backend == "sentencepiece":
            id_ = self.processor.piece_to_id(token)
        else:
            raise RuntimeError
        if id_ is None:
            raise ValueError(f"token {token!r} not found in the collection.")
        return id_

    def check_if_bos_token_used(self, checkpoint_dir: Path) -> bool:
        if not (tokenizer_config_path := checkpoint_dir / "tokenizer_config.json").is_file():
            return False
        with open(tokenizer_config_path, encoding="utf-8") as fp:
            config = json.load(fp)
        # for LlaMA-3 tokenizer there is no `add_bos_token` at all and `tokenizer_class` is only
        # `PreTrainedTokenizerFast`
        if checkpoint_dir.stem.startswith(("Meta-Llama-3", "Llama-3")):
            return True
        if "add_bos_token" in config:
            return config["add_bos_token"]
        # if `add_bos_token` isn't in the config file, but LLaMA tokenizer is used - return True.
        # ex: https://huggingface.co/stabilityai/StableBeluga2/blob/main/tokenizer_config.json#L2
        return config.get("tokenizer_class") == "LlamaTokenizer"

    def encode(
        self,
        string: str,
        device: Optional[torch.device] = None,
        bos: Optional[bool] = None,
        eos: bool = False,
        max_length: int = -1,
    ) -> torch.Tensor:
        if self.backend == "huggingface":
            tokens = self.processor.encode(string).ids
        elif self.backend == "sentencepiece":
            tokens = self.processor.encode(string)
        else:
            raise RuntimeError(f"`{self.backend}` is not supported.")
        if tokens is None:
            raise ValueError("`self.processor` returned tokens of None value.")

        if bos or (bos is None and self.use_bos):
            if self.bos_id is None:
                raise NotImplementedError("This tokenizer does not have a defined bos token.")
            if not tokens or tokens[0] != self.bos_id:
                tokens = [self.bos_id] + tokens
        # if the processor misbehaves and adds `bos` token no matter what
        elif tokens and tokens[0] == self.bos_id:
            tokens = tokens[1:]

        if eos and (not tokens or tokens[-1] != self.eos_id):
            tokens = tokens + [self.eos_id]

        if max_length > 0:
            tokens = tokens[:max_length]
        return torch.tensor(tokens, dtype=torch.int, device=device)

    def decode(self, tensor: torch.Tensor) -> str:
        tokens = [tensor.item()] if tensor.ndim == 0 else tensor.tolist()
        if len(tokens) == 1 and self.apply_decoding_fix:
            dummy_token_id = 33  # \x1e
            dummy_token = self.processor.decode([dummy_token_id])
            return self.processor.decode([dummy_token_id] + tokens)[len(dummy_token) :]
        return self.processor.decode(tokens)

    def decode_stream(self, token_stream: Iterable[torch.Tensor], device: Optional[torch.device] = None) -> Iterator[str]:
        if self.backend == "huggingface":
            try:
                for token in token_stream:
                    yield self.decode(token)
            except KeyboardInterrupt:
                return
        elif self.backend == "sentencepiece":
            # TODO: Is there a way to not have to do this?
            # This may actually affect our tokens per second.

            # sentencepiece does not support decoding token-by-token because it adds spaces based on the surrounding tokens
            # meaning that we need to decode everything each time
            so_far = torch.tensor([], dtype=torch.long, device=device)
            decoded_so_far = ""
            try:
                for token in token_stream:
                    so_far = so_far.to(device=token.device)
                    so_far = torch.cat((so_far, token.view(-1)))
                    decoded_new = self.decode(so_far)
                    yield decoded_new[len(decoded_so_far) :]
                    decoded_so_far = decoded_new
            except KeyboardInterrupt:
                return
        else:
            raise NotImplementedError(self.backend)
