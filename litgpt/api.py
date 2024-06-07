# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
#
# This file implements the LitGPT Python API
from pathlib import Path
from typing import Any, Optional, Tuple, Type, Union

import torch
import lightning as L

from litgpt.model import GPT  # needs to be imported before config
from litgpt.config import Config
from litgpt.generate.base import generate as generate_fn
from litgpt.prompts import load_prompt_style, has_prompt_style, PromptStyle
from litgpt.tokenizer import Tokenizer
from litgpt.utils import (
    extend_checkpoint_dir,
    get_default_supported_precision,
    load_checkpoint,
    parse_devices
)


class LLM:
    def __init__(
        self,
        model: GPT,
        tokenizer: Tokenizer,
        prompt_style: PromptStyle,
        device_type: str,
        devices: Union[int, str] = 1,
        checkpoint_dir: Path = None
    ) -> None:
        self.model = model
        self.preprocessor = Preprocessor(tokenizer)
        self.device_type = device_type
        self.devices = devices
        self.prompt_style = prompt_style
        self.checkpoint_dir = checkpoint_dir

    @classmethod
    def load(
        cls, checkpoint_dir: Path, device_type: str = "cuda", devices: int = 1,
        quantize: Optional[Any] = None, precision: Optional[Any] = None
    ) -> "LLM":

        if devices > 1:
            raise NotImplementedError(
                "Support for multiple devices is currently not implemented."
            )

        checkpoint_dir = extend_checkpoint_dir(Path(checkpoint_dir))
        model, tokenizer, device, devices, prompt_style = cls._setup_model_and_tokenizer(
            checkpoint_dir=checkpoint_dir,
            precision=precision,
            device_type=device_type,
            devices=devices
        )
        return cls(
            model=model, tokenizer=tokenizer, device_type=device_type, devices=devices,
            prompt_style=prompt_style, checkpoint_dir=checkpoint_dir,
        )

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: float = 1.0,
        eos_id: Optional[int] = None,
        return_as_token_ids: bool = False
    ) -> Tuple[str, torch.Tensor]:
        prompt = self.prompt_style.apply(prompt)
        input_ids = self.preprocessor.tokenizer.encode(prompt)
        prompt_length = input_ids.size(0)
        max_returned_tokens = prompt_length + max_new_tokens

        output_ids = generate_fn(
            model=self.model,
            prompt=input_ids.to(self.model.device),
            max_returned_tokens=max_returned_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_id=self.preprocessor.tokenizer.eos_id
        )

        if self.devices > 1:
            raise NotImplementedError(
                "Support for multiple devices is currently not implemented for `generate`."
            )
        for block in self.model.transformer.h:
            block.attn.kv_cache.reset_parameters()

        if return_as_token_ids:
            return output_ids
        else:
            return self.preprocessor.tokenizer.decode(output_ids)

    @classmethod
    def _setup_model_and_tokenizer(
        cls,
        checkpoint_dir: Path,
        precision: Optional[Any],
        device_type: str,
        devices: Union[int, str]
    ) -> Tuple[GPT, Tokenizer, torch.device, PromptStyle]:

        config = Config.from_file(checkpoint_dir / "model_config.yaml")
        torch.set_float32_matmul_precision("high")

        devices = parse_devices(devices)
        precision = precision or get_default_supported_precision(training=False)

        fabric = L.Fabric(
            accelerator=device_type,
            devices=devices,
            precision=precision,
        )
        checkpoint_path = checkpoint_dir / "lit_model.pth"
        tokenizer = Tokenizer(checkpoint_dir)

        prompt_style = (
            load_prompt_style(checkpoint_dir)
            if has_prompt_style(checkpoint_dir)
            else PromptStyle.from_config(config)
        )

        with fabric.init_module(empty_init=(devices > 1)):
            model = GPT(config)
        with fabric.init_tensor():
            # enable the kv cache
            model.set_kv_cache(batch_size=1)
        model.eval()
        model = fabric.setup_module(model)
        load_checkpoint(fabric, model, checkpoint_path)
        return model, tokenizer, fabric.device, devices, prompt_style


class Preprocessor:
    def __init__(self, tokenizer: Tokenizer) -> None:
        self.tokenizer: Tokenizer = tokenizer

    def encode(self, text: str) -> torch.Tensor:
        return self.tokenizer.encode(text, device=self.device)

    def decode(self, token_ids: torch.Tensor) -> str:
        return self.tokenizer.decode(token_ids)
