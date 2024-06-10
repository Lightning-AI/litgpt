# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
#
# This file implements the LitGPT Python API
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import torch
import lightning as L
from lightning.fabric.utilities.device_parser import _normalize_parse_gpu_input_to_list

from litgpt.model import GPT  # needs to be imported before config
from litgpt.config import Config
from litgpt.tokenizer import Tokenizer
from litgpt.generate.base import generate as plain_generate
from litgpt.chat.base import generate as stream_generate
from litgpt.prompts import load_prompt_style, has_prompt_style, PromptStyle
from litgpt.utils import (
    extend_checkpoint_dir,
    get_default_supported_precision,
    load_checkpoint,
)


class LLM:
    def __init__(
        self,
        model: GPT,
        tokenizer: Tokenizer,
        prompt_style: PromptStyle,
        device_type: str,
        devices: Union[int, List[int]] = 1,
        precision: Optional[any] = None,
        checkpoint_dir: Path = None,
        fabric: L.Fabric = None
    ) -> None:
        self.model = model
        self.preprocessor = Preprocessor(tokenizer)
        self.device_type = device_type
        self.devices = devices
        self.prompt_style = prompt_style
        self.checkpoint_dir = checkpoint_dir
        self.fabric = fabric
        self.precision = precision

    @classmethod
    def load(
        cls,
        checkpoint_dir: Path,
        device_type: str = "cuda",
        devices: Union[int, List[int]] = 1,
        quantize: Optional[Any] = None,
        precision: Optional[Any] = None,
    ) -> "LLM":

        num_devices = devices if isinstance(devices, int) else len(devices) if isinstance(devices, list) else 0

        if num_devices > 1:
            raise NotImplementedError(
                "Support for multiple devices is currently not implemented for `load`"
            )

        # TODO: download model if not downloaded already

        checkpoint_dir = extend_checkpoint_dir(Path(checkpoint_dir))
        config = Config.from_file(checkpoint_dir / "model_config.yaml")
        torch.set_float32_matmul_precision("high")

        devices = _normalize_parse_gpu_input_to_list(devices, include_cuda=True, include_mps=True)
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

        with fabric.init_module(empty_init=(num_devices > 1)):
            model = GPT(config)
        with fabric.init_tensor():
            # enable the kv cache
            model.set_kv_cache(batch_size=1)
        model.eval()
        model = fabric.setup_module(model)
        load_checkpoint(fabric, model, checkpoint_path)
        return cls(
            model=model, tokenizer=tokenizer, device_type=device_type, devices=devices,
            prompt_style=prompt_style, checkpoint_dir=checkpoint_dir, fabric=fabric,
            precision=precision
        )

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: float = 1.0,
        eos_id: Optional[int] = None,
        return_as_token_ids: bool = False,
        device_type: Optional[str] = None,
        devices: Union[int, List[int]] = 1,
        precision: Optional[Any] = None
    ) -> Tuple[str, torch.Tensor]:
        prompt = self.prompt_style.apply(prompt)
        input_ids = self.preprocessor.tokenizer.encode(prompt)
        prompt_length = input_ids.size(0)
        max_returned_tokens = prompt_length + max_new_tokens

        self.move_to_device(device_type=device_type, devices=devices, precision=precision)
        self.model.eval()

        if (isinstance(self.devices, int) and self.devices > 1) or (isinstance(self.devices, list) and len(self.devices) > 1):
            raise NotImplementedError(
                "Support for multiple devices is currently not implemented for `load`"
            )

        # TODO: Add streaming later
        #if stream:
        #    generate_fn = stream_generate
        #else:
        #    generate_fn = plain_generate
        generate_fn = plain_generate

        output_ids = generate_fn(
            model=self.model,
            prompt=input_ids.to(self.fabric.device),
            max_returned_tokens=max_returned_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_id=self.preprocessor.tokenizer.eos_id
        )

        for block in self.model.transformer.h:
            block.attn.kv_cache.reset_parameters()

        if return_as_token_ids:
            return output_ids
        else:
            return self.preprocessor.tokenizer.decode(output_ids)

    def move_to_device(self, device_type=None, devices=None, precision=None):
        if device_type is not None:
            self.device_type = device_type
        if devices is not None:
            self.devices = _normalize_parse_gpu_input_to_list(devices, include_cuda=True, include_mps=True)
        if precision is not None:
            self.precision = precision or get_default_supported_precision(training=False)

        if self.devices is not None or self.device_type is not None:
            self.fabric = L.Fabric(
                accelerator=self.device_type,
                devices=self.devices,
                precision=self.precision,
            )

            self.model = self.model.to(self.fabric.device)
            
            # Code below may not be necessary;
            # but moving the model still leaves some GPU memory on the original
            # device unfreed for some reason
            #
            #for name, buffer in self.model.named_buffers():
            #    setattr(self.model, name, buffer.to(self.fabric.device))

            #for block in self.model.transformer.h:
            #    if hasattr(block.attn, "kv_cache"):
            #        block.attn.kv_cache = block.attn.kv_cache.to(self.fabric.device)

            self.model.mask_cache = self.model.mask_cache.to(self.fabric.device)
            torch.cuda.empty_cache()


class Preprocessor:
    def __init__(self, tokenizer: Tokenizer) -> None:
        self.tokenizer: Tokenizer = tokenizer

    def encode(self, text: str) -> torch.Tensor:
        return self.tokenizer.encode(text, device=self.device)

    def decode(self, token_ids: torch.Tensor) -> str:
        return self.tokenizer.decode(token_ids)
