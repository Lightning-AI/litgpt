# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
#
# This file implements the LitGPT Python API
import torch
import lightning as L

from litgpt.model import GPT  # needs to be imported before config
from litgpt.config import Config
from litgpt.prompts import load_prompt_style, has_prompt_style, PromptStyle
from litgpt.tokenizer import Tokenizer
from litgpt.utils import (
    extend_checkpoint_dir,
    get_default_supported_precision,
    load_checkpoint
)


class LLM:
    def __init__(self, model, tokenizer, device, prompt_style):
        self.model = model
        self.preprocessor = Preprocessor(tokenizer)

    @classmethod
    def load(cls, checkpoint_dir, device, quantize=None, precision=None):
        model, tokenizer, device, prompt_style = self._setup_model_and_tokenizer(checkpoint_dir, precision, device)
        return cls(model, tokenizer, device, prompt_style)

    """
    def save(self, checkpoint_dir):
        self.model.save(checkpoint_dir, format=format)
        self.preprocessor.tokenizer.save(checkpoint_dir)

    def generate(self, prompt, **kwargs):
        input_ids = self.preprocessor.tokenizer.encode(prompt)

        output_ids = self.model.generate(input_ids, **kwargs)

        return self.preprocessor.tokenizer.decode(output_ids)
    """

    def _setup_model_and_tokenizer(self, checkpoint_dir, precision, device):
        config = Config.from_file(checkpoint_dir / "model_config.yaml")
        device = torch.device(device)
        torch.set_float32_matmul_precision("high")

        self.precision = precision or get_default_supported_precision(training=False)

        fabric = L.Fabric(
            accelerator=device.type,
            devices=1 if device.type == "cpu" else [device.index],
            precision=precision,
        )
        checkpoint_path = checkpoint_dir / "lit_model.pth"
        tokenizer = Tokenizer(checkpoint_dir)

        prompt_style = (
            load_prompt_style(checkpoint_dir)
            if has_prompt_style(checkpoint_dir)
            else PromptStyle.from_config(config)
        )
        with fabric.init_module(empty_init=True):
            model = GPT(config)
        with fabric.init_tensor():
            # enable the kv cache
            model.set_kv_cache(batch_size=1)
        model.eval()

        model = fabric.setup_module(model)
        load_checkpoint(fabric, model, checkpoint_path)
        return model, tokenizer, fabric.device, prompt_style


class Preprocessor:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def encode(self, text):
        return self.tokenizer.encode(text)

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids)
