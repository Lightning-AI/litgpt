# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
#
# This file implements the LitGPT Python API
from pathlib import Path
import sys
from typing import Any, List, Literal, Optional, Union

import torch
import lightning as L
from lightning.fabric.plugins import BitsandbytesPrecision
from lightning.fabric.accelerators import CUDAAccelerator


from litgpt.model import GPT
from litgpt.config import name_to_config, Config
from litgpt.tokenizer import Tokenizer
from litgpt.generate.sequentially import sequential
from litgpt.generate.base import generate as generate_fn
from litgpt.chat.base import generate as stream_generate_fn
from litgpt.prompts import load_prompt_style, has_prompt_style, PromptStyle
from litgpt.utils import (
    auto_download_checkpoint,
    check_file_size_on_cpu_and_warn,
    extend_checkpoint_dir,
    get_default_supported_precision,
    load_checkpoint,
)


class LLM:
    def __init__(
        self,
        model: GPT,
        preprocessor: None,
        prompt_style: PromptStyle,
        devices: Union[int, List[int]] = None,
        checkpoint_dir: Path = None,
        fabric: L.Fabric = None,
        generate_strategy: Optional[Literal["sequential"]] = None,
        kvcache_initialized: bool = False,
        fixed_kv_cache_size: Union[int, Literal["max_model_supported"], None] = None
    ) -> None:
        self._model = model
        self.preprocessor = preprocessor
        self.devices = devices
        self.prompt_style = prompt_style
        self.checkpoint_dir = checkpoint_dir
        self.fabric = fabric
        self.generate_strategy = generate_strategy
        self.kvcache_initialized = kvcache_initialized
        self.fixed_kv_cache_size = fixed_kv_cache_size
        self.prev_generated_seq_length = 0

    """
    LLM model class for inference, pretraining, and finetuning.

    Example:
        from litgpt.api import LLM

        llm = LLM.load("microsoft/phi-2")
        text = llm.generate("What do Llamas eat?", top_k=1)
        print(text)
    """

    @property
    def model(self):
        if self._model is None:
            raise AttributeError("The model is not initialized yet; use the .distribute() method to initialize the model.")
        return self._model

    @model.setter
    def model(self, content):
        self._model = content

    @classmethod
    def load(
        cls,
        model: str,
        init: Optional[Literal["pretrained", "random"]] = "pretrained",
        tokenizer_dir: Optional[Path] = None,
        access_token: Optional[str] = None,
        distribute: Optional[Literal["auto"]] = "auto"
    ) -> "LLM":
        """
        Loads the LLM from a local directory or model hub.

        Arguments
            model: A local path to a directory containing the model weights or a valid model name.
               You can get a list of valid model names via the `litgpt download list` command line argument.
            init: If "pretrained" (default), downloads the model from the HF Hub if a local model can't be found at the `model`
                directory name; otherwise loads the model from the local directory.
                If "random", initializes the `model` with random weights.
            tokenizer_dir: An optional tokenizer directory if `model` is not a checkpoint directory, or if a user
                wants to use a different tokenizer instead.
            access_token: Optional API token to access models with restrictions when using `init="pretrained"`.
            distribute: If "auto" (default), initializes the model on a single GPU if available and otherwise on the CPU.
                To have more control over the model distribution strategy and utilize multiple GPUs, you can set
                `llm = LLM.load(..., distribute=None)` and call `llm.distribute(...)` manually.
        """

        allowed_init = {"pretrained", "random"}

        if init == "pretrained":
            checkpoint_dir = auto_download_checkpoint(model_name=model, access_token=access_token)
            config = Config.from_file(checkpoint_dir / "model_config.yaml")

        elif init == "random":
            checkpoint_dir = None
            try:
                config = Config.from_name(model)
            except ValueError:
                print(f"Model name {model} is not supported.\n")
                available_models = "\n".join(sorted(name_to_config))
                print(f"Available values:\n{available_models}")
                quit()

        else:
            raise ValueError(f"Invalid init option: {init}. Must be one of {allowed_init}")

        torch.set_float32_matmul_precision("high")

        if tokenizer_dir is not None:
            tokenizer_dir = extend_checkpoint_dir(Path(tokenizer_dir))
            tokenizer = Tokenizer(tokenizer_dir)
        elif checkpoint_dir is not None:
            tokenizer = Tokenizer(checkpoint_dir)
        else:
            raise ValueError("Provide a path to a tokenizer directory via the `tokenizer_dir` setting.")

        if checkpoint_dir is not None:
            prompt_style = (
                load_prompt_style(checkpoint_dir)
                if has_prompt_style(checkpoint_dir)
                else PromptStyle.from_config(config)
            )
        else:
            prompt_style = PromptStyle.from_config(config)

        if distribute == "auto":
            if torch.cuda.is_available():
                accelerator = "cuda"
            elif torch.backends.mps.is_available():
                accelerator = "mps"
            else:
                accelerator = "cpu"

            fabric = L.Fabric(
                accelerator=accelerator,
                devices=1,
                precision=get_default_supported_precision(training=False),
            )

            with fabric.init_module(empty_init=False):
                model = GPT(config)
            model.eval()
            preprocessor = Preprocessor(tokenizer, device=fabric.device)

            if checkpoint_dir is not None:
                checkpoint_path = checkpoint_dir / "lit_model.pth"
                check_file_size_on_cpu_and_warn(checkpoint_path, fabric.device)
                load_checkpoint(fabric, model, checkpoint_path)

            model = fabric.setup_module(model)

        else:
            preprocessor = Preprocessor(tokenizer, device="cuda" if torch.cuda.is_available() else "cpu")
            model = None
            fabric = None

        return cls(
            model=model, preprocessor=preprocessor, prompt_style=prompt_style,
            checkpoint_dir=checkpoint_dir, fabric=fabric, generate_strategy=None,
            kvcache_initialized=False, fixed_kv_cache_size=False
        )

    def distribute(self):
        """In Progress"""
        pass

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: float = 1.0,
        return_as_token_ids: bool = False,
        stream: bool = False
    ) -> Union[str, torch.Tensor]:
        """
        Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.

        Arguments:
            model: The model to use.
            prompt: The prompt string to use for generating the samples.
            max_new_tokens: The maximum number of new tokens to return.
            temperature: Scales the predicted logits by 1 / temperature.
            top_k: If specified, only sample among the tokens with the k highest probabilities.
            top_p: If specified, it represents the cumulative probability threshold to consider in the sampling process.
                In top-p sampling, the next token is sampled from the highest probability tokens
                whose cumulative probability exceeds the threshold `top_p`. When specified,
                it must be `0 <= top_p <= 1`. Here, `top_p=0` is equivalent
                to sampling the most probable token, while `top_p=1` samples from the whole distribution.
                It can be used in conjunction with `top_k` and `temperature` with the following order
                of application:

                1. `top_k` sampling
                2. `temperature` scaling
                3. `top_p` sampling

                For more details, see https://arxiv.org/abs/1904.09751
                or https://huyenchip.com/2024/01/16/sampling.html#top_p
            return_as_token_ids: If True, returns the token IDs as a torch.Tensor. Otherwise, returns the decoded text as a string.
            stream: If True, returns a generator that yields tokens as they are generated.
                At the moment, this setting is slower and may use more memory than the non-streaming version.
                We plan to resolve this in the future.
        """
        assert self.model is not None

        prompt = self.prompt_style.apply(prompt)
        input_ids = self.preprocessor.encode(prompt)
        prompt_length = input_ids.size(0)
        max_returned_tokens = prompt_length + max_new_tokens

        if not self.kvcache_initialized:
            self.model.set_kv_cache(batch_size=1, max_seq_length=max_returned_tokens, device=self.fabric.device)
            self.kvcache_initialized = True

        # Dynamically grow the kv cache size if necessary
        if self.fixed_kv_cache_size is None and self.prev_generated_seq_length < max_returned_tokens:
            self.model.clear_kv_cache()
            self.model.set_kv_cache(batch_size=1, max_seq_length=max_returned_tokens, device=self.fabric.device)

        else:
            for block in self.model.transformer.h:
                block.attn.kv_cache.reset_parameters()

        self.prev_generated_seq_length = max_returned_tokens
        self.model.eval()

        def iterator():
            outputs = stream_generate_fn(
                model=self.model,
                prompt=input_ids,
                max_returned_tokens=max_returned_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                stop_tokens=([self.preprocessor.tokenizer.eos_id],),
            )
            if return_as_token_ids:
                yield from outputs
            else:
                for output in outputs:
                    yield self.preprocessor.tokenizer.decode(output)
            return

        if stream:
            outputs = iterator()
        else:
            outputs = generate_fn(
                model=self.model,
                prompt=input_ids,
                max_returned_tokens=max_returned_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                eos_id=self.preprocessor.tokenizer.eos_id,
                include_prompt=False,
            )

        if stream:
            return outputs
        elif return_as_token_ids:
            return outputs
        else:
            return self.preprocessor.tokenizer.decode(outputs)


class Preprocessor:
    """
    Preprocesser class for tokenization and de-tokenization.
    """

    def __init__(self, tokenizer: Tokenizer, device: str = "cpu") -> None:
        self.tokenizer = tokenizer
        self.device = device

    def encode(self, text: str) -> torch.Tensor:
        return self.tokenizer.encode(text, device=self.device)

    def decode(self, token_ids: torch.Tensor) -> str:
        return self.tokenizer.decode(token_ids)


def calculate_number_of_devices(devices):
    """
    Utility function to calculate the number of devices.
    """
    num_devices = devices if isinstance(devices, int) else len(devices) if isinstance(devices, list) else 0
    return num_devices
