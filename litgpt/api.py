# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
#
# This file implements the LitGPT Python API
import os
from pathlib import Path
from typing import Any, List, Literal, Optional, Union

import torch
import lightning as L


from litgpt.model import GPT  # needs to be imported before config
from litgpt.config import name_to_config, Config
from litgpt.tokenizer import Tokenizer
from litgpt.generate.base import generate as generate_fn
from litgpt.chat.base import generate as stream_generate_fn
from litgpt.prompts import load_prompt_style, has_prompt_style, PromptStyle
from litgpt.utils import (
    check_valid_checkpoint_dir,
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
        devices: Union[int, List[int]] = 1,
        checkpoint_dir: Path = None,
        fabric: L.Fabric = None
    ) -> None:
        self.model = model
        self.preprocessor = Preprocessor(tokenizer, device=fabric.device)
        self.devices = devices
        self.prompt_style = prompt_style
        self.checkpoint_dir = checkpoint_dir
        self.fabric = fabric

    """
    LLM model class for inference, pretraining, and finetuning.

    Example:
        from litgpt.api import LLM

        llm = LLM.load("microsoft/phi-2", accelerator="cuda", devices=1)
        text = llm.generate("What do Llamas eat?", top_k=1)
        print(text)
    """

    @classmethod
    def load(
        cls,
        model: str,
        accelerator: Literal["cpu", "cuda", "auto"] = "auto",
        devices: Union[int, List[int]] = 1,
        quantize: Optional[Literal["bnb.nf4", "bnb.nf4-dq", "bnb.fp4", "bnb.fp4-dq", "bnb.int8"]] = None,
        precision: Optional[Any] = None,
        init: Optional[Literal["pretrained", "random"]] = "pretrained",
        tokenizer_dir: Optional[Path] = None,
        access_token: Optional[str] = None
    ) -> "LLM":
        """
        Loads the LLM from a local directory or model hub.

        Arguments
            model: A local path to a directory containing the model weights or a valid model name.
               You can get a list of valid model names via the `litgpt download list` command line argument.
            accelerator: Which device type to load the model on ("cpu", "gpu", "mps", "cuda", or "auto")
            devices: The number of devices (1, 2, etc.) or device IDs (e.g., [0, 2] to use the first and third GPU).
            quantize: Whether to quantize the model and using which method:
                - bnb.nf4, bnb.nf4-dq, bnb.fp4, bnb.fp4-dq: 4-bit quantization from bitsandbytes
                - bnb.int8: 8-bit quantization from bitsandbytes
                for more details, see https://github.com/Lightning-AI/litgpt/blob/main/tutorials/quantize.md
            precision: Indicates the Fabric precision setting to use.
                For instance, "32-true", "16-mixed", "16-true", "bf16-mixed", "bf16-true".
                For more details, see https://lightning.ai/docs/fabric/stable/api/fabric_args.html#precision
            init: If "pretrained" (default), downloads the model from the HF Hub if a local model can't be found at the `model`
                directory name; otherwise loads the model from the local directory.
                If "random", initializes the `model` with random weights.
            access_token:
                Optional API token to access models with restrictions when using `init="pretrained"`.
            tokenizer_dir: An optional tokenizer directory if `model` is not a checkpoint directory, or if a user
                wants to use a different tokenizer instead.
        """
        allowed_accelerators = {"cpu", "gpu", "cuda", "mps", "auto"}
        if accelerator not in allowed_accelerators:
            raise ValueError(f"Invalid accelerator: {accelerator}. Must be one of {allowed_accelerators}.")

        if accelerator == "auto":
            if torch.cuda.is_available():
                accelerator = "cuda"
            elif torch.backends.mps.is_available():
                accelerator = "mps"
            else:
                accelerator = "cpu"

        num_devices = calculate_number_of_devices(devices)

        if num_devices > 1:
            raise NotImplementedError(
                "Support for multiple devices is currently not implemented, yet."
            )

        allowed_init = {"pretrained", "random"}

        if init == "pretrained":
            from litgpt.scripts.download import download_from_hub  # Moved here due to the circular import issue in LitGPT that we need to solve some time

            checkpoint_dir = extend_checkpoint_dir(Path(model))
            try:
                check_valid_checkpoint_dir(checkpoint_dir, verbose=False, raise_error=True)
            except FileNotFoundError:
                if not access_token:
                    access_token = os.getenv("HF_TOKEN")
                download_from_hub(repo_id=model, access_token=access_token)

            checkpoint_dir = Path("checkpoints") / model
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
        precision = precision or get_default_supported_precision(training=False)

        fabric = L.Fabric(
            accelerator=accelerator,
            devices=devices,
            precision=precision,
        )

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

        with fabric.init_module(empty_init=(num_devices > 1)):
            model = GPT(config)

        with fabric.init_tensor():
            model.set_kv_cache(batch_size=1)

        model.eval()
        model = fabric.setup_module(model)

        if checkpoint_dir is not None:
            checkpoint_path = checkpoint_dir / "lit_model.pth"
            load_checkpoint(fabric, model, checkpoint_path)
        return cls(
            model=model, tokenizer=tokenizer, devices=devices,
            prompt_style=prompt_style, checkpoint_dir=checkpoint_dir, fabric=fabric,
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
        stream: bool = False
    ) -> Union[str, torch.Tensor]:
        """
        Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.

        Arguments:
            model: The model to use.
            prompt: Tensor of shape (T) with indices of the prompt sequence.
            max_returned_tokens: The maximum number of tokens to return (given plus generated).
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
            eos_id: If specified, stop generating any more token once the <eos> token is triggered.
            return_as_token_ids: If true. returns the generated tokens as IDs rather then detokenized text.

        """
        prompt = self.prompt_style.apply(prompt)
        input_ids = self.preprocessor.tokenizer.encode(prompt)
        prompt_length = input_ids.size(0)
        max_returned_tokens = prompt_length + max_new_tokens

        self.model.eval()

        if calculate_number_of_devices(self.devices) > 1:
            raise NotImplementedError(
                "Support for multiple devices is currently not implemented for `generate`"
            )

        def iterator():
            outputs = stream_generate_fn(
                model=self.model,
                prompt=input_ids.to(self.fabric.device),
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
                prompt=input_ids.to(self.fabric.device),
                max_returned_tokens=max_returned_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                eos_id=self.preprocessor.tokenizer.eos_id,
                include_prompt=False,
            )

        for block in self.model.transformer.h:
            block.attn.kv_cache.reset_parameters()

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
