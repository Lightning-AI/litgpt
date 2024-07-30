# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
from pathlib import Path
from pprint import pprint
from typing import Dict, Any, Optional

import lightning as L
from lightning_utilities.core.imports import RequirementCache
import torch

from litgpt.api import LLM

from litgpt.model import GPT  # needs to be imported before config
from litgpt.config import Config
from litgpt.tokenizer import Tokenizer
from litgpt.generate.base import generate as plain_generate
from litgpt.chat.base import generate as stream_generate
from litgpt.prompts import load_prompt_style, has_prompt_style, PromptStyle
from litgpt.utils import (
    auto_download_checkpoint,
    get_default_supported_precision,
    load_checkpoint
)


_LITSERVE_AVAILABLE = RequirementCache("litserve")
if _LITSERVE_AVAILABLE:
    from litserve import LitAPI, LitServer
else:
    LitAPI, LitServer = object, object


class BaseLitAPI(LitAPI):
    def __init__(
        self,
        checkpoint_dir: Path,
        precision: Optional[str] = None,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 1.0,
        max_new_tokens: int = 50
    ) -> None:

        if not _LITSERVE_AVAILABLE:
            raise ImportError(str(_LITSERVE_AVAILABLE))

        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self.precision = precision
        self.temperature = temperature
        self.top_k = top_k
        self.max_new_tokens = max_new_tokens
        self.top_p = top_p

    def setup(self, device: str) -> None:
        if ":" in device:
            accelerator, device = device.split(":")
            device = f"[{int(device)}]"
        else:
            accelerator = device
            device = 1

        print("Initializing model...")
        self.llm = LLM.load(
            self.checkpoint_dir,
            accelerator=accelerator,
            precision=self.precision
        )
        print("Model successfully initialized.")

    def decode_request(self, request: Dict[str, Any]) -> Any:
        # Convert the request payload to your model input.
        prompt = str(request["prompt"])
        return prompt


class SimpleLitAPI(BaseLitAPI):
    def __init__(self,
                 checkpoint_dir: Path,
                 precision: Optional[str] = None,
                 temperature: float = 0.8,
                 top_k: int = 50,
                 top_p: float = 1.0,
                 max_new_tokens: int = 50):
        super().__init__(checkpoint_dir, precision, temperature, top_k, top_p, max_new_tokens)   

    def setup(self, device: str):
        super().setup(device)

    def predict(self, inputs: str) -> Any:
        output = self.llm.generate(
            inputs,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            max_new_tokens=self.max_new_tokens
        )
        return output

    def encode_response(self, output: str) -> Dict[str, Any]:
        # Convert the model output to a response payload.
        return {"output": output}


class StreamLitAPI(BaseLitAPI):
    def __init__(self,
                 checkpoint_dir: Path,
                 precision: Optional[str] = None,
                 temperature: float = 0.8,
                 top_k: int = 50,
                 top_p: float = 1.0,
                 max_new_tokens: int = 50):
        super().__init__(checkpoint_dir, precision, temperature, top_k, top_p, max_new_tokens)   

    def setup(self, device: str):
        super().setup(device)

    def predict(self, inputs: torch.Tensor) -> Any:
        # Run the model on the input and return the output.
        yield from self.llm.generate(
            inputs,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            max_new_tokens=self.max_new_tokens,
            stream=True
        )

    def encode_response(self, output):
        for out in output:
            yield {"output": out}


def run_server(
    checkpoint_dir: Path,
    precision: Optional[str] = None,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 1.0,
    max_new_tokens: int = 50,
    devices: int = 1,
    accelerator: str = "auto",
    port: int = 8000,
    stream: bool = False,
    access_token: Optional[str] = None,
) -> None:
    """Serve a LitGPT model using LitServe.

    Evaluate a model with the LM Evaluation Harness.

    Arguments:
        checkpoint_dir: The checkpoint directory to load the model from.
        precision: Optional precision setting to instantiate the model weights in. By default, this will
            automatically be inferred from the metadata in the given ``checkpoint_dir`` directory.
        temperature: Temperature setting for the text generation. Value above 1 increase randomness.
            Values below 1 decrease randomness.
        top_k: The size of the pool of potential next tokens. Values larger than 1 result in more novel
            generated text but can also lead to more incoherent texts.
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
        max_new_tokens: The number of generation steps to take.
        devices: How many devices/GPUs to use.
        accelerator: The type of accelerator to use. For example, "auto", "cuda", "cpu", or "mps".
            The "auto" setting (default) chooses a GPU if available, and otherwise uses a CPU.
        port: The network port number on which the model is configured to be served.
        stream: Whether to stream the responses.
        access_token: Optional API token to access models with restrictions.
    """
    checkpoint_dir = auto_download_checkpoint(model_name=checkpoint_dir, access_token=access_token)
    pprint(locals())

    if not stream:
        server = LitServer(
            SimpleLitAPI(
                checkpoint_dir=checkpoint_dir,
                precision=precision,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                ),
            accelerator=accelerator,
            devices=devices
            )

    else:
        server = LitServer(
            StreamLitAPI(
                checkpoint_dir=checkpoint_dir,
                precision=precision,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                ),
            accelerator=accelerator,
            devices=devices,
            stream=True
            )

    server.run(port=port, generate_client_file=False)
