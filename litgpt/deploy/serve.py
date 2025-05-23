# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
import json
import sys
from pathlib import Path
from pprint import pprint
from typing import Any, Dict, Literal, Optional

import torch
from lightning_utilities.core.imports import RequirementCache

from litgpt.api import LLM
from litgpt.utils import auto_download_checkpoint

_LITSERVE_AVAILABLE = RequirementCache("litserve")
_JINJA2_AVAILABLE = RequirementCache("jinja2")
if _LITSERVE_AVAILABLE:
    from litserve import LitAPI, LitServer
    from litserve.specs.openai import ChatCompletionRequest, OpenAISpec
else:
    LitAPI, LitServer = object, object


class BaseLitAPI(LitAPI):
    def __init__(
        self,
        checkpoint_dir: Path,
        quantize: Optional[Literal["bnb.nf4", "bnb.nf4-dq", "bnb.fp4", "bnb.fp4-dq", "bnb.int8"]] = None,
        precision: Optional[str] = None,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 1.0,
        max_new_tokens: int = 50,
        devices: int = 1,
    ) -> None:
        if not _LITSERVE_AVAILABLE:
            raise ImportError(str(_LITSERVE_AVAILABLE))

        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self.quantize = quantize
        self.precision = precision
        self.temperature = temperature
        self.top_k = top_k
        self.max_new_tokens = max_new_tokens
        self.top_p = top_p
        self.devices = devices

    def setup(self, device: str) -> None:
        if ":" in device:
            accelerator, device = device.split(":")
            device = f"[{int(device)}]"
        else:
            accelerator = device
            device = 1

        print("Initializing model...", file=sys.stderr)
        self.llm = LLM.load(model=self.checkpoint_dir, distribute=None)

        self.llm.distribute(
            devices=self.devices,
            accelerator=accelerator,
            quantize=self.quantize,
            precision=self.precision,
            generate_strategy="sequential" if self.devices is not None and self.devices > 1 else None,
        )
        print("Model successfully initialized.", file=sys.stderr)

    def decode_request(self, request: Dict[str, Any]) -> Any:
        # Convert the request payload to your model input.
        prompt = str(request["prompt"])
        return prompt


class SimpleLitAPI(BaseLitAPI):
    def __init__(
        self,
        checkpoint_dir: Path,
        quantize: Optional[str] = None,
        precision: Optional[str] = None,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 1.0,
        max_new_tokens: int = 50,
        devices: int = 1,
    ):
        super().__init__(checkpoint_dir, quantize, precision, temperature, top_k, top_p, max_new_tokens, devices)

    def setup(self, device: str):
        super().setup(device)

    def predict(self, inputs: str) -> Any:
        output = self.llm.generate(
            inputs, temperature=self.temperature, top_k=self.top_k, top_p=self.top_p, max_new_tokens=self.max_new_tokens
        )
        return output

    def encode_response(self, output: str) -> Dict[str, Any]:
        # Convert the model output to a response payload.
        return {"output": output}


class StreamLitAPI(BaseLitAPI):
    def __init__(
        self,
        checkpoint_dir: Path,
        quantize: Optional[str] = None,
        precision: Optional[str] = None,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 1.0,
        max_new_tokens: int = 50,
        devices: int = 1,
    ):
        super().__init__(checkpoint_dir, quantize, precision, temperature, top_k, top_p, max_new_tokens, devices)

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
            stream=True,
        )

    def encode_response(self, output):
        for out in output:
            yield {"output": out}


class OpenAISpecLitAPI(BaseLitAPI):
    def __init__(
        self,
        checkpoint_dir: Path,
        quantize: Optional[str] = None,
        precision: Optional[str] = None,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 1.0,
        max_new_tokens: int = 50,
        devices: int = 1,
    ):
        super().__init__(checkpoint_dir, quantize, precision, temperature, top_k, top_p, max_new_tokens, devices)

    def setup(self, device: str):
        super().setup(device)
        if not _JINJA2_AVAILABLE:
            raise ImportError(str(_JINJA2_AVAILABLE))
        from jinja2 import Template

        config_path = self.checkpoint_dir / "tokenizer_config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"Tokenizer config file not found at {config_path}")

        with open(config_path, encoding="utf-8") as fp:
            config = json.load(fp)
            chat_template = config.get("chat_template", None)
            if chat_template is None:
                print("The tokenizer config does not contain chat_template, falling back to a default.")
                chat_template = "{% for m in messages %}{{ m.role }}: {{ m.content }}\n{% endfor %}Assistant: "
            self.chat_template = chat_template

        self.template = Template(self.chat_template)

    def decode_request(self, request: "ChatCompletionRequest") -> Any:
        # Apply chat template to request messages
        return self.template.render(messages=request.messages)

    def predict(self, inputs: str, context: dict) -> Any:
        # Extract parameters from context with fallback to instance attributes
        temperature = context.get("temperature") or self.temperature
        top_p = context.get("top_p", self.top_p) or self.top_p
        max_new_tokens = context.get("max_completion_tokens") or self.max_new_tokens

        # Run the model on the input and return the output.
        yield from self.llm.generate(
            inputs, temperature=temperature, top_k=self.top_k, top_p=top_p, max_new_tokens=max_new_tokens, stream=True
        )


def run_server(
    checkpoint_dir: Path,
    quantize: Optional[Literal["bnb.nf4", "bnb.nf4-dq", "bnb.fp4", "bnb.fp4-dq", "bnb.int8"]] = None,
    precision: Optional[str] = None,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 1.0,
    max_new_tokens: int = 50,
    devices: int = 1,
    accelerator: str = "auto",
    port: int = 8000,
    stream: bool = False,
    openai_spec: bool = False,
    access_token: Optional[str] = None,
) -> None:
    """Serve a LitGPT model using LitServe.

    Evaluate a model with the LM Evaluation Harness.

    Arguments:
        checkpoint_dir: The checkpoint directory to load the model from.
        quantize: Whether to quantize the model and using which method:
            - bnb.nf4, bnb.nf4-dq, bnb.fp4, bnb.fp4-dq: 4-bit quantization from bitsandbytes
            - bnb.int8: 8-bit quantization from bitsandbytes
            for more details, see https://github.com/Lightning-AI/litgpt/blob/main/tutorials/quantize.md
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
        openai_spec: Whether to use the OpenAISpec.
        access_token: Optional API token to access models with restrictions.
    """
    checkpoint_dir = auto_download_checkpoint(model_name=checkpoint_dir, access_token=access_token)
    pprint(locals())

    api_class = OpenAISpecLitAPI if openai_spec else StreamLitAPI if stream else SimpleLitAPI
    server = LitServer(
        api_class(
            checkpoint_dir=checkpoint_dir,
            quantize=quantize,
            precision=precision,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            devices=devices,
        ),
        spec=OpenAISpec() if openai_spec else None,
        accelerator=accelerator,
        devices=1,
        stream=stream,
    )

    server.run(port=port, generate_client_file=False)
