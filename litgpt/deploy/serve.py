from pathlib import Path
from typing import Dict, Any, Optional, Literal
from litgpt.utils import check_valid_checkpoint_dir

import lightning as L
from lightning.fabric.plugins import BitsandbytesPrecision
import torch
from litserve import LitAPI, LitServer

from litgpt.model import GPT
from litgpt.config import Config
from litgpt.tokenizer import Tokenizer
from litgpt.generate.base import generate
from litgpt.prompts import load_prompt_style, has_prompt_style, PromptStyle
from litgpt.utils import load_checkpoint, CLI, get_default_supported_precision


class SimpleLitAPI(LitAPI):
    def __init__(self,
                 checkpoint_dir: Path,
                 precision: Optional[str] = None,
                 quantize: Optional[Literal["bnb.nf4", "bnb.nf4-dq", "bnb.fp4", "bnb.fp4-dq", "bnb.int8"]] = None,
                 temperature: float = 0.8,
                 top_k: int = 200,
                 max_generated_tokens: int = 30) -> None:

        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self.precision = precision
        self.quantize = quantize
        self.temperature = temperature
        self.top_k = top_k
        self.max_generated_tokens = max_generated_tokens

    def setup(self, device: str) -> None:
        # Setup the model so it can be called in `predict`.
        config = Config.from_file(self.checkpoint_dir / "model_config.yaml")
        device = torch.device(device)
        torch.set_float32_matmul_precision("high")

        precision = self.precision or get_default_supported_precision(training=False)
        plugins = None
        if self.quantize is not None and self.quantize.startswith("bnb."):
            if "mixed" in self.precision:
                raise ValueError("Quantization and mixed precision is not supported.")
            dtype = {"16-true": torch.float16, "bf16-true": torch.bfloat16, "32-true": torch.float32}[precision]
            plugins = BitsandbytesPrecision(self.quantize[4:], dtype)
            precision = None
        fabric = L.Fabric(
            accelerator=device.type,
            devices=[device.index],
            precision=precision,
            plugins=plugins,
        )
        checkpoint_path = self.checkpoint_dir / "lit_model.pth"
        self.tokenizer = Tokenizer(self.checkpoint_dir)
        self.prompt_style = (
            load_prompt_style(self.checkpoint_dir)
            if has_prompt_style(self.checkpoint_dir)
            else PromptStyle.from_config(config)
        )
        with fabric.init_module(empty_init=True):
            model = GPT(config)
        with fabric.init_tensor():
            # enable the kv cache
            model.set_kv_cache(batch_size=1)
        model.eval()

        self.model = fabric.setup_module(model)
        load_checkpoint(fabric, self.model, checkpoint_path)
        self.device = fabric.device

    def decode_request(self, request: Dict[str, Any]) -> Any:
        # Convert the request payload to your model input.
        prompt = request["prompt"]
        prompt = self.prompt_style.apply(prompt)
        encoded = self.tokenizer.encode(prompt, device=self.device)
        return encoded

    def predict(self, inputs: torch.Tensor) -> Any:
        # Run the model on the input and return the output.
        prompt_length = inputs.size(0)
        max_returned_tokens = prompt_length + self.max_generated_tokens

        y = generate(
            self.model,
            inputs,
            max_returned_tokens,
            temperature=self.temperature,
            top_k=self.top_k,
            eos_id=self.tokenizer.eos_id
        )

        for block in self.model.transformer.h:
            block.attn.kv_cache.reset_parameters()
        return y

    def encode_response(self, output: torch.Tensor) -> Dict[str, Any]:
        # Convert the model output to a response payload.
        decoded_output = self.tokenizer.decode(output)
        return {"output": decoded_output}


def run_server(
    checkpoint_dir: Path = Path("checkpoints"),
    precision: Optional[str] = None,
    quantize: Optional[Literal["bnb.nf4", "bnb.nf4-dq", "bnb.fp4", "bnb.fp4-dq", "bnb.int8"]] = None,
    temperature: float = 0.8,
    top_k: int = 200,
    max_generated_tokens: int = 50,
    devices: int = 1,
    accelerator: str = "cuda",
    port: int = 8000
) -> None:
    """Serve a LitGPT model using LitServe

    Arguments:
        checkpoint_dir: The checkpoint directory to load the model from.
        precision: Optional precision setting to instantiate the model weights in. By default, this will
            automatically be inferred from the metadata in the given ``checkpoint_dir`` directory.
        quantize: Whether to quantize the model and using which method:
            - bnb.nf4, bnb.nf4-dq, bnb.fp4, bnb.fp4-dq: 4-bit quantization from bitsandbytes
            - bnb.int8: 8-bit quantization from bitsandbytes
            for more details, see https://github.com/Lightning-AI/litgpt/blob/main/tutorials/quantize.md
        temperature: Temperature setting for the text generation. Value above 1 increase randomness.
            Values below 1 decrease randomness.
        top_k: The size of the pool of potential next tokens. Values larger than 1 result in more novel
            generated text but can also lead to more incoherent texts.
        max_generated_tokens: How many new tokens, in addition to the prompt length, to generate.
        devices: How many devices/GPUs to use.
        accelerator: The type of accelerator to use. For example, "cuda" or "cpu".
        port: The network port number on which the model is configured to be served.
    """
    check_valid_checkpoint_dir(checkpoint_dir, model_filename="lit_model.pth")

    server = LitServer(
        SimpleLitAPI(
            checkpoint_dir=checkpoint_dir,
            precision=precision,
            quantize=quantize,
            temperature=temperature,
            top_k=top_k,
            max_generated_tokens=max_generated_tokens,
            ),
        accelerator=accelerator,
        devices=devices)

    server.run(port=port)


if __name__ == "__main__":
    CLI(run_server)
