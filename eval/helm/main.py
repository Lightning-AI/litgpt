# Credits: https://github.com/llm-efficiency-challenge/neurips_llm_efficiency_challenge
import json
import logging
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Literal, Optional

import lightning as L
import torch
import uvicorn
from fastapi import FastAPI
from lightning.fabric.strategies import FSDPStrategy
from pydantic import BaseModel

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_gpt import GPT, Config, Tokenizer
from lit_gpt.model import Block
from lit_gpt.utils import check_valid_checkpoint_dir, lazy_load, quantization

from generate import generate

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI()


class ProcessRequest(BaseModel):
    prompt: str
    num_samples: int = 1
    max_new_tokens: int = 50
    top_k: int = 200
    temperature: float = 0.8
    seed: Optional[int] = None


class Token(BaseModel):
    text: str
    logprob: float
    top_logprob: Dict[str, float]


class ProcessResponse(BaseModel):
    text: str
    tokens: List[Token]
    logprob: float
    request_time: float


class TokenizeRequest(BaseModel):
    text: str
    truncation: bool = True
    max_length: int = 2048


class TokenizeResponse(BaseModel):
    tokens: List[int]
    request_time: float


def main(
    checkpoint_dir: str = "",
    precision: str = "bf16-true",
    device="auto",
    devices: int = 1,
    strategy: str = "auto",
    quantize: Optional[
        Literal[
            "bnb.nf4", "bnb.nf4-dq", "bnb.fp4", "bnb.fp4-dq", "bnb.int8", "gptq.int4"
        ]
    ] = None,
    port=8080,
):
    if strategy == "fsdp":
        strategy = FSDPStrategy(auto_wrap_policy={Block}, cpu_offload=False)
    fabric = L.Fabric(
        devices=devices, accelerator=device, precision=precision, strategy=strategy
    )
    fabric.launch()

    checkpoint_dir = Path(checkpoint_dir)
    check_valid_checkpoint_dir(checkpoint_dir)

    with open(checkpoint_dir / "lit_config.json") as fp:
        config = Config(**json.load(fp))

    checkpoint_path = checkpoint_dir / "lit_model.pth"
    logger.info(f"Loading model {str(checkpoint_path)!r} with {config.__dict__}")
    with fabric.init_module(empty_init=True), quantization(quantize):
        model = GPT(config)

    with lazy_load(checkpoint_path) as checkpoint:
        model.load_state_dict(checkpoint, strict=quantize is None)

    model.eval()
    model = fabric.setup(model)

    tokenizer = Tokenizer(checkpoint_dir)

    @app.post("/process")
    async def process_request(input_data: ProcessRequest) -> ProcessResponse:
        if input_data.seed is not None:
            L.seed_everything(input_data.seed)
        logger.info("Using device: {}".format(fabric.device))
        encoded = tokenizer.encode(
            input_data.prompt, device=fabric.device
        )
        prompt_length = encoded.size(0)
        max_returned_tokens = prompt_length + input_data.max_new_tokens
        assert max_returned_tokens <= model.config.block_size, (
            max_returned_tokens,
            model.config.block_size,
        )  # maximum rope cache length

        t0 = time.perf_counter()
        tokens, logprobs, top_logprobs = generate(
            model,
            encoded,
            max_returned_tokens,
            max_seq_length=max_returned_tokens,
            temperature=input_data.temperature,
            top_k=input_data.top_k,
        )

        t = time.perf_counter() - t0

        model.reset_cache()
        output = tokenizer.decode(tokens)
        tokens_generated = tokens.size(0) - prompt_length
        logger.info(
            f"Time for inference: {t:.02f} sec total, {tokens_generated / t:.02f} tokens/sec"
        )

        logger.info(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")
        generated_tokens = []
        for t, lp, tlp in zip(tokens, logprobs, top_logprobs):
            idx, val = tlp
            tok_str = tokenizer.processor.decode([idx])
            token_tlp = {tok_str: val}
            generated_tokens.append(
                Token(text=tokenizer.decode(t), logprob=lp, top_logprob=token_tlp)
            )
        logprobs_sum = sum(logprobs)
        # Process the input data here
        return ProcessResponse(
            text=output, tokens=generated_tokens, logprob=logprobs_sum, request_time=t
        )

    @app.post("/tokenize")
    async def tokenize(input_data: TokenizeRequest) -> TokenizeResponse:
        logger.info("Using device: {}".format(fabric.device))
        t0 = time.perf_counter()
        encoded = tokenizer.encode(input_data.text, device=fabric.device)
        t = time.perf_counter() - t0
        tokens = encoded.tolist()
        return TokenizeResponse(tokens=tokens, request_time=t)

    uvicorn.run(app, port=port)


if __name__ == "__main__":
    from jsonargparse import CLI

    torch.set_float32_matmul_precision("high")
    warnings.filterwarnings(
        # Triggered internally at ../aten/src/ATen/EmptyTensor.cpp:31
        "ignore",
        message="ComplexHalf support is experimental and many operators don't support it yet",
    )
    CLI(main)
