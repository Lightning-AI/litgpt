from base import generate, prompt_config
from lit_parrot.utils import EmptyInitOnDevice, lazy_load, check_valid_checkpoint_dir
from lit_parrot import Parrot, Tokenizer, Config
import json
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Optional, Tuple, List, Literal

import lightning as L
import torch

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))


def main(
    *,
    top_k: int = 200,
    temperature: float = 0.8,
    checkpoint_dir: Path = Path(
        f"checkpoints/stabilityai/stablelm-tuned-alpha-3b"),
    quantize: Literal["llm.int8", "gptq.int4"] = None,
    adapter_path: Path = Path(
        "out/adapter/alpaca/lit_model_adapter_finetuned.pth"),
) -> None:
    """Starts a conversation with a tuned Parrot model.

    Args:
        top_k: The number of top most probable tokens to consider in the sampling process.
        temperature: A value controlling the randomness of the sampling process. Higher values result in more random
            samples.
        checkpoint_dir: The checkpoint directory to load.
        quantize: Whether to quantize the model and using which method:
            ``"llm.int8"``: LLM.int8() mode,
            ``"gptq.int4"``: GPTQ 4-bit mode.
    """
    check_valid_checkpoint_dir(checkpoint_dir)

    with open(checkpoint_dir / "lit_config.json") as fp:
        config = Config(**json.load(fp))

    fabric = L.Fabric(devices=1)
    dtype = torch.bfloat16 if fabric.device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32

    if quantize == "gptq.int4":
        model_file = "lit_model_gptq.4bit.pth"
        if not (checkpoint_dir / model_file).is_file():
            raise ValueError("Please run `python quantize/gptq.py` first")
    else:
        model_file = "lit_model.pth"
    checkpoint_path = checkpoint_dir / model_file
    print(
        f"Loading model {str(checkpoint_path)!r} with {config.__dict__}", file=sys.stderr)
    with EmptyInitOnDevice(device=fabric.device, dtype=dtype, quantization_mode=quantize):
        model = Parrot(config)
    with lazy_load(checkpoint_dir / "lit_model.pth") as pretrained_checkpoint, lazy_load(
        adapter_path
    ) as adapter_checkpoint:
        # 1. Load the pretrained weights
        model.load_state_dict(pretrained_checkpoint, strict=False)
        # 2. Load the fine-tuned adapter weights
        model.load_state_dict(adapter_checkpoint, strict=False)

    model.eval()
    model = fabric.setup_module(model)

    tokenizer = Tokenizer(checkpoint_dir / "tokenizer.json",
                          checkpoint_dir / "tokenizer_config.json")
    system_prompt, stop_tokens = prompt_config(checkpoint_dir, tokenizer)

    while True:
        try:
            prompt = input(">> Prompt: ")
        except KeyboardInterrupt:
            break
        if not prompt:
            break
        prompt = system_prompt.format(prompt=prompt)
        encoded_prompt = tokenizer.encode(prompt, device=fabric.device)
        max_returned_tokens = model.config.block_size
        y = generate(
            model,
            encoded_prompt,
            max_returned_tokens,
            max_seq_length=max_returned_tokens,
            temperature=temperature,
            top_k=top_k,
            stop_tokens=stop_tokens,
        )
        print(">> Reply: ", end="")
        try:
            tokens_generated = 0
            t0 = time.perf_counter()
            for token in y:
                print(tokenizer.decode(token), end="", flush=True)
                tokens_generated += 1
            t = time.perf_counter() - t0
            model.reset_cache()
            print(
                f"\nTime for inference: {t:.02f} sec total, {tokens_generated / t:.02f} tokens/sec", file=sys.stderr)
        except KeyboardInterrupt:
            # support stopping generation
            pass
        print()


if __name__ == "__main__":
    from jsonargparse import CLI

    torch.set_float32_matmul_precision("high")
    warnings.filterwarnings(
        # Triggered internally at ../aten/src/ATen/EmptyTensor.cpp:31
        "ignore",
        message="ComplexHalf support is experimental and many operators don't support it yet",
    )
    warnings.filterwarnings(
        # Triggered in bitsandbytes/autograd/_functions.py:298
        "ignore",
        message="MatMul8bitLt: inputs will be cast from torch.bfloat16 to float16 during quantization",
    )
    CLI(main)
