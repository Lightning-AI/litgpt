import json
import sys
import time
import warnings
from pathlib import Path
from typing import Optional

import lightning as L
import torch

from generate import generate
from lit_parrot import Tokenizer
from lit_parrot.adapter import Parrot, Config
from lit_parrot.utils import EmptyInitOnDevice, lazy_load, check_valid_checkpoint_dir
from scripts.prepare_alpaca import generate_prompt


def main(
    prompt: str = "What food do lamas eat?",
    input: str = "",
    adapter_path: Path = Path("out/adapter/alpaca/lit_model_adapter_finetuned.pth"),
    checkpoint_dir: Path = Path(f"checkpoints/stabilityai/stablelm-base-alpha-3b"),
    quantize: Optional[str] = None,
    max_new_tokens: int = 100,
    top_k: int = 200,
    temperature: float = 0.8,
) -> None:
    """Generates a response based on a given instruction and an optional input.
    This script will only work with checkpoints from the instruction-tuned Parrot-Adapter model.
    See `finetune_adapter.py`.

    Args:
        prompt: The prompt/instruction (Alpaca style).
        adapter_path: Path to the checkpoint with trained adapter weights, which are the output of
            `finetune_adapter.py`.
        checkpoint_dir: The path to the checkpoint folder with pretrained Parrot weights.
        input: Optional input (Alpaca style).
        quantize: Whether to quantize the model and using which method:
            ``"llm.int8"``: LLM.int8() mode,
            ``"gptq.int4"``: GPTQ 4-bit mode.
        max_new_tokens: The number of generation steps to take.
        top_k: The number of top most probable tokens to consider in the sampling process.
        temperature: A value controlling the randomness of the sampling process. Higher values result in more random
            samples.
    """
    check_valid_checkpoint_dir(checkpoint_dir)

    fabric = L.Fabric(devices=1)
    dtype = torch.bfloat16 if fabric.device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32

    with open(checkpoint_dir / "lit_config.json") as fp:
        config = Config(**json.load(fp))

    print("Loading model ...", file=sys.stderr)
    t0 = time.time()
    with EmptyInitOnDevice(device=fabric.device, dtype=dtype, quantization_mode=quantize):
        model = Parrot(config)
    with lazy_load(checkpoint_dir / "lit_model.pth") as pretrained_checkpoint, lazy_load(
        adapter_path
    ) as adapter_checkpoint:
        # 1. Load the pretrained weights
        model.load_state_dict(pretrained_checkpoint, strict=False)
        # 2. Load the fine-tuned adapter weights
        model.load_state_dict(adapter_checkpoint, strict=False)

    print(f"Time to load model: {time.time() - t0:.02f} seconds.", file=sys.stderr)

    model.eval()
    model = fabric.setup(model)

    tokenizer = Tokenizer(checkpoint_dir / "tokenizer.json", checkpoint_dir / "tokenizer_config.json")
    sample = {"instruction": prompt, "input": input}
    prompt = generate_prompt(sample)
    encoded = tokenizer.encode(prompt, device=model.device)
    prompt_length = encoded.size(0)

    t0 = time.perf_counter()
    y = generate(
        model,
        idx=encoded,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        eos_id=tokenizer.eos_id,
    )
    t = time.perf_counter() - t0

    output = tokenizer.decode(y)
    output = output.split("### Response:")[1].strip()
    print(output)

    tokens_generated = y.size(0) - prompt_length
    print(f"\n\nTime for inference: {t:.02f} sec total, {tokens_generated / t:.02f} tokens/sec", file=sys.stderr)
    if fabric.device.type == "cuda":
        print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB", file=sys.stderr)


if __name__ == "__main__":
    from jsonargparse import CLI

    torch.set_float32_matmul_precision("high")
    warnings.filterwarnings(
        # Triggered internally at ../aten/src/ATen/EmptyTensor.cpp:31
        "ignore",
        message="ComplexHalf support is experimental and many operators don't support it yet",
    )
    CLI(main)
