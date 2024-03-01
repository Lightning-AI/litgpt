# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import sys
import time
from pathlib import Path
from typing import Literal, Optional

import lightning as L
import torch
from lightning.fabric.plugins import BitsandbytesPrecision

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from generate.base import generate
from lit_gpt import Tokenizer
from lit_gpt.lora import GPT, Config, merge_lora_weights
from lit_gpt.utils import CLI, check_valid_checkpoint_dir, get_default_supported_precision, lazy_load
from lit_gpt.data import apply_prompt_template
from lit_gpt.data.alpaca import prompt_template


def main(
    prompt: str = "What food do llamas eat?",
    input: str = "",
    lora_path: Path = Path("out/lora/alpaca/lit_model_lora_finetuned.pth"),
    checkpoint_dir: Path = Path("checkpoints/stabilityai/stablelm-base-alpha-3b"),
    quantize: Optional[Literal["bnb.nf4", "bnb.nf4-dq", "bnb.fp4", "bnb.fp4-dq", "bnb.int8"]] = None,
    max_new_tokens: int = 100,
    top_k: Optional[int] = 200,
    temperature: float = 0.8,
    precision: Optional[str] = None,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_query: bool = True,
    lora_key: bool = False,
    lora_value: bool = True,
    lora_projection: bool = False,
    lora_mlp: bool = False,
    lora_head: bool = False,
) -> None:
    """Generates a response based on a given instruction and an optional input.
    This script will only work with checkpoints from the instruction-tuned GPT-LoRA model.
    See `finetune/lora.py`.

    Args:
        prompt: The prompt/instruction (Alpaca style).
        input: Optional input (Alpaca style).
        lora_path: Path to the checkpoint with trained adapter weights, which are the output of
            `finetune/lora.py`.
        checkpoint_dir: The path to the checkpoint folder with pretrained GPT weights.
        quantize: Whether to quantize the model and using which method:
            - bnb.nf4, bnb.nf4-dq, bnb.fp4, bnb.fp4-dq: 4-bit quantization from bitsandbytes
            - bnb.int8: 8-bit quantization from bitsandbytes
            for more details, see https://github.com/Lightning-AI/lit-gpt/blob/main/tutorials/quantize.md
        max_new_tokens: The number of generation steps to take.
        top_k: The number of top most probable tokens to consider in the sampling process.
        temperature: A value controlling the randomness of the sampling process. Higher values result in more random
            samples.
        precision: Indicates the Fabric precision setting to use.
    """
    precision = precision or get_default_supported_precision(training=False)

    plugins = None
    if quantize is not None and quantize.startswith("bnb."):
        if "mixed" in precision:
            raise ValueError("Quantization and mixed precision is not supported.")
        dtype = {"16-true": torch.float16, "bf16-true": torch.bfloat16, "32-true": torch.float32}[precision]
        plugins = BitsandbytesPrecision(quantize[4:], dtype)
        precision = None

    fabric = L.Fabric(devices=1, precision=precision, plugins=plugins)
    fabric.launch()

    check_valid_checkpoint_dir(checkpoint_dir)

    config = Config.from_json(
        checkpoint_dir / "lit_config.json",
        r=lora_r,
        alpha=lora_alpha,
        dropout=lora_dropout,
        to_query=lora_query,
        to_key=lora_key,
        to_value=lora_value,
        to_projection=lora_projection,
        to_mlp=lora_mlp,
        to_head=lora_head,
    )

    checkpoint_path = checkpoint_dir / "lit_model.pth"

    tokenizer = Tokenizer(checkpoint_dir)

    sample = {"instruction": prompt, "input": input}
    prompt = apply_prompt_template(prompt_template, sample)
    encoded = tokenizer.encode(prompt, device=fabric.device)
    prompt_length = encoded.size(0)
    max_returned_tokens = prompt_length + max_new_tokens

    fabric.print(f"Loading model {str(checkpoint_path)!r} with {config.__dict__}", file=sys.stderr)
    t0 = time.perf_counter()
    with fabric.init_module(empty_init=True):
        model = GPT(config)
    fabric.print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.", file=sys.stderr)
    with fabric.init_tensor():
        # set the max_seq_length to limit the memory usage to what we need
        model.max_seq_length = max_returned_tokens
        # enable the kv cache
        model.set_kv_cache(batch_size=1)
    model.eval()

    t0 = time.perf_counter()
    checkpoint = lazy_load(checkpoint_path)
    lora_checkpoint = lazy_load(lora_path)
    checkpoint.update(lora_checkpoint.get("model", lora_checkpoint))
    model.load_state_dict(checkpoint)
    fabric.print(f"Time to load the model weights: {time.perf_counter() - t0:.02f} seconds.", file=sys.stderr)

    merge_lora_weights(model)
    model = fabric.setup(model)

    L.seed_everything(1234)
    t0 = time.perf_counter()
    y = generate(model, encoded, max_returned_tokens, temperature=temperature, top_k=top_k, eos_id=tokenizer.eos_id)
    t = time.perf_counter() - t0

    output = tokenizer.decode(y)
    output = output.split("### Response:")[1].strip()
    fabric.print(output)

    tokens_generated = y.size(0) - prompt_length
    fabric.print(f"\n\nTime for inference: {t:.02f} sec total, {tokens_generated / t:.02f} tokens/sec", file=sys.stderr)
    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB", file=sys.stderr)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    CLI(main)