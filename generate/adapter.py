import sys
import time
from pathlib import Path
from typing import Literal, Optional

import lightning as L
import torch
from lightning.fabric.plugins import BitsandbytesPrecision
from lightning.fabric.strategies import FSDPStrategy

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from generate.base import generate
from lit_gpt import Tokenizer
from lit_gpt.adapter import GPT, Block, Config
from lit_gpt.utils import check_valid_checkpoint_dir, get_default_supported_precision, gptq_quantization, lazy_load
from scripts.prepare_alpaca import generate_prompt


def main(
    prompt: str = "What food do lamas eat?",
    input: str = "",
    adapter_path: Path = Path("out/adapter/alpaca/lit_model_adapter_finetuned.pth"),
    checkpoint_dir: Path = Path("checkpoints/stabilityai/stablelm-base-alpha-3b"),
    quantize: Optional[Literal["bnb.nf4", "bnb.nf4-dq", "bnb.fp4", "bnb.fp4-dq", "bnb.int8", "gptq.int4"]] = None,
    max_new_tokens: int = 100,
    top_k: int = 200,
    temperature: float = 0.8,
    strategy: str = "auto",
    devices: int = 1,
    precision: Optional[str] = None,
) -> None:
    """Generates a response based on a given instruction and an optional input.
    This script will only work with checkpoints from the instruction-tuned GPT-Adapter model.
    See `finetune/adapter.py`.

    Args:
        prompt: The prompt/instruction (Alpaca style).
        input: Optional input (Alpaca style).
        adapter_path: Path to the checkpoint with trained adapter weights, which are the output of
            `finetune/adapter.py`.
        checkpoint_dir: The path to the checkpoint folder with pretrained GPT weights.
        quantize: Whether to quantize the model and using which method:
            - bnb.nf4, bnb.nf4-dq, bnb.fp4, bnb.fp4-dq: 4-bit quantization from bitsandbytes
            - bnb.int8: 8-bit quantization from bitsandbytes
            - gptq.int4: 4-bit quantization from GPTQ
            for more details, see https://github.com/Lightning-AI/lit-gpt/blob/main/tutorials/quantize.md
        max_new_tokens: The number of generation steps to take.
        top_k: The number of top most probable tokens to consider in the sampling process.
        temperature: A value controlling the randomness of the sampling process. Higher values result in more random
            samples.
        strategy: Indicates the Fabric strategy setting to use.
        devices: How many devices to use.
        precision: Indicates the Fabric precision setting to use.
    """
    precision = precision or get_default_supported_precision(training=False)

    plugins = None
    if quantize is not None:
        if devices > 1:
            raise NotImplementedError(
                "Quantization is currently not supported for multi-GPU training. Please set devices=1 when using the"
                " --quantize flag."
            )
        if quantize.startswith("bnb."):
            if "mixed" in precision:
                raise ValueError("Quantization and mixed precision is not supported.")
            dtype = {"16-true": torch.float16, "bf16-true": torch.bfloat16, "32-true": torch.float32}[precision]
            plugins = BitsandbytesPrecision(quantize[4:], dtype)
            precision = None

    if strategy == "fsdp":
        strategy = FSDPStrategy(auto_wrap_policy={Block}, cpu_offload=False)

    fabric = L.Fabric(devices=devices, precision=precision, strategy=strategy, plugins=plugins)
    fabric.launch()

    check_valid_checkpoint_dir(checkpoint_dir)

    config = Config.from_json(checkpoint_dir / "lit_config.json")

    if quantize is not None and devices > 1:
        raise NotImplementedError
    if quantize == "gptq.int4":
        model_file = "lit_model_gptq.4bit.pth"
        if not (checkpoint_dir / model_file).is_file():
            raise ValueError("Please run `python quantize/gptq.py` first")
    else:
        model_file = "lit_model.pth"
    checkpoint_path = checkpoint_dir / model_file

    fabric.print(f"Loading model {str(checkpoint_path)!r} with {config.__dict__}", file=sys.stderr)
    t0 = time.perf_counter()
    with fabric.init_module(empty_init=True), gptq_quantization(quantize == "gptq.int4"):
        model = GPT(config)
    fabric.print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.", file=sys.stderr)

    t0 = time.perf_counter()
    checkpoint = lazy_load(checkpoint_path)
    adapter_checkpoint = lazy_load(adapter_path)
    checkpoint.update(adapter_checkpoint.get("model", adapter_checkpoint))
    model.load_state_dict(checkpoint)
    fabric.print(f"Time to load the model weights: {time.perf_counter() - t0:.02f} seconds.", file=sys.stderr)

    model.eval()
    model = fabric.setup(model)

    tokenizer = Tokenizer(checkpoint_dir)
    sample = {"instruction": prompt, "input": input}
    prompt = generate_prompt(sample)
    encoded = tokenizer.encode(prompt, device=fabric.device)
    prompt_length = encoded.size(0)
    max_returned_tokens = prompt_length + max_new_tokens

    with fabric.init_tensor():
        # set the max_seq_length to limit the memory usage to what we need
        model.max_seq_length = max_returned_tokens
        # enable the kv cache
        model.set_kv_cache(batch_size=1)

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
    from jsonargparse import CLI

    torch.set_float32_matmul_precision("high")
    CLI(main)
