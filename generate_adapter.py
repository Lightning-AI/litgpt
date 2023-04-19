import sys
import time
import warnings
from pathlib import Path
from typing import Optional

import lightning as L
import torch

from generate import generate
from lit_llama import Tokenizer
from lit_llama.adapter import LLaMA, LLaMAConfig
from lit_llama.utils import EmptyInitOnDevice, lazy_load
from scripts.prepare_alpaca import generate_prompt


def main(
    prompt: str = "What food do lamas eat?",
    input: str = "",
    adapter_path: Optional[Path] = None,
    pretrained_path: Optional[Path] = None,
    tokenizer_path: Optional[Path] = None,
    quantize: Optional[str] = None,
    max_new_tokens: int = 100,
    top_k: int = 200,
    temperature: float = 0.8,
) -> None:
    """Generates a response based on a given instruction and an optional input.
    This script will only work with checkpoints from the instruction-tuned LLaMA-Adapter model.
    See `finetune_adapter.py`.

    Args:
        prompt: The prompt/instruction (Alpaca style).
        adapter_path: Path to the checkpoint with trained adapter weights, which are the output of
            `finetune_adapter.py`.
        input: Optional input (Alpaca style).
        pretrained_path: The path to the checkpoint with pretrained LLaMA weights.
        tokenizer_path: The tokenizer path to load.
        quantize: Whether to quantize the model and using which method:
            ``"llm.int8"``: LLM.int8() mode,
            ``"gptq.int4"``: GPTQ 4-bit mode.
        dtype: The dtype to use during generation.
        max_new_tokens: The number of generation steps to take.
        top_k: The number of top most probable tokens to consider in the sampling process.
        temperature: A value controlling the randomness of the sampling process. Higher values result in more random
            samples.
    """
    if not adapter_path:
        adapter_path = Path("out/adapter/alpaca/lit-llama-adapter-finetuned.pth")
    if not pretrained_path:
        pretrained_path = Path(f"./checkpoints/lit-llama/7B/lit-llama.pth")
    if not tokenizer_path:
        tokenizer_path = Path("./checkpoints/lit-llama/tokenizer.model")
    
    assert adapter_path.is_file()
    assert pretrained_path.is_file()
    assert tokenizer_path.is_file()

    fabric = L.Fabric(accelerator="cuda", devices=1)

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32

    print("Loading model ...", file=sys.stderr)
    t0 = time.time()
    with EmptyInitOnDevice(
        device=fabric.device, dtype=dtype, quantization_mode=quantize
    ):
        model = LLaMA(LLaMAConfig())  # TODO: Support different model sizes

    # 1. Load the pretrained weights
    pretrained_checkpoint = lazy_load(pretrained_path)
    model.load_state_dict(pretrained_checkpoint, strict=False)
        
    # 2. Load the fine-tuned adapter weights
    adapter_checkpoint = lazy_load(adapter_path)
    model.load_state_dict(adapter_checkpoint, strict=False)
        
    print(f"Time to load model: {time.time() - t0:.02f} seconds.", file=sys.stderr)

    model.eval()
    model = fabric.setup_module(model)

    tokenizer = Tokenizer(tokenizer_path)
    sample = {"instruction": prompt, "input": input}
    prompt = generate_prompt(sample)
    encoded = tokenizer.encode(prompt, bos=True, eos=False, device=model.device)

    t0 = time.perf_counter()
    output = generate(
        model,
        idx=encoded,
        max_seq_length=max_new_tokens,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        eos_id=tokenizer.eos_id
    )

    output = tokenizer.decode(output)
    output = output.split("### Response:")[1].strip()

    print(output)
    t = time.perf_counter() - t0

    print(f"\n\nTime for inference: {t:.02f} sec total, {max_new_tokens / t:.02f} tokens/sec", file=sys.stderr)
    print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB", file=sys.stderr)



if __name__ == "__main__":
    from jsonargparse import CLI

    torch.set_float32_matmul_precision("high")
    warnings.filterwarnings(
        # Triggered internally at ../aten/src/ATen/EmptyTensor.cpp:31
        "ignore", 
        message="ComplexHalf support is experimental and many operators don't support it yet"
    )
    CLI(main)
