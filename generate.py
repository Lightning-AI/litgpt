import json
import sys
import time
import warnings
from pathlib import Path
from typing import Optional

import lightning as L
import torch

from lit_parrot import Parrot, Tokenizer, Config
from lit_parrot.utils import EmptyInitOnDevice, lazy_load, check_valid_checkpoint_dir


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    idx: torch.Tensor,
    max_new_tokens: int,
    max_seq_length: int,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    eos_id: Optional[int] = None,
) -> torch.Tensor:
    """Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.

    The implementation of this function is modified from A. Karpathy's nanoGPT.

    Args:
        model: The model to use.
        idx: Tensor of shape (T) with indices of the prompt sequence.
        max_new_tokens: The number of new tokens to generate.
        max_seq_length: The maximum sequence length allowed.
        temperature: Scales the predicted logits by 1 / temperature
        top_k: If specified, only sample among the tokens with the k highest probabilities
        eos_id: If specified, stop generating any more token once the <eos> token is triggered
    """
    # create an empty tensor of the expected final shape and fill in the current tokens
    T = idx.size(0)
    T_new = T + max_new_tokens
    empty = torch.empty(T_new, dtype=idx.dtype, device=idx.device)
    empty[:T] = idx
    idx = empty

    cache_kvs = [
        (
            torch.zeros((1, model.config.n_head, T_new, model.config.n_embd // model.config.n_head), device=idx.device),
            torch.zeros((1, model.config.n_head, T_new, model.config.n_embd // model.config.n_head), device=idx.device),
        )
        for _ in range(model.config.n_layer)
    ]

    input_pos = torch.arange(0, T).to(idx.device)
    input_idx = idx.index_select(0, input_pos)

    # generate max_new_tokens tokens
    for t in range(T, T_new):
        # ignore the not-filled-yet tokens
        idx_cond = idx[:t]
        # if the sequence context is growing too long we must crop it at max_seq_length
        idx_cond = idx_cond if t <= max_seq_length else idx_cond[-max_seq_length:]

        # forward
        logits, cache_kvs = model(input_idx.view(1, -1), input_pos, cache_kvs)
        logits = logits[0, -1] / temperature

        # optionally crop the logits to only the top k options
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits = torch.where(logits < v[[-1]], -float("Inf"), logits)

        probs = torch.nn.functional.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1).to(dtype=idx.dtype)

        # concatenate the new generation
        idx = idx.index_copy(0, input_pos[-1] + 1, idx_next)

        input_pos = input_pos[-1:] + 1
        input_idx = idx.index_select(0, input_pos)

        # if <eos> token is triggered, return the output (stop generation)
        if idx_next == eos_id:
            return idx[: t + 1]  # include the EOS token

    return idx


def main(
    prompt: str = "Hello, my name is",
    *,
    num_samples: int = 1,
    max_new_tokens: int = 50,
    top_k: int = 200,
    temperature: float = 0.8,
    checkpoint_dir: Path = Path(f"checkpoints/stabilityai/stablelm-base-alpha-3b"),
    quantize: Optional[str] = None,
) -> None:
    """Generates text samples based on a pre-trained model and tokenizer.

    Args:
        prompt: The prompt string to use for generating the samples.
        num_samples: The number of text samples to generate.
        max_new_tokens: The number of generation steps to take.
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

    checkpoint_path = checkpoint_dir / "lit_model.pth"
    print(f"Loading model {str(checkpoint_path)!r} with {config.__dict__}", file=sys.stderr)
    t0 = time.time()
    with EmptyInitOnDevice(device=fabric.device, dtype=dtype, quantization_mode=quantize):
        model = Parrot(config)
    with lazy_load(checkpoint_path) as checkpoint:
        model.load_state_dict(checkpoint)
    print(f"Time to load model: {time.time() - t0:.02f} seconds.", file=sys.stderr)

    model.eval()
    model = fabric.setup_module(model)

    tokenizer = Tokenizer(checkpoint_dir / "tokenizer.json", checkpoint_dir / "tokenizer_config.json")
    encoded = tokenizer.encode(prompt, device=fabric.device)
    prompt_length = encoded.size(0)

    L.seed_everything(1234)
    for i in range(num_samples):
        t0 = time.perf_counter()
        y = generate(
            model,
            encoded,
            max_new_tokens,
            model.config.block_size,  # type: ignore[union-attr,arg-type]
            temperature=temperature,
            top_k=top_k,
        )
        t = time.perf_counter() - t0

        print(tokenizer.decode(y))
        tokens_generated = y.size(0) - prompt_length
        print(
            f"Time for inference {i + 1}: {t:.02f} sec total, {tokens_generated / t:.02f} tokens/sec", file=sys.stderr
        )
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
    warnings.filterwarnings(
        # Triggered in bitsandbytes/autograd/_functions.py:298
        "ignore",
        message="MatMul8bitLt: inputs will be cast from torch.bfloat16 to float16 during quantization",
    )
    CLI(main)
