import json
import sys
import warnings
from pathlib import Path
from typing import Optional, Tuple

import lightning as L
import torch

from lit_stablelm import StableLM, Tokenizer, StableLMConfig
from lit_stablelm.utils import EmptyInitOnDevice, lazy_load


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    idx: torch.Tensor,
    max_seq_length: int,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    stop_tokens: Tuple[int, ...] = tuple(),
):
    """Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as possible.

    Args:
        model: The model to use.
        idx: Tensor of shape (T) with indices of the prompt sequence.
        max_seq_length: The maximum sequence length allowed.
        temperature: Scales the predicted logits by 1 / temperature
        top_k: If specified, only sample among the tokens with the k highest probabilities
        stop_tokens: If specified, stop generating any more token once one of this list is generated.
    """
    # create an empty tensor of the expected final shape and fill in the current tokens
    T = idx.size(0)
    assert max_seq_length > T
    for t in range(T, max_seq_length):
        # forward
        logits = model(idx.view(1, -1))
        logits = logits[0, -1] / temperature

        # optionally crop the logits to only the top k options
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[[-1]]] = -float("Inf")

        probs = torch.nn.functional.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        yield idx_next

        # concatenate the new generation
        idx = torch.cat((idx, idx_next), dim=-1)
        # if <eos> token is triggered, return the output (stop generation)
        if idx_next in stop_tokens:
            break


system_prompt = """<|SYSTEM|># StableLM Tuned (Alpha version)
- StableLM is a helpful and harmless open-source AI language model developed by StabilityAI.
- StableLM is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
- StableLM is more than just an information source, StableLM is also able to write poetry, short stories, and make jokes.
- StableLM will refuse to participate in anything that could harm a human.<|USER|>{prompt}<|ASSISTANT|>
"""


def main(
    *,
    top_k: int = 200,
    temperature: float = 0.8,
    ckpt_dir: Path = Path(f"checkpoints/stabilityai/stablelm-tuned-alpha-3b"),
    quantize: Optional[str] = None,
) -> None:
    """Starts a conversation with a tuned StableLM model.

    Args:
        top_k: The number of top most probable tokens to consider in the sampling process.
        temperature: A value controlling the randomness of the sampling process. Higher values result in more random
            samples.
        ckpt_dir: The checkpoint directory to load.
        quantize: Whether to quantize the model and using which method:
            ``"llm.int8"``: LLM.int8() mode,
            ``"gptq.int4"``: GPTQ 4-bit mode.
    """
    if not ckpt_dir.is_dir():
        raise OSError(
            f"`--ckpt_dir={str(ckpt_dir)!r} must be a directory with the lit model checkpoint and configurations. Please,"
            " follow the instructions at"
            " https://github.com/Lightning-AI/lit-stablelm/blob/main/howto/download_weights.md"
        )

    with open(ckpt_dir / "lit_config.json") as fp:
        config = StableLMConfig(**json.load(fp))

    fabric = L.Fabric(devices=1)
    dtype = torch.bfloat16 if fabric.device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32

    checkpoint_path = ckpt_dir / "lit_model.pth"
    print(f"Loading model {str(checkpoint_path)!r} with {config.__dict__}", file=sys.stderr)
    with EmptyInitOnDevice(device=fabric.device, dtype=dtype, quantization_mode=quantize):
        model = StableLM(config)
    with lazy_load(checkpoint_path) as checkpoint:
        model.load_state_dict(checkpoint)

    model.eval()
    model = fabric.setup_module(model)

    tokenizer = Tokenizer(ckpt_dir / "tokenizer.json", ckpt_dir / "tokenizer_config.json")
    stop_tokens = (
        tokenizer.eos_id,
        tokenizer.processor.token_to_id("<|SYSTEM|>"),
        tokenizer.processor.token_to_id("<|ASSISTANT|>"),
        tokenizer.processor.token_to_id("<|USER|>"),
    )

    while True:
        try:
            prompt = input(">> Prompt: ")
        except KeyboardInterrupt:
            break
        if not prompt:
            break
        prompt = system_prompt.format(prompt=prompt)
        encoded_prompt = tokenizer.encode(prompt, device=fabric.device)
        y = generate(
            model,
            encoded_prompt,
            model.config.block_size,  # type: ignore[union-attr,arg-type]
            temperature=temperature,
            top_k=top_k,
            stop_tokens=stop_tokens,
        )
        print(f">> Reply: ", end="")
        try:
            for token in y:
                print(tokenizer.decode(token), end="", flush=True)
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
