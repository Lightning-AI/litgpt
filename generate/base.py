# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import sys
import time
from pathlib import Path
from typing import Any, Literal, Optional, Union, List

import lightning as L
import torch
import torch._dynamo.config
import torch._inductor.config
from lightning.fabric.plugins import BitsandbytesPrecision

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_gpt import GPT, Config, Tokenizer
from lit_gpt.utils import (
    check_valid_checkpoint_dir,
    get_default_supported_precision,
    gptq_quantization,
    load_checkpoint,
    pad_batched_tokens
)


def multinomial_num_samples_1(probs: torch.Tensor) -> torch.Tensor:
    if torch._dynamo.is_compiling():
        # Faster alternative to `torch.multinomial(probs, num_samples=1)` that is also CUDAGraph friendly
        distribution = torch.empty_like(probs).exponential_(1)
        return torch.argmax(probs / distribution, dim=-1, keepdim=True)
    return torch.multinomial(probs, num_samples=1)


def sample(logits: torch.Tensor, temperature: float = 1.0, top_k: Optional[int] = None) -> torch.Tensor:

    logits = logits[:, -1]

    # optionally crop the logits to only the top k options
    if top_k is not None:
        v, i = torch.topk(logits, min(top_k, logits.size(-1)))
        # do not use `torch.where` as in nanogpt because it will repeat top-k collisions
        logits = torch.full_like(logits, float("-inf")).scatter_(-1, i, v)
    # optionally scale the logits and sample from a probability distribution
    if temperature > 0.0:
        probs = torch.nn.functional.softmax(logits / temperature, dim=-1)
        return multinomial_num_samples_1(probs)
    return torch.argmax(logits, dim=-1, keepdim=True)


def next_token(model: GPT, input_pos: torch.Tensor, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
    logits = model(x, input_pos)
    next = sample(logits, **kwargs)
    return next.to(dtype=x.dtype)

@torch.inference_mode()
def generate(
    model: GPT,
    prompts: torch.Tensor,
    max_returned_tokens: int,
    *,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    eos_id: Optional[int] = None,
) -> torch.Tensor:
    
    """Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.

    Is capable of handling batched prompts with left-padding.
    
    The implementation of this function is modified from A. Karpathy's nanoGPT.

    Args:
        model: The model to use.
        prompt: Tensor of shape (T) with indices of the prompt sequence.
        max_returned_tokens: The maximum number of tokens to return (given plus generated).
        temperature: Scales the predicted logits by 1 / temperature.
        top_k: If specified, only sample among the tokens with the k highest probabilities.
        eos_id: If specified, stop generating any more token once the <eos> token is triggered.
    """
    
    # make sure we have a batch dimension, and remember to remove it later if there wasn't one
    resqueeze = False
    if prompts.dim() == 1:
        resqueeze = True
        prompts = prompts.unsqueeze(0)
        
    B = prompts.size(0)
    T = prompts.size(1)
    
    assert max_returned_tokens > T
    if model.max_seq_length < max_returned_tokens - 1:
        # rolling the kv cache based on the `input_pos` value would be necessary. However, doing so would introduce a
        # data dependency on the `input_pos` tensor and impact model compilation. Since this setting is uncommon, we do
        # not support it to avoid negatively impacting the overall speed
        raise NotImplementedError(f"max_seq_length {model.max_seq_length} needs to be >= {max_returned_tokens - 1}")

    device = prompts.device
    dtype = prompts.dtype

    # an output tensor to fill with generated tokens
    outputs_tensor = torch.zeros((B, max_returned_tokens), dtype=dtype, device=device)
    outputs_tensor[:, :T] = prompts

    token = next_token(model, torch.arange(0, T, device=device), prompts, temperature=temperature, top_k=top_k).clone()
    outputs_tensor[torch.arange(0, B, device=device) , T] = token.squeeze()

    input_position = torch.tensor(T + 1, device=device)
    finished = torch.tensor([False] * B, device=device).view(-1)
    max_generation_iters = max_returned_tokens - T + 1

    for _ in range(2, max_generation_iters):

        token = next_token(model, input_position, token, temperature=temperature, top_k=top_k).clone()
        outputs_tensor[~finished, input_position] = token.view(-1)[~finished]
        input_position = input_position.add_(1)
        finished = (token == eos_id).view(-1) | finished
          
        if finished.all():
            break
    
    # removing the artificial batch dimension if there is one for backward compatibility
    if resqueeze:
        outputs_tensor = outputs_tensor.squeeze(0)
    return outputs_tensor



@torch.inference_mode()
def main(
    prompts: Union[str, List] = "What food do llamas eat?",
    *,
    num_samples: int = 1,
    max_new_tokens: int = 512,
    top_k: Optional[int] = 200,
    temperature: float = 0.8,
    checkpoint_dir: Path = Path("checkpoints/stabilityai/stablelm-base-alpha-3b"),
    quantize: Optional[Literal["bnb.nf4", "bnb.nf4-dq", "bnb.fp4", "bnb.fp4-dq", "bnb.int8", "gptq.int4"]] = None,
    precision: Optional[str] = None,
    compile: bool = False,
) -> None:
    
    """Generates text samples based on a pre-trained model and tokenizer.

    Args:
        prompt: The prompt string or list of prompt strings to use for generating the samples.
        num_samples: The number of text samples to generate.
        max_new_tokens: The number of generation steps to take.
        top_k: The number of top most probable tokens to consider in the sampling process.
        temperature: A value controlling the randomness of the sampling process. Higher values result in more random
            samples.
        checkpoint_dir: The checkpoint directory to load.
        quantize: Whether to quantize the model and using which method:
            - bnb.nf4, bnb.nf4-dq, bnb.fp4, bnb.fp4-dq: 4-bit quantization from bitsandbytes
            - bnb.int8: 8-bit quantization from bitsandbytes
            - gptq.int4: 4-bit quantization from GPTQ
            for more details, see https://github.com/Lightning-AI/lit-gpt/blob/main/tutorials/quantize.md
        precision: Indicates the Fabric precision setting to use.
        compile: Whether to compile the model.
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

    check_valid_checkpoint_dir(checkpoint_dir)

    config = Config.from_json(checkpoint_dir / "lit_config.json")

    if quantize == "gptq.int4":
        model_file = "lit_model_gptq.4bit.pth"
        if not (checkpoint_dir / model_file).is_file():
            raise ValueError("Please run `python quantize/gptq.py` first")
    else:
        model_file = "lit_model.pth"
    checkpoint_path = checkpoint_dir / model_file

    tokenizer = Tokenizer(checkpoint_dir)
    if isinstance(prompts, str):
        prompts = [prompts]

    tokenized_prompts = [tokenizer.encode(prompt, device=fabric.device) for prompt in prompts]
    encoded, padding_mask = pad_batched_tokens(tokenized_prompts, pad_id=0, padding_direction="left")
    
    B = len(encoded)   
    T = encoded.size(1)
    
    max_returned_tokens = T + max_new_tokens

    fabric.print(f"Loading model {str(checkpoint_path)!r} with {config.__dict__}", file=sys.stderr)
    t0 = time.perf_counter()
    with fabric.init_module(empty_init=True), gptq_quantization(quantize == "gptq.int4"):
        model = GPT(config)
    fabric.print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.", file=sys.stderr)
    with fabric.init_tensor():
        # set the max_seq_length to limit the memory usage to what we need
        model.max_seq_length = max_returned_tokens
        # enable the kv cache
        model.set_kv_cache(batch_size=B, padding_mask=padding_mask)
    model.eval()

    if compile:
        torch._dynamo.config.automatic_dynamic_shapes = True
        torch._inductor.config.triton.unique_kernel_names = True
        torch._inductor.config.coordinate_descent_tuning = True
        global next_token
        next_token = torch.compile(next_token, mode="reduce-overhead")

    model = fabric.setup_module(model)

    t0 = time.perf_counter()
    load_checkpoint(fabric, model, checkpoint_path)
    fabric.print(f"Time to load the model weights: {time.perf_counter() - t0:.02f} seconds.", file=sys.stderr)

    L.seed_everything(1234)
    for i in range(num_samples):
        t0 = time.perf_counter()
        y = generate(model, encoded, max_returned_tokens, temperature=temperature, top_k=top_k, eos_id=tokenizer.eos_id)
        t = time.perf_counter() - t0
        for block in model.transformer.h:
            block.attn.kv_cache.reset_parameters()

        padding_mask = (y != 0) & (y != tokenizer.eos_id) # filters 0 tokens and eos tokens for printing

        for b in range(B):
            fabric.print(f"\n\033[35mOutput # \033[37m{b + 1}:")
            fabric.print(f"\033[32mTotal tokens: \033[37m{y[b][padding_mask[b]].count_nonzero()}\n")
            fabric.print(f"\033[37m{tokenizer.decode(y[b][padding_mask[b]])}\n")

        tokens_generated = y[padding_mask].count_nonzero() - T * B
        fabric.print(f"Time for inference {i + 1}: {t:.02f} sec total, {tokens_generated / t:.02f} tokens/sec", file=sys.stderr)
        
    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB", file=sys.stderr)


if __name__ == "__main__":
    from jsonargparse import CLI

    torch.set_float32_matmul_precision("high")
    CLI(main)
