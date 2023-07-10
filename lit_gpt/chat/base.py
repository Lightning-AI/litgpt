import json
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Optional, Tuple, List, Literal, Iterator

import lightning as L
import torch

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_gpt import GPT, Tokenizer, Config
from lit_gpt.utils import lazy_load, check_valid_checkpoint_dir, quantization


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    idx: torch.Tensor,
    max_returned_tokens: int,
    max_seq_length: int,
    *,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    stop_tokens: Tuple[List[int], ...] = tuple(),
):
    """Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as possible.

    Args:
        model: The model to use.
        idx: Tensor of shape (T) with indices of the prompt sequence.
        max_returned_tokens: The maximum number of tokens to return (given plus generated).
        max_seq_length: The maximum sequence length allowed. Should be less or equal than the block size.
        temperature: Scales the predicted logits by 1 / temperature
        top_k: If specified, only sample among the tokens with the k highest probabilities
        stop_tokens: If specified, stop generating any more token once one of this list is generated.
    """
    T = idx.size(0)
    assert max_returned_tokens > T
    device = idx.device
    stop_tokens = [torch.tensor(tokens, device=device) for tokens in stop_tokens]
    input_pos = torch.arange(0, T, device=device)

    # buffer holds the tokens that haven't been yield yet
    buffer_length = max((len(tokens) for tokens in stop_tokens), default=1)
    buffer = torch.full((buffer_length,), -999, device=device)  # fill with non-existing token

    if idx.device.type == "xla":
        import torch_xla.core.xla_model as xm

        xm.mark_step()

    yield_i = -1
    # generate up to a fixed number of tokens
    for t in range(max_returned_tokens - T):
        # forward
        logits = model(idx.view(1, -1), max_seq_length, input_pos)
        logits = logits[0, -1] / temperature

        # optionally crop the logits to only the top k options
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits = torch.where(logits < v[[-1]], -float("Inf"), logits)

        probs = torch.nn.functional.softmax(logits, dim=-1)
        idx = torch.multinomial(probs, num_samples=1)

        # advance
        input_pos = input_pos[-1:] + 1

        if idx.device.type == "xla":
            xm.mark_step()

        # concatenate the new generation
        buffer[min(t, buffer_length - 1)] = idx

        # check the stop condition
        for tokens in stop_tokens:
            l = len(tokens)
            if torch.equal(buffer[-l:], tokens):
                # stop token hit, yield any leftovers that aren't part of it
                if buffer_length > l:  # avoid an empty yield
                    yield buffer[:-l]
                return
        # if the buffer is full
        if t - yield_i >= buffer_length:
            # we know this idx is not part of stop tokens, safe to yield
            yield buffer[0]
            # roll once to the left, as next generation will be put at the end
            buffer = torch.roll(buffer, -1, 0)
            yield_i += 1


def decode(fabric: L.Fabric, tokenizer: Tokenizer, token_stream: Iterator[torch.Tensor]) -> int:
    tokens_generated = 0
    if tokenizer.backend == "huggingface":
        for token in token_stream:
            fabric.print(tokenizer.decode(token), end="", flush=True)
            tokens_generated += 1
    elif tokenizer.backend == "sentencepiece":
        # sentencepiece does not support decoding token-by-token because it adds spaces based on the surrounding tokens
        # meaning that we need to decode everything each time
        so_far = torch.tensor([], dtype=torch.long, device=fabric.device)
        decoded_so_far = ""
        for token in token_stream:
            so_far = torch.cat((so_far, token.view(-1)))
            decoded_new = tokenizer.decode(so_far)
            fabric.print(decoded_new[len(decoded_so_far) :], end="", flush=True)
            decoded_so_far = decoded_new
            tokens_generated += 1
    else:
        raise NotImplementedError(tokenizer.backend)
    return tokens_generated


def main(
    *,
    top_k: int = 200,
    temperature: float = 0.8,
    checkpoint_dir: Path = Path(f"checkpoints/stabilityai/stablelm-tuned-alpha-3b"),
    quantize: Literal["llm.int8", "gptq.int4"] = None,
    precision: str = "bf16-true",
) -> None:
    """Starts a conversation with a tuned GPT model.

    Args:
        top_k: The number of top most probable tokens to consider in the sampling process.
        temperature: A value controlling the randomness of the sampling process. Higher values result in more random
            samples.
        checkpoint_dir: The checkpoint directory to load.
        quantize: Whether to quantize the model and using which method:
            ``"llm.int8"``: LLM.int8() mode,
            ``"gptq.int4"``: GPTQ 4-bit mode.
        precision: Indicates the Fabric precision setting to use.
    """
    check_valid_checkpoint_dir(checkpoint_dir)

    with open(checkpoint_dir / "lit_config.json") as fp:
        config = Config(**json.load(fp))

    fabric = L.Fabric(devices=1, precision=precision)

    if quantize == "gptq.int4":
        model_file = "lit_model_gptq.4bit.pth"
        if not (checkpoint_dir / model_file).is_file():
            raise ValueError("Please run `python quantize/gptq.py` first")
    else:
        model_file = "lit_model.pth"
    checkpoint_path = checkpoint_dir / model_file
    fabric.print(f"Loading model {str(checkpoint_path)!r} with {config.__dict__}", file=sys.stderr)
    with fabric.init_module(empty_init=True), quantization(quantize):
        model = GPT(config)
    with lazy_load(checkpoint_path) as checkpoint:
        model.load_state_dict(checkpoint.get("model", checkpoint), strict=quantize is None)

    model.eval()
    model = fabric.setup_module(model)

    tokenizer = Tokenizer(checkpoint_dir)
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
        fabric.print(">> Reply: ", end="")
        try:
            t0 = time.perf_counter()
            tokens_generated = decode(fabric, tokenizer, y)
            t = time.perf_counter() - t0
            model.reset_cache()
            fabric.print(
                f"\nTime for inference: {t:.02f} sec total, {tokens_generated / t:.02f} tokens/sec", file=sys.stderr
            )
        except KeyboardInterrupt:
            # support stopping generation
            pass
        fabric.print()


def prompt_config(checkpoint_dir: Path, tokenizer: Tokenizer) -> Tuple[str, Tuple[List[int], ...]]:
    checkpoint_name = str(checkpoint_dir)
    if re.search(r"stabilityai.*tuned-alpha", checkpoint_name):
        system_prompt = (
            "<|SYSTEM|># StableLM Tuned (Alpha version)\n- StableLM is a helpful and harmless open-source AI language"
            " model developed by StabilityAI.\n- StableLM is excited to be able to help the user, but will refuse to do"
            " anything that could be considered harmful to the user.\n- StableLM is more than just an information"
            " source, StableLM is also able to write poetry, short stories, and make jokes.\n- StableLM will refuse to"
            " participate in anything that could harm a human.<|USER|>{prompt}<|ASSISTANT|>"
        )
        stop_tokens = (
            [tokenizer.eos_id],
            [tokenizer.token_to_id("<|SYSTEM|>")],
            [tokenizer.token_to_id("<|ASSISTANT|>")],
            [tokenizer.token_to_id("<|USER|>")],
        )
        return system_prompt, stop_tokens
    if re.search(r"togethercomputer.*Chat", checkpoint_name):
        system_prompt = "<human>: {prompt}\n<bot>:"
        lt, gt = tokenizer.token_to_id("<"), tokenizer.token_to_id(">:")
        stop_tokens = (
            [tokenizer.eos_id],
            # annoyingly, there's no single stop token for these
            [lt, tokenizer.token_to_id("human"), gt],
            [lt, tokenizer.token_to_id("bot"), gt],
        )
        return system_prompt, stop_tokens
    if re.search(r"togethercomputer.*Instruct", checkpoint_name):
        system_prompt = "Q: {prompt}\nA:"
        colon = tokenizer.token_to_id(":")
        stop_tokens = (
            [tokenizer.eos_id],
            # annoyingly, there's no single stop token for these
            [tokenizer.token_to_id("Q"), colon],
            [tokenizer.token_to_id("Question")],
            [tokenizer.token_to_id("A"), colon],
            [tokenizer.token_to_id("Label"), colon],
            [187, 187],  # '\n', '\n'
            [535],  # '\n\n'
            [2756],  # '\n\n\n'
        )
        return system_prompt, stop_tokens
    if re.search(r"falcon.*-instruct", checkpoint_name):
        # First line could be modified. AFAIK Falcon doesn't impose a specific system prompt
        # The instruction to not prefix its replies doesn't work always, but better than nothing
        system_prompt = "Do not prefix your replies with 'Bot: '\nUser: {prompt}\n"
        # I've also tried just "{prompt}\n" but the model seems to ramble more often
        stop_tokens = (
            [tokenizer.eos_id],
            # the model rarely emits the eos token and instead outputs newlines, but we cannot use them
            # to stop or else things like code generation wouldn't work
            [tokenizer.token_to_id("User"), tokenizer.token_to_id(":")],
            [193, tokenizer.token_to_id("User")],  # 193: '\n'
        )
        return system_prompt, stop_tokens
    if re.search(r"vicuna|longchat", checkpoint_name):
        # https://github.com/lm-sys/FastChat/blob/main/docs/vicuna_weights_version.md#prompt-template
        system_prompt = (
            "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, "
            "detailed, and polite answers to the user's questions. USER: {prompt} ASSISTANT:"
        )
        stop_tokens = ([tokenizer.eos_id],)
        return system_prompt, stop_tokens

    # default format
    return "{prompt}", ([tokenizer.eos_id],)


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
