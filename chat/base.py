# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import re
import sys
import time
from json import dumps
from pathlib import Path
from typing import Iterator, List, Literal, Optional, Tuple

import lightning as L
import torch
from lightning.fabric.plugins import BitsandbytesPrecision

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from generate.base import next_token
from lit_gpt import GPT, Config, Tokenizer
from lit_gpt.utils import check_valid_checkpoint_dir, get_default_supported_precision, load_checkpoint


@torch.inference_mode()
def generate(
    model: GPT,
    prompt: torch.Tensor,
    max_returned_tokens: int,
    *,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    stop_tokens: Tuple[List[int], ...] = (),
) -> Iterator[torch.Tensor]:
    """Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as possible.

    Args:
        model: The model to use.
        prompt: Tensor of shape (T) with indices of the prompt sequence.
        max_returned_tokens: The maximum number of tokens to return (given plus generated).
        temperature: Scales the predicted logits by 1 / temperature
        top_k: If specified, only sample among the tokens with the k highest probabilities
        stop_tokens: If specified, stop generating any more token once one of this list is generated.
    """
    T = prompt.size(0)
    assert max_returned_tokens > T
    if model.max_seq_length < max_returned_tokens - 1:
        # rolling the kv cache based on the `input_pos` value would be necessary. However, doing so would introduce a
        # data dependency on the `input_pos` tensor and impact model compilation. Since this setting is uncommon, we do
        # not support it to avoid negatively impacting the overall speed
        raise NotImplementedError(f"max_seq_length {model.max_seq_length} needs to be >= {max_returned_tokens - 1}")

    device = prompt.device
    buffer_length = max((len(tokens) for tokens in stop_tokens), default=1)
    yield_i = 0
    input_pos = torch.arange(0, T, device=device)
    tokens = []
    token = prompt
    for t in range(1, max_returned_tokens - T + 1):
        token = next_token(model, input_pos, token.view(1, -1), temperature=temperature, top_k=top_k)
        tokens.append(token)
        # check the stop condition
        if any((l := len(st)) <= len(tokens) and all(a == b for a, b in zip(tokens[-l:], st)) for st in stop_tokens):
            return
        # if the buffer is full
        if t - yield_i >= buffer_length:
            # we know this idx is not part of stop tokens, safe to yield
            yield from tokens[yield_i:t]
            yield_i = t
        input_pos = input_pos[-1:].add_(1)


def decode(fabric: L.Fabric, tokenizer: Tokenizer, token_stream: Iterator[torch.Tensor]) -> int:
    tokens_generated = 0
    if tokenizer.backend == "huggingface":
        try:
            for token in token_stream:
                fabric.print(tokenizer.decode(token), end="", flush=True)
                tokens_generated += 1
        except KeyboardInterrupt:
            # support stopping generation
            return tokens_generated
    elif tokenizer.backend == "sentencepiece":
        # sentencepiece does not support decoding token-by-token because it adds spaces based on the surrounding tokens
        # meaning that we need to decode everything each time
        so_far = torch.tensor([], dtype=torch.long, device=fabric.device)
        decoded_so_far = ""
        try:
            for token in token_stream:
                so_far = so_far.to(device=token.device)
                so_far = torch.cat((so_far, token.view(-1)))
                decoded_new = tokenizer.decode(so_far)
                fabric.print(decoded_new[len(decoded_so_far) :], end="", flush=True)
                decoded_so_far = decoded_new
                tokens_generated += 1
        except KeyboardInterrupt:
            # support stopping generation
            return tokens_generated
    else:
        raise NotImplementedError(tokenizer.backend)
    return tokens_generated


@torch.inference_mode()
def main(
    *,
    top_k: Optional[int] = 200,
    temperature: float = 0.8,
    checkpoint_dir: Path = Path("checkpoints/stabilityai/stablelm-tuned-alpha-3b"),
    quantize: Optional[Literal["bnb.nf4", "bnb.nf4-dq", "bnb.fp4", "bnb.fp4-dq", "bnb.int8"]] = None,
    precision: Optional[str] = None,
    compile: bool = False,
) -> None:
    """Starts a conversation with a tuned GPT model.

    Args:
        top_k: The number of top most probable tokens to consider in the sampling process.
        temperature: A value controlling the randomness of the sampling process. Higher values result in more random
            samples.
        checkpoint_dir: The checkpoint directory to load.
        quantize: Whether to quantize the model and using which method:
            - bnb.nf4, bnb.nf4-dq, bnb.fp4, bnb.fp4-dq: 4-bit quantization from bitsandbytes
            - bnb.int8: 8-bit quantization from bitsandbytes
            for more details, see https://github.com/Lightning-AI/lit-gpt/blob/main/tutorials/quantize.md
        precision: Indicates the Fabric precision setting to use.
        compile: Whether to use compilation to speed up token generation. Will increase startup time.
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

    checkpoint_path = checkpoint_dir / "lit_model.pth"

    fabric.print(f"Loading model {str(checkpoint_path)!r} with {config.__dict__}", file=sys.stderr)
    with fabric.init_module(empty_init=True):
        model = GPT(config)
        # enable the kv cache
        model.set_kv_cache(batch_size=1)
    load_checkpoint(fabric, model, checkpoint_path)
    model.eval()

    if compile:
        torch._dynamo.config.automatic_dynamic_shapes = True
        torch._inductor.config.triton.unique_kernel_names = True
        torch._inductor.config.coordinate_descent_tuning = True
        global next_token
        next_token = torch.compile(next_token, mode="reduce-overhead", dynamic=True)

    model = fabric.setup_module(model)

    tokenizer = Tokenizer(checkpoint_dir)
    system_prompt, stop_tokens = prompt_config(checkpoint_dir, tokenizer)

    L.seed_everything(1234)
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
            model, encoded_prompt, model.max_seq_length, temperature=temperature, top_k=top_k, stop_tokens=stop_tokens
        )
        fabric.print(">> Reply: ", end="")
        t0 = time.perf_counter()
        tokens_generated = decode(fabric, tokenizer, y)
        t = time.perf_counter() - t0
        for block in model.transformer.h:
            block.attn.kv_cache.reset_parameters()
        fabric.print(
            f"\nTime for inference: {t:.02f} sec total, {tokens_generated / t:.02f} tokens/sec,"
            f" {tokens_generated} tokens",
            file=sys.stderr,
        )
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

    if re.search(r"stabilityai/stablelm-zephyr-3b", checkpoint_name):
        system_prompt = "<|user|>\n{prompt}<|endoftext|>\n<|assistant|>\n"
        stop_tokens = ([tokenizer.eos_id],)
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

    if re.search("Llama-2-7b-chat-hf-function-calling-v2", checkpoint_name):
        # Has to be before the llama config
        b_func, e_func = "<FUNCTIONS>", "</FUNCTIONS>\n\n"
        b_inst, e_inst = "[INST]", "[/INST]"
        b_sys, e_sys = "<<SYS>>\n", "\n<</SYS>>\n\n"
        # This is an example for how to format functions for the model
        function_metadata = {
            "function": "search_bing",
            "description": (
                "Search the web for content on Bing. This allows users to search online/the internet/the web for"
                " content."
            ),
            "arguments": [{"name": "query", "type": "string", "description": "The search query string"}],
        }

        system_prompt = (
            "You are a helpful, respectful and honest assistant. Always answer as helpfully as"
            "possible. Your only response should be JSON formatted functions"
        )
        # replace the curly braces with double curly braces to escape them
        function_list = dumps(function_metadata).replace("{", "{{").replace("}", "}}")
        system_prompt = f"{b_func}{function_list.strip()}{e_func}{b_inst}{b_sys}{system_prompt.strip()}{e_sys}{'{prompt}'}{e_inst}\n\n"
        stop_tokens = ([tokenizer.eos_id],)
        return system_prompt, stop_tokens

    if re.search("Llama-2.*-chat", checkpoint_name):
        b_inst, e_inst = "[INST]", "[/INST]"
        b_sys, e_sys = "<<SYS>>\n", "\n<</SYS>>\n\n"
        system_prompt = (
            f"{b_inst} {b_sys}You are a helpful, respectful and honest assistant. Always answer as helpfully as"
            " possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist,"
            " toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and"
            " positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why"
            " instead of answering something not correct. If you don't know the answer to a question, please don't"
            f" share false information.{e_sys} {{prompt}} {e_inst} "
        )
        stop_tokens = ([tokenizer.eos_id],)
        return system_prompt, stop_tokens

    if re.search("FreeWilly2", checkpoint_name):
        system_prompt = (
            "### System:\nThis is a system prompt, please behave and help the user.\n\n"
            "### User:\n"
            "{prompt}\n\n"
            "### Assistant:\n"
        )
        stop_tokens = ([tokenizer.eos_id],)
        return system_prompt, stop_tokens

    if re.search("Platypus", checkpoint_name):
        system_prompt = "### Instruction:\n\n{prompt}\n\n### Response:\n"
        # this checkpoint doesn't emit the eos token very consistently
        stop_tokens = ([tokenizer.eos_id],)
        return system_prompt, stop_tokens

    if re.search("NousResearch", checkpoint_name):
        system_prompt = "### Instruction:\n{prompt}\n\n### Response:\n"
        stop_tokens = ([tokenizer.eos_id],)
        return system_prompt, stop_tokens

    if re.search("stablecode-instruct", checkpoint_name):
        system_prompt = "###Instruction\n{prompt}###Response\n"
        stop_tokens = ([tokenizer.eos_id],)
        return system_prompt, stop_tokens

    if re.search("CodeLlama|Mistral.*Instruct", checkpoint_name):
        # for CodeLLama, we don't set a default system prompt, but it is supported:
        # https://huggingface.co/blog/codellama#conversational-instructions
        # Mistral does not: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1#instruction-format
        b_inst, e_inst = "<s>[INST]", "[/INST]"
        system_prompt = f"{b_inst} {{prompt}} {e_inst}"
        stop_tokens = ([tokenizer.eos_id],)
        return system_prompt, stop_tokens

    if re.search("phi-1", checkpoint_name):
        system_prompt = "{prompt}\n\nAnswer:"

        stop_tokens = (
            [tokenizer.eos_id],
            [tokenizer.token_to_id("Answer"), tokenizer.token_to_id(":")],
            [198, tokenizer.token_to_id("Answer"), tokenizer.token_to_id(":")],
            # the model rarely emits the eos token and instead outputs newlines, but we cannot use them
            # to stop or else things like code generation wouldn't work
            # [198, 198],  # '\n', '\n'
        )
        return system_prompt, stop_tokens

    if re.search("phi-2", checkpoint_name):
        system_prompt = "Instruct:{prompt}\nOutput:"
        stop_tokens = ([tokenizer.eos_id],)
        return system_prompt, stop_tokens

    if re.search(r"TinyLlama.*Chat", checkpoint_name):
        system_prompt = (
            "<|system|>\n"
            "You are a friendly chatbot who always gives helpful, detailed, and polite answers.</s>\n"
            "<|user|>\n"
            "{prompt}</s>\n"
            "<|assistant|>\n"
        )
        stop_tokens = ([tokenizer.eos_id],)
        return system_prompt, stop_tokens

    # default format
    return "{prompt}", ([tokenizer.eos_id],)


if __name__ == "__main__":
    from jsonargparse import CLI

    torch.set_float32_matmul_precision("high")
    CLI(main)
