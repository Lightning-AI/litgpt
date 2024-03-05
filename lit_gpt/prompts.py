# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
from abc import abstractmethod
from json import dumps
from typing import TYPE_CHECKING, Dict, List, Type, Tuple, Union

import lit_gpt.config
from lit_gpt.config import Config

if TYPE_CHECKING:
    from lit_gpt import Tokenizer


class PromptStyle:
    """Base interface for prompt styles."""
    @abstractmethod
    def apply(self, prompt: str, **kwargs: str) -> str:
        return prompt

    def stop_tokens(self, tokenizer: "Tokenizer") -> Tuple[List[int], ...]:
        return ([tokenizer.eos_id], )

    @classmethod
    def from_name(cls, name: str) -> "PromptStyle":
        return prompt_styles[name]()

    @classmethod
    def from_config(cls, config: Config) -> "PromptStyle":
        if config.name not in model_name_to_prompt_style:
            # TODO: Warning here? Error?
            return Default()
        return prompt_styles[model_name_to_prompt_style[config.name]]()


class Default(PromptStyle):
    def apply(self, prompt: str, **kwargs: str) -> str:
        return prompt

    def stop_tokens(self, tokenizer: "Tokenizer") -> Tuple[List[int], ...]:
        return ([tokenizer.eos_id], )


class Alpaca(PromptStyle):
    def apply(self, prompt: str, **kwargs: str) -> str:
        if kwargs.get("input"):
            return (
                "Below is an instruction that describes a task, paired with an input that provides further context. "
                "Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{prompt}\n\n### Input:\n{kwargs['input']}\n\n### Response:\n"
            )
        return (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{prompt}\n\n### Response:\n"
        )


class FLAN(PromptStyle):
    def apply(self, prompt: str, **kwargs: str) -> str:
        return (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{prompt}\n\n### Response:\n"
        )


class Longform(PromptStyle):
    def apply(self, prompt: str, **kwargs: str) -> str:
        return (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{prompt}\n\n### Response:\n"
        )


class StableLMAlpha(PromptStyle):
    def apply(self, prompt: str, **kwargs: str) -> str:
        return (
            "<|SYSTEM|># StableLM Tuned (Alpha version)\n- StableLM is a helpful and harmless open-source AI language"
            " model developed by StabilityAI.\n- StableLM is excited to be able to help the user, but will refuse to do"
            " anything that could be considered harmful to the user.\n- StableLM is more than just an information"
            " source, StableLM is also able to write poetry, short stories, and make jokes.\n- StableLM will refuse to"
            f" participate in anything that could harm a human.<|USER|>{prompt}<|ASSISTANT|>"
        )

    def stop_tokens(self, tokenizer: "Tokenizer") -> Tuple[List[int], ...]:
        return (
            [tokenizer.eos_id],
            [tokenizer.token_to_id("<|SYSTEM|>")],
            [tokenizer.token_to_id("<|ASSISTANT|>")],
            [tokenizer.token_to_id("<|USER|>")],
        )


class StableLMZephyr(PromptStyle):
    def apply(self, prompt: str, **kwargs: str) -> str:
        return f"<|user|>\n{prompt}<|endoftext|>\n<|assistant|>\n"


class TogetherComputerChat(PromptStyle):
    def apply(self, prompt: str, **kwargs: str) -> str:
        return f"<human>: {prompt}\n<bot>:"

    def stop_tokens(self, tokenizer: "Tokenizer") -> Tuple[List[int], ...]:
        lt, gt = tokenizer.token_to_id("<"), tokenizer.token_to_id(">:")
        return (
            [tokenizer.eos_id],
            # annoyingly, there's no single stop token for these
            [lt, tokenizer.token_to_id("human"), gt],
            [lt, tokenizer.token_to_id("bot"), gt],
        )


class TogetherComputerInstruct(PromptStyle):
    def apply(self, prompt: str, **kwargs: str) -> str:
        return f"Q: {prompt}\nA:"

    def stop_tokens(self, tokenizer: "Tokenizer") -> Tuple[List[int], ...]:
        colon = tokenizer.token_to_id(":")
        return (
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


class Falcon(PromptStyle):
    def apply(self, prompt: str, **kwargs: str) -> str:
        # First line could be modified. AFAIK Falcon doesn't impose a specific system prompt
        # The instruction to not prefix its replies doesn't work always, but better than nothing
        # I've also tried just "{prompt}\n" but the model seems to ramble more often
        return f"Do not prefix your replies with 'Bot: '\nUser: {prompt}\n"

    def stop_tokens(self, tokenizer: "Tokenizer") -> Tuple[List[int], ...]:
        return (
            [tokenizer.eos_id],
            # the model rarely emits the eos token and instead outputs newlines, but we cannot use them
            # to stop or else things like code generation wouldn't work
            [tokenizer.token_to_id("User"), tokenizer.token_to_id(":")],
            [193, tokenizer.token_to_id("User")],  # 193: '\n'
        )


class Vicuna(PromptStyle):
    def apply(self, prompt: str, **kwargs: str) -> str:
        # https://github.com/lm-sys/FastChat/blob/main/docs/vicuna_weights_version.md#prompt-template
        return (
            "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, "
            f"detailed, and polite answers to the user's questions. USER: {prompt} ASSISTANT:"
        )


class Llama2FunctionCalling(PromptStyle):
    def apply(self, prompt: str, **kwargs: str) -> str:
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
        return (
            f"{b_func}{function_list.strip()}{e_func}{b_inst}{b_sys}"
            f"{system_prompt.strip()}"
            f"{e_sys}{prompt}{e_inst}\n\n"
        )


class Llama2(PromptStyle):
    def apply(self, prompt: str, **kwargs: str) -> str:
        b_inst, e_inst = "[INST]", "[/INST]"
        b_sys, e_sys = "<<SYS>>\n", "\n<</SYS>>\n\n"
        return (
            f"{b_inst} {b_sys}You are a helpful, respectful and honest assistant. Always answer as helpfully as"
            " possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist,"
            " toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and"
            " positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why"
            " instead of answering something not correct. If you don't know the answer to a question, please don't"
            f" share false information.{e_sys} {prompt} {e_inst} "
        )


class FreeWilly2(PromptStyle):
    def apply(self, prompt: str, **kwargs: str) -> str:
        return (
            "### System:\nThis is a system prompt, please behave and help the user.\n\n"
            "### User:\n"
            f"{prompt}\n\n"
            "### Assistant:\n"
        )


class Platypus(PromptStyle):
    def apply(self, prompt: str, **kwargs: str) -> str:
        return f"### Instruction:\n\n{prompt}\n\n### Response:\n"


class NousResearch(PromptStyle):
    def apply(self, prompt: str, **kwargs: str) -> str:
        return f"### Instruction:\n{prompt}\n\n### Response:\n"


class StableCode(PromptStyle):
    def apply(self, prompt: str, **kwargs: str) -> str:
        return f"###Instruction\n{prompt}###Response\n"


class CodeLlama(PromptStyle):
    def apply(self, prompt: str, **kwargs: str) -> str:
        # for CodeLLama, we don't set a default system prompt, but it is supported:
        # https://huggingface.co/blog/codellama#conversational-instructions
        # Mistral does not: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1#instruction-format
        b_inst, e_inst = "<s>[INST]", "[/INST]"
        return f"{b_inst} {prompt} {e_inst}"


class Phi1(PromptStyle):
    def apply(self, prompt: str, **kwargs: str) -> str:
        return f"{prompt}\n\nAnswer:"

    def stop_tokens(self, tokenizer: "Tokenizer") -> Tuple[List[int], ...]:
        return (
            [tokenizer.eos_id],
            [tokenizer.token_to_id("Answer"), tokenizer.token_to_id(":")],
            [198, tokenizer.token_to_id("Answer"), tokenizer.token_to_id(":")],
            # the model rarely emits the eos token and instead outputs newlines, but we cannot use them
            # to stop or else things like code generation wouldn't work
            # [198, 198],  # '\n', '\n'
        )


class Phi2(PromptStyle):
    def apply(self, prompt: str, **kwargs: str) -> str:
        return f"Instruct:{prompt}\nOutput:"


class TinyLlama(PromptStyle):
    def apply(self, prompt: str, **kwargs: str) -> str:
        return (
            "<|system|>\n"
            "You are a friendly chatbot who always gives helpful, detailed, and polite answers.</s>\n"
            "<|user|>\n"
            f"{prompt}</s>\n"
            "<|assistant|>\n"
        )


class Gemma(PromptStyle):
    def apply(self, prompt: str, **kwargs: str) -> str:
        return f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"


# Maps prompt style names to PromptStyle classes
prompt_styles: Dict[str, Type[PromptStyle]] = {
    # Dataset-specific prompt styles
    "alpaca": Alpaca,
    "flan": FLAN,
    "longform": Longform,
    # Model-specific prompt styles
    "stablelm-alpha": StableLMAlpha,
    "stablelm-zephyr": StableLMZephyr,
    "togethercomputer-chat": TogetherComputerChat,
    "togethercomputer-instruct": TogetherComputerInstruct,
    "falcon": Falcon,
    "vicuna": Vicuna,
    "llama2-function-calling": Llama2FunctionCalling,
    "llama2": Llama2,
    "freewilly2": FreeWilly2,
    "platypus": Platypus,
    "nous-research": NousResearch,
    "stablecode": StableCode,
    "codellama": CodeLlama,
    "phi-1": Phi1,
    "phi-2": Phi2,
    "tinyllama": TinyLlama,
    "gemma": Gemma,
}

# TODO: Double check the names are all Config.name and not Config.hf_config.name
# Maps HF model names to prompt style names
model_name_to_prompt_style = {
    "stablelm-tuned-alpha-3b": "stablelm-alpha",
    "stablelm-tuned-alpha-7b": "stablelm-alpha",
    "stablelm-zephyr-3b": "stablelm-zephyr",
    "falcon-7b-instruct": "falcon",
    "falcon-40b-instruct": "falcon",
    "vicuna-7b-v1.3": "vicuna",
    "vicuna-13b-v1.3": "vicuna",
    "vicuna-33b-v1.3": "vicuna",
    "vicuna-7b-v1.5": "vicuna",
    "vicuna-7b-v1.5-16k": "vicuna",
    "vicuna-13b-v1.5": "vicuna",
    "vicuna-13b-v1.5-16k": "vicuna",
    "longchat-7b-16k": "vicuna",
    "longchat-13b-16k": "vicuna",
    "Nous-Hermes-llama-2-7b": "nous-research",
    "Nous-Hermes-13b": "nous-research",
    "Nous-Hermes-Llama2-13b": "nous-research",
    "Llama-2-7b-chat-hf": "llama2",
    "Llama-2-13b-chat-hf": "llama2",
    "Llama-2-70b-chat-hf": "llama2",
    "gemma-2b-it": "gemma",
    "gemma-7b-it": "gemma",
    "FreeWilly2": "freewilly2",
    "CodeLlama-7b-Instruct-hf": "codellama",
    "CodeLlama-13b-Instruct-hf": "codellama",
    "CodeLlama-34b-Instruct-hf": "codellama",
    "CodeLlama-70b-Instruct-hf": "codellama",
    "stablecode-instruct-alpha-3b": "stablecode",
    "phi-1_5": "phi-1",
    "phi-2": "phi-2",
    "Mistral-7B-Instruct-v0.1": "codellama",
    "Mistral-7B-Instruct-v0.2": "codellama",
    "Mixtral-8x7B-Instruct-v0.1": "codellama",
    "TinyLlama-1.1B-chat": "tinyllama",
    "TinyLlama-1.1B-Chat-v1.0": "tinyllama",
    "Llama-2-7b-chat-hf-function-calling-v2": "llama2-function-calling",
}

for template in ("RedPajama-INCITE-{}-3B-v1", "RedPajama-INCITE-7B-{}", "RedPajama-INCITE-{}-7B-v0.1"):
    model_name_to_prompt_style[template.format("Chat")] = "togethercomputer-chat"
    model_name_to_prompt_style[template.format("Instruct")] = "togethercomputer-instruct"

for c in lit_gpt.config.platypus:
    model_name_to_prompt_style[c["name"]] = "platypus"
