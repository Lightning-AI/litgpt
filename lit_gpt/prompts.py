# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
from abc import abstractmethod
from typing import Dict, List, Type

from lit_gpt import Tokenizer


class PromptStyle:
    @abstractmethod
    def apply(self, prompt: str, **kwargs: str) -> str:
        return prompt

    def stop_tokens(self, tokenizer: Tokenizer) -> List[int]:
        return [tokenizer.eos_id]

    @classmethod
    def from_name(cls, name: str) -> "PromptStyle":
        return prompt_styles[name]()


class Alpaca(PromptStyle):
    def apply(self, prompt: str, **kwargs: str):
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


class Stability(PromptStyle):
    def apply(self, prompt: str, **kwargs: str):
        return (
            "<|SYSTEM|># StableLM Tuned (Alpha version)\n- StableLM is a helpful and harmless open-source AI language"
            " model developed by StabilityAI.\n- StableLM is excited to be able to help the user, but will refuse to do"
            " anything that could be considered harmful to the user.\n- StableLM is more than just an information"
            " source, StableLM is also able to write poetry, short stories, and make jokes.\n- StableLM will refuse to"
            f" participate in anything that could harm a human.<|USER|>{prompt}<|ASSISTANT|>"
        )

    def stop_tokens(self, tokenizer: Tokenizer) -> Tuple[List[int]]:
        # TODO: Why a tuple?
        return (
            [tokenizer.eos_id],
            [tokenizer.token_to_id("<|SYSTEM|>")],
            [tokenizer.token_to_id("<|ASSISTANT|>")],
            [tokenizer.token_to_id("<|USER|>")],
        )


prompt_styles: Dict[str, Type[PromptStyle]] = {
    "alpaca": Alpaca,
    "stability": Stability,
}



model_name_to_prompt_style = {
}