# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
from abc import abstractmethod
from typing import TYPE_CHECKING, Dict, List, Type, Tuple

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


# Maps prompt style names to PromptStyle classes
prompt_styles: Dict[str, Type[PromptStyle]] = {
    # Dataset-specific prompt styles
    "alpaca": Alpaca,
    "flan": FLAN,
    "longform": Longform,
    # TODO: Model-specific prompt styles
}
