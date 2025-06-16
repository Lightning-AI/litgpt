# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from litgpt.api import LLM
    from litgpt.config import Config
    from litgpt.model import GPT
    from litgpt.prompts import PromptStyle
    from litgpt.tokenizer import Tokenizer


def __getattr__(name):
    if name == "LLM":
        from litgpt.api import LLM

        return LLM
    elif name == "Config":
        from litgpt.config import Config

        return Config
    elif name == "GPT":
        from litgpt.model import GPT

        return GPT
    elif name == "PromptStyle":
        from litgpt.prompts import PromptStyle

        return PromptStyle
    elif name == "Tokenizer":
        from litgpt.tokenizer import Tokenizer

        return Tokenizer

    # Handle the modules that used to be available immediately after the top-level import
    elif name == "api":
        import litgpt.api as api

        return api
    elif name == "config":
        import litgpt.config as config

        return config
    elif name == "model":
        import litgpt.model as model

        return model
    elif name == "prompts":
        import litgpt.prompts as prompts

        return prompts
    elif name == "tokenizer":
        import litgpt.tokenizer as tokenizer

        return tokenizer

    # If the attribute is not found, raise an AttributeError
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Suppress excessive warnings, see https://github.com/pytorch/pytorch/issues/111632
pattern = re.compile(".*Profiler function .* will be ignored")
logging.getLogger("torch._dynamo.variables.torch").addFilter(lambda record: not pattern.search(record.getMessage()))

# Avoid printing state-dict profiling output at the WARNING level when saving a checkpoint
logging.getLogger("torch.distributed.fsdp._optim_utils").disabled = True
logging.getLogger("torch.distributed.fsdp._debug_utils").disabled = True

__all__ = ["LLM", "GPT", "Config", "PromptStyle", "Tokenizer"]
