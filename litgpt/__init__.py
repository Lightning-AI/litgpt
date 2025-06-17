# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import importlib
import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from litgpt.api import LLM
    from litgpt.config import Config
    from litgpt.model import GPT
    from litgpt.prompts import PromptStyle
    from litgpt.tokenizer import Tokenizer


_LAZY_IMPORTS = {
    "LLM": "litgpt.api",
    "Config": "litgpt.config",
    "GPT": "litgpt.model",
    "PromptStyle": "litgpt.prompts",
    "Tokenizer": "litgpt.tokenizer",
    "api": "litgpt.api",
    "chat": "litgpt.chat",
    "config": "litgpt.config",
    "generate": "litgpt.generate",
    "lora": "litgpt.lora",
    "model": "litgpt.model",
    "prompts": "litgpt.prompts",
    "scripts": "litgpt.scripts",
    "tokenizer": "litgpt.tokenizer",
    "utils": "litgpt.utils",
}

def __getattr__(name):
    """
    Allow for lazy imports of all litgpt submodules,
    as well as some selected top-level attributes.
    """
    if name in _LAZY_IMPORTS:
        module_name = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_name)
        if not module_name.endswith(name):
            return getattr(module, name)
        return module

    # If the attribute is not found, raise an AttributeError
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def __dir__():
    """
    Return a list of all attributes in the litgpt module.
    """
    return sorted(list(_LAZY_IMPORTS.keys()) + list(globals().keys()))

# Suppress excessive warnings, see https://github.com/pytorch/pytorch/issues/111632
pattern = re.compile(".*Profiler function .* will be ignored")
logging.getLogger("torch._dynamo.variables.torch").addFilter(lambda record: not pattern.search(record.getMessage()))

# Avoid printing state-dict profiling output at the WARNING level when saving a checkpoint
logging.getLogger("torch.distributed.fsdp._optim_utils").disabled = True
logging.getLogger("torch.distributed.fsdp._debug_utils").disabled = True

__all__ = ["LLM", "GPT", "Config", "PromptStyle", "Tokenizer"]
