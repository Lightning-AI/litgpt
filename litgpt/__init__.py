# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import logging
import re
import warnings

from litgpt.model import GPT  # needs to be imported before config
from litgpt.config import Config
from litgpt.prompts import PromptStyle
from litgpt.tokenizer import Tokenizer
from litgpt.utils import check_library_version

# Suppress excessive warnings, see https://github.com/pytorch/pytorch/issues/111632
pattern = re.compile(".*Profiler function .* will be ignored")
logging.getLogger("torch._dynamo.variables.torch").addFilter(lambda record: not pattern.search(record.getMessage()))

# Avoid printing state-dict profiling output at the WARNING level when saving a checkpoint
logging.getLogger("torch.distributed.fsdp._optim_utils").disabled = True
logging.getLogger("torch.distributed.fsdp._debug_utils").disabled = True

supported, version = check_library_version("bitsandbytes", "0.42.0")
if version and not supported:
    warnings.warn(
        f"You have bitsandbytes v{version} installed but "
        "LitGPT only supports bitsandbytes v0.42.0. "
        "This may result in errors when using quantization."
    )

__all__ = ["GPT", "Config", "PromptStyle", "Tokenizer"]
