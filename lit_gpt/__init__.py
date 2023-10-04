from lit_gpt.model import GPT
from lit_gpt.config import Config
from lit_gpt.tokenizer import Tokenizer

from lightning_utilities.core.imports import RequirementCache

_LIGHTNING_AVAILABLE = RequirementCache("lightning>=2.1.0.dev0")
if not bool(_LIGHTNING_AVAILABLE):
    raise ImportError(
        "Lit-GPT requires lightning==2.1. Please run:\n"
        f" pip uninstall -y lightning; pip install -r requirements.txt\n{str(_LIGHTNING_AVAILABLE)}"
    )


__all__ = ["GPT", "Config", "Tokenizer"]
