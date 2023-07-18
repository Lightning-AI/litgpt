from lit_gpt.model import GPT
from lit_gpt.config import Config
from lit_gpt.tokenizer import Tokenizer

from lightning_utilities.core.imports import RequirementCache

if not bool(RequirementCache("torch>=2.1.0dev")):
    raise ImportError(
        "Lit-GPT requires torch nightly (future torch 2.1). Please follow the installation instructions in the"
        " repository README.md"
    )
if not bool(RequirementCache("lightning>=2.1.0.dev0")):
    raise ImportError(
        "Lit-GPT requires Lightning nightly (future lightning 2.1). Please run:\n"
        " pip uninstall -y lightning; pip install -r requirements.txt"
    )


__all__ = ["GPT", "Config", "Tokenizer"]
