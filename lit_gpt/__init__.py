from lit_gpt.model import GPT, build_rope_cache, apply_rope
from lit_gpt.config import Config
from lit_gpt.tokenizer import Tokenizer

from lightning_utilities.core.imports import RequirementCache

if not bool(RequirementCache("torch>=2.1.0dev")):
    raise ImportError(
        "Lit-GPT requires torch nightly (future torch 2.1). Please follow the installation instructions in the"
        " repository README.md"
    )
