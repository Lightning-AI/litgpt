from lit_parrot.model import Parrot, build_rope_cache, apply_rope
from lit_parrot.config import Config
from lit_parrot.tokenizer import Tokenizer

from lightning_utilities.core.imports import RequirementCache

if not bool(RequirementCache("torch>=2.1.0dev")):
    raise ImportError(
        "Lit-Parrot requires torch nightly (future torch 2.1). Please follow the installation instructions in the"
        " repository README.md"
    )
