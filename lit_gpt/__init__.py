from lightning_utilities.core.imports import RequirementCache, package_available

if not bool(RequirementCache("torch>=2.1.0dev")):
    raise ImportError(
        "Lit-GPT requires torch nightly (future torch 2.1). Please follow the installation instructions in the"
        " repository README.md"
    )
if (
    # environments like colab often come with these pre-installed. since there's a 1:1 mapping of torch version to
    # these, their nightly would need to be installed too. even though lit-gpt doesn't import or use them, they get
    # imported by the dependency chain of lightning
    (package_available("torchaudio") and RequirementCache("torchaudio<2.1.0"))
    or (package_available("torchvision") and RequirementCache("torchvision<0.16.0"))
    or (package_available("torchtext") and RequirementCache("torchtext<0.16.0"))
):
    raise ImportError(
        "You are running in an environment that already had torchvision, torchtext, or torchaudio installed. These will"
        " not work with torch nightly. Since Lit-GPT doesn't use them, please run:\n pip uninstall -y torchaudio"
        " torchtext torchvision"
    )
_LIGHTNING_AVAILABLE = RequirementCache("lightning>=2.1.0.dev0")
if not bool(_LIGHTNING_AVAILABLE):
    raise ImportError(
        "Lit-GPT requires Lightning nightly (future lightning 2.1). Please run:\n"
        f" pip uninstall -y lightning; pip install -r requirements.txt\n{str(_LIGHTNING_AVAILABLE)}"
    )


from lit_gpt.model import GPT
from lit_gpt.config import Config
from lit_gpt.tokenizer import Tokenizer

__all__ = ["GPT", "Config", "Tokenizer"]
