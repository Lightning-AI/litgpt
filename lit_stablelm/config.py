from dataclasses import dataclass
from typing import Optional

from typing_extensions import Self

from lit_stablelm.utils import find_multiple


@dataclass
class Config:
    block_size: int = 4096
    vocab_size: int = 50254
    padding_multiple: int = 512
    padded_vocab_size: Optional[int] = None
    n_layer: int = 16
    n_head: int = 32
    n_embd: int = 4096
    rotary_percentage: float = 0.25
    parallel_residual: bool = True

    def __post_init__(self):
        if self.padded_vocab_size is None:
            self.padded_vocab_size = find_multiple(self.vocab_size, self.padding_multiple)

    @classmethod
    def from_name(cls, name: str) -> Self:
        return cls(**configs[name])


# fmt: off

########################
# Stability AI StableLM
########################
configs = {
    # https://huggingface.co/stabilityai/stablelm-base-alpha-3b/blob/main/config.json
    "stablelm-base-alpha-3b": dict(padding_multiple=512),
    # https://huggingface.co/stabilityai/stablelm-base-alpha-7b/blob/main/config.json
    "stablelm-base-alpha-7b": dict(n_head=48, n_embd=6144, padding_multiple=256),
    # https://huggingface.co/stabilityai/stablelm-tuned-alpha-3b/blob/main/config.json
    "stablelm-tuned-alpha-3b": dict(n_head=32, padding_multiple=512),
    # https://huggingface.co/stabilityai/stablelm-tuned-alpha-7b/blob/main/config.json
    "stablelm-tuned-alpha-7b": dict(n_head=48, n_embd=6144, padding_multiple=256),
}

####################
# EleutherAI Pythia
####################
configs.update({
    # https://huggingface.co/EleutherAI/pythia-70m/blob/main/config.json
    "pythia-70m": dict(block_size=2048, n_layer=6, n_embd=512, n_head=8, padding_multiple=128),
    # https://huggingface.co/EleutherAI/pythia-160m/blob/main/config.json
    "pythia-160m": dict(block_size=2048, n_layer=12, n_embd=768, n_head=12, padding_multiple=128),
    # https://huggingface.co/EleutherAI/pythia-410m/blob/main/config.json
    "pythia-410m": dict(block_size=2048, n_layer=24, n_embd=1024, n_head=16, padding_multiple=128),
    # https://huggingface.co/EleutherAI/pythia-1b/blob/main/config.json
    "pythia-1b": dict(block_size=2048, n_layer=16, n_embd=8192, n_head=8, padding_multiple=128),
    # https://huggingface.co/EleutherAI/pythia-1.4b/blob/main/config.json
    "pythia-1.4b": dict(block_size=2048, n_layer=24, n_embd=8192, n_head=16, padding_multiple=128),
    # https://huggingface.co/EleutherAI/pythia-2.8b/blob/main/config.json
    "pythia-2.8b": dict(block_size=2048, n_layer=32, n_embd=10240, n_head=32, padding_multiple=128),
    # https://huggingface.co/EleutherAI/pythia-6.9b/blob/main/config.json
    "pythia-6.9b": dict(block_size=2048, n_layer=32, n_embd=16384, n_head=32, padding_multiple=128),
    # https://huggingface.co/EleutherAI/pythia-12b/blob/main/config.json
    "pythia-12b": dict(block_size=2048, n_layer=36, n_embd=20480, n_head=40, padding_multiple=128),
})
for k in list(configs):
    if k.startswith("pythia"):
        configs[k + "-deduped"] = configs[k]


####################################
# togethercomputer RedPajama INCITE
####################################
configs.update({
    # https://huggingface.co/togethercomputer/RedPajama-INCITE-Base-3B-v1/blob/main/config.json
    "RedPajama-INCITE-Base-3B-v1": dict(block_size=2048, n_layer=32, n_embd=2560, n_head=32, padding_multiple=256, rotary_percentage=1.0, parallel_residual=False),
    # https://huggingface.co/togethercomputer/RedPajama-INCITE-Chat-3B-v1/blob/main/config.json
    "RedPajama-INCITE-Chat-3B-v1": dict(block_size=2048, n_layer=32, n_embd=2560, n_head=32, padding_multiple=256, rotary_percentage=1.0, parallel_residual=False),
    # https://huggingface.co/togethercomputer/RedPajama-INCITE-Instruct-3B-v1/blob/main/config.json
    "RedPajama-INCITE-Instruct-3B-v1": dict(block_size=2048, n_layer=32, n_embd=2560, n_head=32, padding_multiple=256, rotary_percentage=1.0, parallel_residual=False),
    # https://huggingface.co/togethercomputer/RedPajama-INCITE-Base-7B-v0.1/blob/main/config.json
    "RedPajama-INCITE-Base-7B-v0.1": dict(block_size=2048, n_layer=32, n_embd=4096, n_head=32, padding_multiple=256, rotary_percentage=1.0, parallel_residual=False),
    # https://huggingface.co/togethercomputer/RedPajama-INCITE-Chat-7B-v0.1/blob/main/config.json
    "RedPajama-INCITE-Chat-7B-v0.1": dict(block_size=2048, n_layer=32, n_embd=4096, n_head=32, padding_multiple=256, rotary_percentage=1.0, parallel_residual=False),
    # https://huggingface.co/togethercomputer/RedPajama-INCITE-Instruct-7B-v0.1/blob/main/config.json
    "RedPajama-INCITE-Instruct-7B-v0.1": dict(block_size=2048, n_layer=32, n_embd=4096, n_head=32, padding_multiple=256, rotary_percentage=1.0, parallel_residual=False),
})
