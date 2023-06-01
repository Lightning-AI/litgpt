from dataclasses import dataclass
from typing import Optional, Any

from typing_extensions import Self

from lit_parrot.utils import find_multiple


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
    bias: bool = True
    multi_query: bool = False
    shared_attention_norm: bool = False

    def __post_init__(self):
        if self.padded_vocab_size is None:
            self.padded_vocab_size = find_multiple(self.vocab_size, self.padding_multiple)
        self.head_size = self.n_embd // self.n_head

    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> Self:
        conf_dict = configs[name].copy()
        conf_dict.update(kwargs)
        return cls(**conf_dict)


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
pythia = {
    # https://huggingface.co/EleutherAI/pythia-70m/blob/main/config.json
    "pythia-70m": dict(block_size=2048, n_layer=6, n_embd=512, n_head=8, padding_multiple=128),
    # https://huggingface.co/EleutherAI/pythia-160m/blob/main/config.json
    "pythia-160m": dict(block_size=2048, n_layer=12, n_embd=768, n_head=12, padding_multiple=128),
    # https://huggingface.co/EleutherAI/pythia-410m/blob/main/config.json
    "pythia-410m": dict(block_size=2048, n_layer=24, n_embd=1024, n_head=16, padding_multiple=128),
    # https://huggingface.co/EleutherAI/pythia-1b/blob/main/config.json
    "pythia-1b": dict(block_size=2048, n_layer=16, n_embd=2048, n_head=8, padding_multiple=128),
    # https://huggingface.co/EleutherAI/pythia-1.4b/blob/main/config.json
    "pythia-1.4b": dict(block_size=2048, n_layer=24, n_embd=2048, n_head=16, padding_multiple=128),
    # https://huggingface.co/EleutherAI/pythia-2.8b/blob/main/config.json
    "pythia-2.8b": dict(block_size=2048, n_layer=32, n_embd=2560, n_head=32, padding_multiple=128),
    # https://huggingface.co/EleutherAI/pythia-6.9b/blob/main/config.json
    "pythia-6.9b": dict(block_size=2048, n_layer=32, n_embd=4096, n_head=32, padding_multiple=256),
    # https://huggingface.co/EleutherAI/pythia-12b/blob/main/config.json
    "pythia-12b": dict(block_size=2048, n_layer=36, n_embd=5120, n_head=40, padding_multiple=512),
}
configs.update(pythia)
pythia_deduped = {f"{k}-deduped": pythia[k] for k in pythia}
configs.update(pythia_deduped)


####################################
# togethercomputer RedPajama INCITE
####################################
redpajama_incite = {
    # https://huggingface.co/togethercomputer/RedPajama-INCITE-Base-3B-v1/blob/main/config.json
    "RedPajama-INCITE-{}-3B-v1": dict(block_size=2048, n_layer=32, n_embd=2560, n_head=32, padding_multiple=256, rotary_percentage=1.0, parallel_residual=False),
    # https://huggingface.co/togethercomputer/RedPajama-INCITE-Base-7B-v0.1/blob/main/config.json
    "RedPajama-INCITE-{}-7B-v0.1": dict(block_size=2048, n_layer=32, n_embd=4096, n_head=32, padding_multiple=256, rotary_percentage=1.0, parallel_residual=False),
}
for k in list(redpajama_incite):
    for kind in ("Base", "Chat", "Instruct"):
        configs[k.format(kind)] = redpajama_incite[k]


#################
# TII UAE Falcon
#################
falcon = {
    # https://huggingface.co/tiiuae/falcon-7b/blob/main/config.json
    "falcon-7b{}": dict(
        block_size=2048,
        padded_vocab_size=65024,
        n_layer=32,
        n_head=71,
        n_embd=4544,
        rotary_percentage=1.0,
        parallel_residual=True,
        multi_query=True,
        bias=False,
        # this is not in the config, but in the original model implementation, only for this config
        shared_attention_norm=True
    ),
    # https://huggingface.co/tiiuae/falcon-40b/blob/main/config.json
    "falcon-40b{}": dict(
        block_size=2048,
        padded_vocab_size=65024,
        n_layer=80,
        n_head=128,
        n_embd=8192,
        rotary_percentage=1.0,
        parallel_residual=True,
        multi_query=True,  # strangely, the 40b config doesn't specify this, but it is
        bias=False,
        # FIXME: n_head_kv=8
    ),
}
for k in list(falcon):
    for kind in ("", "-instruct"):
        configs[k.format(kind)] = falcon[k]
