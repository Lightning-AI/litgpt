from dataclasses import dataclass
from typing import Optional, Any

from typing_extensions import Self

from lit_gpt.utils import find_multiple


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
    # to use multi-head attention (MHA), set this to `n_head` (default)
    # to use multi-query attention (MQA), set this to 1
    # to use grouped-query attention (GQA), set this to a value in between
    # Example with `n_head=4`
    # ┌───┐┌───┐┌───┐┌───┐     ┌───┐    ┌───┐             ┌───┐
    # │ v ││ v ││ v ││ v │     │ v │    │ v │             │ v │
    # └───┘└───┘└───┘└───┘     └───┘    └───┘             └───┘
    #   │    │    │    │         │        │                 │
    # ┌───┐┌───┐┌───┐┌───┐     ┌───┐    ┌───┐             ┌───┐
    # │ k ││ k ││ k ││ k │     │ k │    │ k │             │ k │
    # └───┘└───┘└───┘└───┘     └───┘    └───┘             └───┘
    #   │    │    │    │      ┌──┴──┐  ┌──┴──┐      ┌────┬──┴─┬────┐
    # ┌───┐┌───┐┌───┐┌───┐  ┌───┐┌───┐┌───┐┌───┐  ┌───┐┌───┐┌───┐┌───┐
    # │ q ││ q ││ q ││ q │  │ q ││ q ││ q ││ q │  │ q ││ q ││ q ││ q │
    # └───┘└───┘└───┘└───┘  └───┘└───┘└───┘└───┘  └───┘└───┘└───┘└───┘
    # ◀──────────────────▶  ◀──────────────────▶  ◀──────────────────▶
    #         MHA                    GQA                   MQA
    #   n_query_groups=4       n_query_groups=2      n_query_groups=1
    #
    # credit https://arxiv.org/pdf/2305.13245.pdf
    n_query_groups: Optional[int] = None
    shared_attention_norm: bool = False

    def __post_init__(self):
        if self.padded_vocab_size is None:
            self.padded_vocab_size = find_multiple(self.vocab_size, self.padding_multiple)
        assert self.n_embd % self.n_head == 0
        if self.n_query_groups is not None:
            assert self.n_head % self.n_query_groups == 0
        else:
            self.n_query_groups = self.n_head

    @property
    def head_size(self) -> int:
        return self.n_embd // self.n_head

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
    # https://huggingface.co/togethercomputer/RedPajama-INCITE-7B-Base/blob/main/config.json
    "RedPajama-INCITE-7B-{}": dict(block_size=2048, n_layer=32, n_embd=4096, n_head=32, padding_multiple=256, rotary_percentage=1.0, parallel_residual=False),
    # this redirects to the checkpoint above. kept for those who had the old weights already downloaded
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
        n_query_groups=1,
        bias=False,
        # this is not in the config, but in the original model implementation, only for this config
        shared_attention_norm=True
    ),
    # https://huggingface.co/tiiuae/falcon-40b/blob/main/config.json
    "falcon-40b{}": dict(
        block_size=2048,
        padded_vocab_size=65024,
        n_layer=60,
        n_head=128,
        n_embd=8192,
        rotary_percentage=1.0,
        parallel_residual=True,
        n_query_groups=8,
        bias=False,
    ),
}
for k in list(falcon):
    for kind in ("", "-instruct"):
        configs[k.format(kind)] = falcon[k]
