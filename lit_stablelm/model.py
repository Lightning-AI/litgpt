"""Full definition of a GPT NeoX Language Model, all of it in this single file.

Based on the nanoGPT implementation: https://github.com/karpathy/nanoGPT and
https://github.com/EleutherAI/gpt-neox/tree/main/megatron/model.
"""
# mypy: ignore-errors
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F
from typing_extensions import Self

from lit_stablelm.utils import find_multiple


@dataclass
class StableLMConfig:
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
        return cls(**stablelm_configs[name])


stablelm_configs = {
    # Stability AI StableLM
    # https://huggingface.co/stabilityai/stablelm-base-alpha-3b/blob/main/config.json
    "stablelm-base-alpha-3b": dict(padding_multiple=512),
    # https://huggingface.co/stabilityai/stablelm-base-alpha-7b/blob/main/config.json
    "stablelm-base-alpha-7b": dict(n_head=48, n_embd=6144, padding_multiple=256),
    # https://huggingface.co/stabilityai/stablelm-tuned-alpha-3b/blob/main/config.json
    "stablelm-tuned-alpha-3b": dict(n_head=32, padding_multiple=512),
    # https://huggingface.co/stabilityai/stablelm-tuned-alpha-7b/blob/main/config.json
    "stablelm-tuned-alpha-7b": dict(n_head=48, n_embd=6144, padding_multiple=256),
    # EleutherAI Pythia
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
    # togethercomputer
    # https://huggingface.co/togethercomputer/RedPajama-INCITE-Base-3B-v1/blob/main/config.json
    "RedPajama-INCITE-Base-3B-v1": dict(
        block_size=2048,
        n_layer=32,
        n_embd=2560,
        n_head=32,
        padding_multiple=256,
        rotary_percentage=1.0,
        parallel_residual=False,
    ),
    # https://huggingface.co/togethercomputer/RedPajama-INCITE-Chat-3B-v1/blob/main/config.json
    "RedPajama-INCITE-Chat-3B-v1": dict(
        block_size=2048,
        n_layer=32,
        n_embd=2560,
        n_head=32,
        padding_multiple=256,
        rotary_percentage=1.0,
        parallel_residual=False,
    ),
    # https://huggingface.co/togethercomputer/RedPajama-INCITE-Instruct-3B-v1/blob/main/config.json
    "RedPajama-INCITE-Instruct-3B-v1": dict(
        block_size=2048,
        n_layer=32,
        n_embd=2560,
        n_head=32,
        padding_multiple=256,
        rotary_percentage=1.0,
        parallel_residual=False,
    ),
    # https://huggingface.co/togethercomputer/RedPajama-INCITE-Base-7B-v0.1/blob/main/config.json
    "RedPajama-INCITE-Base-7B-v0.1": dict(
        block_size=2048,
        n_layer=32,
        n_embd=4096,
        n_head=32,
        padding_multiple=256,
        rotary_percentage=1.0,
        parallel_residual=False,
    ),
    # https://huggingface.co/togethercomputer/RedPajama-INCITE-Chat-7B-v0.1/blob/main/config.json
    "RedPajama-INCITE-Chat-7B-v0.1": dict(
        block_size=2048,
        n_layer=32,
        n_embd=4096,
        n_head=32,
        padding_multiple=256,
        rotary_percentage=1.0,
        parallel_residual=False,
    ),
    # https://huggingface.co/togethercomputer/RedPajama-INCITE-Instruct-7B-v0.1/blob/main/config.json
    "RedPajama-INCITE-Instruct-7B-v0.1": dict(
        block_size=2048,
        n_layer=32,
        n_embd=4096,
        n_head=32,
        padding_multiple=256,
        rotary_percentage=1.0,
        parallel_residual=False,
    ),
}
for k in list(stablelm_configs):
    if k.startswith("pythia"):
        stablelm_configs[k + "-deduped"] = stablelm_configs[k]


class StableLM(nn.Module):
    def __init__(self, config: StableLMConfig) -> None:
        super().__init__()
        assert config.padded_vocab_size is not None
        self.config = config

        self.lm_head = nn.Linear(config.n_embd, config.padded_vocab_size, bias=False)
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.padded_vocab_size, config.n_embd),
                h=nn.ModuleList(Block(config) for _ in range(config.n_layer)),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            # https://huggingface.co/stabilityai/stablelm-base-alpha-3b/blob/main/config.json#L10
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
            # https://huggingface.co/stabilityai/stablelm-base-alpha-3b/blob/main/config.json#L12
            module.eps = 1e-5

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        _, t = idx.size()
        assert (
            t <= self.config.block_size
        ), f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        # forward the model itself
        x = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        logits = self.lm_head(x)  # (b, t, vocab_size)

        return logits

    @classmethod
    def from_name(cls, name: str) -> Self:
        return cls(StableLMConfig.from_name(name))


class Block(nn.Module):
    def __init__(self, config: StableLMConfig) -> None:
        super().__init__()
        self.norm_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.norm_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

        self.parallel_residual = config.parallel_residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.parallel_residual:
            x = x + self.attn(self.norm_1(x)) + self.mlp(self.norm_2(x))
        else:
            x = x + self.attn(self.norm_1(x))
            x = x + self.mlp(self.norm_2(x))
        return x


class CausalSelfAttention(nn.Module):
    def __init__(self, config: StableLMConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # key, query, value projections for all heads, but in a batch
        self.attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=True)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=True)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.block_size = config.block_size
        self.rotary_percentage = config.rotary_percentage
        self.rope_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        qkv = self.attn(x)
        head_size = C // self.n_head
        qkv = qkv.view(B, T, self.n_head, 3 * head_size).transpose(1, 2)
        q, k, v = qkv.split(head_size, dim=-1)  # (B, nh, T, hs)

        n_elem = int(self.rotary_percentage * head_size)
        if self.rope_cache is None:
            self.rope_cache = build_rope_cache(self.block_size, n_elem, x.dtype, x.device)
        cos, sin = self.rope_cache
        cos, sin = cos[:T], sin[:T]

        q_roped = apply_rope(q[..., :n_elem], cos, sin)
        k_roped = apply_rope(k[..., :n_elem], cos, sin)
        q = torch.cat((q_roped, q[..., n_elem:]), dim=-1)
        k = torch.cat((k_roped, k[..., n_elem:]), dim=-1)

        # efficient attention using Flash Attention CUDA kernels
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True, scale=1.0 / math.sqrt(head_size)
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.proj(y)

        return y


class MLP(nn.Module):
    def __init__(self, config: StableLMConfig) -> None:
        super().__init__()
        hidden_dim = 4 * config.n_embd
        self.fc = nn.Linear(config.n_embd, hidden_dim, bias=True)
        self.proj = nn.Linear(hidden_dim, config.n_embd, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # gpt-neox style MLP
        x = self.fc(x)
        x = F.gelu(x)
        x = self.proj(x)
        return x


def build_rope_cache(
    seq_len: int, n_elem: int, dtype: torch.dtype, device: torch.device, base: int = 10000
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Enhanced Transformer with Rotary Position Embedding.

    Derived from: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/
    transformers/rope/__init__.py. MIT License:
    https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/license.
    """
    # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
    theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, dtype=dtype, device=device) / n_elem))

    # Create position indexes `[0, 1, ..., seq_len - 1]`
    seq_idx = torch.arange(seq_len, dtype=dtype, device=device)

    # Calculate the product of position index and $\theta_i$
    idx_theta = torch.outer(seq_idx, theta).float().repeat(1, 2)

    cos, sin = torch.cos(idx_theta), torch.sin(idx_theta)

    # this is to mimic the behaviour of complex32, else we will get different results
    if dtype in (torch.float16, torch.bfloat16, torch.int8):
        return cos.half(), sin.half()
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    head_size = x.size(-1)
    x1 = x[..., : head_size // 2]  # (B, nh, T, hs/2)
    x2 = x[..., head_size // 2 :]  # (B, nh, T, hs/2)
    rotated = torch.cat((-x2, x1), dim=-1)  # (B, nh, T, hs)
    roped = (x * cos) + (rotated * sin)
    return roped.type_as(x)
