"""
Full definition of a LLaMA Language Model, all of it in this single file.
Based on the nanoGPT implementation: https://github.com/karpathy/nanoGPT.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


def build_rope_cache(seq_len, n_elem, dtype, device, base=10000):
    """
    Derived from: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/transformers/rope/__init__.py
    MIT License: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/license
    """
    # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
    theta = 1. / (base ** (torch.arange(0, n_elem, 2, dtype=dtype, device=device) / n_elem))

    # Create position indexes `[0, 1, ..., seq_len - 1]`
    seq_idx = torch.arange(seq_len, device=device, dtype=dtype)

    # Calculate the product of position index and $\theta_i$
    idx_theta = torch.outer(seq_idx, theta)

    # Cache them
    cache = torch.polar(torch.ones_like(idx_theta), idx_theta)  # complex64
    return cache


def apply_rope(x: torch.Tensor, rope_cache: torch.Tensor):
    x = x.transpose(1, 2)
    
    # truncate to support variable sizes
    T = x.size(1)
    rope_cache = rope_cache[:T]
    
    xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    rope_cache = rope_cache.view(1, xc.size(1), 1, xc.size(3))
    x_out = torch.view_as_real(xc * rope_cache).flatten(3)
    return x_out.transpose(1, 2).type_as(x)


class RMSNorm(nn.Module):
    """
    Derived from https://github.com/bzhangGo/rmsnorm/blob/master/rmsnorm_torch.py
    BSD 3-Clause License: https://github.com/bzhangGo/rmsnorm/blob/master/LICENSE
    """

    def __init__(self, size, dim=-1, eps=1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(size))
        self.eps = eps
        self.dim = dim

    def forward(self, x):
        # NOTE: the original RMSNorm paper implementation is not equivalent
        # norm_x = x.norm(2, dim=self.dim, keepdim=True)
        # rms_x = norm_x * d_x ** (-1. / 2)
        # x_normed = x / (rms_x + self.eps)
        norm_x = torch.mean(x*x, dim=self.dim, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        return self.scale * x_normed


class CausalSelfAttention(nn.Module):

    def __init__(self, config, rope_cache):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.rope_cache = rope_cache

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        head_size = C // self.n_head
        k = k.view(B, T, self.n_head, head_size).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, head_size).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, head_size).transpose(1, 2) # (B, nh, T, hs)

        q = apply_rope(q, self.rope_cache)
        k = apply_rope(k, self.rope_cache)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # efficient attention using Flash Attention CUDA kernels
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.c_proj(y)

        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        hidden_dim = 4 * config.n_embd
        n_hidden = int(2 * hidden_dim / 3)
        N = 256
        # ensure n_hidden is multiple of N
        n_hidden = ((n_hidden - 1) // N) * N + N

        self.c_fc1   = nn.Linear(config.n_embd, n_hidden, bias=False)
        self.c_fc2   = nn.Linear(config.n_embd, n_hidden, bias=False)
        self.c_proj  = nn.Linear(n_hidden, config.n_embd, bias=False)

    def forward(self, x):
        x = F.silu(self.c_fc1(x)) * self.c_fc2(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):

    def __init__(self, config, rope_cache):
        super().__init__()
        self.rms_1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config, rope_cache)
        self.rms_2 = RMSNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.rms_1(x))
        x = x + self.mlp(self.rms_2(x))
        return x


@dataclass
class LLaMAConfig:
    block_size: int = 4096  # 7B
    vocab_size: int = 32000
    n_layer: int = 32
    n_head: int = 32
    n_embd: int = 4096


class LLaMA(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.rope_cache = build_rope_cache(
            seq_len=config.block_size,
            n_elem=config.n_embd // config.n_head,
            dtype=self.lm_head.weight.dtype,
            device=self.lm_head.weight.device,
        )

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([Block(config, self.rope_cache) for _ in range(config.n_layer)]),
            ln_f = RMSNorm(config.n_embd),
        ))

        # init all weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02/math.sqrt(2 * self.config.n_layer))
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02/math.sqrt(2 * self.config.n_layer))

    def forward(self, idx):
        _, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        # forward the LLaMA model itself
        x = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        logits = self.lm_head(x)

        return logits

    def step(self, idx, targets):
        logits = self(idx)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return loss
