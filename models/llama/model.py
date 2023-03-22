# TODO: replace with Apache licensed version
# Taken from minimal-lama repo

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

MULTIPLE_OF = 256


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    vocab_size: int = 32000
    norm_eps: float = 1e-6
    max_seq_length: int = 2048


DEBUG_CONFIG = ModelArgs(
    dim=32,
    n_layers=10,
    n_heads=4,
    vocab_size=32000,
)
LLAMA_7B_CONFIG = ModelArgs(
    dim=4096,
    n_layers=32,
    n_heads=32,
    vocab_size=32000,
)
LLAMA_13B_CONFIG = ModelArgs(
    dim=5120,
    n_layers=40,
    n_heads=40,
    vocab_size=32000,
)

LLAMA_CONFIG_DICT = {
    "7B": LLAMA_7B_CONFIG,
    "13B": LLAMA_13B_CONFIG,
    "debug": DEBUG_CONFIG,
}


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim, hidden_dim=4 * args.dim, multiple_of=MULTIPLE_OF
        )
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, cos, sin, mask: Optional[torch.Tensor]):
        h = x + self.attention.forward(self.attention_norm(x), cos, sin, mask)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_local_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
        )
        self.wk = nn.Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
        )
        self.wv = nn.Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
        )
        self.wo = nn.Linear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False,
        )

    def forward(self, x: torch.Tensor, cos, sin, mask: Optional[torch.Tensor]):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

        keys = xk[:, :seqlen]
        values = xv[:, :seqlen]

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, slen, cache_len + slen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, slen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        return self.wo(output)


def apply_rotary_pos_emb(q, k, cos, sin):
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def precompute_cos_sin(seq_len, dim, dtype, device, base=10000):
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(seq_len, device=device, dtype=dtype)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1).to(device)
    cos_cached = emb.cos()[None, :, None, :]
    sin_cached = emb.sin()[None, :, None, :]
    return cos_cached, sin_cached


class LLAMA(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)
        self.cos_cached, self.sin_cached = precompute_cos_sin(
            self.params.max_seq_length,
            self.params.dim // self.params.n_heads,
            dtype=self.tok_embeddings.weight.dtype,
            device=self.tok_embeddings.weight.device,
        )

    def forward(self, tokens: torch.Tensor):
        _bsz, seq_len = tokens.shape
        h = self.tok_embeddings(tokens)
        # Note
        # cos = self.cos_cached[:, start_pos : start_pos + seq_len].to(h.dtype)
        # sin = self.sin_cached[:, start_pos : start_pos + seq_len].to(h.dtype)
        cos = self.cos_cached[:, :seq_len].type(h.dtype)
        sin = self.sin_cached[:, :seq_len].type(h.dtype)

        mask = torch.full((1, 1, seq_len, seq_len), float("-inf"), device=tokens.device)
        mask = torch.triu(mask, diagonal=1).type_as(h)

        for layer in self.layers:
            h = layer(h, cos, sin, mask)
        h = self.norm(h)
        output = self.output(h)
        return output.float()

    @torch.inference_mode()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.

        Args:
            idx: Tensor of shape (B, T) with indices of the prompt sequence.
            max_new_tokens: The number of new tokens to generate.
            temperature: Scales the predicted logits by 1 / temperature
            top_k: If specified, only sample among the tokens with the k highest probabilities.

        The implementation of this function is modified from A. Karpathy's nanoGPT.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at max_seq_length
            idx_cond = idx if idx.size(1) <= self.params.max_seq_length else idx[:, -self.params.max_seq_length:]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
