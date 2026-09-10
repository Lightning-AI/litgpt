"""Standalone benchmark: torchembed fused RoPE vs. litgpt plain-PyTorch RoPE.

Reproduces the numbers cited in the PR description.

Hardware used for reference results: NVIDIA GB10, bfloat16.
Run with:
    python benchmarks/bench_torchembed_rope.py

Requirements:
    pip install torchembed
    CUDA GPU with triton support
"""

import time

import torch

from litgpt.model import apply_rope

try:
    from torchembed.positional import RotaryEmbedding as TorchembedRotaryEmbedding
except ImportError as e:
    raise SystemExit("torchembed is not installed. Run: pip install torchembed") from e


def _time_fn(fn, *args, warmup: int = 5, iters: int = 50) -> float:
    """Return median wall-clock time in milliseconds."""
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        start = time.perf_counter()
        fn(*args)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)

    times.sort()
    return times[len(times) // 2]


def _litgpt_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Thin wrapper matching the litgpt apply_rope calling convention."""
    apply_rope(q, cos, sin)
    apply_rope(k, cos, sin)


def _build_litgpt_cos_sin(seq_len: int, rope_n_elem: int, base: int, device: torch.device, dtype: torch.dtype):
    """Replicate litgpt's build_rope_cache (simplified, standard RoPE)."""
    inv_freq = 1.0 / (base ** (torch.arange(0, rope_n_elem, 2, device=device).float() / rope_n_elem))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    # litgpt expects (1, T, rope_n_elem)
    cos = emb.cos().unsqueeze(0).to(dtype)
    sin = emb.sin().unsqueeze(0).to(dtype)
    return cos, sin


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required for this benchmark.")

    device = torch.device("cuda")
    dtype = torch.bfloat16

    batch = 4
    n_heads = 32
    d_qk = 128  # head dim / rope_n_elem
    base = 10_000

    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"dtype={dtype}, batch={batch}, n_heads={n_heads}, d_qk={d_qk}")
    print()
    print(f"{'seq_len':>8}  {'litgpt (ms)':>12}  {'torchembed (ms)':>16}  {'speedup':>8}")
    print("-" * 52)

    rope_tc = TorchembedRotaryEmbedding(dim=d_qk, max_seq_len=8192, base=base, use_fused=True).to(device)

    for seq_len in [512, 1024, 2048, 4096, 8192]:
        q = torch.randn(batch, n_heads, seq_len, d_qk, device=device, dtype=dtype)
        k = torch.randn(batch, n_heads, seq_len, d_qk, device=device, dtype=dtype)

        cos, sin = _build_litgpt_cos_sin(seq_len, d_qk, base, device, dtype)

        t_litgpt = _time_fn(_litgpt_rope, q, k, cos, sin)
        t_tc = _time_fn(rope_tc, q, k)

        speedup = t_litgpt / t_tc
        print(f"{seq_len:>8}  {t_litgpt:>12.3f}  {t_tc:>16.3f}  {speedup:>7.2f}x")


if __name__ == "__main__":
    main()
