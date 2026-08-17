# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

"""Profiles the memory behavior of `chunked_cross_entropy` (litgpt/utils.py) across a sweep of
`chunk_size` values, to give reproducible evidence for the memory-management concerns raised in
https://github.com/Lightning-AI/litgpt/issues/2190.

Usage:
    python -m litgpt.scripts.profile_memory --output-dir docs/profiling
"""

import argparse
import json
import tempfile
from pathlib import Path

import torch

from litgpt.utils import chunked_cross_entropy

DEFAULT_CHUNK_SIZES = [0, 32, 64, 128, 256, 512]


def _peak_bytes_from_memory_timeline(timeline_path: Path) -> int:
    with open(timeline_path) as f:
        times, sizes_by_category = json.load(f)
    return max((sum(sizes) for sizes in sizes_by_category), default=0)


def profile_chunk_size(chunk_size: int, batch_size: int, seq_length: int, vocab_size: int, scratch_dir: Path):
    logits = torch.randn(batch_size, seq_length, vocab_size, requires_grad=True)
    targets = torch.randint(0, vocab_size, (batch_size, seq_length))

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        profile_memory=True,
        record_shapes=True,
        with_stack=True,
    ) as prof:
        loss = chunked_cross_entropy(logits, targets, chunk_size=chunk_size)
        loss.backward()

    # the memory timeline JSON is an intermediate artifact used only to compute peak_bytes below;
    # it's not committed, so it's written to a scratch dir rather than the (committed) output dir.
    timeline_path = scratch_dir / f"memory_timeline_chunk_{chunk_size}.json"
    prof.export_memory_timeline(str(timeline_path), device="cpu")
    peak_bytes = _peak_bytes_from_memory_timeline(timeline_path)
    table = prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=15)
    return peak_bytes, table


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--chunk-sizes", type=int, nargs="+", default=DEFAULT_CHUNK_SIZES)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-length", type=int, default=2048)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--output-dir", type=Path, default=Path("docs/profiling"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    tables = {}
    with tempfile.TemporaryDirectory() as scratch_dir:
        scratch_dir = Path(scratch_dir)
        for chunk_size in args.chunk_sizes:
            print(
                f"Profiling chunk_size={chunk_size} (B={args.batch_size}, T={args.seq_length}, V={args.vocab_size})..."
            )
            peak_bytes, table = profile_chunk_size(
                chunk_size, args.batch_size, args.seq_length, args.vocab_size, scratch_dir
            )
            results[chunk_size] = peak_bytes
            tables[chunk_size] = table
            print(f"  peak profiler-tracked memory: {peak_bytes / 1e6:.1f} MB")

    (args.output_dir / "op_table.md").write_text(
        "\n\n".join(
            f"### chunk_size={chunk_size}\n\n```\n{table}\n```" for chunk_size, table in tables.items()
        )
    )

    _plot(results, args.output_dir / "peak_memory_vs_chunk_size.png", args.batch_size, args.seq_length, args.vocab_size)
    print(f"\nWrote {args.output_dir / 'peak_memory_vs_chunk_size.png'} and {args.output_dir / 'op_table.md'}")


def _plot(results: dict, output_path: Path, batch_size: int, seq_length: int, vocab_size: int):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    chunk_sizes = list(results.keys())
    peak_mb = [results[c] / 1e6 for c in chunk_sizes]
    labels = ["unchunked" if c == 0 else str(c) for c in chunk_sizes]

    surface = "#fcfcfb"
    ink_primary = "#0b0b0b"
    ink_muted = "#898781"
    gridline = "#e1e0d9"
    bar_color = "#2a78d6"

    fig, ax = plt.subplots(figsize=(7, 4.5))
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)
    ax.bar(labels, peak_mb, color=bar_color, width=0.6)
    ax.set_xlabel("cross_entropy_chunk_size", color=ink_primary)
    ax.set_ylabel("Peak profiler-tracked CPU memory (MB)", color=ink_primary)
    ax.set_title(
        f"chunked_cross_entropy backward-pass memory\n(B={batch_size}, T={seq_length}, V={vocab_size}, CPU)",
        color=ink_primary,
    )
    ax.tick_params(colors=ink_muted)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(gridline)
    ax.spines["bottom"].set_color(gridline)
    ax.yaxis.grid(True, color=gridline, linewidth=1)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, facecolor=surface)


if __name__ == "__main__":
    main()
