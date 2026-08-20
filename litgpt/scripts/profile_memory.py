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


def profile_chunk_size(
    chunk_size: int,
    batch_size: int,
    seq_length: int,
    vocab_size: int,
    scratch_dir: Path,
    device: str = "cpu",
    export_memory_plot: Path = None,
):
    logits = torch.randn(batch_size, seq_length, vocab_size, requires_grad=True, device=device)
    targets = torch.randint(0, vocab_size, (batch_size, seq_length), device=device)

    activities = [torch.profiler.ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    with torch.profiler.profile(
        activities=activities,
        profile_memory=True,
        record_shapes=True,
        with_stack=export_memory_plot is not None,
    ) as prof:
        loss = chunked_cross_entropy(logits, targets, chunk_size=chunk_size)
        loss.backward()
        if device == "cuda":
            torch.cuda.synchronize()

    if device == "cuda":
        peak_bytes = torch.cuda.max_memory_allocated()
        sort_key = "self_cuda_memory_usage"
    else:
        # the memory timeline JSON is an intermediate artifact used only to compute peak_bytes below;
        # it's not committed, so it's written to a scratch dir rather than the (committed) output dir.
        timeline_path = scratch_dir / f"memory_timeline_chunk_{chunk_size}.json"
        prof.export_memory_timeline(str(timeline_path), device="cpu")
        peak_bytes = _peak_bytes_from_memory_timeline(timeline_path)
        sort_key = "self_cpu_memory_usage"

    if export_memory_plot is not None:
        # .html embeds a base64 PNG of the same memory-timeline plot shown in the PyTorch profiler docs;
        # .png is written as its own file for embedding directly in docs/PRs.
        html_path = export_memory_plot.with_suffix(".html")
        prof.export_memory_timeline(str(html_path), device=f"{device}:0" if device == "cuda" else device)
        _png_from_memory_timeline_html(html_path, export_memory_plot)

    table = prof.key_averages().table(sort_by=sort_key, row_limit=15)
    return peak_bytes, table


def _png_from_memory_timeline_html(html_path: Path, png_path: Path):
    import base64
    import re

    html = html_path.read_text()
    match = re.search(r"data:image/png;base64,([A-Za-z0-9+/=]+)", html)
    png_path.write_bytes(base64.b64decode(match.group(1)))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--chunk-sizes", type=int, nargs="+", default=DEFAULT_CHUNK_SIZES)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-length", type=int, default=2048)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--output-dir", type=Path, default=Path("docs/profiling"))
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--memory-plot-chunk-sizes",
        type=int,
        nargs="*",
        default=[0, 128],
        help="chunk sizes to additionally export a memory-timeline plot (PNG) for",
    )
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but no CUDA device is available")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.device == "cuda":
        print(f"Profiling on {torch.cuda.get_device_name(0)}")

    results = {}
    tables = {}
    with tempfile.TemporaryDirectory() as scratch_dir:
        scratch_dir = Path(scratch_dir)
        for chunk_size in args.chunk_sizes:
            print(
                f"Profiling chunk_size={chunk_size} (B={args.batch_size}, T={args.seq_length}, V={args.vocab_size}, device={args.device})..."
            )
            tag = "unchunked" if chunk_size == 0 else f"chunk{chunk_size}"
            export_memory_plot = (
                args.output_dir / f"memory_timeline_{tag}.png" if chunk_size in args.memory_plot_chunk_sizes else None
            )
            peak_bytes, table = profile_chunk_size(
                chunk_size,
                args.batch_size,
                args.seq_length,
                args.vocab_size,
                scratch_dir,
                device=args.device,
                export_memory_plot=export_memory_plot,
            )
            results[chunk_size] = peak_bytes
            tables[chunk_size] = table
            print(f"  peak profiler-tracked memory: {peak_bytes / 1e6:.1f} MB")

    (args.output_dir / "op_table.md").write_text(
        "\n\n".join(f"### chunk_size={chunk_size}\n\n```\n{table}\n```" for chunk_size, table in tables.items())
    )

    plot_name = "peak_memory_vs_chunk_size.png" if args.device == "cpu" else "peak_memory_vs_chunk_size_gpu.png"
    _plot(
        results,
        args.output_dir / plot_name,
        args.batch_size,
        args.seq_length,
        args.vocab_size,
        device=args.device,
    )
    print(f"\nWrote {args.output_dir / plot_name} and {args.output_dir / 'op_table.md'}")


def _plot(results: dict, output_path: Path, batch_size: int, seq_length: int, vocab_size: int, device: str = "cpu"):
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

    device_label = "CPU" if device == "cpu" else torch.cuda.get_device_name(0)
    mem_label = "Peak profiler-tracked CPU memory (MB)" if device == "cpu" else "Peak CUDA memory allocated (MB)"

    fig, ax = plt.subplots(figsize=(7, 4.5))
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)
    ax.bar(labels, peak_mb, color=bar_color, width=0.6)
    ax.set_xlabel("cross_entropy_chunk_size", color=ink_primary)
    ax.set_ylabel(mem_label, color=ink_primary)
    ax.set_title(
        f"chunked_cross_entropy backward-pass memory\n(B={batch_size}, T={seq_length}, V={vocab_size}, {device_label})",
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
