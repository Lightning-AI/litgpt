# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

"""Regression tests for pretrain.validate with uneven / empty validation shards (#2319)."""

from __future__ import annotations

import threading
from typing import Any

import torch
import torch.nn as nn
from lightning import Fabric
from torch.utils.data import DataLoader

from litgpt.pretrain import validate


class _TinyLM(nn.Module):
    """Minimal causal LM stand-in for validate()."""

    def __init__(self, vocab_size: int = 8, n_embd: int = 4, max_seq_length: int = 2):
        super().__init__()
        self.max_seq_length = max_seq_length
        self.embed = nn.Embedding(vocab_size, n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(input_ids))


def _seq_batches(num_batches: int, seq_len: int = 3, vocab: int = 8) -> DataLoader:
    """Build a DataLoader yielding ``[B, T]`` long tensors (not TensorDataset tuples)."""

    class _Rows(torch.utils.data.Dataset):
        def __init__(self, data: torch.Tensor):
            self.data = data

        def __len__(self) -> int:
            return self.data.size(0)

        def __getitem__(self, index: int) -> torch.Tensor:
            return self.data[index]

    if num_batches == 0:
        data = torch.empty(0, seq_len, dtype=torch.long)
    else:
        data = torch.randint(0, vocab, (num_batches, seq_len), dtype=torch.long)
    return DataLoader(_Rows(data), batch_size=1)


def test_validate_empty_dataloader_returns_nan():
    """Empty validation set must not crash with torch.stack([])."""
    fabric = Fabric(accelerator="cpu", devices=1)
    fabric.launch()
    model = fabric.setup(_TinyLM())
    loader = _seq_batches(0)

    val_loss = validate(fabric, model, loader, max_iters=2, verbose=False)

    assert torch.isnan(val_loss).item()
    assert val_loss.device.type == fabric.device.type


def test_validate_with_batches_returns_finite_loss():
    fabric = Fabric(accelerator="cpu", devices=1)
    fabric.launch()
    model = fabric.setup(_TinyLM())
    loader = _seq_batches(3)

    val_loss = validate(fabric, model, loader, max_iters=2, verbose=False)

    assert torch.isfinite(val_loss).item()
    assert val_loss.ndim == 0


def test_validate_lockstep_logic_matches_issue_workaround():
    """Drive two ranks through the same MIN-reduce protocol as validate().

    Verifies the synchronization contract that prevents deadlock when shards are
    uneven, including the case a naive empty-skip mishandles.
    """
    # Rank 0: 2 batches; rank 1: 0 → zero synchronized forwards.
    remaining = [2, 0]
    collective_forwards = 0
    forward_counts = [0, 0]
    for _ in range(4):
        votes = [1.0 if remaining[r] > 0 else 0.0 for r in range(2)]
        if min(votes) == 0:
            break
        for rank in range(2):
            remaining[rank] -= 1
            forward_counts[rank] += 1
        collective_forwards += 1

    assert collective_forwards == 0
    assert forward_counts == [0, 0]

    # Partial overlap: rank0=3, rank1=1 → exactly one synchronized step.
    remaining = [3, 1]
    collective_forwards = 0
    forward_counts = [0, 0]
    for _ in range(4):
        votes = [1.0 if remaining[r] > 0 else 0.0 for r in range(2)]
        if min(votes) == 0:
            break
        for rank in range(2):
            remaining[rank] -= 1
            forward_counts[rank] += 1
        collective_forwards += 1

    assert collective_forwards == 1
    assert forward_counts == [1, 1]
    assert remaining == [2, 0]


class _BarrierReduceFabric:
    """Minimal Fabric stand-in: all_reduce(min) + barrier across threads.

    A forward that also all_reduces deadlocks if ranks skip model() unequally —
    the same failure mode as FSDP when validation shards are uneven (#2319).
    """

    def __init__(self, world_size: int, timeout: float = 5.0):
        self.world_size = world_size
        self.device = torch.device("cpu")
        self._timeout = timeout
        self._barrier = threading.Barrier(world_size, timeout=timeout)
        self._lock = threading.Lock()
        self._votes: list[torch.Tensor] = []
        self._reduced: torch.Tensor | None = None
        self._reduce_barrier = threading.Barrier(world_size, timeout=timeout)
        self.local = threading.local()

    @property
    def global_rank(self) -> int:
        return self.local.rank

    def barrier(self) -> None:
        self._barrier.wait()

    def print(self, *args: Any, **kwargs: Any) -> None:
        pass

    def all_reduce(self, tensor: torch.Tensor, reduce_op: str = "mean") -> torch.Tensor:
        """Match Fabric: reduce in-place and return the same tensor."""
        assert reduce_op == "min"
        with self._lock:
            self._votes.append(tensor.detach().clone())
            if len(self._votes) == self.world_size:
                self._reduced = torch.stack(self._votes).min()
                self._votes.clear()
        self._reduce_barrier.wait()
        assert self._reduced is not None
        # Copy under lock so a slow reader cannot see the next step's value.
        with self._lock:
            tensor.copy_(self._reduced)
        # Second wait so every rank finishes reading before the next reduce.
        self._reduce_barrier.wait()
        return tensor


class _CollectiveTinyLM(_TinyLM):
    def __init__(self, fabric: _BarrierReduceFabric):
        super().__init__()
        self.fabric = fabric
        self.forward_calls = 0

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        self.forward_calls += 1
        # Collective every forward — skipping this on some ranks hangs the barrier.
        token = torch.ones((), device=self.fabric.device)
        self.fabric.all_reduce(token, reduce_op="min")
        return super().forward(input_ids)


def _run_validate_on_rank(
    fabric: _BarrierReduceFabric,
    rank: int,
    num_batches: int,
    max_iters: int,
    results: dict[int, Any],
    errors: dict[int, BaseException],
) -> None:
    fabric.local.rank = rank
    model = _CollectiveTinyLM(fabric)
    loader = _seq_batches(num_batches)
    try:
        loss = validate(fabric, model, loader, max_iters=max_iters, verbose=False)
        results[rank] = {"loss": loss, "calls": model.forward_calls}
    except BaseException as exc:  # noqa: BLE001 - surface any hang/crash to the parent
        errors[rank] = exc


def test_validate_uneven_shards_does_not_deadlock_threaded():
    """Rank 0 has data, rank 1 empty: both finish; equal collective call counts."""
    fabric = _BarrierReduceFabric(world_size=2)
    results: dict[int, Any] = {}
    errors: dict[int, BaseException] = {}
    threads = [
        threading.Thread(
            target=_run_validate_on_rank,
            args=(fabric, rank, batches, 4, results, errors),
            daemon=True,
        )
        for rank, batches in ((0, 2), (1, 0))
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
        assert not t.is_alive(), "validate() hung — ranks likely skipped a collective unequally"

    assert errors == {}
    assert results[0]["calls"] == results[1]["calls"] == 0
    assert torch.isnan(results[0]["loss"]).item()
    assert torch.isnan(results[1]["loss"]).item()


def test_validate_partial_overlap_stops_together_threaded():
    """Rank 0 has 3 batches, rank 1 has 1: both take exactly one model step."""
    fabric = _BarrierReduceFabric(world_size=2)
    results: dict[int, Any] = {}
    errors: dict[int, BaseException] = {}
    threads = [
        threading.Thread(
            target=_run_validate_on_rank,
            args=(fabric, rank, batches, 5, results, errors),
            daemon=True,
        )
        for rank, batches in ((0, 3), (1, 1))
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
        assert not t.is_alive(), "validate() hung on partial-overlap shards"

    assert errors == {}
    assert results[0]["calls"] == results[1]["calls"] == 1
    assert torch.isfinite(results[0]["loss"]).item()
    assert torch.isfinite(results[1]["loss"]).item()


def test_naive_empty_skip_deadlocks_without_min_reduce():
    """Control: skipping model() on the empty rank hangs a collective forward.

    Documents why a bare ``if not losses: return nan`` is insufficient under FSDP.
    """
    fabric = _BarrierReduceFabric(world_size=2, timeout=1.0)
    hung = threading.Event()

    def rank0() -> None:
        fabric.local.rank = 0
        model = _CollectiveTinyLM(fabric)
        # Pretend we have a batch and enter forward (collective).
        try:
            model(torch.zeros(1, 2, dtype=torch.long))
        except threading.BrokenBarrierError:
            hung.set()

    def rank1_skips() -> None:
        fabric.local.rank = 1
        # Naive empty-rank path: never call model(), only barrier at the end.
        try:
            fabric.barrier()
        except threading.BrokenBarrierError:
            hung.set()

    t0 = threading.Thread(target=rank0, daemon=True)
    t1 = threading.Thread(target=rank1_skips, daemon=True)
    t0.start()
    t1.start()
    t0.join(timeout=3)
    t1.join(timeout=3)
    assert hung.is_set() or t0.is_alive() or t1.is_alive()
