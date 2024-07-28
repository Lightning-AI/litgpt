# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
import pytest

from litgpt.args import TrainArgs


def test_compute_warmup_iters():
    # warmup disabled
    train = TrainArgs(lr_warmup_steps=0, lr_warmup_fraction=0)
    assert train.warmup_iters(devices=1, max_iters=1000, train_dataloader=range(10)) == 0

    # lr_warmup_steps and lr_warmup_fraction both are not allowed
    with pytest.raises(ValueError, match="Can't provide both `--train.lr_warmup_fraction`"):
        TrainArgs(lr_warmup_steps=1, lr_warmup_fraction=0.2)

    # lr_warmup_fraction invalid range
    with pytest.raises(ValueError, match=" must be between 0 and 1"):
        TrainArgs(lr_warmup_steps=0, lr_warmup_fraction=1.1)

    # lr_warmup_steps
    train = TrainArgs(global_batch_size=1, micro_batch_size=1, lr_warmup_steps=100, lr_warmup_fraction=0)
    assert train.warmup_iters(devices=1, max_iters=1000, train_dataloader=range(10)) == 100
    # lr_warmup_steps multiplied by accumulation factor
    train.global_batch_size = 4
    assert train.warmup_iters(devices=1, max_iters=1000, train_dataloader=range(10)) == 400
    assert train.warmup_iters(devices=2, max_iters=1000, train_dataloader=range(10)) == 200
    # lr_warmup_steps truncated by max iters
    assert train.warmup_iters(devices=1, max_iters=120, train_dataloader=range(10)) == 120

    # lr_warmup_fraction
    train = TrainArgs(global_batch_size=1, micro_batch_size=1, lr_warmup_steps=0, lr_warmup_fraction=0.3)
    assert train.warmup_iters(devices=1, max_iters=1000, train_dataloader=range(100)) == 30
    # lr_warmup_fraction truncated by max iters
    assert train.warmup_iters(devices=1, max_iters=20, train_dataloader=range(100)) == 20
    # lr_warmup_fraction rounds up
    assert train.warmup_iters(devices=1, max_iters=1000, train_dataloader=range(5)) == 2
