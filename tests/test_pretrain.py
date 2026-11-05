# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import math

from litgpt.pretrain import get_lr


def test_get_lr():
    learning_rate, min_lr = 0.1, 0.01
    warmup_iters, max_iters = 10, 100

    # linear warmup
    assert get_lr(learning_rate, 0, warmup_iters, max_iters, min_lr) == 0.0
    assert get_lr(learning_rate, 5, warmup_iters, max_iters, min_lr) == 0.05
    # cosine decay
    assert get_lr(learning_rate, warmup_iters, warmup_iters, max_iters, min_lr) == learning_rate
    assert math.isclose(get_lr(learning_rate, max_iters, warmup_iters, max_iters, min_lr), min_lr)
    # past the end of training
    assert get_lr(learning_rate, max_iters + 1, warmup_iters, max_iters, min_lr) == min_lr


def test_get_lr_when_warmup_covers_the_whole_run():
    # `it == warmup_iters == max_iters` used to divide by zero in the cosine decay
    assert get_lr(0.1, 10, 10, 10, 0.01) == 0.01
    assert get_lr(0.1, 20, 10, 10, 0.01) == 0.01
    # warmup still applies before the end of the run
    assert get_lr(0.1, 5, 10, 10, 0.01) == 0.05
