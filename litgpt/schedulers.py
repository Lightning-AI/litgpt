import math

# learning rate decay scheduler (cosine with linear warmup)
def get_lr(
    learning_rate: float, it: int, warmup_iters: int, max_iters: int, min_lr: float
) -> float:
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > max_iters, return min learning rate
    if it > max_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


def get_lr_decay_stage(
    learning_rate: float,
    it: int,
    decay_start: int,
    min_lr: float,
    gradient_accumulation_iters: int,
    decay_hyperparam: int = 50
) -> float:
    # learning_rate: the max lr to start from.
    # it: current iter
    # decay_start: the iter at which decay begins
    # min_lr: the minimum LR
    if it < decay_start:
        return learning_rate

    decay_ratio = 0.5 ** ((it - decay_start) / (decay_hyperparam * gradient_accumulation_iters))
    decayed_lr = learning_rate * decay_ratio
    return decayed_lr


def get_lr_linear_decay(
    learning_rate: float, it: int, warmup_iters: int, max_iters: int, min_lr: float
) -> float:
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    return learning_rate * (1 - it / max_iters) + min_lr
