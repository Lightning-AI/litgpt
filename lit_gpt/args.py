from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class TrainArgs:
    save_interval: int = 1000
    log_interval: int = 1
    global_batch_size: int = 64
    micro_batch_size: int = 4
    lr_warmup_epochs: int = 2
    epochs: int = 5
    train_epoch_size: int = 50000

    def max_iters(self, devices: int) -> int:
        max_iters = self.epochs * self.train_epoch_size // devices // self.micro_batch_size
        assert max_iters > 0
        return max_iters

    def gradient_accumulation_iters(self, devices: int) -> int:
        gradient_accumulation_iters = self.global_batch_size // devices // self.micro_batch_size
        assert gradient_accumulation_iters > 0
        return gradient_accumulation_iters

    def batch_size(self, devices: int) -> int:
        batch_size = self.global_batch_size // devices
        assert batch_size > 0
        return batch_size


@dataclass
class EvalArgs:
    interval: int = 600
    max_new_tokens: int = 100
    iters: int = 100


@dataclass
class OptimizationArgs:
    learning_rate: float = 1e-3
    weight_decay: float = 0.02


@dataclass
class DataArgs:
    max_seq_length: Optional[int] = None  # set value to truncate


@dataclass
class IOArgs:
    data_dir: Path = Path("data/alpaca")
    checkpoint_dir: Path = Path("checkpoints/stabilityai/stablelm-base-alpha-3b")
    out_dir: Path = Path("out/adapter/alpaca")
