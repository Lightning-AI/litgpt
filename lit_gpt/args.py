from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class TrainArgs:
    """Training related arguments"""

    save_interval: int = 1000
    """Number of optimizer steps between checkpoints"""
    log_interval: int = 1
    """Number of iterations between logging calls"""
    global_batch_size: int = 64
    """Number of samples between optimizer steps across data-parallel ranks"""
    micro_batch_size: int = 4
    """Number of samples per data-parallel rank"""
    lr_warmup_epochs: int = 2
    """Number of epochs with learning rate warmup active"""
    lr_warmup_steps: int = 100
    """Number of optimizer steps with learning rate warmup active"""
    epochs: int = 5
    """Number of epochs to run"""
    epoch_size: int = 50000
    """Size of the epoch"""

    def max_iters(self, devices: int) -> int:
        """Number of iterations"""
        max_iters = self.epochs * self.epoch_size // devices // self.micro_batch_size
        assert max_iters > 0
        return max_iters

    def gradient_accumulation_iters(self, devices: int) -> int:
        """Number of iterations between gradient synchronizations"""
        gradient_accumulation_iters = self.batch_size(devices) // self.micro_batch_size
        assert gradient_accumulation_iters > 0
        return gradient_accumulation_iters

    def batch_size(self, devices: int) -> int:
        """Number of samples between optimizer steps per data-parallel rank"""
        batch_size = self.global_batch_size // devices
        assert batch_size > 0
        return batch_size


@dataclass
class EvalArgs:
    """Evaluation related arguments"""

    interval: int = 600
    """Number of optimizer steps between evaluation calls"""
    max_new_tokens: int = 100
    """Number of tokens to generate"""
    max_iters: int = 100
    """Number of iterations"""


@dataclass
class OptimizationArgs:
    """Optimization related arguments"""

    learning_rate: float = 1e-3
    weight_decay: float = 0.02


@dataclass
class DataArgs:
    """Data related arguments"""

    max_seq_length: Optional[int] = None
    """Limits the length of samples. Off by default"""


@dataclass
class IOArgs:
    """Inputs and outputs related arguments"""

    data_dir: Path = Path("data/alpaca")
    """Where to read data from"""
    checkpoint_dir: Path = Path("checkpoints/stabilityai/stablelm-base-alpha-3b")
    """Where to read weights and tokenizer data from"""
    out_dir: Path = Path("out/adapter/alpaca")
    """Where to save artifacts"""
