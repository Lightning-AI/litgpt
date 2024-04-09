# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class TrainArgs:
    """Training-related arguments"""

    save_interval: Optional[int] = 1000
    """Number of optimizer steps between saving checkpoints"""
    log_interval: int = 1
    """Number of iterations between logging calls"""
    global_batch_size: int = 64
    """Number of samples between optimizer steps across data-parallel ranks"""
    micro_batch_size: int = 4
    """Number of samples per data-parallel rank"""
    lr_warmup_steps: int = 100
    """Number of iterations with learning rate warmup active"""
    epochs: Optional[int] = None
    """Number of epochs to train on"""
    # TODO: `pretrain` is the only script using `max_tokens` explicitly. replace it with epoch_size*epochs?
    max_tokens: Optional[int] = None
    """Total number of tokens to train on"""
    max_steps: Optional[int] = None
    """Limits the number of optimizer steps to run"""
    max_seq_length: Optional[int] = None
    """Limits the length of samples"""
    tie_embeddings: Optional[bool] = None
    """Whether to tie the embedding weights with the language modeling head weights"""

    # Optimization args
    learning_rate: float = 1e-3
    weight_decay: float = 0.02
    beta1: float = 0.9
    beta2: float = 0.95
    max_norm: Optional[float] = None
    min_lr: float = 6e-5

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
    """Evaluation-related arguments"""

    interval: int = 600
    """Number of optimizer steps between evaluation calls"""
    max_new_tokens: Optional[int] = None
    """Number of tokens to generate"""
    max_iters: int = 100
    """Number of iterations"""

@dataclass
class GaLoreArgs:
    """GaLore-related arguments"""

    use_galore: bool = False,
    """Whether to enable GaLore (GaLore is applied to all linear layers)."""
    galore_8bit: bool = False,
    """Whether to use the 8-bit GaLore AdamW optimizer instead of the Galore AdamW optimizer."""
    galore_r: int = 128,
    """GaLore rank"""
    galore_update_proj_gap: int = 200,
    """GaLore hyperparameter"""
    galore_scale: float = 0.25,
    """GaLore scale factor"""
    galore_proj_type: Literal["std", "reverse_std"] = "std",
    """GaLore projection type"""
