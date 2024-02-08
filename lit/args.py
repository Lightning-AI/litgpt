from dataclasses import dataclass

@dataclass
class FinetuneArgs:
    max_iters: int = 512
    """Maximum number of iterations"""

@dataclass
class DataArgs:
    dataset: str = "alpaca"
    """The dataset to load"""