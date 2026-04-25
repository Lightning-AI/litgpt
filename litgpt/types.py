# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
"""Type aliases used across LitGPT modules."""

from typing import Literal

# Logger-related types
LoggerChoice = Literal["csv", "tensorboard", "wandb", "mlflow", "litlogger"]
"""Valid logger choices for experiment tracking.

Available options:
- "csv": Local CSV file logging (default for most scripts)
- "tensorboard": TensorBoard visualization (default for pretrain)
- "wandb": Weights & Biases cloud tracking
- "mlflow": MLflow experiment tracking
- "litlogger": Lightning.ai native tracking
"""
