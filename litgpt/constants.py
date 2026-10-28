# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
"""Centralized package availability constants for optional dependencies."""

from lightning_utilities.core.imports import RequirementCache

# Logger-related constants
_SUPPORTED_LOGGERS: tuple[str, ...] = ("csv", "tensorboard", "wandb", "mlflow", "litlogger")

# Logger-related optional dependencies
_LITLOGGER_AVAILABLE = RequirementCache("litlogger>=0.1.7")
_TENSORBOARD_AVAILABLE = RequirementCache("tensorboard")
_WANDB_AVAILABLE = RequirementCache("wandb")
_MLFLOW_AVAILABLE = RequirementCache("mlflow")
_MLFLOW_SKINNY_AVAILABLE = RequirementCache("mlflow-skinny")

# PyTorch version-specific constants
_TORCH_EQUAL_2_7 = RequirementCache("torch>=2.7.0,<2.8")
_TORCH_EQUAL_2_8 = RequirementCache("torch>=2.8.0,<2.9")

# Other optional dependencies
_REQUESTS_AVAILABLE = RequirementCache("requests")
_THUNDER_AVAILABLE = RequirementCache("thunder")
_TRITON_AVAILABLE = RequirementCache("triton")
_BITANDBYTES_AVAILABLE = RequirementCache("bitsandbytes")
_BITANDBYTES_AVAILABLE_NOT_EQUAL_0_42_0 = RequirementCache("bitsandbytes != 0.42.0")
_LITDATA_AVAILABLE = RequirementCache("litdata")
_LITSERVE_AVAILABLE = RequirementCache("litserve")
_JINJA2_AVAILABLE = RequirementCache("jinja2")
_SAFETENSORS_AVAILABLE = RequirementCache("safetensors")
_HF_TRANSFER_AVAILABLE = RequirementCache("hf_transfer")
