# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0
# Adapted for standalone use
from collections import deque
from typing import Deque, Optional

import torch
from lightning.fabric.loggers import Logger

from lit_parrot import Parrot

GPU_AVAILABLE_FLOPS = {
    # source: https://resources.nvidia.com/en-us-tensor-core/nvidia-tensor-core-gpu-datasheet
    # nvidia publishes spec sheet with a 2x sparsity factor
    "h100-sxm": {
        "64-true": 67e12,
        "32-true": 67e12,
        "16-true": 1.979e15 / 2,
        "16-mixed": 1.979e15 / 2,
        "bf16-true": 1.979e15 / 2,
        "bf16-mixed": 1.979e15 / 2,
        "8-true": 3.958e15 / 2,
        "8-mixed": 3.958e15 / 2,
    },
    "h100-pcie": {
        "64-true": 51e12,
        "32-true": 51e12,
        "16-true": 1.513e15 / 2,
        "16-mixed": 1.513e15 / 2,
        "bf16-true": 1.513e15 / 2,
        "bf16-mixed": 1.513e15 / 2,
        "8-true": 3.026e15 / 2,
        "8-mixed": 3.026e15 / 2,
    },
    # source: https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf
    # sxm and pcie have same flop counts
    "a100": {
        "64-true": 19.5e12,
        "32-true": 19.5e12,
        "16-true": 312e12,
        "16-mixed": 312e12,
        "bf16-true": 312e12,
        "bf16-mixed": 312e12,
    },
    # source: https://images.nvidia.com/content/technologies/volta/pdf/volta-v100-datasheet-update-us-1165301-r5.pdf
    "v100-sxm": {"64-true": 7.8e12, "32-true": 15.7e12, "16-true": 125e12, "16-mixed": 125e12},
    "v100-pcie": {"64-true": 7e12, "32-true": 14e12, "16-true": 112e12, "16-mixed": 112e12},
    "v100s-pcie": {"64-true": 8.2e12, "32-true": 16.4e12, "16-true": 130e12, "16-mixed": 130e12},
    # source: https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet-951643.pdf
    # sxm and pcie have same flop counts
    "t4": {"32-true": 8.1e12, "16-true": 65e12, "16-mixed": 65e12, "8-true": 130e12, "int4": 260e12},
    # https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/quadro-product-literature/quadro-rtx-5000-data-sheet-us-nvidia-704120-r4-web.pdf
    "quadro rtx 5000": {"32-true": 11.2e12, "16-true": 89.2e12, "16-mixed": 89.2e12},
}


def get_gpu_flops_available(precision: str):
    # Return 0 if no CUDA device (e.g., when running with CPU only)
    if not torch.cuda.is_available():
        return 0

    # torch.cuda.get_device_name() ex output: 'NVIDIA A100-SXM4-40GB'
    device_name = torch.cuda.get_device_name().lower()
    if "h100" in device_name and "hbm3" in device_name:
        device_name = "h100-sxm"
    elif "h100" in device_name and ("pcie" in device_name or "hbm2e" in device_name):
        device_name = "h100-pcie"
    elif "a100" in device_name:
        device_name = "a100"
    elif "v100-sxm" in device_name:
        device_name = "v100-sxm"
    elif "v100-pcie" in device_name:
        device_name = "v100-pcie"
    elif "t4" in device_name:
        device_name = "t4"
    elif "quadro rtx 5000" in device_name:
        device_name = "quadro rtx 5000"
    else:
        device_name = None

    gpu_flops_available = None
    if device_name is not None:
        try:
            gpu_flops_available = int(GPU_AVAILABLE_FLOPS[device_name][precision])
        except KeyError:
            raise KeyError(
                f"gpu_flop count not found for {device_name} with precision: {precision}; "
                "MFU cannot be calculated and reported. gpu_flops_available can be manually "
                "overridden by setting gpu_flops_available in SpeedMonitor."
            )

    return gpu_flops_available


class SpeedMonitor:
    """Logs the training throughput and utilization.

    +-------------------------------------+-----------------------------------------------------------+
    | Key                                 | Logged data                                               |
    +=====================================+===========================================================+
    |                                     | Rolling average (over `window_size` most recent           |
    | `throughput/batches_per_sec`        | batches) of the number of batches processed per second    |
    |                                     |                                                           |
    +-------------------------------------+-----------------------------------------------------------+
    |                                     | Rolling average (over `window_size` most recent           |
    | `throughput/samples_per_sec`        | batches) of the number of samples processed per second    |
    |                                     |                                                           |
    +-------------------------------------+-----------------------------------------------------------+
    |                                     | Rolling average (over `window_size` most recent           |
    | `throughput/tokens_per_sec`         | batches) of the number of tokens processed per second.    |
    |                                     | This may include padding depending on dataset             |
    +-------------------------------------+-----------------------------------------------------------+
    |                                     | Estimates flops by `flops_per_batch * batches_per_sec`    |
    | `throughput/flops_per_sec`          |                                                           |
    |                                     |                                                           |
    +-------------------------------------+-----------------------------------------------------------+
    | `throughput/device/batches_per_sec` | `throughput/batches_per_sec` divided by world size        |
    +-------------------------------------+-----------------------------------------------------------+
    | `throughput/device/samples_per_sec` | `throughput/samples_per_sec` divided by world size        |
    +-------------------------------------+-----------------------------------------------------------+
    |                                     | `throughput/tokens_per_sec` divided by world size. This   |
    | `throughput/device/tokens_per_sec`  | may include pad tokens depending on dataset               |
    |                                     |                                                           |
    +-------------------------------------+-----------------------------------------------------------+
    |                                     | `throughput/flops_per_sec` divided by world size. Only    |
    | `throughput/device/flops_per_sec`   | logged when model has attribute `flops_per_batch`         |
    |                                     |                                                           |
    +-------------------------------------+-----------------------------------------------------------+
    |                                     | `throughput/device/flops_per_sec` divided by world size.  |
    | `throughput/device/mfu`             |                                                           |
    |                                     |                                                           |
    +-------------------------------------+-----------------------------------------------------------+
    | `time/train`                        | Total elapsed training time                               |
    +-------------------------------------+-----------------------------------------------------------+
    | `time/val`                          | Total elapsed validation time                             |
    +-------------------------------------+-----------------------------------------------------------+
    | `time/total`                        | Total elapsed time (time/train + time/val)                |
    +-------------------------------------+-----------------------------------------------------------+

    Args:
        window_size (int, optional): Number of batches to use for a rolling average of throughput.
            Defaults to 100.
        time_unit (str, optional): Time unit to use for `time` logging. Can be one of
            'seconds', 'minutes', 'hours', or 'days'. Defaults to 'hours'.
    """

    def __init__(self, logger: Logger, precision: str, window_size: int = 100, time_unit: str = "hours"):
        self.logger = logger
        self.gpu_flops_available = get_gpu_flops_available(precision)

        # Track the batch num samples and wct to compute throughput over a window of batches
        self.history_samples: Deque[int] = deque(maxlen=window_size + 1)
        self.history_wct: Deque[float] = deque(maxlen=window_size + 1)
        self.history_flops: Deque[float] = deque(maxlen=window_size + 1)

        self.divider = 1
        if time_unit == "seconds":
            self.divider = 1
        elif time_unit == "minutes":
            self.divider = 60
        elif time_unit == "hours":
            self.divider = 60 * 60
        elif time_unit == "days":
            self.divider = 60 * 60 * 24
        else:
            raise ValueError(
                f'Invalid time_unit: {time_unit}. Must be one of "seconds", "minutes", "hours", or "days".'
            )

        # Keep track of time spent evaluating
        self.total_eval_wct = 0.0
        self.step = -1

    def batch_end(
        self,
        samples: int,  # total samples seen (per device)
        train_elapsed: float,  # total training time (seconds)
        world_size: int,
        flops_per_batch: Optional[float] = None,  # flops per batch (per device)
        max_seq_length: Optional[int] = None,
    ):
        self.history_samples.append(samples)
        self.history_wct.append(train_elapsed)

        self.step += 1
        step = self.step

        if len(self.history_wct) == self.history_wct.maxlen:
            elapsed_batches = len(self.history_samples) - 1
            elapsed_samples = int(self.history_samples[-1]) - int(self.history_samples[0])
            elapsed_wct = self.history_wct[-1] - self.history_wct[0]
            samples_per_sec = elapsed_samples * world_size / elapsed_wct
            dev_samples_per_sec = elapsed_samples / elapsed_wct
            self.logger.log_metrics({"throughput/batches_per_sec": elapsed_batches * world_size / elapsed_wct}, step)
            self.logger.log_metrics({"throughput/samples_per_sec": samples_per_sec}, step)
            self.logger.log_metrics({"throughput/device/batches_per_sec": elapsed_batches / elapsed_wct}, step)
            self.logger.log_metrics({"throughput/device/samples_per_sec": dev_samples_per_sec}, step)

            # Assumes no padding.
            if max_seq_length is not None:
                # Only applicable to seq data / models
                self.logger.log_metrics({"throughput/tokens_per_sec": samples_per_sec * max_seq_length}, step)
                self.logger.log_metrics(
                    {"throughput/device/tokens_per_sec": dev_samples_per_sec * max_seq_length}, step
                )

        if flops_per_batch is not None:
            # sum of flops per batch across ranks
            self.history_flops.append(flops_per_batch * world_size)

        if len(self.history_flops) == self.history_flops.maxlen:
            elapsed_flops = sum(self.history_flops) - self.history_flops[0]
            elapsed_wct = self.history_wct[-1] - self.history_wct[0]
            flops_per_sec = elapsed_flops / elapsed_wct
            device_flops_per_sec = flops_per_sec / world_size
            self.logger.log_metrics({"throughput/flops_per_sec": flops_per_sec}, step)
            self.logger.log_metrics({"throughput/device/flops_per_sec": device_flops_per_sec}, step)
            if self.gpu_flops_available:
                mfu = device_flops_per_sec / self.gpu_flops_available
                self.logger.log_metrics({"throughput/device/mfu": mfu}, step)

        self.logger.log_metrics(
            {
                "time/train": train_elapsed / self.divider,
                "time/val": self.total_eval_wct / self.divider,
                "time/total": (train_elapsed + self.total_eval_wct) / self.divider,
            },
            step,
        )
        self.logger.log_metrics({"samples": samples}, step)

    def eval_end(self, eval_elapsed: float):
        self.total_eval_wct += eval_elapsed  # seconds


def total_flops(model: Parrot):
    n_params = sum(p.numel() for p in model.parameters())
    # credit: https://github.com/mosaicml/llm-foundry/blob/main/scripts/train/benchmarking/collect_results.py#L144-L156
    # mfu is approximated using thoughtput and param count
    # the number of parameters is approximately the number of multiply-accumulates (MAC) in the network
    # each MAC has 2 FLOPs - we multiply by 2 ie 2 * n_param
    # there are 3 passes of a NN (fwd, bwd, delta) - we multiply by 3 ie 2 * 3 * n_param
    # this gets us FLOPs / token
    flops_per_token = 2 * n_params
    flops_per_seq = flops_per_token * model.config.block_size
    # there are 2 FLOPS per mac; there is A=Q*K^T and out=A*V ops (ie mult by 2)
    attn_flops_per_seq = model.config.n_layer * 2 * 2 * (model.config.n_embd * (model.config.block_size**2))
    # there are 2 ops in bwd pass and 1 in fwd pass so we mult by 3
    total_flops = 3 * (flops_per_seq + attn_flops_per_seq)
    return total_flops
