# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

from litgpt.utils import _RunIf
from lightning.fabric.plugins.precision.bitsandbytes import _BITSANDBYTES_AVAILABLE


@_RunIf(min_cuda_gpus=1)
def test_gpu_ci_installs_bitsandbytes():
    assert _BITSANDBYTES_AVAILABLE, str(_BITSANDBYTES_AVAILABLE)
