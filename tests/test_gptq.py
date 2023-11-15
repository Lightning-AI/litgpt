import lightning as L
import pytest
import torch
from lightning.fabric.utilities.imports import _TORCH_GREATER_EQUAL_2_2


@pytest.mark.skipif(_TORCH_GREATER_EQUAL_2_2, reason="Core dumped")
def test_gptq_blockwise_quantization():
    from quantize.gptq import _TRITON_AVAILABLE

    if not _TRITON_AVAILABLE:
        pytest.skip(str(_TRITON_AVAILABLE))

    from lit_gpt import GPT

    fabric = L.Fabric(devices=1)
    with fabric.init_module(empty_init=False):
        model = GPT.from_name("pythia-70m", n_layer=2)
        x = torch.randint(0, 10, (2, model.config.block_size))

    from quantize.gptq import blockwise_quantization

    blockwise_quantization(model, x, fabric.device)
