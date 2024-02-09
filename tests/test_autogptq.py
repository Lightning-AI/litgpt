import json
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import Mock

import pytest
import torch
from conftest import RunIf


@RunIf(min_cuda_gpus=1)
@pytest.mark.parametrize("mlp_class", ("GptNeoxMLP", "LLaMAMLP", "LLaMAMoE"))
@pytest.mark.parametrize("bits", [2, 3, 4, 8], ids=[f"{bit}bit" for bit in (2, 3, 4, 8)])
@pytest.mark.parametrize("group_size", [32, 128], ids=[f"{gs}group_size" for gs in (32, 128)])
def test_autogptq_quantization_mlp_layers(tmp_path, fake_checkpoint_dir, monkeypatch, bits, group_size, mlp_class):

    import math
    from dataclasses import asdict
    from functools import reduce

    from lit_gpt import GPT
    from lit_gpt.config import Config
    from quantize import autogptq as module

    # Prepare calibration data
    data = [
        {"input_ids": torch.tensor([0, 1, 2]), "labels": torch.tensor([1, 2, 3])},
        {"input_ids": torch.tensor([1, 2, 3]), "labels": torch.tensor([2, 3, 4])},
    ]
    torch.save(data, tmp_path / "test.pt")

    # Prepare model's config
    config = Config(
        block_size=128,
        vocab_size=50,
        n_layer=2,
        n_head=4,
        n_embd=8,
        intermediate_size=32,
        _mlp_class=mlp_class,
    )
    config_path = fake_checkpoint_dir / "lit_config.json"
    config_path.write_text(json.dumps(asdict(config)))

    # Mock weights loading and a tokenizer
    monkeypatch.setattr(module, "load_checkpoint", Mock())
    tokenizer_mock = Mock()
    tokenizer_mock.eos_id.return_value = 0
    monkeypatch.setattr(module, "Tokenizer", tokenizer_mock)

    # Run quantization, check that the time for quantization is printed
    stdout = StringIO()
    with redirect_stdout(stdout):
        module.main(data_dir=tmp_path, checkpoint_dir=fake_checkpoint_dir, bits=bits, group_size=group_size)
    assert "Quantization time" in stdout.getvalue()

    # Assert that the quantized model weights are saved
    assert "lit_model_gptq.4bit.pth" in [p.name for p in fake_checkpoint_dir.glob("*")]

    # --- Validate the saved quantized weights ---
    quantized_state_dict = torch.load(fake_checkpoint_dir / "lit_model_gptq.4bit.pth")

    # Create a reference model to check that the saved quantized weights have a proper shape
    reference_model = GPT(config)
    reference_model.config.model_type = None
    # Retrieve `inside_layer_modules` - they control what layers inside each Transformer Block
    # are quantized
    autogptq = module.AutoGPTQ(reference_model, quantized=False, quantize_config=None)
    inside_layer_modules = autogptq.inside_layer_modules
    inside_layer_modules = sum(inside_layer_modules, [])

    for layer_name, quant_tensor in quantized_state_dict.items():
        reference_layer = reduce(getattr, layer_name.split(".")[:-1], reference_model)
        if not any(ilm in layer_name for ilm in inside_layer_modules):
            assert not layer_name.endswith((".qweight", ".qzeros", ".scales", "g_idx"))
            assert quant_tensor.dtype != torch.int32
            assert quant_tensor.shape == reference_layer.weight.shape
        else:
            assert layer_name.endswith((".qweight", ".qzeros", ".scales", ".g_idx", ".bias"))
            if layer_name.endswith(".qweight"):
                assert quant_tensor.dtype == torch.int32
                assert quant_tensor.shape == (reference_layer.in_features // 32 * bits, reference_layer.out_features)
            elif layer_name.endswith(".qzeros"):
                assert quant_tensor.dtype == torch.int32
                assert quant_tensor.shape == (
                    math.ceil(reference_layer.in_features / group_size),
                    reference_layer.out_features // 32 * bits,
                )
            elif layer_name.endswith(".scales"):
                assert quant_tensor.dtype == torch.float16
                assert quant_tensor.shape == (
                    math.ceil(reference_layer.in_features / group_size),
                    reference_layer.out_features,
                )
            elif layer_name.endswith(".g_idx"):
                assert quant_tensor.dtype == torch.int32
                assert quant_tensor.shape == (reference_layer.in_features,)
            else:
                # bias is not quantized and is created by AutoGPTQ despite the config
                continue
