from contextlib import nullcontext
from functools import reduce

import pytest
import torch
from conftest import RunIf


@RunIf(min_cuda_gpus=1)
@pytest.mark.parametrize("bits", [2, 3, 4, 8], ids=[f"{bit}bit" for bit in (2, 3, 4, 8)])
@pytest.mark.parametrize("group_size", [32, 128], ids=[f"{gs}group_size" for gs in (32, 128)])
@pytest.mark.parametrize("use_triton", (True, False), ids=["use_triton", "dont_use_triton"])
@pytest.mark.parametrize("mlp_class", ("GptNeoxMLP", "LLaMAMLP", "LLaMAMoE"))
def test_autogptq_quantization_mlp_layers(
    tmp_path, fake_checkpoint_dir, monkeypatch, bits, group_size, use_triton, mlp_class
):
    if use_triton and bits == 3:
        pytest.skip("Triton doesn't support 3bit precision.")

    import json
    import math
    from contextlib import redirect_stdout
    from dataclasses import asdict
    from io import StringIO
    from unittest.mock import Mock

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
        padded_vocab_size=10_000,
        n_layer=2,
        n_embd=2 * group_size,
        n_head=8,
        n_query_groups=2,
        intermediate_size=group_size,
        _mlp_class=mlp_class,
        n_expert=4 if mlp_class == "LLaMAMoE" else 0,
        n_expert_per_token=2 if mlp_class == "LLaMAMoE" else 0,
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
        module.main(
            data_dir=tmp_path,
            checkpoint_dir=fake_checkpoint_dir,
            bits=bits,
            group_size=group_size,
            use_triton=use_triton,
        )
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

    for layer_name, weight in quantized_state_dict.items():
        reference_layer = reduce(getattr, layer_name.split(".")[:-1], reference_model)
        if not any(ilm in layer_name for ilm in inside_layer_modules):
            assert layer_name.endswith((".weight", ".bias"))
            assert weight.dtype != torch.int32
            assert weight.shape == reference_layer.weight.shape
        else:
            assert layer_name.endswith((".qweight", ".qzeros", ".scales", ".g_idx", ".bias"))
            if layer_name.endswith(".qweight"):
                assert weight.dtype == torch.int32
                assert weight.shape == (reference_layer.in_features // 32 * bits, reference_layer.out_features)
            elif layer_name.endswith(".qzeros"):
                assert weight.dtype == torch.int32
                assert weight.shape == (
                    math.ceil(reference_layer.in_features / group_size),
                    reference_layer.out_features // 32 * bits,
                )
            elif layer_name.endswith(".scales"):
                assert weight.dtype == torch.float16
                assert weight.shape == (
                    math.ceil(reference_layer.in_features / group_size),
                    reference_layer.out_features,
                )
            elif layer_name.endswith(".g_idx"):
                assert weight.dtype == torch.int32
                assert weight.shape == (reference_layer.in_features,)
            else:
                # bias is not quantized and is created by AutoGPTQ despite the config
                continue


@RunIf(min_cuda_gpus=1)
@pytest.mark.parametrize("kernel", ("cuda", "exllama", "exllamav2", "triton"))
@pytest.mark.parametrize("bits", [2, 3, 4, 8], ids=[f"{bit}bit" for bit in (2, 3, 4, 8)])
@pytest.mark.parametrize("group_size", [32, 128], ids=[f"{gs}group_size" for gs in (32, 128)])
@pytest.mark.parametrize("mlp_class", ("GptNeoxMLP", "LLaMAMLP", "LLaMAMoE"))
def test_autogptq_convert_layers(kernel, bits, group_size, mlp_class):

    import importlib
    from functools import reduce

    from auto_gptq.modeling._base import BaseQuantizeConfig

    from lit_gpt import GPT
    from lit_gpt.config import Config
    from quantize.autogptq import AutoGPTQ

    # Prepare model's config
    config = Config(
        padded_vocab_size=10_000,
        n_layer=2,
        n_embd=2 * group_size,
        n_head=8,
        n_query_groups=2,
        intermediate_size=group_size,
        _mlp_class=mlp_class,
        n_expert=4 if mlp_class == "LLaMAMoE" else 0,
        n_expert_per_token=2 if mlp_class == "LLaMAMoE" else 0,
    )

    # Create a model: it has to be on a GPU and with float16 precision
    device = "cuda:0"
    model = GPT(config).to(device=device, dtype=torch.float16)
    model.config.model_type = None  # used in .from_pretrained and .from_quantized
    model.config.pad_token_id = None  # ._prepare_examples_for_quantization
    model.config.eos_token_id = 0  # _prepare_examples_for_quantization
    model.config.use_cache = False  # for quantization it's disabled anyway

    # Wrap the model in AutoGPTQ as it allows to convert "nn.Linear" layers to "QuantLinear"
    quantize_config = BaseQuantizeConfig(bits=bits, group_size=group_size)
    autogptq_model = AutoGPTQ(model, quantized=True, quantize_config=quantize_config)

    # Some kernels support only specific set of precision. The code has to tell about it.
    # We should check it.
    skip_test = False
    if (kernel == "triton" and bits == 3) or (kernel in ("exllama", "exllamav2") and bits != 4):
        skip_test = True
    with pytest.raises(ValueError, match="doesn't support") if skip_test else nullcontext():
        autogptq_model.convert_model_to_quantized(kernel)
    if skip_test:
        pytest.skip(f"Kernel `{kernel}` doesn't support {bits}bit precision.")

    # Convert layers and run obligatory "post_init" method: initializes kernel's buffers
    autogptq_model.post_init()

    # Check that all the target layer were converted
    inside_layer_modules = autogptq_model.inside_layer_modules
    inside_layer_modules = sum(inside_layer_modules, [])

    QuantLinear = importlib.import_module(f"auto_gptq.nn_modules.qlinear.qlinear_{kernel}").QuantLinear

    for layer_name in autogptq_model.model.state_dict():
        module = reduce(getattr, layer_name.split(".")[:-1], autogptq_model.model)
        if any(ilm in layer_name for ilm in inside_layer_modules):
            assert layer_name.endswith((".qweight", ".qzeros", ".scales", ".g_idx", ".bias")), layer_name
            assert isinstance(module, QuantLinear), layer_name
        else:
            assert layer_name.endswith((".weight", ".bias")), layer_name
            assert not isinstance(module, QuantLinear), layer_name

    # Run a forward pass, it should not fail
    x = torch.tensor([[9856, 23, 491, 1536, 304], [23, 345, 65, 123, 321]], dtype=torch.int32, device=device)
    autogptq_model(x)
