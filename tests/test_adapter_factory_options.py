# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import pytest
import torch

from litgpt.adapter_v2 import AdapterV2Linear
from litgpt.lora import LoRALinear, LoRAQKVLinear

KINDS = ["linear", "qkv_mha", "qkv_gqa", "adapter_v2"]


def make_layer(kind, **factory_kwargs):
    if kind == "linear":
        return LoRALinear(4, 8, r=2, lora_alpha=3, **factory_kwargs)
    if kind == "adapter_v2":
        return AdapterV2Linear(4, 8, **factory_kwargs)
    groups = 2 if kind == "qkv_mha" else 1
    return LoRAQKVLinear(
        4,
        4 + 4 * groups,
        head_size=2,
        n_head=2,
        n_query_groups=groups,
        r=2,
        lora_alpha=3,
        enable_lora=(True, False, True),
        **factory_kwargs,
    )


def initialize_adapter(layer):
    with torch.no_grad():
        if hasattr(layer, "lora_B"):
            layer.lora_B.uniform_(-0.2, 0.2)
        else:
            layer.adapter_bias.fill_(0.1)
            layer.adapter_scale.fill_(0.9)


def assert_factory_parameters(layer, device, dtype):
    assert all(parameter.device == torch.device(device) for parameter in layer.parameters())
    assert all(parameter.dtype == dtype for parameter in layer.parameters())


@pytest.fixture(autouse=True)
def preserve_random_state():
    with torch.random.fork_rng():
        torch.manual_seed(123)
        yield


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16])
def test_explicit_dtype_matches_converting_the_complete_layer(kind, dtype):
    layer = make_layer(kind, device="cpu", dtype=dtype)
    reference = make_layer(kind).to(dtype=dtype)
    initialize_adapter(reference)
    layer.load_state_dict(reference.state_dict())
    assert_factory_parameters(layer, "cpu", dtype)
    # Adapter V2's constructor must retain its original frozen-parameter defaults.
    assert {name: p.requires_grad for name, p in layer.named_parameters()} == {
        name: p.requires_grad for name, p in reference.named_parameters()
    }
    layer.requires_grad_(True)
    reference.requires_grad_(True)
    optimizer = torch.optim.SGD(layer.parameters(), lr=0.05, momentum=0.8)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.05, momentum=0.8)

    for _ in range(2):
        inputs = torch.randn(2, 3, 4, dtype=dtype, requires_grad=True)
        reference_inputs = inputs.detach().clone().requires_grad_(True)
        output = layer(inputs)
        expected = reference(reference_inputs)
        assert output.dtype == dtype
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        output.square().mean().backward()
        expected.square().mean().backward()
        torch.testing.assert_close(inputs.grad, reference_inputs.grad, rtol=0, atol=0)
        for parameter, reference_parameter in zip(layer.parameters(), reference.parameters()):
            torch.testing.assert_close(parameter.grad, reference_parameter.grad, rtol=0, atol=0)
        optimizer.step()
        reference_optimizer.step()
        optimizer.zero_grad()
        reference_optimizer.zero_grad()
        for parameter, reference_parameter in zip(layer.parameters(), reference.parameters()):
            torch.testing.assert_close(parameter, reference_parameter, rtol=0, atol=0)


@pytest.mark.parametrize("kind", KINDS)
def test_explicit_meta_initialization_and_materialization(kind):
    layer = make_layer(kind, device="meta", dtype=torch.float64)
    assert_factory_parameters(layer, "meta", torch.float64)
    inputs = torch.ones(2, 3, 4, device="meta", dtype=torch.float64, requires_grad=True)
    layer(inputs).sum().backward()
    assert inputs.grad.device.type == "meta"

    # Materialize a freshly initialized layer as in deferred model initialization.
    restored = make_layer(kind, device="meta", dtype=torch.float64).to_empty(device="cpu")
    reference = make_layer(kind).to(dtype=torch.float64)
    initialize_adapter(reference)
    restored.load_state_dict(reference.state_dict())
    assert_factory_parameters(restored, "cpu", torch.float64)
    sample = torch.randn(2, 3, 4, dtype=torch.float64)
    torch.testing.assert_close(restored(sample), reference(sample), rtol=0, atol=0)


@pytest.mark.parametrize("kind", KINDS)
def test_explicit_cpu_device_overrides_meta_context(kind):
    with torch.device("meta"):
        layer = make_layer(kind, device="cpu", dtype=torch.float64)
    assert_factory_parameters(layer, "cpu", torch.float64)
    inputs = torch.ones(2, 3, 4, dtype=torch.float64, requires_grad=True)
    layer(inputs).sum().backward()
    assert torch.isfinite(inputs.grad).all()


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("explicit_none", [False, True])
def test_omitted_factory_options_preserve_global_defaults(kind, explicit_none):
    original_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        with torch.device("meta"):
            kwargs = {"device": None, "dtype": None} if explicit_none else {}
            layer = make_layer(kind, **kwargs)
        assert_factory_parameters(layer, "meta", torch.float64)
    finally:
        torch.set_default_dtype(original_dtype)
