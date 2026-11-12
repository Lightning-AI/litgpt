# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

from collections import OrderedDict
from copy import deepcopy

import pytest
import torch
from torch import nn

from litgpt.lora import GPT, Config, LoRALinear, LoRAQKVLinear, mark_only_lora_as_trainable


@pytest.fixture(autouse=True)
def _restore_rng():
    with torch.random.fork_rng():
        torch.manual_seed(17)
        yield


def _set_reference_policy(model, policy):
    for name, parameter in model.named_parameters():
        enabled = "lora_" in name
        if policy == "all":
            enabled |= "bias" in name
        elif policy == "lora_only":
            enabled |= name.endswith(".linear.bias")
        parameter.requires_grad_(enabled)


def _assert_gradients_and_step(model, reference, optimizer, reference_optimizer):
    for (name, parameter), (_, expected) in zip(model.named_parameters(), reference.named_parameters()):
        assert parameter.requires_grad == expected.requires_grad, name
        if expected.grad is None:
            assert parameter.grad is None, name
        else:
            assert parameter.grad is not None, name
            torch.testing.assert_close(parameter.grad, expected.grad)
    optimizer.step()
    reference_optimizer.step()
    for parameter, expected in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(parameter, expected)


@pytest.mark.parametrize("kind", ["linear", "mha", "gqa"])
@pytest.mark.parametrize("policy", ["none", "lora_only", "all"])
@pytest.mark.parametrize("bias", [False, True])
def test_lora_bias_policy_updates_only_selected_parameters(kind, policy, bias):
    if kind == "linear":
        layer = LoRALinear(4, 8, r=2, bias=bias)
    else:
        groups = 2 if kind == "mha" else 1
        layer = LoRAQKVLinear(
            4,
            4 + 4 * groups,
            head_size=2,
            n_head=2,
            n_query_groups=groups,
            r=2,
            enable_lora=(True, False, True),
            bias=bias,
        )
    model = nn.Sequential(OrderedDict(adapter=layer, other=nn.Linear(layer.linear.out_features, 2)))
    reference = deepcopy(model)
    _set_reference_policy(reference, policy)
    mark_only_lora_as_trainable(model, bias=policy)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
    initial = {name: value.detach().clone() for name, value in model.named_parameters()}

    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        reference_optimizer.zero_grad(set_to_none=True)
        inputs = torch.randn(2, 3, 4, requires_grad=True)
        reference_inputs = inputs.detach().clone().requires_grad_()
        output = model(inputs)
        reference_output = reference(reference_inputs)
        torch.testing.assert_close(output, reference_output)
        output.square().mean().backward()
        reference_output.square().mean().backward()
        torch.testing.assert_close(inputs.grad, reference_inputs.grad)
        _assert_gradients_and_step(model, reference, optimizer, reference_optimizer)

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            torch.testing.assert_close(parameter, initial[name], rtol=0, atol=0)
    if bias and policy != "none":
        assert not torch.equal(model.adapter.linear.bias, initial["adapter.linear.bias"])


@pytest.mark.parametrize("kind", ["linear", "qkv"])
@pytest.mark.parametrize("policy", ["none", "lora_only", "all"])
def test_lora_bias_policy_preserves_rank_zero_wrappers(kind, policy):
    if kind == "linear":
        layer = LoRALinear(4, 8, r=0, bias=True)
    else:
        layer = LoRAQKVLinear(4, 8, head_size=2, n_head=2, n_query_groups=1, r=0, enable_lora=True, bias=True)
    mark_only_lora_as_trainable(layer, bias=policy)
    assert not layer.linear.weight.requires_grad
    assert layer.linear.bias.requires_grad == (policy != "none")


@pytest.mark.parametrize("policy", ["none", "lora_only", "all"])
def test_lora_gpt_bias_policy_matches_explicit_training_selection(policy):
    model = GPT(
        Config(
            n_layer=1,
            n_head=2,
            n_embd=8,
            n_query_groups=1,
            block_size=8,
            vocab_size=8,
            padding_multiple=1,
            bias=True,
            lm_head_bias=True,
            lora_r=2,
            lora_query=True,
            lora_value=True,
            lora_projection=True,
            lora_mlp=True,
            lora_head=True,
        )
    )
    reference = deepcopy(model)
    _set_reference_policy(reference, policy)
    mark_only_lora_as_trainable(model, bias=policy)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
    input_ids = torch.tensor([[1, 2, 3, 4], [2, 4, 6, 7]])
    targets = torch.tensor([[2, 3, 4, 5], [4, 6, 7, 1]])

    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        reference_optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids)
        reference_logits = reference(input_ids)
        loss = nn.functional.cross_entropy(logits.flatten(0, 1), targets.flatten())
        reference_loss = nn.functional.cross_entropy(reference_logits.flatten(0, 1), targets.flatten())
        torch.testing.assert_close(loss, reference_loss)
        loss.backward()
        reference_loss.backward()
        _assert_gradients_and_step(model, reference, optimizer, reference_optimizer)
