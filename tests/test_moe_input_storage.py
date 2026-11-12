# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

from copy import deepcopy

import pytest
import torch
from torch.profiler import ProfilerActivity, profile

from litgpt.config import Config
from litgpt.model import GPT, LLaMAMoE


def _config(shared, grouped=False):
    return Config(
        block_size=32,
        n_layer=2,
        n_embd=16,
        n_head=4,
        vocab_size=16,
        padded_vocab_size=16,
        intermediate_size=24,
        mlp_class_name="LLaMAMoE",
        moe_intermediate_size=24,
        n_expert=4,
        n_expert_per_token=2,
        n_shared_expert=shared,
        n_expert_groups=2 if grouped else None,
        n_topk_groups=1 if grouped else None,
        n_topk_scores_per_group=1 if grouped else None,
        norm_topk_prob=grouped,
        routed_scaling_factor=1.25,
    )


def _model(shared, grouped=False, dtype=torch.float32):
    torch.manual_seed(23)
    model = LLaMAMoE(_config(shared, grouped)).to(dtype)
    if grouped:
        torch.nn.init.uniform_(model.gate.weight, -0.2, 0.2)
    return model


def _dense_reference(model, x):
    """Evaluate every expert, then select and weight each token's routed results."""
    flat = x.reshape(-1, x.size(-1))
    if model.config.n_expert_groups:
        weights, indices = model.gate(flat)
    else:
        logits = model.gate(flat)
        selected, indices = logits.topk(model.config.n_expert_per_token)
        weights = selected.softmax(dim=-1, dtype=torch.float32).to(x.dtype)
    weights = weights * model.config.routed_scaling_factor
    all_outputs = torch.stack([expert(flat) for expert in model.experts], dim=1)
    selected_outputs = all_outputs.gather(1, indices.unsqueeze(-1).expand(-1, -1, x.size(-1)))
    routed = (selected_outputs * weights.unsqueeze(-1)).sum(dim=1)
    output = routed.reshape_as(x)
    if model.config.n_shared_expert:
        output = output + model.shared_experts(x)
    return output


@pytest.mark.parametrize("shared", [None, 1, 2])
@pytest.mark.parametrize("requires_grad", [False, True])
def test_moe_does_not_allocate_a_full_input_copy(shared, requires_grad):
    model = _model(shared)
    x = torch.randn(2, 257, 16, requires_grad=requires_grad)
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True, profile_memory=True) as prof:
        with torch.set_grad_enabled(requires_grad):
            output = model(x)
    copies = [
        event.cpu_memory_usage
        for event in prof.events()
        if event.name == "aten::clone" and event.input_shapes[0] == list(x.shape)
    ]
    assert sum(copies) == 0
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("shared", [1, 2])
@pytest.mark.parametrize("grouped", [False, True])
def test_shared_experts_reuse_the_saved_input_storage(shared, grouped):
    model = _model(shared, grouped)
    x = torch.randn(2, 257, 16, requires_grad=True)
    saved_storages = set()

    def pack(tensor):
        if tensor.numel() == x.numel() and tensor.shape[-1] == x.shape[-1]:
            saved_storages.add(tensor.untyped_storage().data_ptr())
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        output = model(x)
    assert saved_storages == {x.untyped_storage().data_ptr()}
    output.square().mean().backward()
    assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize("shared", [None, 1, 2])
@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_moe_outputs_gradients_and_updates_match_dense_reference(shared, grouped, dtype):
    model = _model(shared, grouped, dtype)
    reference = deepcopy(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.003, momentum=0.9)
    x = torch.randn(2, 11, 16, dtype=dtype, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    original = x.detach().clone()
    for _ in range(3):
        optimizer.zero_grad()
        reference_optimizer.zero_grad()
        x.grad = reference_x.grad = None
        actual = model(x)
        expected = _dense_reference(reference, reference_x)
        torch.testing.assert_close(actual, expected)
        actual.square().mean().backward()
        expected.square().mean().backward()
        torch.testing.assert_close(x.grad, reference_x.grad)
        for parameter, ref_parameter in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(parameter.grad, ref_parameter.grad)
        optimizer.step()
        reference_optimizer.step()
        for parameter, ref_parameter in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(parameter, ref_parameter)
        torch.testing.assert_close(x, original, rtol=0, atol=0)


@pytest.mark.parametrize("shared", [None, 1, 2])
def test_native_gpt_moe_training_matches_explicit_expert_reference(shared):
    torch.manual_seed(41)
    model = GPT(_config(shared))
    reference = deepcopy(model)
    for block in reference.transformer.h:
        module = block.mlp
        module.forward = lambda x, module=module: _dense_reference(module, x)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01, momentum=0.9)
    tokens = torch.randint(0, 16, (2, 11))
    for _ in range(2):
        optimizer.zero_grad()
        reference_optimizer.zero_grad()
        logits = model(tokens)
        expected = reference(tokens)
        torch.testing.assert_close(logits, expected)
        torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 16), tokens[:, 1:].reshape(-1)).backward()
        torch.nn.functional.cross_entropy(expected[:, :-1].reshape(-1, 16), tokens[:, 1:].reshape(-1)).backward()
        for parameter, ref_parameter in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(parameter.grad, ref_parameter.grad)
        optimizer.step()
        reference_optimizer.step()
    for parameter, ref_parameter in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(parameter, ref_parameter)
