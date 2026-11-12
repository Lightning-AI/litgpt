# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

from copy import deepcopy

import pytest
import torch

from litgpt.config import Config
from litgpt.model import GPT
from litgpt.utils import instantiate_torch_optimizer

OPTIMIZER_NAMES = ["SGD", "AdamW", "RMSprop", "Adagrad"]


def optimizer_config(name):
    options = {"lr": 0.05, "weight_decay": 0.02}
    if name in {"SGD", "RMSprop"}:
        options["momentum"] = 0.6
    elif name == "AdamW":
        options["betas"] = (0.8, 0.9)
    else:
        options["lr_decay"] = 0.1
    return {"class_path": f"torch.optim.{name}", "init_args": options}


@pytest.mark.parametrize("name", OPTIMIZER_NAMES)
def test_runtime_options_do_not_mutate_the_callers_configuration(name):
    config = optimizer_config(name)
    original = deepcopy(config)
    original_options = config["init_args"]
    parameter = torch.nn.Parameter(torch.ones(3))
    optimizer = instantiate_torch_optimizer(config, [parameter], lr=0.003, weight_decay=0.1, unsupported_option=True)
    assert optimizer.defaults["lr"] == 0.003
    assert optimizer.defaults["weight_decay"] == 0.1
    assert "unsupported_option" not in optimizer.defaults
    assert config == original
    assert config["init_args"] is original_options


@pytest.mark.parametrize("name", ["SGD", "AdamW"])
@pytest.mark.parametrize("separate_groups", [False, True])
def test_reusing_configuration_preserves_native_gpt_updates_and_resume(name, separate_groups, tmp_path):
    torch.manual_seed(19)
    config = optimizer_config(name)
    original = deepcopy(config)
    first = instantiate_torch_optimizer(config, [torch.nn.Parameter(torch.ones(2))], lr=0.003, weight_decay=0.1)
    assert first.defaults["lr"] == 0.003

    model = GPT(Config(n_layer=1, n_embd=8, n_head=2, block_size=6, padded_vocab_size=16))
    reference = deepcopy(model)

    def parameters(module):
        if not separate_groups:
            return module.parameters()
        return [
            {"params": list(module.transformer.parameters())},
            {"params": list(module.lm_head.parameters()), "lr": 0.02},
        ]

    actual_optimizer = instantiate_torch_optimizer(config, parameters(model))
    reference_optimizer = getattr(torch.optim, name)(parameters(reference), **original["init_args"])
    generator = torch.Generator().manual_seed(27)
    for step in range(4):
        inputs = torch.randint(0, 16, (2, 5), generator=generator)
        targets = torch.randint(0, 16, (2, 5), generator=generator)
        actual_optimizer.zero_grad()
        reference_optimizer.zero_grad()
        actual_loss = torch.nn.functional.cross_entropy(model(inputs).reshape(-1, 16), targets.reshape(-1))
        expected_loss = torch.nn.functional.cross_entropy(reference(inputs).reshape(-1, 16), targets.reshape(-1))
        torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
        actual_loss.backward()
        expected_loss.backward()
        for actual, expected in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(actual.grad, expected.grad, rtol=0, atol=0)
        actual_optimizer.step()
        reference_optimizer.step()
        torch.testing.assert_close(model.state_dict(), reference.state_dict(), rtol=0, atol=0)
        torch.testing.assert_close(actual_optimizer.state_dict(), reference_optimizer.state_dict(), rtol=0, atol=0)
        if step == 1:
            checkpoint = tmp_path / "training.pth"
            torch.save({"model": model.state_dict(), "optimizer": actual_optimizer.state_dict()}, checkpoint)
            state = torch.load(checkpoint, weights_only=True)
            resumed = GPT(model.config)
            resumed.load_state_dict(state["model"])
            resumed_optimizer = instantiate_torch_optimizer(config, parameters(resumed))
            resumed_optimizer.load_state_dict(state["optimizer"])
            model, actual_optimizer = resumed, resumed_optimizer
    assert config == original


@pytest.mark.parametrize("name", OPTIMIZER_NAMES)
def test_rejected_runtime_options_leave_configuration_usable(name):
    config = optimizer_config(name)
    original = deepcopy(config)
    parameter = torch.nn.Parameter(torch.ones(3))
    with pytest.raises(ValueError, match="learning rate"):
        instantiate_torch_optimizer(config, [parameter], lr=-0.1)
    assert config == original
    recovered = instantiate_torch_optimizer(config, [parameter])
    assert recovered.defaults["lr"] == original["init_args"]["lr"]


def test_two_configurations_sharing_options_remain_independent():
    options = {"lr": 0.1, "momentum": 0.8}
    first_config = {"class_path": "torch.optim.SGD", "init_args": options}
    second_config = {"class_path": "torch.optim.RMSprop", "init_args": options}
    instantiate_torch_optimizer(first_config, [torch.nn.Parameter(torch.ones(2))], lr=0.01)
    parameter = torch.nn.Parameter(torch.ones(2))
    reference = torch.nn.Parameter(torch.ones(2))
    actual_optimizer = instantiate_torch_optimizer(second_config, [parameter])
    expected_optimizer = torch.optim.RMSprop([reference], lr=0.1, momentum=0.8)
    for gradient in [0.2, -0.5, 0.7]:
        parameter.grad = torch.full_like(parameter, gradient)
        reference.grad = torch.full_like(reference, gradient)
        actual_optimizer.step()
        expected_optimizer.step()
        torch.testing.assert_close(parameter, reference, rtol=0, atol=0)
    assert first_config["init_args"] is second_config["init_args"] is options
    assert options == {"lr": 0.1, "momentum": 0.8}


@pytest.mark.parametrize("name", OPTIMIZER_NAMES)
def test_string_configuration_still_filters_runtime_options(name):
    parameter = torch.nn.Parameter(torch.ones(3))
    optimizer = instantiate_torch_optimizer(name, [parameter], lr=0.03, unsupported_option=True)
    reference = getattr(torch.optim, name)([torch.nn.Parameter(torch.ones(3))], lr=0.03)
    assert optimizer.defaults == reference.defaults
