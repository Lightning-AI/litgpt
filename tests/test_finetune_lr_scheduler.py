# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import importlib
import io
import math

import pytest
import torch


@pytest.mark.parametrize("variant", ["lora", "lora_legacy", "full", "adapter", "adapter_v2"])
@pytest.mark.parametrize("warmup_steps", [0, 2])
def test_finetune_lr_scheduler_trajectory(variant, warmup_steps):
    get_lr_scheduler = importlib.import_module(f"litgpt.finetune.{variant}").get_lr_scheduler
    parameters = [torch.nn.Parameter(torch.ones(1, dtype=torch.float64)) for _ in range(2)]
    base_lrs = [0.1, 0.025]
    optimizer = torch.optim.SGD([{"params": [p], "lr": lr} for p, lr in zip(parameters, base_lrs)])
    scheduler = get_lr_scheduler(optimizer, warmup_steps=warmup_steps, max_steps=8)
    expected_parameters = [1.0, 1.0]

    for step in range(9):
        if step < warmup_steps:
            factor = step / warmup_steps
        else:
            factor = (1 + math.cos(math.pi * (step - warmup_steps) / (8 - warmup_steps))) / 2
        expected_lrs = [lr * factor for lr in base_lrs]
        assert scheduler.get_last_lr() == pytest.approx(expected_lrs)
        for parameter, expected in zip(parameters, expected_parameters):
            assert parameter.item() == pytest.approx(expected)
        if step < 8:
            for parameter in parameters:
                parameter.grad = torch.ones_like(parameter)
            optimizer.step()
            scheduler.step()
            expected_parameters = [value - lr for value, lr in zip(expected_parameters, expected_lrs)]


@pytest.mark.parametrize("variant", ["lora", "lora_legacy", "full", "adapter", "adapter_v2"])
@pytest.mark.parametrize("warmup_steps", [0, 2])
def test_finetune_lr_scheduler_checkpoint_resume(variant, warmup_steps):
    get_lr_scheduler = importlib.import_module(f"litgpt.finetune.{variant}").get_lr_scheduler
    parameter = torch.nn.Parameter(torch.ones(1, dtype=torch.float64))
    optimizer = torch.optim.SGD([parameter], lr=0.1, momentum=0.8)
    scheduler = get_lr_scheduler(optimizer, warmup_steps=warmup_steps, max_steps=8)
    for _ in range(3):
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        scheduler.step()

    checkpoint = io.BytesIO()
    torch.save(
        {"parameter": parameter.detach(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()},
        checkpoint,
    )
    checkpoint.seek(0)
    state = torch.load(checkpoint, weights_only=True)
    restored_parameter = torch.nn.Parameter(state["parameter"])
    restored_optimizer = torch.optim.SGD([restored_parameter], lr=0.1, momentum=0.8)
    restored_scheduler = get_lr_scheduler(restored_optimizer, warmup_steps=warmup_steps, max_steps=8)
    restored_optimizer.load_state_dict(state["optimizer"])
    restored_scheduler.load_state_dict(state["scheduler"])

    for step in range(3, 8):
        assert restored_scheduler.get_last_lr() == scheduler.get_last_lr()
        for current_parameter, current_optimizer, current_scheduler in (
            (parameter, optimizer, scheduler),
            (restored_parameter, restored_optimizer, restored_scheduler),
        ):
            current_parameter.grad = torch.full_like(current_parameter, (step + 1) / 10)
            current_optimizer.step()
            current_scheduler.step()
        torch.testing.assert_close(restored_parameter, parameter, rtol=0, atol=0)
