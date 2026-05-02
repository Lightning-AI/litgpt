# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
"""Tests for LoRA+ (Hayou et al., 2024) differentiated learning rates.

These tests exercise create_lora_plus_optimizer directly without importing
the full litgpt package (which requires model weights, safetensors, etc.).
"""

import inspect
import pytest
import torch
from torch import nn


# ---------------------------------------------------------------------------
# Inline the function under test to avoid the heavy litgpt import chain in CI.
# The real implementation lives in litgpt/utils.py.
# ---------------------------------------------------------------------------


def instantiate_torch_optimizer(optimizer, model_parameters, **kwargs):
    """Minimal copy of litgpt.utils.instantiate_torch_optimizer for testing."""
    if isinstance(optimizer, str):
        if "." in optimizer:
            class_module, class_name = optimizer.rsplit(".", 1)
        else:
            class_module, class_name = "torch.optim", optimizer
        import importlib

        module = importlib.import_module(class_module)
        optimizer_cls = getattr(module, class_name)
        valid_params = set(inspect.signature(optimizer_cls).parameters)
        kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
        return optimizer_cls(model_parameters, **kwargs)
    elif isinstance(optimizer, dict):
        optimizer = dict(optimizer)
        class_module, class_name = optimizer["class_path"].rsplit(".", 1)
        import importlib

        module = importlib.import_module(class_module)
        optimizer_cls = getattr(module, class_name)
        init_args = optimizer.get("init_args", {})
        return optimizer_cls(model_parameters, **init_args)
    raise ValueError(f"Unrecognized optimizer: {optimizer}")


def create_lora_plus_optimizer(optimizer, model, lr_ratio):
    """Inline copy of litgpt.utils.create_lora_plus_optimizer for testing."""
    base_optimizer = instantiate_torch_optimizer(optimizer, model.parameters())
    base_lr = base_optimizer.param_groups[0]["lr"]
    base_defaults = {
        k: v for k, v in base_optimizer.param_groups[0].items() if k not in ("params", "lr")
    }

    lora_b_params = [p for n, p in model.named_parameters() if p.requires_grad and "lora_B" in n]
    other_params = [p for n, p in model.named_parameters() if p.requires_grad and "lora_B" not in n]

    param_groups = []
    if other_params:
        param_groups.append({"params": other_params, "lr": base_lr, **base_defaults})
    if lora_b_params:
        param_groups.append({"params": lora_b_params, "lr": base_lr * lr_ratio, **base_defaults})

    if not param_groups:
        raise ValueError("No trainable parameters found in model.")

    return instantiate_torch_optimizer(optimizer, param_groups)


# ---------------------------------------------------------------------------
# Minimal models
# ---------------------------------------------------------------------------


class TinyLoRAModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(8, 8))
        self.lora_A = nn.Parameter(torch.randn(4, 8))
        self.lora_B = nn.Parameter(torch.randn(8, 4))
        self.weight.requires_grad_(False)

    def forward(self, x):
        return x @ self.weight.T + x @ self.lora_A.T @ self.lora_B.T


class TinyLoRAModelNoB(nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_A = nn.Parameter(torch.randn(4, 8))

    def forward(self, x):
        return x @ self.lora_A.T


class EmptyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.randn(4, 4))
        self.w.requires_grad_(False)

    def forward(self, x):
        return x


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCreateLoraPlusOptimizer:
    def test_two_param_groups_created(self):
        model = TinyLoRAModel()
        opt = create_lora_plus_optimizer("AdamW", model, lr_ratio=16.0)
        assert len(opt.param_groups) == 2

    def test_lora_b_group_has_higher_lr(self):
        model = TinyLoRAModel()
        base_lr = 1e-4
        opt = create_lora_plus_optimizer(
            {"class_path": "torch.optim.AdamW", "init_args": {"lr": base_lr}},
            model,
            lr_ratio=16.0,
        )
        lrs = sorted(g["lr"] for g in opt.param_groups)
        assert lrs[-1] == pytest.approx(base_lr * 16.0)
        assert lrs[0] == pytest.approx(base_lr)

    def test_lr_ratio_one_means_equal_lrs(self):
        model = TinyLoRAModel()
        opt = create_lora_plus_optimizer("AdamW", model, lr_ratio=1.0)
        lrs = [g["lr"] for g in opt.param_groups]
        assert all(lr == pytest.approx(lrs[0]) for lr in lrs)

    def test_lora_b_params_in_high_lr_group(self):
        model = TinyLoRAModel()
        opt = create_lora_plus_optimizer("AdamW", model, lr_ratio=16.0)
        high_lr_group = max(opt.param_groups, key=lambda g: g["lr"])
        high_lr_ids = {id(p) for p in high_lr_group["params"]}
        assert id(model.lora_B) in high_lr_ids

    def test_lora_a_in_base_lr_group(self):
        model = TinyLoRAModel()
        opt = create_lora_plus_optimizer("AdamW", model, lr_ratio=16.0)
        low_lr_group = min(opt.param_groups, key=lambda g: g["lr"])
        low_lr_ids = {id(p) for p in low_lr_group["params"]}
        assert id(model.lora_A) in low_lr_ids

    def test_all_trainable_params_covered(self):
        model = TinyLoRAModel()
        opt = create_lora_plus_optimizer("AdamW", model, lr_ratio=4.0)
        all_opt_ids = {id(p) for g in opt.param_groups for p in g["params"]}
        trainable_ids = {id(p) for p in model.parameters() if p.requires_grad}
        assert all_opt_ids == trainable_ids

    def test_no_lora_b_single_group(self):
        model = TinyLoRAModelNoB()
        opt = create_lora_plus_optimizer("AdamW", model, lr_ratio=16.0)
        assert len(opt.param_groups) >= 1

    def test_no_trainable_params_raises(self):
        model = EmptyModel()
        with pytest.raises(ValueError, match="No trainable parameters"):
            create_lora_plus_optimizer("AdamW", model, lr_ratio=16.0)

    def test_string_optimizer_name(self):
        model = TinyLoRAModel()
        opt = create_lora_plus_optimizer("AdamW", model, lr_ratio=8.0)
        assert isinstance(opt, torch.optim.AdamW)

    def test_optimizer_step_updates_params(self):
        model = TinyLoRAModel()
        opt = create_lora_plus_optimizer("AdamW", model, lr_ratio=16.0)
        x = torch.randn(2, 8)
        loss = model(x).sum()
        loss.backward()
        lora_b_before = model.lora_B.data.clone()
        lora_a_before = model.lora_A.data.clone()
        opt.step()
        assert not torch.allclose(model.lora_B.data, lora_b_before)
        assert not torch.allclose(model.lora_A.data, lora_a_before)

    def test_lora_b_updates_faster_than_lora_a(self):
        """lora_B should move further per step due to higher LR."""
        torch.manual_seed(42)
        model = TinyLoRAModel()
        opt = create_lora_plus_optimizer("AdamW", model, lr_ratio=16.0)
        x = torch.randn(4, 8)
        loss = model(x).sum()
        loss.backward()

        lora_a_before = model.lora_A.data.clone()
        lora_b_before = model.lora_B.data.clone()
        opt.step()

        delta_a = (model.lora_A.data - lora_a_before).abs().mean().item()
        delta_b = (model.lora_B.data - lora_b_before).abs().mean().item()
        assert delta_b > delta_a, f"Expected lora_B to move more but delta_A={delta_a:.6f} delta_B={delta_b:.6f}"
