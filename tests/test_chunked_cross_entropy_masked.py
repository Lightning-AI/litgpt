# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import importlib

import pytest
import torch
from lightning.fabric import Fabric
from torch.utils.data import DataLoader

from litgpt import GPT, Config
from litgpt.args import EvalArgs
from litgpt.utils import chunked_cross_entropy


@pytest.mark.parametrize("as_list", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("ignore_index", [-100, -1])
@pytest.mark.parametrize("chunk_size", [0, 3])
def test_fully_masked_cross_entropy_preserves_zero_gradients(as_list, dtype, ignore_index, chunk_size):
    torch.manual_seed(42)
    model = torch.nn.Linear(4, 7, dtype=dtype)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    before = [parameter.detach().clone() for parameter in model.parameters()]
    logits = model(torch.randn(2, 5, 4, dtype=dtype))
    targets = torch.full((2, 5), ignore_index)
    if as_list:
        logits = list(logits.split(3, dim=1))
    loss = chunked_cross_entropy(logits, targets, chunk_size=chunk_size, ignore_index=ignore_index)
    assert loss.item() == 0
    loss.backward()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.count_nonzero(parameter.grad) == 0
    optimizer.step()
    for parameter, original in zip(model.parameters(), before):
        torch.testing.assert_close(parameter, original, rtol=0, atol=0)


@pytest.mark.parametrize("as_list", [False, True])
def test_empty_unchunked_targets_preserve_gradient_connection(as_list):
    logits = torch.randn(2, 0, 7, requires_grad=True)
    targets = torch.empty((2, 0), dtype=torch.long)
    loss = chunked_cross_entropy([logits] if as_list else logits, targets, chunk_size=0)
    assert loss.item() == 0
    loss.backward()
    assert logits.grad is not None
    assert logits.grad.shape == logits.shape


@pytest.mark.parametrize("as_list", [False, True])
def test_unchunked_loss_does_not_hide_nonfinite_valid_logits(as_list):
    logits = torch.full((1, 2, 7), float("nan"), requires_grad=True)
    targets = torch.tensor([[1, 2]])
    loss = chunked_cross_entropy(list(logits.split(1, dim=1)) if as_list else logits, targets, chunk_size=0)
    assert torch.isnan(loss)


@pytest.mark.parametrize("variant", ["full", "lora", "lora_legacy", "adapter", "adapter_v2"])
def test_finetune_validation_keeps_masked_batch_finite(variant):
    validate = importlib.import_module(f"litgpt.finetune.{variant}").validate
    torch.manual_seed(42)
    model = GPT(Config(block_size=8, n_layer=1, n_embd=8, n_head=4, padded_vocab_size=16))
    fabric = Fabric(accelerator="cpu", devices=1)
    samples = [
        {"input_ids": torch.tensor([1, 2, 3, 4]), "labels": torch.full((4,), -100)},
        {"input_ids": torch.tensor([5, 6, 7, 8]), "labels": torch.tensor([-100, 6, 7, 8])},
    ]
    loader = DataLoader(samples, batch_size=1)
    with torch.no_grad():
        valid_logits = model(samples[1]["input_ids"].unsqueeze(0))[:, :-1]
        valid_loss = torch.nn.functional.cross_entropy(valid_logits.reshape(-1, 16), samples[1]["labels"][1:])
    loss = validate(fabric, model, loader, EvalArgs(max_iters=2), verbose=False)
    torch.testing.assert_close(loss, valid_loss / 2)
