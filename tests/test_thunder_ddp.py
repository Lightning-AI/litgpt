import sys
from pathlib import Path

import pytest
import torch
from tests.conftest import RunIf
from lightning import Fabric

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from extensions.thunder.strategies.thunder_ddp import ThunderDDPStrategy
from extensions.thunder.strategies.thunder_fsdp import ThunderFSDPStrategy


@RunIf(thunder=True)
def test_thunder_strategy_input_parsing():
    with pytest.raises(ValueError, match="doesn't have an effect with `jit=False"):
        ThunderDDPStrategy(jit=False, executors=("python",))


@RunIf(min_cuda_gpus=2, thunder=True, standalone=True)
@pytest.mark.parametrize("choice", ["ddp", "thunder_ddp", "fsdp", "thunder_fsdp"])
def test_no_backward_sync(choice):
    if choice == "thunder_ddp":
        strategy = ThunderDDPStrategy()
    elif choice == "thunder_fsdp":
        strategy = ThunderFSDPStrategy()
    else:
        strategy = choice

    fabric = Fabric(devices=2, accelerator="cuda", strategy=strategy)
    fabric.launch()

    # account for sharding in the case of FSDP
    out_features = 1 if "ddp" in choice else fabric.world_size
    
    model = torch.nn.Linear(1, out_features, bias=False, device=fabric.device)
    x = torch.randn(1, 1, device=fabric.device)
    model = fabric.setup(model)

    # 6 iters, 3 grad accumulation iters
    for i, enabled in enumerate((True, True, False, True, True, False), 1):
        x = torch.tensor([i * (fabric.local_rank + 1)], device=fabric.device, dtype=torch.float32)

        with fabric.no_backward_sync(model, enabled):
            y = model(x)
            fabric.backward(y.sum())
        if not enabled:
            # Math for the first 3 iters
            #
            # DistributedDataParallel
            # (1*1+2*1+3*1 + 1*2+2*2+3*2) / 2       = 9
            #  ^^^^^^^^^^^   ^^^^^^^^^^^  ^^^
            #  rank0         rank1        allreduce
            #
            # thunder.distributed.ddp
            # ((1*1+2*1) + (1*2+2*2)) / 2        + (3*1 + 3*2)  / 2        = 9
            #   ^^^^^^^     ^^^^^^^   ^^^           ^^^   ^^^   ^^^
            #   rank0       rank1     allreduce1    rank0 rank1 allreduce2
            assert model.weight.grad.shape.numel() == 1, model.weight.grad.shape
            assert model.weight.grad.item() == (9.0 if i == 3 else 22.5)
            assert not hasattr(model.weight, "_thunder_fsdp_unsharded_grad")
            model.weight.grad = None
        elif choice == "thunder_fsdp":
            assert model.weight._thunder_fsdp_unsharded_grad.shape == (2, 1)
            assert model.weight.grad is None


@RunIf(min_cuda_gpus=2, thunder=True, standalone=True)
@pytest.mark.parametrize("jit", (False, True))
def test_jit_before_setup(jit):
    import thunder

    fabric = Fabric(devices=2, accelerator="cuda", strategy=ThunderDDPStrategy(jit=jit))
    fabric.launch()

    x = torch.randn(1, 1, device=fabric.device)
    model = torch.nn.Linear(1, 2, bias=False, device=fabric.device)

    tmodel = thunder.jit(model)
    fmodel = fabric.setup(tmodel)
    fmodel(x)

    assert "all_reduce" in thunder.last_backward_traces(tmodel)[-1].python()


@RunIf(min_cuda_gpus=1, thunder=True)
def test_setup_already_traced():
    import thunder

    device = torch.device("cuda")
    x = torch.randn(1, 1, device=device)
    model = torch.nn.Linear(1, 2, bias=False, device=device)

    strategy = ThunderDDPStrategy()

    tmodel = thunder.jit(model)
    tmodel(x)
    with pytest.raises(RuntimeError, match="already called"):
        strategy.setup_module(tmodel)
