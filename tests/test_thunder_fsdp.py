import os
import re
import sys
from pathlib import Path
from typing import Optional, Tuple, Union

import pytest
import torch
from lightning.fabric import Fabric
from lightning.fabric.utilities.imports import _TORCH_GREATER_EQUAL_2_3

from conftest import RunIf

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from extensions.thunder.strategies.thunder_fsdp import ThunderFSDPStrategy
from extensions.thunder.strategies.utils import _validate_executors


@RunIf(thunder=True)
def test_thunder_strategy_input_parsing():
    from thunder import pythonex
    from thunder.distributed import FSDPBucketingStrategy, FSDPType

    strategy = ThunderFSDPStrategy(bucketing_strategy="BlOcK", executors=("python",), sharding_strategy="zero3")
    assert strategy.bucketing_strategy is FSDPBucketingStrategy.BLOCK
    assert strategy.executors == (pythonex,)
    assert strategy.sharding_strategy is FSDPType.ZERO3


@RunIf(thunder=True)
def test_validate_executors():
    from thunder import pythonex, pytorch_executor

    assert _validate_executors(None) is None
    assert _validate_executors((pythonex, pytorch_executor)) == (pythonex, pytorch_executor)
    assert _validate_executors(("python", "torch")) == (pythonex, pytorch_executor)
    assert _validate_executors(("python", pytorch_executor)) == (pythonex, pytorch_executor)
    with pytest.raises(ValueError, match=re.escape("not find the executors ['foo', 'bar'] in")):
        assert _validate_executors(("python", "foo", pytorch_executor, "bar"))


@RunIf(thunder=True)
def test_save_checkpoint_invalid_settings_raise(tmp_path):
    strategy = ThunderFSDPStrategy(state_dict_type="full")
    with pytest.raises(TypeError, match="not supported"):
        strategy.save_checkpoint(tmp_path, {}, storage_options=object())

    with pytest.raises(IsADirectoryError, match="path exists"):
        strategy.save_checkpoint(tmp_path, {})

    model = torch.nn.Linear(1, 1)
    with pytest.raises(ValueError, match="Could not find"):
        strategy.save_checkpoint(tmp_path / "foo", {})

    model.use_fsdp = True
    with pytest.raises(ValueError, match="Found multiple"):
        strategy.save_checkpoint(tmp_path / "foo", {"model1": model, "model2": model})

    with pytest.raises(ValueError, match="at least a model"):
        strategy.load_checkpoint(tmp_path / "foo", {})

    with pytest.raises(ValueError, match="must be a single file"):
        strategy.load_checkpoint(tmp_path, model)

    optimizer = torch.optim.Adam(model.parameters())
    with pytest.raises(NotImplementedError, match="not supported"):
        strategy.load_checkpoint(tmp_path, optimizer)

    with pytest.raises(ValueError, match="Found multiple"):
        strategy.load_checkpoint(tmp_path / "foo", {"model1": model, "model2": model})

    with pytest.raises(ValueError, match="Could not find"):
        strategy.load_checkpoint(tmp_path / "foo", {"foo": 1})


class Submodule(torch.nn.Module):
    def __init__(self, h: int):
        super().__init__()
        self.l = torch.nn.Linear(4, h * 2, bias=False)

    def forward(self, x):
        # defined just because preprocessing fails otherwise
        ...


class MyModel(torch.nn.Module):
    def __init__(self, h: int):
        super().__init__()
        self.register_buffer("buf", torch.tensor(0))
        self.l = torch.nn.Linear(2, h)
        self.inner = Submodule(h)

    def forward(self):
        # defined just because preprocessing fails otherwise
        ...

    def reset_parameters(self):
        self.buf = torch.empty_like(self.buf)


@RunIf(min_cuda_gpus=2, thunder=True, standalone=True)
def test_materialize_meta_tensors():
    strategy = ThunderFSDPStrategy()
    fabric = Fabric(accelerator="cuda", devices=2, strategy=strategy)
    fabric.launch()

    with fabric.init_module(empty_init=True):
        model = MyModel(2)

    model = fabric.setup(model)
    # all parameters were moved
    assert len(list(model.parameters())) == 3
    assert all(p.device.type == "cuda" for p in model.parameters())
    # buffers were moved too
    assert model.buf.device.type == "cuda"


class StatefulThing:
    def state_dict(self):
        return {"thing": 1}

    def load_state_dict(self, state_dict):
        assert state_dict == self.state_dict()


class TensorLike:
    def __init__(self, device: Optional[Union[str, torch.device]] = None, shape: Optional[Tuple[int, ...]] = None):
        self.device = torch.device(device) if device is not None else None
        self.shape = torch.Size(shape) if shape is not None else None

    def __eq__(self, other):
        return (
            isinstance(other, torch.Tensor)
            and (self.device is None or other.device == self.device)
            and (self.shape is None or other.shape == self.shape)
        )


@RunIf(min_cuda_gpus=2, thunder=True, standalone=True)
def test_save_load_full_checkpoint(tmp_path):
    strategy = ThunderFSDPStrategy(state_dict_type="full", broadcast_from=0)
    fabric = Fabric(accelerator="cuda", devices=2, strategy=strategy)
    fabric.launch()

    model = MyModel(4)
    expected = model.state_dict()

    # save a sharded model
    model = fabric.setup(model)
    state = {"model": model, "stateful": StatefulThing(), "primitive": 123}
    checkpoint_path = tmp_path / "foo"
    fabric.save(checkpoint_path, state)

    # assert the file contents
    if fabric.global_rank == 0:
        checkpoint = torch.load(checkpoint_path)
        # cpu_offload is enabled by default
        assert checkpoint == {
            "model": {
                "buf": TensorLike("cpu", tuple()),
                "inner.l.weight": TensorLike("cpu", (8, 4)),
                "l.bias": TensorLike("cpu", (4,)),
                "l.weight": TensorLike("cpu", (4, 2)),
            },
            "stateful": {"thing": 1},
            "primitive": 123,
        }
        torch.testing.assert_close(checkpoint["model"], expected)

    # load its weights into a different sharded model
    model = MyModel(4)
    model = fabric.setup(model)
    state = {"model": model, "stateful": StatefulThing(), "primitive": 321}
    fabric.load(checkpoint_path, state)

    from thunder.distributed import _unshard_params

    # unshard this model's parameters to compare with the original state dict before sharding
    _unshard_params(model, model.process_group_for_ddp, True)
    # we loaded rank 0's weights, so this would fail in the other ranks
    if fabric.global_rank == 0:
        actual = model.state_dict()
        # `_unshard_params` doesnt offload buffers at the moment
        assert actual["buf"].device.type == "cuda"
        actual["buf"] = actual["buf"].to(device="cpu")
        torch.testing.assert_close(actual, expected)
    assert state["primitive"] == 123


@RunIf(min_cuda_gpus=2, thunder=True, standalone=True)
def test_load_full_checkpoint_only_model(tmp_path):
    strategy = ThunderFSDPStrategy()
    fabric = Fabric(accelerator="cuda", devices=2, strategy=strategy)
    fabric.launch()

    checkpoint_path = tmp_path / "foo"
    checkpoint_path = fabric.broadcast(checkpoint_path)
    if fabric.global_rank == 0:
        model = MyModel(4)
        expected = model.state_dict()
        torch.save(expected, checkpoint_path)
    fabric.barrier()
    expected = torch.load(checkpoint_path)

    # before sharding
    model = MyModel(4)
    fabric.load_raw(checkpoint_path, model)
    torch.testing.assert_close(model.state_dict(), expected)

    # after sharding
    model = MyModel(4)
    model = fabric.setup(model)
    fabric.load_raw(checkpoint_path, model)
    from thunder.distributed import _unshard_params

    # unshard this model's parameters to compare with the original state dict before sharding
    _unshard_params(model, model.process_group_for_ddp, True)
    actual = model.state_dict()
    # `_unshard_params` doesnt offload buffers at the moment
    assert actual["buf"].device.type == "cuda"
    actual["buf"] = actual["buf"].to(device="cpu")
    torch.testing.assert_close(actual, expected)


def distributed_ckpt_to_regular(path):
    """From ``torch.distributed.checkpoint.format_utils.dcp_to_torch_save``."""
    from torch.distributed.checkpoint import FileSystemReader
    from torch.distributed.checkpoint.state_dict_loader import _load_state_dict

    if _TORCH_GREATER_EQUAL_2_3:
        from torch.distributed.checkpoint.format_utils import _EmptyStateDictLoadPlanner
    else:
        from torch.distributed.checkpoint._traverse import set_element
        from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
        from torch.distributed.checkpoint.metadata import TensorStorageMetadata

        class _EmptyStateDictLoadPlanner(DefaultLoadPlanner):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

            def set_up_planner(self, state_dict, metadata, is_coordinator):
                assert not state_dict
                # rebuild the state dict from the metadata
                for k, v in metadata.state_dict_metadata.items():
                    if isinstance(v, TensorStorageMetadata):
                        v = torch.empty(v.size, dtype=v.properties.dtype)
                    if k in metadata.planner_data:
                        set_element(state_dict, metadata.planner_data[k], v)
                    else:
                        state_dict[k] = v
                super().set_up_planner(state_dict, metadata, is_coordinator)

    state_dict = {}
    storage_reader = FileSystemReader(path)
    _load_state_dict(state_dict, storage_reader=storage_reader, planner=_EmptyStateDictLoadPlanner(), no_dist=True)
    return state_dict


@RunIf(min_cuda_gpus=2, thunder=True, standalone=True)
def test_save_load_sharded_checkpoint(tmp_path):
    strategy = ThunderFSDPStrategy(state_dict_type="sharded", broadcast_from=0)
    fabric = Fabric(accelerator="cuda", devices=2, strategy=strategy)
    fabric.launch()

    model = MyModel(4)
    expected = model.state_dict()

    # save a sharded model
    model = fabric.setup(model)
    state = {"model": model, "stateful": StatefulThing(), "primitive": 123}
    fabric.save(tmp_path, state)

    # assert the file contents
    if fabric.global_rank == 0:
        assert set(os.listdir(tmp_path)) == {"meta.pt", "__1_0.distcp", "__0_0.distcp", ".metadata"}

        metadata = torch.load(tmp_path / "meta.pt")
        assert metadata == {"stateful": {"thing": 1}, "primitive": 123}

        checkpoint = distributed_ckpt_to_regular(tmp_path)
        # cpu_offload is enabled by default
        assert checkpoint == {
            "model": {
                "buf": TensorLike("cpu", tuple()),
                "inner.l.weight": TensorLike("cpu", (8, 4)),
                "l.bias": TensorLike("cpu", (4,)),
                "l.weight": TensorLike("cpu", (4, 2)),
            }
        }
        torch.testing.assert_close(checkpoint["model"], expected)

    # load its weights into a different sharded model
    model = MyModel(4)
    model = fabric.setup(model)
    state = {"model": model, "stateful": StatefulThing(), "primitive": 321}
    fabric.load(tmp_path, state)

    from thunder.distributed import _unshard_params

    # unshard this model's parameters to compare with the original state dict before sharding
    _unshard_params(model, model.process_group_for_ddp, True)
    # we loaded rank 0's weights, so this would fail in the other ranks
    if fabric.global_rank == 0:
        actual = model.state_dict()
        # `_unshard_params` doesnt offload buffers at the moment
        assert actual["buf"].device.type == "cuda"
        actual["buf"] = actual["buf"].to(device="cpu")
        torch.testing.assert_close(actual, expected)
    assert state["primitive"] == 123
