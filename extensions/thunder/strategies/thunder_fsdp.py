"""Fabric Strategy to support Thunder FSDP: To be upstreamed into Fabric eventually."""

import shutil
from contextlib import ExitStack, nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, ContextManager, Dict, List, Literal, Optional, Tuple, Union

import torch
from lightning.fabric.accelerators.accelerator import Accelerator
from lightning.fabric.plugins.environments.cluster_environment import ClusterEnvironment
from lightning.fabric.plugins.io.checkpoint_io import CheckpointIO
from lightning.fabric.plugins.precision import Precision
from lightning.fabric.strategies.launchers.subprocess_script import _SubprocessScriptLauncher
from lightning.fabric.strategies.parallel import ParallelStrategy
from lightning.fabric.strategies.strategy import TBroadcast, _apply_filter, _Sharded, _validate_keys_for_strict_loading
from lightning.fabric.utilities.distributed import (
    ReduceOp,
    _distributed_is_initialized,
    _get_default_process_group_backend_for_device,
    _init_dist_connection,
    _sync_ddp_if_available,
)
from lightning.fabric.utilities.imports import _TORCH_GREATER_EQUAL_2_2
from lightning.fabric.utilities.load import _METADATA_FILENAME, _move_state_into
from lightning.fabric.utilities.rank_zero import rank_zero_only
from lightning.fabric.utilities.seed import reset_seed
from lightning.fabric.utilities.types import _PATH, _Stateful
from lightning_utilities.core.imports import RequirementCache
from lightning_utilities.core.rank_zero import rank_zero_only as utils_rank_zero_only
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from typing_extensions import override

from .utils import _validate_executors

if TYPE_CHECKING:
    from thunder import Executor
    from thunder.distributed import FSDPBucketingStrategy, FSDPType
    from thunder.distributed.checkpoint import StateDictOptions

    _FSDP_TYPE = Union[FSDPType, Literal["ZERO2", "ZERO3"]]
    _BUCKETING_STRATEGY = Union[FSDPBucketingStrategy, Literal["NONE", "LAYER", "BLOCK"]]


_THUNDER_AVAILABLE = RequirementCache("lightning-thunder", "thunder")


class ThunderFSDPStrategy(ParallelStrategy, _Sharded):
    def __init__(
        self,
        accelerator: Optional[Accelerator] = None,
        parallel_devices: Optional[List[torch.device]] = None,
        cluster_environment: Optional[ClusterEnvironment] = None,
        checkpoint_io: Optional[CheckpointIO] = None,
        precision: Optional[Precision] = None,
        sharding_strategy: "_FSDP_TYPE" = "ZERO3",
        bucketing_strategy: "_BUCKETING_STRATEGY" = "NONE",
        executors: Optional[Tuple[Union["Executor", str], ...]] = None,
        state_dict_type: Literal["full", "sharded"] = "sharded",
        **kwargs: Any,
    ):
        if not _TORCH_GREATER_EQUAL_2_2:
            raise ImportError("Thunder's FSDP strategy requires PyTorch 2.2 or higher.")
        if not _THUNDER_AVAILABLE:
            raise ModuleNotFoundError(str(_THUNDER_AVAILABLE))
        super().__init__(accelerator=accelerator, checkpoint_io=checkpoint_io, precision=precision)
        self.parallel_devices = parallel_devices
        self.cluster_environment: Optional[ClusterEnvironment] = cluster_environment
        from thunder.distributed import FSDPBucketingStrategy, FSDPType

        self.sharding_strategy = (
            FSDPType[sharding_strategy.upper()] if isinstance(sharding_strategy, str) else sharding_strategy
        )
        self.bucketing_strategy = (
            FSDPBucketingStrategy[bucketing_strategy.upper()]
            if isinstance(bucketing_strategy, str)
            else bucketing_strategy
        )
        self.executors = _validate_executors(executors)
        self._state_dict_type = state_dict_type
        self._fsdp_kwargs = kwargs

    @property
    @override
    def root_device(self) -> torch.device:
        assert self.parallel_devices is not None
        return self.parallel_devices[self.local_rank]

    @property
    def num_nodes(self) -> int:
        return 1

    @property
    def num_processes(self) -> int:
        return len(self.parallel_devices) if self.parallel_devices is not None else 0

    @property
    @override
    def distributed_sampler_kwargs(self) -> Dict[str, Any]:
        return {"num_replicas": self.num_nodes * self.num_processes, "rank": self.global_rank}

    @override
    def _configure_launcher(self) -> None:
        assert self.cluster_environment is not None
        if not self.cluster_environment.creates_processes_externally:
            self._launcher = _SubprocessScriptLauncher(self.cluster_environment, self.num_processes, self.num_nodes)

    @override
    def setup_environment(self) -> None:
        super().setup_environment()
        self._setup_distributed()

    @override
    def setup_module(self, module: Module) -> Module:
        import thunder

        module = thunder.distributed.fsdp(
            module,
            device=self.root_device,
            sharding_strategy=self.sharding_strategy,
            bucketing_strategy=self.bucketing_strategy,
            **self._fsdp_kwargs,
        )

        # NOTE @IvanYaschuck says that `fsdp(jit(model))` could be supported in the future so that the user owns the `jit` call.
        # we would still `jit(fsdp(undo_jit(jit(model))))` internally
        return thunder.jit(module, executors=self.executors)

    @override
    def module_to_device(self, module: Module) -> None:
        pass

    @override
    def module_init_context(self, empty_init: Optional[bool] = None) -> ContextManager:
        precision_init_ctx = self.precision.module_init_context()
        module_sharded_ctx = self.module_sharded_context()
        stack = ExitStack()
        if empty_init:
            # Materialization happens in `setup`. When modules get wrapped by FSDP
            stack.enter_context(torch.device("meta"))
        stack.enter_context(precision_init_ctx)
        stack.enter_context(module_sharded_ctx)
        return stack

    @override
    def module_sharded_context(self) -> ContextManager:
        return nullcontext()

    @override
    def all_reduce(
        self, tensor: Tensor, group: Optional[Any] = None, reduce_op: Optional[Union[ReduceOp, str]] = "mean"
    ) -> Tensor:
        if isinstance(tensor, Tensor):
            return _sync_ddp_if_available(tensor, group, reduce_op=reduce_op)
        return tensor

    @override
    def barrier(self, *args: Any, **kwargs: Any) -> None:
        if not _distributed_is_initialized():
            return
        if torch.distributed.get_backend() == "nccl":
            torch.distributed.barrier(device_ids=[self.root_device.index])
        else:
            torch.distributed.barrier()

    @override
    def broadcast(self, obj: TBroadcast, src: int = 0) -> TBroadcast:
        if not _distributed_is_initialized():
            return obj

        obj = [obj]
        torch.distributed.broadcast_object_list(obj, src)
        return obj[0]

    @override
    def clip_gradients_norm(
        self,
        module: Module,
        optimizer: Optimizer,
        max_norm: Union[float, int],
        norm_type: Union[float, int] = 2.0,
        error_if_nonfinite: bool = True,
    ) -> Tensor:
        raise NotImplementedError

    @override
    def save_checkpoint(
        self,
        path: _PATH,
        state: Dict[str, Union[Module, Optimizer, Any]],
        storage_options: Optional[Any] = None,
        filter: Optional[Dict[str, Callable[[str, Any], bool]]] = None,
    ) -> None:
        if storage_options is not None:
            raise TypeError(
                "`FSDPStrategy.save_checkpoint(..., storage_options=...)` is not supported because"
                " `FSDPStrategy` does not use the `CheckpointIO`."
            )
        if filter is not None:
            raise NotImplementedError("Filtering checkpoint paths is not implemented")

        # broadcast the path from rank 0 to ensure all the states are saved in a common path
        path = Path(self.broadcast(path))
        if path.is_dir() and self._state_dict_type == "full" and not _is_sharded_checkpoint(path):
            raise IsADirectoryError(f"The checkpoint path exists and is a directory: {path}")

        from thunder.distributed.checkpoint import StateDictOptions, has_fsdp_modules, save

        modules = [module for module in state.values() if has_fsdp_modules(module)]
        if len(modules) == 0:
            raise ValueError(
                "Could not find a FSDP model in the provided checkpoint state. Please provide the model as"
                " part of the state like so: `save_checkpoint(..., state={'model': model, ...})`. Make sure"
                " you set up the model (and optimizers if any) through the strategy before saving the checkpoint."
            )
        if len(modules) > 1:
            raise ValueError(
                "Found multiple FSDP models in the given state. Saving checkpoints with FSDP is"
                " currently limited to a single model per checkpoint. To save multiple models, call the"
                " save method for each model separately with a different path."
            )

        if self._state_dict_type == "sharded":
            if _is_full_checkpoint(path):
                path.unlink()
            path.mkdir(parents=True, exist_ok=True)

            options = StateDictOptions(full_state_dict=False, cpu_offload=True, rank0_only=False)
            converted_state, metadata = _get_state_dict(state, filter, options, self.local_rank)
            save(converted_state, path)
            if self.global_rank == 0:
                torch.save(metadata, path / _METADATA_FILENAME)

        elif self._state_dict_type == "full":
            if _is_sharded_checkpoint(path):
                shutil.rmtree(path)

            options = StateDictOptions(full_state_dict=True, cpu_offload=True, rank0_only=True)
            converted_state, metadata = _get_state_dict(state, filter, options, self.local_rank)
            converted_state.update(metadata)
            if self.global_rank == 0:
                torch.save(converted_state, path)
        else:
            raise ValueError(f"Unknown state_dict_type: {self._state_dict_type}")

    @override
    def load_checkpoint(
        self,
        path: _PATH,
        state: Optional[Union[Module, Optimizer, Dict[str, Union[Module, Optimizer, Any]]]] = None,
        strict: bool = True,
    ) -> Dict[str, Any]:
        if not state:
            raise ValueError(
                f"Got `FSDPStrategy.load_checkpoint(..., state={state!r})` but a state with at least"
                " a model instance to reload is required. Pass it in like so:"
                " `FSDPStrategy.load_checkpoint(..., state={'model': model, ...})`"
            )
        # broadcast the path from rank 0 to ensure all the states are loaded from a common path
        path = Path(self.broadcast(path))

        from thunder.distributed.checkpoint import StateDictOptions, has_fsdp_modules, load, load_model_state_dict

        if isinstance(state, Module):
            if not _is_full_checkpoint(path):
                raise ValueError(
                    "Failed to load checkpoint directly into the model. The given path must be a single file"
                    f" containing the full state dict: {path}"
                )
            state_dict = torch.load(str(path), mmap=True, map_location="cpu")
            options = StateDictOptions(full_state_dict=True, cpu_offload=True, strict=strict, rank0_only=False)
            load_model_state_dict(state_dict, _unwrap_tom(state), options, self.local_rank)
            return {}

        if isinstance(state, Optimizer):
            raise NotImplementedError(
                "Loading a single optimizer object from a checkpoint is not supported yet with the FSDP strategy."
            )

        modules = {key: module for key, module in state.items() if has_fsdp_modules(module)}
        if len(modules) == 0:
            raise ValueError(
                "Could not find a FSDP model in the provided checkpoint state. Please provide the model as"
                " part of the state like so: `load_checkpoint(..., state={'model': model, ...})`. Make sure"
                " you set up the model (and optimizers if any) through the strategy before loading the checkpoint."
            )
        if len(modules) > 1:
            raise ValueError(
                "Found multiple FSDP models in the given state. Loading checkpoints with FSDP is"
                " currently limited to a single model per checkpoint. To load multiple models, call the"
                " load method for each model separately with a different path."
            )
        optimizers = {key: optim for key, optim in state.items() if isinstance(optim, Optimizer)}
        module_key, module = list(modules.items())[0]
        module = _unwrap_tom(module)

        if _is_sharded_checkpoint(path):
            options = StateDictOptions(full_state_dict=False, cpu_offload=True, strict=strict, rank0_only=False)
            # Load the DCP state dict, which requires a holder state dict
            converted_state, _ = _get_state_dict(state, None, options, self.local_rank)
            load(converted_state, path)
            load_model_state_dict(converted_state[module_key], module, options, self.local_rank)

            # Load metadata (anything not a module or optimizer)
            metadata = torch.load(path / _METADATA_FILENAME)
            requested_metadata_keys = state.keys() - modules.keys() - optimizers.keys()
            _validate_keys_for_strict_loading(requested_metadata_keys, metadata.keys(), strict=strict)
            for key in requested_metadata_keys:
                if key not in metadata:
                    continue
                state[key] = metadata.pop(key)
            # return the remaining metadata that wasn't requested as part of `state`
            return metadata

        if _is_full_checkpoint(path):
            options = StateDictOptions(full_state_dict=True, cpu_offload=True, strict=strict, rank0_only=False)
            if not options.rank0_only or self.local_rank == 0:
                map_location = "cpu" if options.cpu_offload else None
                checkpoint = torch.load(str(path), mmap=True, map_location=map_location)
                load_model_state_dict(checkpoint[module_key], module, options, self.local_rank)
            else:
                checkpoint = {}

            requested_metadata_keys = state.keys() - modules.keys() - optimizers.keys()
            _validate_keys_for_strict_loading(requested_metadata_keys, checkpoint.keys(), strict=strict)
            # Load metadata (anything not a module or optimizer)
            _move_state_into(source=checkpoint, destination=state, keys=requested_metadata_keys)
            # return the remaining metadata that wasn't requested as part of `state`
            return checkpoint

        raise ValueError(
            f"The path {str(path)!r} does not point to a valid checkpoint. Make sure the path points to either a"
            " directory with FSDP checkpoint shards, or a single file with a full checkpoint."
        )

    def _setup_distributed(self) -> None:
        reset_seed()
        self._set_world_ranks()
        process_group_backend = _get_default_process_group_backend_for_device(self.root_device)
        assert self.cluster_environment is not None
        _init_dist_connection(self.cluster_environment, process_group_backend)

    def _set_world_ranks(self) -> None:
        if self.cluster_environment is not None:
            self.cluster_environment.set_global_rank(self.node_rank * self.num_processes + self.local_rank)
            self.cluster_environment.set_world_size(self.num_nodes * self.num_processes)
        # `LightningEnvironment.set_global_rank` will do this too, but we cannot rely on that implementation detail
        # additionally, for some implementations, the setter is a no-op, so it's safer to access the getter
        rank_zero_only.rank = utils_rank_zero_only.rank = self.global_rank


def _is_sharded_checkpoint(path: Path) -> bool:
    """A heuristic check to determine whether the path points to a directory with checkpoint shards."""
    return path.is_dir() and (path / _METADATA_FILENAME).is_file()


def _is_full_checkpoint(path: Path) -> bool:
    return path.is_file()


def _get_state_dict(
    state: Dict[str, Any],
    filter: Optional[Dict[str, Callable[[str, Any], bool]]],
    options: "StateDictOptions",
    rank: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    from thunder.distributed.checkpoint import get_model_state_dict

    # replace the modules and optimizer objects in the state with their local state dict
    # and separate the user's metadata
    converted_state: Dict[str, Any] = {}
    metadata: Dict[str, Any] = {}
    for key, obj in state.items():
        converted: Any
        if isinstance(obj, Module):
            converted = get_model_state_dict(_unwrap_tom(obj), options, rank)
            target_dict = converted_state
        elif isinstance(obj, Optimizer):
            # TODO: optimizer support
            converted = obj.state_dict()
            target_dict = converted_state
        else:  # everything not a module or optimizer is considered metadata
            converted = obj.state_dict() if isinstance(obj, _Stateful) else obj
            target_dict = metadata
        _apply_filter(key, filter or {}, converted, target_dict)

    return converted_state, metadata


def _unwrap_tom(obj: object) -> object:
    # TODO: this unwrap won't be required when Fabric's `_unwrap_objects` supports Thunder
    from thunder import ThunderModule

    if isinstance(obj, ThunderModule):
        return obj._model
    return obj
