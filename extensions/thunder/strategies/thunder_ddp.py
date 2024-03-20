"""Fabric Strategy to support Thunder DDP: To be upstreamed into Fabric eventually."""

from contextlib import nullcontext
from datetime import timedelta
from typing import TYPE_CHECKING, Any, ContextManager, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed
from lightning.fabric.accelerators.accelerator import Accelerator
from lightning.fabric.plugins.collectives.torch_collective import default_pg_timeout
from lightning.fabric.plugins.environments.cluster_environment import ClusterEnvironment
from lightning.fabric.plugins.io.checkpoint_io import CheckpointIO
from lightning.fabric.plugins.precision import Precision
from lightning.fabric.strategies.launchers.subprocess_script import _SubprocessScriptLauncher
from lightning.fabric.strategies.parallel import ParallelStrategy
from lightning.fabric.strategies.strategy import TBroadcast, _BackwardSyncControl
from lightning.fabric.utilities.distributed import (
    ReduceOp,
    _distributed_is_initialized,
    _get_default_process_group_backend_for_device,
    _init_dist_connection,
    _sync_ddp_if_available,
)
from lightning.fabric.utilities.rank_zero import rank_zero_only
from lightning_utilities.core.imports import RequirementCache
from lightning_utilities.core.rank_zero import rank_zero_only as utils_rank_zero_only
from torch import Tensor
from torch.nn import Module
from typing_extensions import override

from .utils import _validate_executors

if TYPE_CHECKING:
    from thunder import Executor


_THUNDER_AVAILABLE = RequirementCache("lightning-thunder", "thunder")


class ThunderDDPStrategy(ParallelStrategy):
    def __init__(
        self,
        accelerator: Optional[Accelerator] = None,
        parallel_devices: Optional[List[torch.device]] = None,
        cluster_environment: Optional[ClusterEnvironment] = None,
        checkpoint_io: Optional[CheckpointIO] = None,
        precision: Optional[Precision] = None,
        executors: Optional[Tuple[Union["Executor", str], ...]] = None,
        process_group_backend: Optional[str] = None,
        timeout: Optional[timedelta] = default_pg_timeout,
        **kwargs: Any,
    ):
        if not _THUNDER_AVAILABLE:
            raise ModuleNotFoundError(str(_THUNDER_AVAILABLE))
        super().__init__(accelerator=accelerator, checkpoint_io=checkpoint_io, precision=precision)
        self.parallel_devices = parallel_devices
        self.cluster_environment: Optional[ClusterEnvironment] = cluster_environment

        self.executors = _validate_executors(executors)
        self._num_nodes = 1
        self._process_group_backend: Optional[str] = process_group_backend
        self._timeout: Optional[timedelta] = timeout
        self._backward_sync_control = _DDPBackwardSyncControl()
        self._ddp_kwargs = kwargs

    @property
    @override
    def root_device(self) -> torch.device:
        assert self.parallel_devices is not None
        return self.parallel_devices[self.local_rank]

    @property
    def num_nodes(self) -> int:
        return self._num_nodes

    @num_nodes.setter
    def num_nodes(self, num_nodes: int) -> None:
        # note that world ranks is related to num_nodes, when resetting it, need to reset world ranks
        self._num_nodes = num_nodes

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

    @property
    def process_group_backend(self) -> Optional[str]:
        return self._process_group_backend

    @override
    def _configure_launcher(self) -> None:
        assert self.cluster_environment is not None
        self._launcher = _SubprocessScriptLauncher(self.cluster_environment, self.num_processes, self.num_nodes)

    @override
    def setup_environment(self) -> None:
        super().setup_environment()
        self._setup_distributed()

    @override
    def setup_module(self, module: Module) -> Module:
        import thunder

        module = thunder.distributed.ddp(module, **self._ddp_kwargs)

        return thunder.jit(module, executors=self.executors)

    @override
    def module_to_device(self, module: Module) -> None:
        module.to(self.root_device)

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

    def _setup_distributed(self) -> None:
        self._set_world_ranks()
        self._process_group_backend = self._get_process_group_backend()
        assert self.cluster_environment is not None
        _init_dist_connection(self.cluster_environment, self._process_group_backend, timeout=self._timeout)

    def _get_process_group_backend(self) -> str:
        return self._process_group_backend or _get_default_process_group_backend_for_device(self.root_device)

    def _set_world_ranks(self) -> None:
        if self.cluster_environment is not None:
            self.cluster_environment.set_global_rank(self.node_rank * self.num_processes + self.local_rank)
            self.cluster_environment.set_world_size(self.num_nodes * self.num_processes)
        # `LightningEnvironment.set_global_rank` will do this too, but we cannot rely on that implementation detail
        # additionally, for some implementations, the setter is a no-op, so it's safer to access the getter
        rank_zero_only.rank = utils_rank_zero_only.rank = self.global_rank


class _DDPBackwardSyncControl(_BackwardSyncControl):
    def __init__(self):
        self._enabled = False

    @override
    def no_backward_sync(self, module: Module, enabled: bool) -> ContextManager:
        if not getattr(module, "use_ddp", False):
            raise TypeError(
                "Blocking backward sync is only possible if the module passed to"
                f" `{self.__class__.__name__}.no_backward_sync` is applied DDP."
                f" Got: {module.__class__.__name__}."
            )

        # see https://github.com/Lightning-AI/lightning-thunder/issues/2085
        # for why we cannot just return `module.no_sync()`
        from thunder.distributed import skip_data_parallel_grad_sync

        previous, self._enabled = self._enabled, enabled
        if enabled:
            return skip_data_parallel_grad_sync()
        if not enabled and previous:
            return _AllReduceGradsContextManager(module)
        return nullcontext()


class _AllReduceGradsContextManager:
    def __init__(self, module: Module) -> None:
        self._module = module

    @override
    def __enter__(self) -> None:
        from thunder.distributed import _sync_grads

        _sync_grads(self._module)

    @override
    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        pass
