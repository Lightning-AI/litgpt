from typing import Any

from lightning_utilities.core.rank_zero import rank_prefixed_message
import lightning as L


def rank_print(fabric: L.Fabric, message: object, *, flush: bool = True, **kwargs: Any) -> None:
    if fabric.local_rank == 0:
        message = str(message)
        # let each host print, but only on rank 0
        message = rank_prefixed_message(message, fabric.global_rank)
        # TPU VM will only print when the script finishes if `flush=False`
        print(message, flush=flush, **kwargs)
