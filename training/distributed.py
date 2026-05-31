"""
Distributed training primitives for VULGARIS.

Transport: torch.distributed with gloo backend (CPU/TCP — no CUDA required).
The model stays in numpy throughout; torch is used only as a communication
bus for gradient allreduce and parameter broadcast.

Typical workflow
----------------
    # Each worker process:
    from training.distributed import init_process_group, is_initialized
    from training.distributed import broadcast_parameters, allreduce_gradients

    init_process_group()          # reads RANK/WORLD_SIZE/MASTER_ADDR from env
    broadcast_parameters(model)   # sync weights from rank 0 at startup

    # Inside training loop, after loss.backward():
    allreduce_gradients(model.parameters())
    optimizer.step()

Environment variables (set by torchrun automatically)
------------------------------------------------------
    MASTER_ADDR   : IP of rank-0 node (default "127.0.0.1")
    MASTER_PORT   : free TCP port       (default "29500")
    RANK          : global rank of this process
    WORLD_SIZE    : total number of processes
    LOCAL_RANK    : rank within this node (used for GPU assignment if needed)
from __future__ import annotations

"""


import datetime
import os
import numpy as np
from typing import Iterable, Optional

from engine.tensor import Parameter

# Module-level state — set by init_process_group()
_initialized:  bool = False
_rank:         int  = 0
_world_size:   int  = 1


# ──────────────────────────────────────────────────────────────────────────────
# Lifecycle
# ──────────────────────────────────────────────────────────────────────────────

def init_process_group(
    backend: str = "gloo",
    init_method: str = "env://",
    timeout_s: int = 300,
) -> None:
    """
    Initialise the process group.

    Parameters
    ----------
    backend     : "gloo" (CPU/TCP, default) or "nccl" (GPU, requires CUDA).
    init_method : "env://" reads MASTER_ADDR, MASTER_PORT, RANK, WORLD_SIZE
                  from environment (set automatically by torchrun).
    timeout_s   : collective operation timeout in seconds.

    Raises
    ------
    RuntimeError if torch is not installed.
    """
    global _initialized, _rank, _world_size

    try:
        import torch.distributed as dist
    except ImportError:
        raise RuntimeError(
            "torch.distributed is required for distributed training.\n"
            "Install PyTorch: pip install torch"
        )

    if dist.is_initialized():
        _rank       = dist.get_rank()
        _world_size = dist.get_world_size()
        _initialized = True
        return

    dist.init_process_group(
        backend=backend,
        init_method=init_method,
        timeout=datetime.timedelta(seconds=timeout_s),
    )

    _rank       = dist.get_rank()
    _world_size = dist.get_world_size()
    _initialized = True


def destroy_process_group() -> None:
    """Tear down the process group cleanly (call at end of training)."""
    global _initialized
    if _initialized:
        try:
            import torch.distributed as dist
            if dist.is_initialized():
                dist.destroy_process_group()
        except Exception:
            pass
        _initialized = False


# ──────────────────────────────────────────────────────────────────────────────
# Identity queries
# ──────────────────────────────────────────────────────────────────────────────

def is_initialized() -> bool:
    return _initialized

def get_rank() -> int:
    """Global rank of this process (0 … world_size-1)."""
    return _rank

def get_world_size() -> int:
    """Total number of processes."""
    return _world_size

def is_main_process() -> bool:
    """True for rank 0 only — use to guard checkpointing / logging."""
    return _rank == 0


# ──────────────────────────────────────────────────────────────────────────────
# Collective operations
# ──────────────────────────────────────────────────────────────────────────────

def barrier() -> None:
    """Block until all ranks reach this call."""
    if not _initialized or _world_size == 1:
        return
    import torch.distributed as dist
    dist.barrier()


def broadcast_parameters(model, src: int = 0) -> None:
    """
    Copy rank `src` parameters to all other ranks.

    Call once after init_process_group() and before the first train_step()
    so all workers start from identical weights regardless of random seed.
    """
    if not _initialized or _world_size == 1:
        return

    import torch
    import torch.distributed as dist

    for param in model.parameters():
        # numpy float64 → torch float64 → broadcast → back to numpy
        t = torch.from_numpy(param.data)
        dist.broadcast(t, src=src)
        if _rank != src:
            param.data = t.numpy()


def allreduce_gradients(
    params: Iterable[Parameter],
    divide: bool = True,
) -> None:
    """
    Sum parameter gradients across all ranks (allreduce SUM), then
    optionally divide by world_size to produce the mean gradient.

    Call after loss.backward() and the NaN/Inf guard, before optimizer.step().

    Parameters
    ----------
    params  : iterable of Parameter (e.g. model.parameters())
    divide  : if True, divide the reduced gradient by world_size so that
              the gradient magnitude is independent of the number of workers.
    """
    if not _initialized or _world_size == 1:
        return

    import torch
    import torch.distributed as dist

    for param in params:
        if param.grad is None:
            continue
        t = torch.from_numpy(param.grad)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        if divide:
            t.div_(_world_size)
        param.grad = t.numpy()


def allreduce_scalar(value: float, op: str = "mean") -> float:
    """
    Reduce a scalar (e.g. loss value) across ranks for logging.

    op : "mean" | "sum" | "max"
    """
    if not _initialized or _world_size == 1:
        return value

    import torch
    import torch.distributed as dist

    t = torch.tensor([value], dtype=torch.float64)
    if op == "max":
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
    else:
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        if op == "mean":
            t /= _world_size
    return float(t[0])
