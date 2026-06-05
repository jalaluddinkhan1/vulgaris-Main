"""
engine_torch/tensor.py — Tensor with .data shim for numpy compatibility.

The .data property is the central compatibility mechanism:
  READ  : returns tensor.detach().cpu().numpy()  (zero-copy view when possible)
  WRITE : accepts numpy array, converts to torch and updates storage in-place

This makes all existing module code that reads .data (shape checks, numpy ops)
work without changes.  Operations that go through Tensor.__add__, __matmul__ etc.
ARE traced by PyTorch autograd.  Operations that call np.exp(t.data) are NOT
traced — but they return the correct value.
"""
from __future__ import annotations

import numpy as np
from typing import Optional, Tuple

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    raise ImportError(
        "engine_torch requires PyTorch.  Install with:\n"
        "    pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
        "or: pip install 'vulgaris[export]'"
    )


# Default device — override with VULGARIS_DEVICE env var
import os as _os
_DEFAULT_DEVICE = _os.environ.get("VULGARIS_DEVICE", "cpu")
_DEFAULT_DTYPE  = torch.float32


class Tensor:
    """
    Thin wrapper around torch.Tensor that exposes a .data numpy property
    for backward compatibility with module code that accesses .data directly.

    The internal _t attribute is the real torch.Tensor — all autograd lives there.
    """

    # ── Construction ─────────────────────────────────────────────────────

    def __init__(
        self,
        data,
        requires_grad: bool = False,
        # Ignored in torch backend — PyTorch builds its own graph
        _children=(),
        _op: str = "",
        name: str = "",
    ):
        if isinstance(data, torch.Tensor):
            self._t = data.to(dtype=_DEFAULT_DTYPE)
        elif isinstance(data, np.ndarray):
            self._t = torch.from_numpy(data.astype(np.float32)).to(_DEFAULT_DEVICE)
        else:
            self._t = torch.tensor(data, dtype=_DEFAULT_DTYPE, device=_DEFAULT_DEVICE)

        if requires_grad:
            self._t = self._t.requires_grad_(True)

        # Compat fields — ignored in torch backend, kept so code that sets them
        # (e.g. _backward = ...) doesn't crash
        self._backward = lambda: None
        self._children = _children
        self._op       = _op

    # ── .data shim (key compatibility mechanism) ─────────────────────────

    @property
    def data(self) -> np.ndarray:
        """READ: returns numpy view of the tensor (detached, cpu)."""
        return self._t.detach().cpu().numpy()

    @data.setter
    def data(self, value: np.ndarray):
        """WRITE: update torch storage from numpy array in-place."""
        with torch.no_grad():
            new = torch.from_numpy(np.asarray(value, dtype=np.float32)).to(
                self._t.device
            )
            self._t.copy_(new)

    # ── .grad shim ───────────────────────────────────────────────────────

    @property
    def grad(self) -> Optional[np.ndarray]:
        g = self._t.grad
        return g.detach().cpu().numpy() if g is not None else None

    @grad.setter
    def grad(self, value):
        if value is None:
            self._t.grad = None
        else:
            self._t.grad = torch.from_numpy(
                np.asarray(value, dtype=np.float32)
            ).to(self._t.device)

    # ── Shape / ndim ─────────────────────────────────────────────────────

    @property
    def shape(self) -> torch.Size:
        return self._t.shape

    @property
    def ndim(self) -> int:
        return self._t.ndim

    @property
    def requires_grad(self) -> bool:
        return self._t.requires_grad

    @requires_grad.setter
    def requires_grad(self, v: bool):
        self._t.requires_grad_(v)

    # ── Arithmetic (traced by PyTorch autograd) ──────────────────────────

    def _wrap(self, t: torch.Tensor) -> "Tensor":
        return Tensor(t)

    def __add__(self, other):
        o = other._t if isinstance(other, Tensor) else torch.tensor(float(other))
        return self._wrap(self._t + o)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        o = other._t if isinstance(other, Tensor) else torch.tensor(float(other))
        return self._wrap(self._t - o)

    def __rsub__(self, other):
        o = other._t if isinstance(other, Tensor) else torch.tensor(float(other))
        return self._wrap(o - self._t)

    def __mul__(self, other):
        o = other._t if isinstance(other, Tensor) else torch.tensor(float(other))
        return self._wrap(self._t * o)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        o = other._t if isinstance(other, Tensor) else torch.tensor(float(other))
        return self._wrap(self._t / o)

    def __matmul__(self, other):
        o = other._t if isinstance(other, Tensor) else other
        return self._wrap(self._t @ o)

    def __neg__(self):
        return self._wrap(-self._t)

    # ── Activations ──────────────────────────────────────────────────────

    def relu(self):
        return self._wrap(torch.relu(self._t))

    def sigmoid(self):
        return self._wrap(torch.sigmoid(self._t))

    def tanh(self):
        return self._wrap(torch.tanh(self._t))

    def silu(self):
        return self._wrap(torch.nn.functional.silu(self._t))

    def softmax(self, axis=-1):
        return self._wrap(torch.softmax(self._t, dim=axis))

    # ── Reductions ───────────────────────────────────────────────────────

    def sum(self, axis=None, keepdims=False):
        if axis is None:
            return self._wrap(self._t.sum())
        return self._wrap(self._t.sum(dim=axis, keepdim=keepdims))

    def mean(self, axis=None, keepdims=False):
        if axis is None:
            return self._wrap(self._t.mean())
        return self._wrap(self._t.mean(dim=axis, keepdim=keepdims))

    # ── Shape ops ────────────────────────────────────────────────────────

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return self._wrap(self._t.reshape(shape))

    def transpose(self, *axes):
        if len(axes) == 2:
            return self._wrap(self._t.transpose(axes[0], axes[1]))
        # Numpy-style: (0,2,1) etc.
        return self._wrap(self._t.permute(axes))

    # ── Backward ─────────────────────────────────────────────────────────

    def backward(self, grad=None):
        if grad is None:
            self._t.sum().backward()
        else:
            g = torch.from_numpy(np.asarray(grad, np.float32))
            self._t.backward(g)

    # ── Repr ─────────────────────────────────────────────────────────────

    def __repr__(self):
        return f"Tensor(shape={tuple(self._t.shape)}, backend=torch)"


# ── Parameter ────────────────────────────────────────────────────────────────

class Parameter(Tensor):
    """Trainable parameter — same API as engine.tensor.Parameter."""

    def __init__(self, data, name: str = ""):
        super().__init__(data, requires_grad=True)
        self.name = name


# ── Factory functions ─────────────────────────────────────────────────────────

def zeros(*shape, **kwargs) -> Tensor:
    return Tensor(torch.zeros(*shape, dtype=_DEFAULT_DTYPE, device=_DEFAULT_DEVICE))

def ones(*shape, **kwargs) -> Tensor:
    return Tensor(torch.ones(*shape, dtype=_DEFAULT_DTYPE, device=_DEFAULT_DEVICE))

def randn(*shape, **kwargs) -> Tensor:
    return Tensor(torch.randn(*shape, dtype=_DEFAULT_DTYPE, device=_DEFAULT_DEVICE))

def rand(*shape, **kwargs) -> Tensor:
    return Tensor(torch.rand(*shape, dtype=_DEFAULT_DTYPE, device=_DEFAULT_DEVICE))

def cat(tensors, axis=0) -> Tensor:
    ts = [t._t if isinstance(t, Tensor) else t for t in tensors]
    return Tensor(torch.cat(ts, dim=axis))

def stack(tensors, axis=0) -> Tensor:
    ts = [t._t if isinstance(t, Tensor) else t for t in tensors]
    return Tensor(torch.stack(ts, dim=axis))
