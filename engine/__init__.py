"""
VULGARIS autograd engine.

Provides numpy-based reverse-mode automatic differentiation,
the Module base class, and standard layer implementations.
"""

from __future__ import annotations

from .tensor import Tensor, Parameter, zeros, ones, randn, rand, cat, stack
from .module import Module
from .layers import (
    Linear,
    LayerNorm,
    RMSNorm,
    Dropout,
    Conv1d,
    Embedding,
    Sequential,
    SwiGLU,
)
from . import ops

__all__ = [
    "Tensor", "Parameter",
    "zeros", "ones", "randn", "rand", "cat", "stack",
    "Module",
    "Linear", "LayerNorm", "RMSNorm", "Dropout",
    "Conv1d", "Embedding", "Sequential", "SwiGLU",
    "ops",
]
