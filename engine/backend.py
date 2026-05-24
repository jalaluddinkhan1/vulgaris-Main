"""
Backend dispatcher for VULGARIS execution engine.

Priority: triton (CUDA/ROCm GPU) > numba (CPU JIT) > numpy (fallback)
Override with env var: VULGARIS_BACKEND=numpy|numba|triton
"""
import os
import numpy as np

_ENV = os.environ.get("VULGARIS_BACKEND", "auto").lower()
_RESOLVED = None


def get_backend() -> str:
    global _RESOLVED
    if _RESOLVED is not None:
        return _RESOLVED

    if _ENV != "auto":
        _RESOLVED = _ENV
        return _RESOLVED

    # Try triton (requires GPU + torch)
    try:
        import triton          # noqa: F401
        import torch           # noqa: F401
        if torch.cuda.is_available() or torch.version.hip is not None:
            _RESOLVED = "triton"
            return _RESOLVED
    except ImportError:
        pass

    # Try numba (CPU JIT)
    try:
        import numba           # noqa: F401
        _RESOLVED = "numba"
        return _RESOLVED
    except ImportError:
        pass

    _RESOLVED = "numpy"
    return _RESOLVED


def reset_backend():
    """Force re-detection (useful after installing new packages)."""
    global _RESOLVED
    _RESOLVED = None
