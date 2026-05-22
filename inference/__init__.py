"""
VULGARIS inference package.

Provides streaming single-step inference and a FastAPI REST server.
"""

from __future__ import annotations

from .streaming import StreamingInference

__all__ = ["StreamingInference", "InferenceServer"]


def __getattr__(name: str):
    if name == "InferenceServer":
        from .server import InferenceServer
        return InferenceServer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
