"""
VULGARIS inference package.

Provides streaming single-step inference, speculative autoregressive rollout,
a shared-memory event buffer, and a FastAPI REST server.
"""

from __future__ import annotations

from .streaming    import StreamingInference
from .speculative  import SpeculativeRollout
from .event_buffer import EventBuffer

__all__ = ["StreamingInference", "SpeculativeRollout", "EventBuffer", "InferenceServer"]


def __getattr__(name: str):
    if name == "InferenceServer":
        from .server import InferenceServer
        return InferenceServer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
