"""
VULGARIS production serving utilities.

Auth, structured JSON logging, Prometheus-style metrics, graceful degradation,
canary deployment, live checkpoint hot-swap, and model version registry.
"""

from __future__ import annotations

from .auth          import require_api_key, is_valid_key, reload_keys
from .logging_setup import setup_logging, get_logger
from .metrics       import REGISTRY, Counter, Gauge, Histogram, MetricsRegistry
from .metrics       import requests_total, requests_errors, request_latency, active_requests
from .degradation   import DegradationController, DegradationLevel, DeploymentMode, CanaryController
from .versioning    import ModelVersionRegistry
from .audit         import AuditLogger
from .migration     import migrate_checkpoint, HotSwapAdapter, get_checkpoint_version

__all__ = [
    "require_api_key", "is_valid_key", "reload_keys",
    "setup_logging", "get_logger",
    "REGISTRY", "Counter", "Gauge", "Histogram", "MetricsRegistry",
    "requests_total", "requests_errors", "request_latency", "active_requests",
    "DegradationController", "DegradationLevel", "DeploymentMode", "CanaryController",
    "ModelVersionRegistry",
    "AuditLogger",
    "migrate_checkpoint", "HotSwapAdapter", "get_checkpoint_version",
]
