"""API key authentication for VULGARIS inference server."""
import os
from typing import Callable
from functools import wraps


_API_KEYS: set = set()


def _load_keys():
    raw = os.environ.get("VULGARIS_API_KEYS", "")
    if raw:
        _API_KEYS.update(k.strip() for k in raw.split(",") if k.strip())


_load_keys()


def reload_keys():
    """Reload API keys from environment (call after os.environ changes)."""
    _API_KEYS.clear()
    _load_keys()


def is_valid_key(api_key: str) -> bool:
    if not _API_KEYS:
        return True  # open access when no keys configured
    return api_key in _API_KEYS


def require_api_key(request):
    """Extract and validate API key from request headers. Returns (valid: bool, key: str)."""
    key = request.headers.get("X-API-Key", "")
    return is_valid_key(key), key
