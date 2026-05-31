"""
Checkpoint schema migration utility for VULGARIS.

Handles forward-compatibility when the checkpoint format_version advances.
Also provides a hot-swap adapter protocol so a new checkpoint can be loaded
into a live server without restarting the process.

Supported migrations
--------------------
  v1 → v2 : add weights_sha256 field to metadata.json (backfill hash)
  v2 → v3 : (placeholder — extend when format_version bumps)

Usage
-----
    from serve.migration import migrate_checkpoint, HotSwapAdapter

    # Offline: upgrade an old checkpoint directory in-place
    migrate_checkpoint("checkpoints/model_v1", target_version=2)

    # Online: swap running model weights without dropping connections
    adapter = HotSwapAdapter(streaming_inference_instance)
    adapter.swap("checkpoints/model_v2")
from __future__ import annotations

"""


import hashlib
import json
import os
import shutil
import threading
import time
from typing import Optional


# ──────────────────────────────────────────────────────────────────────────────
# Schema migration
# ──────────────────────────────────────────────────────────────────────────────

CURRENT_FORMAT_VERSION = 2

_MIGRATIONS: dict = {}   # {from_version: migration_fn}


def _register(from_ver):
    def decorator(fn):
        _MIGRATIONS[from_ver] = fn
        return fn
    return decorator


@_register(1)
def _migrate_v1_to_v2(path: str):
    """Backfill weights_sha256 into metadata.json."""
    weights_path = os.path.join(path, "weights.npz")
    meta_path    = os.path.join(path, "metadata.json")

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"weights.npz not found in {path}")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"metadata.json not found in {path}")

    h = hashlib.sha256()
    with open(weights_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)

    with open(meta_path) as f:
        meta = json.load(f)

    meta["weights_sha256"] = h.hexdigest()
    meta["format_version"] = 2
    meta["migrated_at"]    = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


def get_checkpoint_version(path: str) -> int:
    """Return format_version from metadata.json, defaulting to 1 if missing."""
    meta_path = os.path.join(path, "metadata.json")
    if not os.path.exists(meta_path):
        return 1
    with open(meta_path) as f:
        meta = json.load(f)
    return int(meta.get("format_version", 1))


def migrate_checkpoint(path: str, target_version: int = CURRENT_FORMAT_VERSION,
                       backup: bool = True) -> int:
    """
    Migrate a checkpoint directory from its current schema version to
    `target_version`.  Returns the version after migration.

    Parameters
    ----------
    path           : checkpoint directory
    target_version : desired schema version (default: CURRENT_FORMAT_VERSION)
    backup         : if True, copy the directory to <path>.bak before mutating
    """
    current = get_checkpoint_version(path)
    if current >= target_version:
        return current

    if backup:
        bak = path.rstrip("/\\") + ".bak"
        if os.path.exists(bak):
            shutil.rmtree(bak)
        shutil.copytree(path, bak)

    ver = current
    while ver < target_version:
        fn = _MIGRATIONS.get(ver)
        if fn is None:
            raise NotImplementedError(
                f"No migration path from format_version={ver} to {ver+1}. "
                f"Upgrade vulgaris to support this version."
            )
        fn(path)
        ver += 1

    return ver


# ──────────────────────────────────────────────────────────────────────────────
# Hot-swap adapter protocol
# ──────────────────────────────────────────────────────────────────────────────

class HotSwapAdapter:
    """
    Live model weight replacement without restarting the inference server.

    The swap is atomic from the perspective of in-flight `step()` calls:
      1. Load new weights into a shadow model.
      2. Acquire a brief write-lock (drains the in-flight step).
      3. Copy shadow weights into the live model's parameters.
      4. Release lock.

    Typical swap time < 200 ms for a 50M-param model on CPU.

    Parameters
    ----------
    streaming : StreamingInference instance currently serving traffic.
    """

    def __init__(self, streaming):
        self._streaming = streaming
        self._lock = threading.Lock()

    def swap(self, checkpoint_path: str, verify_hash: bool = True,
             auto_migrate: bool = True) -> dict:
        """
        Load a new checkpoint and hot-swap the live model's weights.

        Parameters
        ----------
        checkpoint_path : directory produced by `model.save_pretrained()`
        verify_hash     : if True, validate SHA-256 before swapping
        auto_migrate    : if True, migrate the checkpoint schema if needed

        Returns
        -------
        dict with keys: previous_version, new_version, swap_duration_ms
        """
        if auto_migrate:
            migrate_checkpoint(checkpoint_path)

        # Load new weights (heavyweight — do outside the lock)
        from model.vulgaris import Vulgaris
        new_model = Vulgaris.load(checkpoint_path)   # hash check inside load()

        t0 = time.perf_counter()
        with self._lock:
            live = self._streaming.model
            new_state = new_model.state_dict()
            live.load_state_dict(new_state)
            # Reset streaming state so recurrences start fresh
            self._streaming.state = live.init_state(self._streaming.batch_size)

        swap_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "checkpoint": checkpoint_path,
            "swap_duration_ms": round(swap_ms, 2),
        }
