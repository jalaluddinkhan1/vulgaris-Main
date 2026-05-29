"""Append-only per-prediction audit logger for VULGARIS."""
import hashlib
import json
import os
import threading
import time
from typing import Dict, List, Optional


class AuditLogger:
    """
    Writes one JSONL record per prediction to an append-only audit file.

    Each record contains:
      ts              — UTC timestamp (float)
      input_sha256    — SHA-256 of the raw input bytes (hex)
      prediction      — model output (list)
      interval_90     — [lo, hi] conformal prediction interval, if available
      rules_fired     — list of rule IDs that triggered
      causal_parents  — list of causal parent node IDs from CRG
      safety_override — True if SafetyHead blocked/modified the output
      weights_sha256  — SHA-256 of active model weights snapshot (hex)
      domain_idx      — domain adapter index used
      step            — inference step counter

    Usage
    -----
        logger = AuditLogger("audit/predictions.jsonl")
        logger.record(
            x_raw=x_np,
            prediction=pred,
            domain_idx=0,
            step=engine.step_count,
        )
        recent = logger.tail(20)
    """

    def __init__(self, path: str, weights_sha256: Optional[str] = None):
        self.path = path
        self._lock = threading.Lock()
        self._count = 0
        self._weights_sha256 = weights_sha256 or ""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        # Touch file so tail() works even before the first record
        if not os.path.exists(path):
            open(path, "a").close()

    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _sha256(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    # ──────────────────────────────────────────────────────────────────────

    def update_weights_sha256(self, sha: str):
        """Call after a weight update/checkpoint load to refresh the hash."""
        with self._lock:
            self._weights_sha256 = sha

    @staticmethod
    def compute_weights_sha256(params) -> str:
        """Compute SHA-256 over all parameter arrays (pass list of Parameter)."""
        import numpy as np
        h = hashlib.sha256()
        for p in params:
            h.update(np.asarray(p.data).tobytes())
        return h.hexdigest()

    # ──────────────────────────────────────────────────────────────────────

    def record(
        self,
        x_raw,
        prediction,
        step: int,
        domain_idx: int = 0,
        interval_90: Optional[List[float]] = None,
        rules_fired: Optional[List[str]] = None,
        causal_parents: Optional[List[int]] = None,
        safety_override: bool = False,
    ):
        """Write one audit record.  Thread-safe; non-blocking on fast paths."""
        import numpy as np
        x_bytes = np.asarray(x_raw).astype(np.float32).tobytes()
        input_hash = self._sha256(x_bytes)

        pred_list = (
            prediction.tolist()
            if hasattr(prediction, "tolist")
            else list(prediction)
        )

        record: Dict = {
            "ts": time.time(),
            "input_sha256": input_hash,
            "prediction": pred_list,
            "interval_90": interval_90,
            "rules_fired": rules_fired or [],
            "causal_parents": causal_parents or [],
            "safety_override": safety_override,
            "weights_sha256": self._weights_sha256,
            "domain_idx": domain_idx,
            "step": step,
        }

        line = json.dumps(record, separators=(",", ":")) + "\n"
        with self._lock:
            with open(self.path, "a") as f:
                f.write(line)
            self._count += 1

    # ──────────────────────────────────────────────────────────────────────

    def tail(self, n: int = 100) -> List[Dict]:
        """Return the last `n` records (reads from file, newest last)."""
        with self._lock:
            try:
                with open(self.path, "r") as f:
                    lines = f.readlines()
            except FileNotFoundError:
                return []
        records = []
        for line in lines[-n:]:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return records

    def count(self) -> int:
        """Number of records written in this process lifetime."""
        with self._lock:
            return self._count

    def status(self) -> Dict:
        return {
            "path": self.path,
            "records_written": self._count,
            "weights_sha256": self._weights_sha256,
        }
