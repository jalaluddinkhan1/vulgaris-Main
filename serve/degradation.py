"""Graceful degradation controller for VULGARIS inference."""
import time
import threading
from enum import Enum
from typing import Optional


class DegradationLevel(Enum):
    FULL = "full"           # All features active
    REDUCED = "reduced"     # Skip ESE + DAH, return cached outputs
    ALERT_ONLY = "alert_only"  # Only anomaly scores, no classification


class DegradationController:
    """
    Monitors p95 latency and error rate; auto-downgrades serving level.

    Thresholds (configurable):
      p95 latency > 2.0s → REDUCED
      p95 latency > 5.0s or error_rate > 0.1 → ALERT_ONLY
      Recovery: 60s window below threshold → upgrade
    """

    def __init__(
        self,
        p95_reduced_threshold: float = 2.0,
        p95_alert_threshold: float = 5.0,
        error_rate_alert_threshold: float = 0.1,
        recovery_window_s: float = 60.0,
    ):
        self.p95_reduced_threshold = p95_reduced_threshold
        self.p95_alert_threshold = p95_alert_threshold
        self.error_rate_alert_threshold = error_rate_alert_threshold
        self.recovery_window_s = recovery_window_s

        self._level = DegradationLevel.FULL
        self._lock = threading.Lock()
        self._latencies: list = []
        self._error_window: list = []  # (timestamp, is_error)
        self._max_window = 500
        self._last_degraded_at: Optional[float] = None

    def record(self, latency_s: float, is_error: bool = False):
        with self._lock:
            self._latencies.append(latency_s)
            if len(self._latencies) > self._max_window:
                self._latencies.pop(0)
            now = time.time()
            self._error_window.append((now, is_error))
            # Keep last 60s
            cutoff = now - 60.0
            self._error_window = [(t, e) for t, e in self._error_window if t > cutoff]
            self._update_level()

    def _update_level(self):
        if len(self._latencies) < 10:
            return
        import numpy as np
        p95 = float(np.percentile(self._latencies[-200:], 95))

        errors = [e for _, e in self._error_window]
        error_rate = sum(errors) / max(len(errors), 1)

        now = time.time()
        if p95 > self.p95_alert_threshold or error_rate > self.error_rate_alert_threshold:
            if self._level != DegradationLevel.ALERT_ONLY:
                self._level = DegradationLevel.ALERT_ONLY
                self._last_degraded_at = now
        elif p95 > self.p95_reduced_threshold:
            if self._level == DegradationLevel.FULL:
                self._level = DegradationLevel.REDUCED
                self._last_degraded_at = now
        else:
            # Attempt recovery
            if self._last_degraded_at and (now - self._last_degraded_at) > self.recovery_window_s:
                if self._level == DegradationLevel.ALERT_ONLY:
                    self._level = DegradationLevel.REDUCED
                    self._last_degraded_at = now
                elif self._level == DegradationLevel.REDUCED:
                    self._level = DegradationLevel.FULL
                    self._last_degraded_at = None

    @property
    def level(self) -> DegradationLevel:
        return self._level

    def status(self) -> dict:
        with self._lock:
            import numpy as np
            p95 = float(np.percentile(self._latencies[-200:], 95)) if len(self._latencies) >= 10 else 0.0
            errors = [e for _, e in self._error_window]
            return {
                "level": self._level.value,
                "p95_latency_s": round(p95, 4),
                "error_rate": round(sum(errors) / max(len(errors), 1), 4),
                "total_recorded": len(self._latencies),
            }
