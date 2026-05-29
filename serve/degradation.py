"""Graceful degradation controller for VULGARIS inference."""
import time
import threading
from enum import Enum
from typing import Callable, Dict, List, Optional


class DeploymentMode(Enum):
    PRIMARY = "primary"   # Live traffic; predictions acted upon
    CANARY  = "canary"    # Fraction of traffic; compare vs primary
    SHADOW  = "shadow"    # Receives all traffic but predictions not acted upon


class CanaryController:
    """
    Routes a configurable percentage of requests to a canary model and logs
    all shadow predictions for offline comparison.

    Shadow mode: the controller receives every prediction but never signals
    `should_use_output` — outputs are logged to `_shadow_log` for later
    analysis without affecting live serving.

    Canary mode: `canary_pct`% of requests are flagged for use; the rest
    fall back to the primary.  Agreement rate vs primary is tracked.

    Usage
    -----
        ctrl = CanaryController(mode=DeploymentMode.CANARY, canary_pct=5.0)
        if ctrl.should_use_output():
            result = canary_model.predict(x)
            ctrl.log_shadow(primary_pred, result['prediction'], agree=np.allclose(...))
        else:
            result = primary_model.predict(x)
    """

    def __init__(
        self,
        mode: DeploymentMode = DeploymentMode.PRIMARY,
        canary_pct: float = 5.0,
        max_shadow_log: int = 10_000,
    ):
        self.mode = mode
        self.canary_pct = canary_pct
        self._max_shadow_log = max_shadow_log
        self._lock = threading.Lock()
        self._shadow_log: List[Dict] = []
        self._total_requests: int = 0
        self._canary_requests: int = 0
        self._agree_count: int = 0

    def set_mode(self, mode: DeploymentMode):
        with self._lock:
            self.mode = mode

    def should_use_output(self) -> bool:
        """
        Returns True if this request should use the canary/shadow model output.
        - PRIMARY: always False (primary handles everything)
        - CANARY:  True for `canary_pct`% of requests (random sampling)
        - SHADOW:  always False (log-only, never acted upon)
        """
        with self._lock:
            self._total_requests += 1
            if self.mode == DeploymentMode.PRIMARY:
                return False
            if self.mode == DeploymentMode.SHADOW:
                return False
            # CANARY: probabilistic routing
            import numpy as np
            use = bool(np.random.rand() * 100.0 < self.canary_pct)
            if use:
                self._canary_requests += 1
            return use

    def log_shadow(
        self,
        primary_pred,
        canary_pred,
        agree: bool = False,
        metadata: Optional[Dict] = None,
    ):
        """Record one shadow comparison entry."""
        with self._lock:
            if agree:
                self._agree_count += 1
            if len(self._shadow_log) < self._max_shadow_log:
                entry: Dict = {
                    "ts": time.time(),
                    "primary": primary_pred.tolist() if hasattr(primary_pred, "tolist") else primary_pred,
                    "canary":  canary_pred.tolist()  if hasattr(canary_pred,  "tolist") else canary_pred,
                    "agree":   agree,
                }
                if metadata:
                    entry.update(metadata)
                self._shadow_log.append(entry)

    def shadow_stats(self) -> Dict:
        with self._lock:
            n = max(len(self._shadow_log), 1)
            agree_rate = self._agree_count / n
            return {
                "mode": self.mode.value,
                "total_requests": self._total_requests,
                "canary_requests": self._canary_requests,
                "shadow_entries": len(self._shadow_log),
                "agree_rate": round(agree_rate, 4),
                "canary_pct": self.canary_pct,
            }

    def get_shadow_log(self, last_n: int = 100) -> List[Dict]:
        with self._lock:
            return list(self._shadow_log[-last_n:])


# ──────────────────────────────────────────────────────────────────────────────


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


# ──────────────────────────────────────────────────────────────────────────────


class MemoryPressurePolicy:
    """
    Defines what gets evicted first when the process is under memory pressure.

    Eviction priority (lowest memory cost first):
      1. Adapter weight cache  — cheapest: re-load from disk
      2. HMB working buffer    — moderate: loses recent context
      3. HMB archive           — expensive: loses long-term memory

    Call `register_*` from the inference server to wire in real evictors.
    `check_and_evict()` is called by the server's health-check loop.
    """

    RSS_WARN_MB  = 3_000    # warn above 3 GB RSS
    RSS_EVICT_MB = 4_500    # start evicting above 4.5 GB RSS

    def __init__(self):
        self._adapter_evictor:  Optional[Callable[[], int]] = None
        self._hmb_buf_evictor:  Optional[Callable[[], int]] = None
        self._hmb_arch_evictor: Optional[Callable[[], int]] = None

    def register_adapter_evictor(self, fn: Callable[[], int]):
        """fn() evicts adapter cache entries and returns bytes freed."""
        self._adapter_evictor = fn

    def register_hmb_buffer_evictor(self, fn: Callable[[], int]):
        """fn() clears oldest half of HMB working buffer; returns bytes freed."""
        self._hmb_buf_evictor = fn

    def register_hmb_archive_evictor(self, fn: Callable[[], int]):
        """fn() evicts lowest-uncertainty archive entries; returns bytes freed."""
        self._hmb_arch_evictor = fn

    @staticmethod
    def _rss_mb() -> float:
        try:
            import psutil
            return psutil.Process().memory_info().rss / (1024 * 1024)
        except ImportError:
            return 0.0   # psutil not available — skip pressure check

    def check_and_evict(self) -> dict:
        """
        Check current RSS.  Evict in priority order until RSS drops below
        RSS_WARN_MB.  Returns dict with bytes_freed and actions taken.
        """
        rss = self._rss_mb()
        actions = []
        freed = 0

        if rss < self.RSS_WARN_MB:
            return {"rss_mb": rss, "actions": [], "bytes_freed": 0}

        for label, evictor in [
            ("adapter_cache",  self._adapter_evictor),
            ("hmb_buffer",     self._hmb_buf_evictor),
            ("hmb_archive",    self._hmb_arch_evictor),
        ]:
            if evictor is None:
                continue
            try:
                b = evictor()
                freed += b
                actions.append(label)
            except Exception:
                pass
            rss = self._rss_mb()
            if rss < self.RSS_WARN_MB:
                break

        return {"rss_mb": rss, "actions": actions, "bytes_freed": freed}


# ──────────────────────────────────────────────────────────────────────────────


class ThermalThrottleController:
    """
    Reduces inference throughput when the host CPU/GPU runs hot.

    Monitors temperature via psutil (CPU) or nvidia-smi (GPU).
    When temperature exceeds `warn_c`, inserts inter-step sleep.
    When temperature exceeds `throttle_c`, doubles the sleep.

    Temperatures are polled lazily (every `poll_interval` steps) to avoid
    making the per-step critical path dependent on a syscall.
    """

    def __init__(
        self,
        warn_c: float = 80.0,
        throttle_c: float = 90.0,
        warn_sleep_s: float = 0.005,
        throttle_sleep_s: float = 0.020,
        poll_interval: int = 100,
    ):
        self.warn_c           = warn_c
        self.throttle_c       = throttle_c
        self.warn_sleep_s     = warn_sleep_s
        self.throttle_sleep_s = throttle_sleep_s
        self.poll_interval    = poll_interval
        self._step            = 0
        self._current_temp    = 0.0
        self._sleep_s         = 0.0

    @staticmethod
    def _read_cpu_temp() -> float:
        try:
            import psutil
            temps = psutil.sensors_temperatures()
            if not temps:
                return 0.0
            # Use first available sensor package
            for readings in temps.values():
                for r in readings:
                    if r.current > 0:
                        return float(r.current)
        except Exception:
            pass
        return 0.0

    def maybe_throttle(self):
        """Call once per inference step.  Sleeps if temperature is high."""
        self._step += 1
        if self._step % self.poll_interval == 0:
            temp = self._read_cpu_temp()
            self._current_temp = temp
            if temp >= self.throttle_c:
                self._sleep_s = self.throttle_sleep_s
            elif temp >= self.warn_c:
                self._sleep_s = self.warn_sleep_s
            else:
                self._sleep_s = 0.0

        if self._sleep_s > 0.0:
            time.sleep(self._sleep_s)

    def status(self) -> dict:
        return {
            "cpu_temp_c": self._current_temp,
            "sleep_per_step_s": self._sleep_s,
            "throttling": self._sleep_s > 0.0,
        }
