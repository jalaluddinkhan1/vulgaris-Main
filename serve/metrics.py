"""In-process metrics for VULGARIS (no prometheus_client dependency)."""
import time
import threading
from collections import defaultdict
from typing import Dict, List, Optional


class Counter:
    def __init__(self, name: str, help_text: str = ""):
        self.name = name
        self.help_text = help_text
        self._value: float = 0.0
        self._lock = threading.Lock()

    def inc(self, amount: float = 1.0):
        with self._lock:
            self._value += amount

    def get(self) -> float:
        return self._value

    def exposition(self) -> str:
        lines = []
        if self.help_text:
            lines.append(f"# HELP {self.name} {self.help_text}")
        lines.append(f"# TYPE {self.name} counter")
        lines.append(f"{self.name} {self._value}")
        return "\n".join(lines)


class Gauge:
    def __init__(self, name: str, help_text: str = ""):
        self.name = name
        self.help_text = help_text
        self._value: float = 0.0
        self._lock = threading.Lock()

    def set(self, value: float):
        with self._lock:
            self._value = value

    def inc(self, amount: float = 1.0):
        with self._lock:
            self._value += amount

    def dec(self, amount: float = 1.0):
        with self._lock:
            self._value -= amount

    def get(self) -> float:
        return self._value

    def exposition(self) -> str:
        lines = []
        if self.help_text:
            lines.append(f"# HELP {self.name} {self.help_text}")
        lines.append(f"# TYPE {self.name} gauge")
        lines.append(f"{self.name} {self._value}")
        return "\n".join(lines)


class Histogram:
    DEFAULT_BUCKETS = [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]

    def __init__(self, name: str, help_text: str = "", buckets: Optional[List[float]] = None):
        self.name = name
        self.help_text = help_text
        self._buckets = sorted(buckets or self.DEFAULT_BUCKETS)
        self._counts = [0] * len(self._buckets)
        self._inf_count = 0
        self._sum: float = 0.0
        self._total: int = 0
        self._lock = threading.Lock()
        self._window: List[float] = []
        self._window_max = 1000

    def observe(self, value: float):
        with self._lock:
            self._sum += value
            self._total += 1
            self._window.append(value)
            if len(self._window) > self._window_max:
                self._window.pop(0)
            for i, b in enumerate(self._buckets):
                if value <= b:
                    self._counts[i] += 1
            self._inf_count += 1

    def p95(self) -> float:
        with self._lock:
            if not self._window:
                return 0.0
            import numpy as np
            return float(np.percentile(self._window, 95))

    def p99(self) -> float:
        with self._lock:
            if not self._window:
                return 0.0
            import numpy as np
            return float(np.percentile(self._window, 99))

    def exposition(self) -> str:
        lines = []
        if self.help_text:
            lines.append(f"# HELP {self.name} {self.help_text}")
        lines.append(f"# TYPE {self.name} histogram")
        for b, c in zip(self._buckets, self._counts):
            lines.append(f'{self.name}_bucket{{le="{b}"}} {c}')
        lines.append(f'{self.name}_bucket{{le="+Inf"}} {self._inf_count}')
        lines.append(f"{self.name}_sum {self._sum}")
        lines.append(f"{self.name}_count {self._total}")
        return "\n".join(lines)


class MetricsRegistry:
    def __init__(self):
        self._metrics: Dict[str, object] = {}

    def counter(self, name: str, help_text: str = "") -> Counter:
        if name not in self._metrics:
            self._metrics[name] = Counter(name, help_text)
        return self._metrics[name]  # type: ignore

    def gauge(self, name: str, help_text: str = "") -> Gauge:
        if name not in self._metrics:
            self._metrics[name] = Gauge(name, help_text)
        return self._metrics[name]  # type: ignore

    def histogram(self, name: str, help_text: str = "", buckets: Optional[List[float]] = None) -> Histogram:
        if name not in self._metrics:
            self._metrics[name] = Histogram(name, help_text, buckets)
        return self._metrics[name]  # type: ignore

    def exposition_text(self) -> str:
        parts = [m.exposition() for m in self._metrics.values()]
        return "\n".join(parts) + "\n"


# Singleton registry
REGISTRY = MetricsRegistry()

# Pre-defined metrics
requests_total = REGISTRY.counter("vulgaris_requests_total", "Total inference requests")
requests_errors = REGISTRY.counter("vulgaris_requests_errors_total", "Total inference errors")
request_latency = REGISTRY.histogram("vulgaris_request_latency_seconds", "Inference latency in seconds")
active_requests = REGISTRY.gauge("vulgaris_active_requests", "Currently active requests")
model_version = REGISTRY.gauge("vulgaris_model_version_info", "Loaded model version")
