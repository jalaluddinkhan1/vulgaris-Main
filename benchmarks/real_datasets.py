"""Dataset loaders for VULGARIS benchmarking (with synthetic fallback)."""
import os
import numpy as np
from typing import Dict, Optional, Tuple


class ETTDataset:
    """
    Electricity Transformer Temperature (ETT) dataset loader.
    Falls back to synthetic sinusoidal data if CSV not found.
    """
    name = "ETT"

    def __init__(self, csv_path: Optional[str] = None, seq_len: int = 96, n_channels: int = 7):
        self.csv_path = csv_path
        self.seq_len = seq_len
        self.n_channels = n_channels
        self._data: Optional[np.ndarray] = None

    def load(self) -> np.ndarray:
        """Returns (N, C, T) array."""
        if self.csv_path and os.path.exists(self.csv_path):
            try:
                import csv
                rows = []
                with open(self.csv_path) as f:
                    reader = csv.reader(f)
                    next(reader)  # skip header
                    for row in reader:
                        try:
                            rows.append([float(v) for v in row[1:self.n_channels + 1]])
                        except (ValueError, IndexError):
                            pass
                data = np.array(rows, dtype=np.float32).T  # (C, total_T)
                return self._to_windows(data)
            except Exception:
                pass
        return self._synthetic()

    def _to_windows(self, data: np.ndarray) -> np.ndarray:
        C, total_T = data.shape
        n_windows = total_T // self.seq_len
        out = np.zeros((n_windows, C, self.seq_len), dtype=np.float32)
        for i in range(n_windows):
            out[i] = data[:, i * self.seq_len:(i + 1) * self.seq_len]
        return out

    def _synthetic(self) -> np.ndarray:
        rng = np.random.default_rng(42)
        N = 500
        t = np.linspace(0, 4 * np.pi, self.seq_len)
        data = np.zeros((N, self.n_channels, self.seq_len), dtype=np.float32)
        for c in range(self.n_channels):
            freq = 0.5 + c * 0.3
            phase = rng.uniform(0, 2 * np.pi)
            signal = np.sin(freq * t + phase) + 0.1 * rng.normal(size=(N, self.seq_len))
            data[:, c, :] = signal.astype(np.float32)
        return data


class NABDataset:
    """
    Numenta Anomaly Benchmark dataset loader.
    Falls back to synthetic anomaly-injected data.
    """
    name = "NAB"

    def __init__(self, csv_path: Optional[str] = None, seq_len: int = 64, n_channels: int = 1):
        self.csv_path = csv_path
        self.seq_len = seq_len
        self.n_channels = n_channels

    def load(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (X: (N, C, T), labels: (N,)) where labels=1 means anomaly window."""
        if self.csv_path and os.path.exists(self.csv_path):
            try:
                rows = []
                with open(self.csv_path) as f:
                    import csv
                    reader = csv.reader(f)
                    next(reader)
                    for row in reader:
                        try:
                            rows.append(float(row[1]))
                        except (ValueError, IndexError):
                            pass
                series = np.array(rows, dtype=np.float32)
                return self._to_windows_with_labels(series)
            except Exception:
                pass
        return self._synthetic()

    def _to_windows_with_labels(self, series: np.ndarray):
        T = self.seq_len
        n = len(series) // T
        X = np.zeros((n, 1, T), dtype=np.float32)
        labels = np.zeros(n, dtype=np.int32)
        for i in range(n):
            window = series[i * T:(i + 1) * T]
            X[i, 0] = window
            z = (window - window.mean()) / (window.std() + 1e-8)
            labels[i] = int(np.any(np.abs(z) > 3.0))
        return X, labels

    def _synthetic(self):
        rng = np.random.default_rng(0)
        N = 300
        T = self.seq_len
        t = np.arange(T, dtype=np.float32)
        X = np.zeros((N, 1, T), dtype=np.float32)
        labels = np.zeros(N, dtype=np.int32)
        for i in range(N):
            signal = np.sin(0.2 * t) + 0.05 * rng.normal(size=T)
            if rng.random() < 0.15:
                spike_pos = rng.integers(T // 4, 3 * T // 4)
                signal[spike_pos:spike_pos + 3] += rng.uniform(3, 6)
                labels[i] = 1
            X[i, 0] = signal.astype(np.float32)
        return X, labels


class EdgeTelemetryDataset:
    """
    Synthetic edge/IoT/telecom telemetry dataset with labeled events.
    Generates 5 event types matching VULGARIS network_telecom_test.py.
    """
    name = "EdgeTelemetry"

    def __init__(
        self,
        csv_path: Optional[str] = None,
        n_channels: int = 9,
        seq_len: int = 50,
        n_samples: int = 1000,
        n_classes: int = 5,
        seed: int = 42,
    ):
        self.csv_path = csv_path
        self.n_channels = n_channels
        self.seq_len = seq_len
        self.n_samples = n_samples
        self.n_classes = n_classes
        self.seed = seed

    def load(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (X: (N, C, T), y: (N,) class labels)."""
        if self.csv_path and os.path.exists(self.csv_path):
            try:
                data = np.load(self.csv_path)
                return data["X"].astype(np.float32), data["y"].astype(np.int32)
            except Exception:
                pass
        return self._synthetic()

    def _synthetic(self) -> Tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(self.seed)
        C, T = self.n_channels, self.seq_len
        N = self.n_samples
        X = np.zeros((N, C, T), dtype=np.float32)
        y = np.zeros(N, dtype=np.int32)
        per_class = N // self.n_classes

        def gen(cls_idx, n):
            out = rng.normal(0, 0.1, (n, C, T)).astype(np.float32)
            t = np.arange(T, dtype=np.float32)
            if cls_idx == 0:  # normal
                for c in range(C):
                    out[:, c] += np.sin(0.1 * t + c)
            elif cls_idx == 1:  # congestion: rising load
                out[:, :3] += np.linspace(0, 2, T)
            elif cls_idx == 2:  # hw_fault: channel dropout + spike
                drop_ch = rng.integers(0, C)
                out[:, drop_ch] = 0
                spike_t = rng.integers(T // 3, 2 * T // 3, size=n)
                for i, st in enumerate(spike_t):
                    out[i, 0, st] += 5.0
            elif cls_idx == 3:  # ddos: all channels high frequency
                for c in range(C):
                    out[:, c] += np.sin(0.8 * t + c * 0.3) * 2
            elif cls_idx == 4:  # routing_loop: oscillation
                for c in range(C):
                    out[:, c] += np.sin(0.3 * t) * np.sin(0.07 * t + c)
            return out

        idx = 0
        for cls in range(self.n_classes):
            n = per_class if cls < self.n_classes - 1 else N - idx
            X[idx:idx + n] = gen(cls, n)
            y[idx:idx + n] = cls
            idx += n

        return X, y


def load_benchmark_suite(
    ett_path: Optional[str] = None,
    nab_path: Optional[str] = None,
    telemetry_path: Optional[str] = None,
    seq_len: int = 50,
    n_channels: int = 9,
) -> Dict[str, dict]:
    """
    Load all benchmark datasets.
    Returns dict of {name: {"X_train", "X_test", "y_train", "y_test"}}.
    """
    suite = {}

    # ETT
    ett = ETTDataset(csv_path=ett_path, seq_len=seq_len, n_channels=min(7, n_channels))
    X_ett = ett.load()
    n = len(X_ett)
    split = int(n * 0.8)
    suite["ETT"] = {
        "X_train": X_ett[:split],
        "X_test": X_ett[split:],
        "y_train": np.zeros(split, dtype=np.float32),
        "y_test": np.zeros(n - split, dtype=np.float32),
    }

    # NAB
    nab = NABDataset(csv_path=nab_path, seq_len=seq_len)
    X_nab, y_nab = nab.load()
    n = len(X_nab)
    split = int(n * 0.8)
    suite["NAB"] = {
        "X_train": X_nab[:split],
        "X_test": X_nab[split:],
        "y_train": y_nab[:split].astype(np.float32),
        "y_test": y_nab[split:].astype(np.float32),
    }

    # EdgeTelemetry
    tel = EdgeTelemetryDataset(
        csv_path=telemetry_path, n_channels=n_channels, seq_len=seq_len
    )
    X_tel, y_tel = tel.load()
    n = len(X_tel)
    split = int(n * 0.8)
    suite["EdgeTelemetry"] = {
        "X_train": X_tel[:split],
        "X_test": X_tel[split:],
        "y_train": y_tel[:split].astype(np.float32),
        "y_test": y_tel[split:].astype(np.float32),
    }

    return suite
