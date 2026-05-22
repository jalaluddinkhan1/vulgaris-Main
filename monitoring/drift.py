"""Distribution drift detection for VULGARIS (pure numpy)."""
import numpy as np
from typing import Optional, Tuple
from collections import deque


class DriftDetector:
    """
    Detects distribution shift between reference window and current window.

    Three statistics (all pure numpy):
      - KS statistic (Kolmogorov-Smirnov, per-feature max)
      - Linear MMD (Maximum Mean Discrepancy with RBF kernel approximation)
      - Wasserstein-1D (Earth Mover's Distance, per-feature mean)

    Usage:
        detector = DriftDetector(window_size=200, n_features=64)
        detector.set_reference(reference_samples)  # (N, d)
        result = detector.update(new_batch)         # (M, d)
        if result["drift_detected"]:
            ...
    """

    def __init__(
        self,
        window_size: int = 200,
        n_features: int = 64,
        ks_threshold: float = 0.1,
        mmd_threshold: float = 0.05,
        wasserstein_threshold: float = 0.1,
        alpha: float = 0.05,
    ):
        self.window_size = window_size
        self.n_features = n_features
        self.ks_threshold = ks_threshold
        self.mmd_threshold = mmd_threshold
        self.wasserstein_threshold = wasserstein_threshold
        self.alpha = alpha

        self._reference: Optional[np.ndarray] = None
        self._current_window: deque = deque(maxlen=window_size)
        self._history: list = []

    def set_reference(self, samples: np.ndarray):
        """Set reference distribution. samples: (N, d)"""
        self._reference = samples.astype(np.float32)
        self._current_window.clear()

    def update(self, samples: np.ndarray) -> dict:
        """
        Add samples to current window and compute drift statistics.
        samples: (M, d) or (M,) for 1D
        Returns dict with drift statistics and detection flag.
        """
        if samples.ndim == 1:
            samples = samples.reshape(-1, 1)
        samples = samples.astype(np.float32)

        for s in samples:
            self._current_window.append(s)

        if self._reference is None or len(self._current_window) < 10:
            return {
                "drift_detected": False,
                "ks_stat": 0.0,
                "mmd_stat": 0.0,
                "wasserstein_stat": 0.0,
                "n_current": len(self._current_window),
                "n_reference": 0,
            }

        current = np.array(list(self._current_window))
        reference = self._reference

        ks_stat = self._ks_statistic(reference, current)
        mmd_stat = self._mmd(reference, current)
        w_stat = self._wasserstein_1d(reference, current)

        drift_detected = (
            ks_stat > self.ks_threshold
            or mmd_stat > self.mmd_threshold
            or w_stat > self.wasserstein_threshold
        )

        result = {
            "drift_detected": drift_detected,
            "ks_stat": float(ks_stat),
            "mmd_stat": float(mmd_stat),
            "wasserstein_stat": float(w_stat),
            "n_current": len(current),
            "n_reference": len(reference),
        }
        self._history.append(result)
        return result

    def _ks_statistic(self, ref: np.ndarray, cur: np.ndarray) -> float:
        """Per-feature KS statistic, return max across features."""
        n_feat = min(ref.shape[1] if ref.ndim > 1 else 1,
                     cur.shape[1] if cur.ndim > 1 else 1)
        if ref.ndim == 1:
            ref = ref.reshape(-1, 1)
        if cur.ndim == 1:
            cur = cur.reshape(-1, 1)

        ks_vals = []
        for f in range(n_feat):
            r = np.sort(ref[:, f])
            c = np.sort(cur[:, f])
            # Two-sample KS: compare ECDFs
            combined = np.concatenate([r, c])
            combined_sorted = np.sort(combined)
            # CDF values
            r_cdf = np.searchsorted(r, combined_sorted, side='right') / len(r)
            c_cdf = np.searchsorted(c, combined_sorted, side='right') / len(c)
            ks_vals.append(float(np.max(np.abs(r_cdf - c_cdf))))

        return float(np.max(ks_vals)) if ks_vals else 0.0

    def _mmd(self, ref: np.ndarray, cur: np.ndarray, n_samples: int = 100) -> float:
        """Linear-time MMD with RBF kernel approximation."""
        if ref.ndim == 1:
            ref = ref.reshape(-1, 1)
        if cur.ndim == 1:
            cur = cur.reshape(-1, 1)

        # Subsample for efficiency
        n_r = min(n_samples, len(ref))
        n_c = min(n_samples, len(cur))
        r = ref[np.random.choice(len(ref), n_r, replace=False)]
        c = cur[np.random.choice(len(cur), n_c, replace=False)]

        # Median heuristic for bandwidth
        all_samples = np.concatenate([r, c])
        dists = np.linalg.norm(all_samples[:, None] - all_samples[None, :], axis=-1)
        sigma = float(np.median(dists[dists > 0])) + 1e-8

        def rbf(a, b):
            d2 = np.sum((a[:, None] - b[None, :]) ** 2, axis=-1)
            return np.exp(-d2 / (2 * sigma ** 2))

        k_rr = rbf(r, r).mean()
        k_cc = rbf(c, c).mean()
        k_rc = rbf(r, c).mean()

        mmd2 = float(k_rr + k_cc - 2 * k_rc)
        return max(0.0, mmd2) ** 0.5

    def _wasserstein_1d(self, ref: np.ndarray, cur: np.ndarray) -> float:
        """Mean Wasserstein-1D across features."""
        if ref.ndim == 1:
            ref = ref.reshape(-1, 1)
        if cur.ndim == 1:
            cur = cur.reshape(-1, 1)

        n_feat = min(ref.shape[1], cur.shape[1])
        w_vals = []
        for f in range(n_feat):
            r = np.sort(ref[:, f])
            c = np.sort(cur[:, f])
            # Interpolate to common grid
            n = max(len(r), len(c))
            r_interp = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(r)), r)
            c_interp = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(c)), c)
            w_vals.append(float(np.mean(np.abs(r_interp - c_interp))))

        return float(np.mean(w_vals)) if w_vals else 0.0

    def recent_drift_rate(self, window: int = 20) -> float:
        """Fraction of recent updates that detected drift."""
        recent = self._history[-window:]
        if not recent:
            return 0.0
        return sum(1 for r in recent if r["drift_detected"]) / len(recent)

    def reset_reference(self):
        """Promote current window to new reference."""
        if len(self._current_window) > 0:
            self._reference = np.array(list(self._current_window), dtype=np.float32)
            self._current_window.clear()
