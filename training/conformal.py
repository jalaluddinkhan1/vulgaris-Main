import numpy as np
from collections import deque
from typing import Optional, Tuple


class NonStationaryConformal:
    """
    Non-stationary conformal predictor using exponential forgetting (EnbPI-style).

    Calibration scores: s_t = |y_t - ŷ_t| / σ_t
    Weights: w_t = exp(-λ*(T-t)), exponentially decaying for older scores.
    Weighted quantile at level α.
    Prediction interval: [ŷ - q̂*σ, ŷ + q̂*σ]
    """

    def __init__(
        self,
        alpha: float = 0.05,
        forgetting_factor: float = 0.01,
        max_calibration_size: int = 2000,
    ):
        self.alpha = alpha
        self.forgetting_factor = forgetting_factor
        self.max_calibration_size = max_calibration_size

        # Deque of (score, weight) tuples; weight is renewed on each update
        self._scores: deque = deque(maxlen=max_calibration_size)
        # Track coverage: bool per prediction
        self._coverage_history: deque = deque(maxlen=500)
        # Cache the weighted quantile; invalidated when scores change
        self._quantile_cache: Optional[float] = None
        self._quantile_dirty: bool = True

    # ──────────────────────────────────────────────────────────────────────

    def update(self, y_pred: np.ndarray, y_true: np.ndarray, sigma: np.ndarray):
        """
        Add new calibration observation(s) and decay existing weights.

        y_pred, y_true, sigma: 1-D arrays of the same length (or scalars).
        """
        y_pred = np.atleast_1d(np.asarray(y_pred, dtype=np.float64)).flatten()
        y_true = np.atleast_1d(np.asarray(y_true, dtype=np.float64)).flatten()
        sigma  = np.atleast_1d(np.asarray(sigma,  dtype=np.float64)).flatten()

        # Decay all existing weights by exp(-λ) per new data point added
        decay = np.exp(-self.forgetting_factor)

        # Compute new scores s = |y - ŷ| / (σ + ε)
        n_new = len(y_pred)
        new_scores = np.abs(y_true - y_pred) / (sigma + 1e-8)

        # Decay existing weights
        updated = []
        for s, w in self._scores:
            updated.append((s, w * (decay ** n_new)))
        self._scores.clear()
        self._scores.extend(updated)

        # Append new scores with weight 1.0
        for s_val in new_scores:
            self._scores.append((float(s_val), 1.0))

        self._quantile_dirty = True

    # ──────────────────────────────────────────────────────────────────────

    def predict_interval(
        self, y_pred: np.ndarray, sigma: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (lower, upper) prediction interval arrays.
        lower = y_pred - q̂ * sigma
        upper = y_pred + q̂ * sigma
        """
        y_pred = np.asarray(y_pred, dtype=np.float64)
        sigma  = np.asarray(sigma,  dtype=np.float64)
        q = self.current_quantile()
        lower = y_pred - q * sigma
        upper = y_pred + q * sigma
        return lower, upper

    # ──────────────────────────────────────────────────────────────────────

    def coverage_loss(
        self, y_pred: np.ndarray, y_true: np.ndarray, sigma: np.ndarray
    ) -> float:
        """
        Scalar coverage gap: relu(target_coverage - actual_coverage).
        target_coverage = 1 - alpha.
        """
        y_pred = np.atleast_1d(np.asarray(y_pred, dtype=np.float64))
        y_true = np.atleast_1d(np.asarray(y_true, dtype=np.float64))
        sigma  = np.atleast_1d(np.asarray(sigma,  dtype=np.float64))

        q = self.current_quantile()
        lower = y_pred - q * sigma
        upper = y_pred + q * sigma

        covered = np.mean((y_true >= lower) & (y_true <= upper))
        # Update coverage history
        for c in ((y_true >= lower) & (y_true <= upper)):
            self._coverage_history.append(bool(c))

        target = 1.0 - self.alpha
        gap = max(0.0, target - float(covered))
        return gap

    # ──────────────────────────────────────────────────────────────────────

    def current_quantile(self) -> float:
        """
        Weighted quantile at level (1 - alpha).

        q̂ = inf{q : Σ_t w_t * 1[s_t ≤ q] / Σ w_t ≥ 1-α}
        """
        if not self._quantile_dirty and self._quantile_cache is not None:
            return self._quantile_cache

        if len(self._scores) == 0:
            self._quantile_cache = 1.0
            self._quantile_dirty = False
            return 1.0

        scores_arr = np.array([s for s, _ in self._scores], dtype=np.float64)
        weights_arr = np.array([w for _, w in self._scores], dtype=np.float64)

        # Normalise weights
        w_sum = weights_arr.sum()
        if w_sum < 1e-12:
            self._quantile_cache = float(np.max(scores_arr))
            self._quantile_dirty = False
            return self._quantile_cache

        w_norm = weights_arr / w_sum

        # Sort by score ascending
        order = np.argsort(scores_arr)
        scores_sorted = scores_arr[order]
        w_sorted = w_norm[order]

        # Cumulative weighted fraction
        cum_w = np.cumsum(w_sorted)

        target = 1.0 - self.alpha
        # Find smallest score s.t. cum_w >= target
        idx = np.searchsorted(cum_w, target)
        if idx >= len(scores_sorted):
            idx = len(scores_sorted) - 1

        self._quantile_cache = float(scores_sorted[idx])
        self._quantile_dirty = False
        return self._quantile_cache

    # ──────────────────────────────────────────────────────────────────────

    def current_coverage(self) -> float:
        """Fraction of recent predictions that contained the true y."""
        if len(self._coverage_history) == 0:
            return 1.0
        return float(np.mean(list(self._coverage_history)))

    # ──────────────────────────────────────────────────────────────────────

    def is_calibrated(self) -> bool:
        """
        True if enough calibration data exists and empirical coverage is
        within 2 percentage points of the target (1 - alpha).
        """
        if len(self._scores) < 50:
            return False
        cov = self.current_coverage()
        target = 1.0 - self.alpha
        return abs(cov - target) <= 0.02
