"""Baseline models for VULGARIS benchmarking."""
import numpy as np
from typing import Dict, List, Optional, Tuple


class LastValue:
    """Predict last observed value (persistence baseline)."""
    name = "LastValue"

    def fit(self, X: np.ndarray, y: np.ndarray): pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        # X: (B, C, T) → return last timestep mean across channels
        return X[:, :, -1].mean(axis=1, keepdims=True).repeat(
            max(1, y_dim := 1), axis=1
        ) if X.ndim == 3 else X[:, -1:]

    def predict_flat(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 3:
            return X[:, 0, -1]  # first channel, last timestep
        return X[:, -1]


class MovingAverage:
    """Moving average over last k timesteps."""
    name = "MovingAverage"

    def __init__(self, k: int = 5):
        self.k = k

    def fit(self, X: np.ndarray, y: np.ndarray): pass

    def predict_flat(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 3:
            return X[:, 0, -self.k:].mean(axis=-1)
        return X[:, -self.k:].mean(axis=-1)


class ExponentialSmoothing:
    """Single exponential smoothing."""
    name = "ExponentialSmoothing"

    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha

    def fit(self, X: np.ndarray, y: np.ndarray): pass

    def predict_flat(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 3:
            series = X[:, 0, :]  # (B, T)
        else:
            series = X
        B, T = series.shape
        result = np.zeros(B, dtype=np.float32)
        for b in range(B):
            s = series[b, 0]
            for t in range(1, T):
                s = self.alpha * series[b, t] + (1 - self.alpha) * s
            result[b] = s
        return result


class ARIMA_lite:
    """AR(p) model fit via least squares (no differencing, no MA)."""
    name = "ARIMA_lite"

    def __init__(self, p: int = 5):
        self.p = p
        self._coefs: Optional[np.ndarray] = None

    def _build_matrix(self, series: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Build Toeplitz-style regression matrix from 1D series."""
        T = len(series)
        if T <= self.p:
            return np.zeros((1, self.p), dtype=np.float32), series[-1:]
        X_rows = []
        y_rows = []
        for t in range(self.p, T):
            X_rows.append(series[t - self.p:t][::-1])
            y_rows.append(series[t])
        return np.array(X_rows, dtype=np.float32), np.array(y_rows, dtype=np.float32)

    def fit(self, X: np.ndarray, y: np.ndarray):
        if X.ndim == 3:
            series = X[:, 0, :].mean(axis=0)  # average over batch, use first channel
        elif X.ndim == 2:
            series = X.mean(axis=0)
        else:
            series = X
        X_reg, y_reg = self._build_matrix(series)
        if len(X_reg) > self.p:
            # Ridge least squares: (X^T X + λI)^{-1} X^T y
            lam = 1e-3
            A = X_reg.T @ X_reg + lam * np.eye(self.p, dtype=np.float32)
            b = X_reg.T @ y_reg
            try:
                self._coefs = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                self._coefs = np.zeros(self.p, dtype=np.float32)
        else:
            self._coefs = np.zeros(self.p, dtype=np.float32)

    def predict_flat(self, X: np.ndarray) -> np.ndarray:
        if self._coefs is None:
            if X.ndim == 3:
                return X[:, 0, -1]
            return X[:, -1]
        if X.ndim == 3:
            series = X[:, 0, :]  # (B, T)
        else:
            series = X
        B, T = series.shape
        result = np.zeros(B, dtype=np.float32)
        p = min(self.p, T)
        for b in range(B):
            window = series[b, -p:][::-1]
            c = self._coefs[:p]
            result[b] = float(window @ c)
        return result


class LSTMLite:
    """Single-layer LSTM implemented in pure numpy."""
    name = "LSTMLite"

    def __init__(self, input_size: int = 1, hidden_size: int = 16, seed: int = 42):
        rng = np.random.default_rng(seed)
        scale = 0.1
        # Gates: [i, f, g, o] — each (hidden, input) and (hidden, hidden)
        self.Wx = rng.normal(0, scale, (4 * hidden_size, input_size)).astype(np.float32)
        self.Wh = rng.normal(0, scale, (4 * hidden_size, hidden_size)).astype(np.float32)
        self.b = np.zeros(4 * hidden_size, dtype=np.float32)
        self.hidden_size = hidden_size
        self.input_size = input_size

    def fit(self, X: np.ndarray, y: np.ndarray): pass  # No training — random init baseline

    def _step(self, x_t: np.ndarray, h: np.ndarray, c: np.ndarray):
        gates = self.Wx @ x_t + self.Wh @ h + self.b
        H = self.hidden_size
        i = 1 / (1 + np.exp(-gates[:H]))
        f = 1 / (1 + np.exp(-gates[H:2*H]))
        g = np.tanh(gates[2*H:3*H])
        o = 1 / (1 + np.exp(-gates[3*H:]))
        c_new = f * c + i * g
        h_new = o * np.tanh(c_new)
        return h_new, c_new

    def predict_flat(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 3:
            series = X[:, :self.input_size, :].transpose(0, 2, 1)  # (B, T, input_size)
        elif X.ndim == 2:
            series = X[:, :, None]  # (B, T, 1)
        else:
            series = X[:, :, None]
        B, T, _ = series.shape
        result = np.zeros(B, dtype=np.float32)
        for b in range(B):
            h = np.zeros(self.hidden_size, dtype=np.float32)
            c = np.zeros(self.hidden_size, dtype=np.float32)
            for t in range(T):
                h, c = self._step(series[b, t], h, c)
            # Linear readout: mean of hidden state
            result[b] = float(h.mean())
        return result


def _mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def run_baseline_comparison(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    vulgaris_preds: Optional[np.ndarray] = None,
) -> Dict[str, dict]:
    """
    Fit and evaluate all baselines.
    X: (B, C, T), y: (B,) scalar targets
    Returns dict of {model_name: {"mse": ..., "mae": ...}}
    """
    baselines = [
        LastValue(),
        MovingAverage(k=5),
        ExponentialSmoothing(alpha=0.3),
        ARIMA_lite(p=5),
        LSTMLite(input_size=min(X_train.shape[1], 4) if X_train.ndim == 3 else 1),
    ]

    results: Dict[str, dict] = {}
    y_test_flat = y_test.flatten()

    for model in baselines:
        try:
            model.fit(X_train, y_train)
            preds = model.predict_flat(X_test)
            preds_flat = preds.flatten()[:len(y_test_flat)]
            results[model.name] = {
                "mse": _mse(y_test_flat, preds_flat),
                "mae": _mae(y_test_flat, preds_flat),
            }
        except Exception as e:
            results[model.name] = {"mse": float("nan"), "mae": float("nan"), "error": str(e)}

    if vulgaris_preds is not None:
        vp = vulgaris_preds.flatten()[:len(y_test_flat)]
        results["VULGARIS"] = {
            "mse": _mse(y_test_flat, vp),
            "mae": _mae(y_test_flat, vp),
        }

    return results
