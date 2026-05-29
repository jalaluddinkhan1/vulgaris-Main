import numpy as np
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

from engine.tensor import Tensor, zeros
from model.vulgaris import Vulgaris, VulgarisState


class RunningNormalizer:
    """Online Welford algorithm for running mean/variance."""

    def __init__(self, dim: int, momentum: float = 0.01):
        self.dim = dim
        self.momentum = momentum
        self.mean = np.zeros(dim, dtype=np.float64)
        self.var = np.ones(dim, dtype=np.float64)
        self.count = 0

    def update(self, x: np.ndarray):
        # x: (..., dim) — flatten all leading dims
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        flat = x.reshape(-1, self.dim)
        for sample in flat:
            self.count += 1
            delta = sample - self.mean
            self.mean += delta / self.count
            delta2 = sample - self.mean
            # Welford M2 update via momentum for stability in streaming
            self.var = (1.0 - self.momentum) * self.var + self.momentum * delta * delta2
        self.var = np.maximum(self.var, 1e-8)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)

    def denormalize(self, x: np.ndarray) -> np.ndarray:
        return x * (np.sqrt(self.var) + 1e-8) + self.mean


class QuantizedWeights:
    """INT8 weight quantization for edge deployment."""

    def __init__(self, scale: float, zero_point: int, data: np.ndarray):
        self.scale = scale
        self.zero_point = zero_point
        self.data = data  # int8 array

    @staticmethod
    def quantize(weights: np.ndarray, bits: int = 8) -> 'QuantizedWeights':
        # Symmetric per-tensor quantization
        max_val = np.max(np.abs(weights))
        max_val = max(max_val, 1e-8)
        max_int = (1 << (bits - 1)) - 1  # 127 for int8
        scale = max_val / max_int
        q = np.round(weights / scale).clip(-max_int - 1, max_int).astype(np.int8)
        return QuantizedWeights(scale=scale, zero_point=0, data=q)

    def dequantize(self) -> np.ndarray:
        return self.data.astype(np.float32) * self.scale


class _PoisoningDetector:
    """
    Online Z-score sensor for telemetry poisoning.

    Maintains per-channel running mean/std (EMA).  Any sample with
    |z-score| > z_thresh is clamped and flagged.
    """

    def __init__(self, n_channels: int, z_thresh: float = 6.0,
                 momentum: float = 0.05):
        self.z_thresh = z_thresh
        self.momentum = momentum
        self._mean = np.zeros(n_channels, dtype=np.float64)
        self._std  = np.ones(n_channels, dtype=np.float64)
        self._n    = 0

    def check_and_sanitize(self, x: np.ndarray) -> Tuple[np.ndarray, bool]:
        """x: (batch, channels). Returns (sanitized_x, poisoning_flag)."""
        if self._n < 30:
            self._update(x)
            return x.copy(), False

        z = (x - self._mean) / (self._std + 1e-8)
        flagged = bool(np.any(np.abs(z) > self.z_thresh))
        x_clean = np.clip(x,
                          self._mean - self.z_thresh * self._std,
                          self._mean + self.z_thresh * self._std)
        self._update(x_clean)
        return x_clean, flagged

    def _update(self, x: np.ndarray):
        a = self.momentum
        self._mean = (1 - a) * self._mean + a * x.mean(axis=0)
        self._std  = (1 - a) * self._std  + a * (x.std(axis=0) + 1e-8)
        self._n   += x.shape[0]


class StreamingInference:
    def __init__(self, model: Vulgaris, batch_size: int = 1,
                 normalize_input: bool = True, mode: str = 'predictive',
                 z_thresh: float = 6.0,
                 delta_threshold: Optional[float] = None):
        self.model = model
        self.model.training = False
        self.batch_size = batch_size
        self.normalize_input = normalize_input
        self.mode = mode

        in_channels = model.config.input_dim
        self.normalizer = RunningNormalizer(dim=in_channels)
        self._poisoning_detector = _PoisoningDetector(in_channels, z_thresh=z_thresh)
        self.state: VulgarisState = model.init_state(batch_size)
        self.latency_buffer: deque = deque(maxlen=100)
        self.step_count: int = 0
        self._quantized_weights: Dict[str, QuantizedWeights] = {}
        self._record_ese: bool = False
        self._use_safety: bool = False
        self._last_timestamp: Optional[float] = None
        self._ooo_count: int = 0
        self._poison_count: int = 0
        # Delta-threshold event gating: skip forward pass when sensors are static
        self._delta_threshold: Optional[float] = delta_threshold
        self._last_x: Optional[np.ndarray] = None
        self._last_pred: Optional[np.ndarray] = None
        self._last_uncertainty: float = 0.0
        self._skipped_count: int = 0

    def step(self, x_t: np.ndarray, domain_idx: int = 0,
             timestamp: Optional[float] = None) -> dict:
        """
        x_t       : (batch, in_channels) single timestep
        timestamp : monotonic sensor timestamp (seconds).  If provided and
                    smaller than the last seen timestamp, the step is still
                    processed (event-driven systems can deliver out-of-order)
                    but `ooo_event=True` is set in the returned dict.
        """
        t0 = time.perf_counter()

        # --- Malformed packet guard ---
        x = np.asarray(x_t, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]

        expected = self.model.config.input_dim
        if x.shape[-1] != expected:
            raise ValueError(
                f"Input has {x.shape[-1]} channels; model expects {expected}"
            )

        # NaN / Inf → replace with running mean (safe fallback)
        bad = ~np.isfinite(x)
        if bad.any():
            x = np.where(bad, self.normalizer.mean, x)

        # --- Delta-threshold event gating ---
        if (self._delta_threshold is not None
                and self._last_x is not None
                and self._last_pred is not None
                and np.abs(x - self._last_x).max() < self._delta_threshold):
            self._skipped_count += 1
            latency_ms = (time.perf_counter() - t0) * 1000.0
            self.latency_buffer.append(latency_ms)
            return {
                'prediction': self._last_pred.copy(),
                'uncertainty': self._last_uncertainty,
                'latency_ms': latency_ms,
                'step': self.step_count,
                'ooo_event': False,
                'poisoning_suspected': False,
                'skipped': True,
            }

        # --- Out-of-order timestamp detection ---
        ooo = False
        if timestamp is not None:
            if self._last_timestamp is not None and timestamp < self._last_timestamp:
                self._ooo_count += 1
                ooo = True
            # Always advance to the maximum seen so gaps don't permanently block
            self._last_timestamp = max(
                timestamp,
                self._last_timestamp if self._last_timestamp is not None else timestamp,
            )

        # --- Telemetry poisoning guard ---
        x, poisoned = self._poisoning_detector.check_and_sanitize(x)
        if poisoned:
            self._poison_count += 1

        # Cache pre-normalization value for delta-threshold gating
        self._last_x = x.copy()

        if self.normalize_input:
            self.normalizer.update(x)
            x = self.normalizer.normalize(x)

        x_tensor = Tensor(x, requires_grad=False)
        output_t, new_state = self.model.step(x_tensor, self.state, domain_idx=domain_idx)
        self.state = new_state
        self.step_count += 1

        pred = output_t.data.copy()  # (batch, output_dim)

        # Uncertainty: variance across output dimensions as a simple proxy
        uncertainty = float(np.var(pred))
        self._last_pred = pred
        self._last_uncertainty = uncertainty

        latency_ms = (time.perf_counter() - t0) * 1000.0
        self.latency_buffer.append(latency_ms)

        return {
            'prediction': pred,
            'uncertainty': uncertainty,
            'latency_ms': latency_ms,
            'step': self.step_count,
            'ooo_event': ooo,
            'poisoning_suspected': poisoned,
            'skipped': False,
        }

    def process_window(self, x_window: np.ndarray, domain_idx: int = 0) -> dict:
        # x_window: (batch, in_channels, T)
        t0 = time.perf_counter()
        x = np.asarray(x_window, dtype=np.float64)
        if x.ndim == 2:
            x = x[None, :, :]  # (1, in_channels, T)

        T = x.shape[2]
        predictions = []
        for t in range(T):
            x_t = x[:, :, t]  # (batch, in_channels)
            result = self.step(x_t, domain_idx=domain_idx)
            predictions.append(result['prediction'])

        total_latency_ms = (time.perf_counter() - t0) * 1000.0
        preds = np.stack(predictions, axis=-1)  # (batch, output_dim, T)

        return {
            'predictions': preds,
            'final_prediction': predictions[-1],
            'uncertainty': float(np.var(preds)),
            'total_latency_ms': total_latency_ms,
            'steps': T,
        }

    def reset_state(self):
        self.state = self.model.init_state(self.batch_size)
        self.step_count = 0

    def switch_mode(self, mode: str):
        self.mode = mode
        if mode == 'diagnostic':
            self._record_ese = True
            self.model.training = True  # enable ESE recording
        elif mode == 'prescriptive':
            self._use_safety = True
            self._record_ese = False
            self.model.training = False
        else:  # 'predictive'
            self._record_ese = False
            self._use_safety = False
            self.model.training = False

    def set_domain(self, domain_idx: int):
        self.model.set_domain(domain_idx)

    def get_latency_stats(self) -> dict:
        if not self.latency_buffer:
            return {'mean_ms': 0.0, 'p50_ms': 0.0, 'p95_ms': 0.0, 'p99_ms': 0.0, 'max_ms': 0.0}
        arr = np.array(self.latency_buffer)
        return {
            'mean_ms': float(np.mean(arr)),
            'p50_ms': float(np.percentile(arr, 50)),
            'p95_ms': float(np.percentile(arr, 95)),
            'p99_ms': float(np.percentile(arr, 99)),
            'max_ms': float(np.max(arr)),
        }

    def get_health_stats(self) -> dict:
        """Return security and data-quality counters alongside latency stats."""
        return {
            **self.get_latency_stats(),
            'ooo_events_total': self._ooo_count,
            'poisoning_events_total': self._poison_count,
            'steps_processed': self.step_count,
            'steps_skipped': self._skipped_count,
        }

    def quantize_model(self, bits: int = 8):
        # Quantize all Linear weight matrices in the model
        from engine.layers import Linear as EngineLinear
        for name, module in self.model._modules.items():
            self._quantize_module_recursive(name, module, bits)

    def _quantize_module_recursive(self, prefix: str, module, bits: int):
        from engine.layers import Linear as EngineLinear
        # Check if this module has a weight attribute that is a Parameter
        if hasattr(module, 'weight') and hasattr(module.weight, 'data'):
            key = prefix + '.weight'
            w = module.weight.data.astype(np.float32)
            qw = QuantizedWeights.quantize(w, bits=bits)
            self._quantized_weights[key] = qw
            # Replace weight data with dequantized version for inference
            module.weight.data = qw.dequantize().astype(np.float64)
        # Recurse into submodules
        if hasattr(module, '_modules'):
            for name, submod in module._modules.items():
                if submod is not None:
                    self._quantize_module_recursive(prefix + '.' + name, submod, bits)
