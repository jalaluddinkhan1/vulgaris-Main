"""
Parallel prefix scan for SSM linear recurrences.

Recurrence: h[t] = a[t] * h[t-1] + b[t]

Implementations:
  numpy  — Hillis-Steele O(T log T) work, vectorised over B and D
  numba  — same algorithm, JIT-compiled with SIMD
  triton — GPU kernel, parallelised over B*D with sequential T per thread

All share the same interface:
    h = parallel_scan_ssm(a, b, h_init=None)
    a, b  : (B, T, D)  float32 numpy arrays
    h_init: (B, D)     float32 numpy array or None
    returns h: (B, T, D) float32 numpy array
"""
import numpy as np
from engine.backend import get_backend


# ---------------------------------------------------------------------------
# Numpy — Hillis-Steele inclusive prefix scan
# ---------------------------------------------------------------------------

def _numpy_scan(a: np.ndarray, b: np.ndarray, h_init=None) -> np.ndarray:
    B, T, D = a.shape
    fa = a.astype(np.float32, copy=True)
    fb = b.astype(np.float32, copy=True)

    if h_init is not None:
        fb[:, 0, :] += fa[:, 0, :] * h_init.astype(np.float32)

    stride = 1
    while stride < T:
        fa_prev = fa.copy()
        fb_prev = fb.copy()
        idx = np.arange(stride, T)
        src = idx - stride
        # Composition: (fa[t], fb[t]) after stride steps starting from src
        fa[:, idx] = fa_prev[:, idx] * fa_prev[:, src]
        fb[:, idx] = fa_prev[:, idx] * fb_prev[:, src] + fb_prev[:, idx]
        stride <<= 1

    return fb   # fb[t] = h[t] (h_init absorbed into fb[0])


# ---------------------------------------------------------------------------
# Numba — JIT-compiled version of the same algorithm
# ---------------------------------------------------------------------------

def _numba_scan(a: np.ndarray, b: np.ndarray, h_init=None) -> np.ndarray:
    try:
        from engine.backends.numba_ops import ssm_scan_numba
        return ssm_scan_numba(a, b, h_init)
    except Exception:
        return _numpy_scan(a, b, h_init)


# ---------------------------------------------------------------------------
# Triton — GPU kernel dispatch
# ---------------------------------------------------------------------------

def _triton_scan(a: np.ndarray, b: np.ndarray, h_init=None) -> np.ndarray:
    try:
        from engine.backends.triton_ops.ssm_scan import ssm_scan_triton
        return ssm_scan_triton(a, b, h_init)
    except Exception:
        return _numpy_scan(a, b, h_init)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parallel_scan_ssm(
    a: np.ndarray,
    b: np.ndarray,
    h_init: np.ndarray = None,
) -> np.ndarray:
    """
    Compute all hidden states of h[t] = a[t]*h[t-1] + b[t] in parallel.

    Parameters
    ----------
    a      : (B, T, D) float32 — per-step decay factors
    b      : (B, T, D) float32 — per-step inputs
    h_init : (B, D)    float32 — initial state h[-1], default zeros

    Returns
    -------
    h : (B, T, D) float32 — all hidden states
    """
    backend = get_backend()
    if backend == "triton":
        return _triton_scan(a, b, h_init)
    if backend == "numba":
        return _numba_scan(a, b, h_init)
    return _numpy_scan(a, b, h_init)


def parallel_scan_ssm_backward(
    a: np.ndarray,
    h: np.ndarray,
    grad_h: np.ndarray,
) -> tuple:
    """
    Adjoint of parallel_scan_ssm via reverse-time scan.

    Given forward: h[t] = a[t]*h[t-1] + b[t]
    Backward:
      grad_b[t] = lambda[t]
      grad_a[t] = lambda[t] * h[t-1]
      lambda[t-1] += a[t] * lambda[t]

    Parameters
    ----------
    a      : (B, T, D) — forward decay factors
    h      : (B, T, D) — forward hidden states (h[t])
    grad_h : (B, T, D) — upstream gradient dL/dh[t]

    Returns
    -------
    grad_a : (B, T, D)
    grad_b : (B, T, D)
    """
    B, T, D = a.shape
    lam = np.zeros((B, D), dtype=np.float32)  # lambda accumulator
    grad_a = np.zeros_like(a)
    grad_b = np.zeros_like(a)

    for t in range(T - 1, -1, -1):
        lam = lam + grad_h[:, t, :]           # accumulate upstream
        grad_b[:, t, :] = lam
        h_prev = h[:, t - 1, :] if t > 0 else np.zeros((B, D), dtype=np.float32)
        grad_a[:, t, :] = lam * h_prev
        lam = lam * a[:, t, :]               # propagate backwards

    return grad_a, grad_b
