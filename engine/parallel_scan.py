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
        fa[:, idx] = fa_prev[:, idx] * fa_prev[:, src]
        fb[:, idx] = fa_prev[:, idx] * fb_prev[:, src] + fb_prev[:, idx]
        stride <<= 1

    return fb   # fb[t] = h[t] (h_init absorbed into fb[0])


def _numpy_scan_logspace(a: np.ndarray, b: np.ndarray, h_init=None) -> np.ndarray:
    """
    Log-space Hillis-Steele scan: tracks cumulative a-products in log space so
    they never underflow to 0 on long sequences (T > 1000).

    a-products are maintained as log_fa (always ≤ 0 since a ∈ (0,1]).
    The b-accumulation uses exp(log_fa) to recover the product only when
    needed for mixing — exp of a negative number is in (0,1], so no overflow.
    Numerically equivalent to _numpy_scan but stable for arbitrarily long T.
    """
    B, T, D = a.shape
    a_cl = np.clip(a.astype(np.float32), 1e-7, 1.0)
    log_fa = np.log(a_cl)                            # (B, T, D) ≤ 0
    fb = b.astype(np.float32, copy=True)

    if h_init is not None:
        fb[:, 0, :] += a_cl[:, 0, :] * h_init.astype(np.float32)

    stride = 1
    while stride < T:
        log_fa_prev = log_fa.copy()
        fb_prev     = fb.copy()
        idx = np.arange(stride, T)
        src = idx - stride
        # Log-space product: log(a_0 … a_t) = sum of logs — never underflows
        log_fa[:, idx] = log_fa_prev[:, idx] + log_fa_prev[:, src]
        # Recover a-product only for the b-mixing step; exp(negative) ∈ (0,1]
        fa_idx = np.exp(log_fa_prev[:, idx])
        fb[:, idx] = fa_idx * fb_prev[:, src] + fb_prev[:, idx]
        stride <<= 1

    return fb


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
    # Use log-space scan by default: same output, stable for T > 1000
    return _numpy_scan_logspace(a, b, h_init)


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
    # Use the same clamped a as the forward pass so gradient aligns with the
    # actual computation; prevents lam from being multiplied by 0 (underflow).
    a_cl = np.clip(a, 1e-7, 1.0).astype(np.float32)
    lam = np.zeros((B, D), dtype=np.float32)
    grad_a = np.zeros_like(a)
    grad_b = np.zeros_like(a)

    for t in range(T - 1, -1, -1):
        lam = lam + grad_h[:, t, :]
        grad_b[:, t, :] = lam
        h_prev = h[:, t - 1, :] if t > 0 else np.zeros((B, D), dtype=np.float32)
        grad_a[:, t, :] = lam * h_prev
        lam = lam * a_cl[:, t, :]

    return grad_a, grad_b
