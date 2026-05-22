"""
Mathematical operations with correct gradients for the VULGARIS engine.
All functions return Tensor objects with _backward hooks for reverse-mode autodiff.
"""

import numpy as np
from typing import Tuple, Optional
from scipy.linalg import expm
from .tensor import Tensor, Parameter


# ---------------------------------------------------------------------------
# 1. conv1d_forward
# ---------------------------------------------------------------------------

def conv1d_forward(x: Tensor, weight: Tensor, bias: Optional[Tensor],
                   stride: int, padding: int, groups: int) -> Tensor:
    """
    1D convolution using im2col.
    x     : (B, C_in, L)
    weight: (C_out, C_in//groups, K)
    bias  : (C_out,) or None
    Returns (B, C_out, L_out) where L_out = (L + 2*padding - K) // stride + 1
    """
    B, C_in, L = x.data.shape
    C_out, C_in_g, K = weight.data.shape
    assert C_in_g == C_in // groups

    L_out = (L + 2 * padding - K) // stride + 1

    # Pad input
    if padding > 0:
        x_pad = np.pad(x.data, ((0, 0), (0, 0), (padding, padding)), mode="constant")
    else:
        x_pad = x.data

    # im2col: (B, C_in, L_out, K) -> (B, C_in*K, L_out)
    col = np.zeros((B, C_in, K, L_out), dtype=np.float64)
    for k in range(K):
        col[:, :, k, :] = x_pad[:, :, k: k + stride * L_out: stride]
    # col shape: (B, C_in, K, L_out) -> (B, C_in*K, L_out)
    col = col.reshape(B, C_in * K, L_out)

    # weight reshape: (C_out, C_in*K) per group, we handle groups
    out_data = np.zeros((B, C_out, L_out), dtype=np.float64)
    group_size_in = C_in // groups
    group_size_out = C_out // groups

    for g in range(groups):
        w_g = weight.data[g * group_size_out: (g + 1) * group_size_out]  # (C_out//g, C_in//g, K)
        w_g_flat = w_g.reshape(group_size_out, group_size_in * K)         # (C_out//g, C_in_g*K)
        col_g = col[:, g * group_size_in * K: (g + 1) * group_size_in * K, :]  # (B, C_in_g*K, L_out)
        # (B, C_out//g, L_out) = w_g_flat @ col_g
        out_data[:, g * group_size_out: (g + 1) * group_size_out, :] = np.einsum("oi,bil->bol", w_g_flat, col_g)

    if bias is not None:
        out_data += bias.data[None, :, None]

    children = [x, weight] + ([bias] if bias is not None else [])
    rg = any(c.requires_grad for c in children)
    out = Tensor(out_data, requires_grad=rg, _children=tuple(children), _op="conv1d")

    def _back():
        g_out = out.grad if out.grad is not None else np.ones(out_data.shape)  # (B, C_out, L_out)

        if bias is not None and bias.requires_grad:
            gb = g_out.sum(axis=(0, 2))
            bias.grad = bias.grad + gb if bias.grad is not None else gb

        if weight.requires_grad:
            gw = np.zeros_like(weight.data)
            for gg in range(groups):
                g_out_g = g_out[:, gg * group_size_out: (gg + 1) * group_size_out, :]  # (B, C_out//g, L_out)
                col_g = col[:, gg * group_size_in * K: (gg + 1) * group_size_in * K, :]  # (B, C_in_g*K, L_out)
                # gw_g: (C_out//g, C_in_g*K)
                gw_g = np.einsum("bol,bil->oi", g_out_g, col_g)
                gw[gg * group_size_out: (gg + 1) * group_size_out] = gw_g.reshape(
                    group_size_out, group_size_in, K)
            weight.grad = weight.grad + gw if weight.grad is not None else gw

        if x.requires_grad:
            gx_pad = np.zeros((B, C_in, L + 2 * padding), dtype=np.float64)
            for gg in range(groups):
                g_out_g = g_out[:, gg * group_size_out: (gg + 1) * group_size_out, :]  # (B, C_out//g, L_out)
                w_g = weight.data[gg * group_size_out: (gg + 1) * group_size_out]
                w_g_flat = w_g.reshape(group_size_out, group_size_in * K)
                # col_grad_g: (B, C_in_g*K, L_out)
                col_grad_g = np.einsum("oi,bol->bil", w_g_flat, g_out_g)
                col_grad_g = col_grad_g.reshape(B, group_size_in, K, L_out)
                for k in range(K):
                    start = k
                    end = k + stride * L_out
                    gx_pad[:, gg * group_size_in: (gg + 1) * group_size_in, start:end:stride] += col_grad_g[:, :, k, :]
            # Remove padding
            gx = gx_pad[:, :, padding: padding + L] if padding > 0 else gx_pad
            x.grad = x.grad + gx if x.grad is not None else gx

    out._backward = _back
    return out


# ---------------------------------------------------------------------------
# 2. ssm_parallel_scan  (associative scan on pairs)
# ---------------------------------------------------------------------------

def ssm_parallel_scan(A_diag: Tensor, B_vec: Tensor, C_vec: Tensor,
                      x: Tensor = None, delta: Tensor = None) -> Tuple[Tensor, Tensor]:
    """
    Associative parallel scan for SSM.
    A_diag: (B, L, N) — diagonal transition (already discretized, i.e. exp(-exp(A_log)*dt))
    B_vec : (B, L, N) — input contribution (already = B*u for the step)
    C_vec : (B, L, N) — output projection per step

    Each element is (a_t, b_t) in the monoid (A, B), with op:
        (a1, b1) ⊕ (a2, b2) = (a2*a1, a2*b1 + b2)

    h_t = A_t * h_{t-1} + B_t,  y_t = (C_t * h_t).sum(-1)
    Returns (y, h_last): y is (B, L), h_last is (B, N).
    """
    a_np = A_diag.data  # (B, L, N)
    b_np = B_vec.data   # (B, L, N)
    c_np = C_vec.data   # (B, L, N)
    B_sz, L, N = a_np.shape

    # Forward scan — store all intermediate states for backward
    # h[0] = B_vec[0], h[t] = A[t]*h[t-1] + B[t]
    h = np.zeros((B_sz, L, N), dtype=np.float64)
    h[:, 0, :] = b_np[:, 0, :]
    for t in range(1, L):
        h[:, t, :] = a_np[:, t, :] * h[:, t - 1, :] + b_np[:, t, :]

    y_np = (c_np * h).sum(axis=-1)  # (B, L)
    h_last_np = h[:, -1, :]          # (B, N)

    rg = A_diag.requires_grad or B_vec.requires_grad or C_vec.requires_grad
    y_out = Tensor(y_np, requires_grad=rg, _children=(A_diag, B_vec, C_vec), _op="ssm_scan_y")
    h_out = Tensor(h_last_np, requires_grad=rg, _children=(A_diag, B_vec, C_vec), _op="ssm_scan_h")

    def _back():
        dy = y_out.grad if y_out.grad is not None else np.zeros(y_np.shape)   # (B, L)
        dh_last = h_out.grad if h_out.grad is not None else np.zeros(h_last_np.shape)  # (B, N)

        dC = dy[:, :, None] * h          # (B, L, N)
        dh = np.zeros_like(h)            # (B, L, N)
        dh[:, :, :] = dy[:, :, None] * c_np  # contribution from y
        dh[:, -1, :] += dh_last          # contribution from h_last

        dA = np.zeros_like(a_np)
        dB = np.zeros_like(b_np)

        # Reverse scan
        for t in range(L - 1, -1, -1):
            # dh[t] is current gradient wrt h[t]
            dB[:, t, :] = dh[:, t, :]
            dA[:, t, :] = dh[:, t, :] * (h[:, t - 1, :] if t > 0 else np.zeros((B_sz, N)))
            if t > 0:
                dh[:, t - 1, :] += dh[:, t, :] * a_np[:, t, :]

        if A_diag.requires_grad:
            A_diag.grad = A_diag.grad + dA if A_diag.grad is not None else dA
        if B_vec.requires_grad:
            B_vec.grad = B_vec.grad + dB if B_vec.grad is not None else dB
        if C_vec.requires_grad:
            C_vec.grad = C_vec.grad + dC if C_vec.grad is not None else dC

    y_out._backward = _back
    h_out._backward = lambda: None  # backward already handled by y_out's _back via shared closure
    # We attach same backward to h_out but guard against double-call
    _back_called = [False]

    def _back_shared():
        if not _back_called[0]:
            _back_called[0] = True
            _back()

    y_out._backward = _back_shared
    h_out._backward = _back_shared

    return y_out, h_out


# ---------------------------------------------------------------------------
# 3. selective_scan_step  (single-step streaming)
# ---------------------------------------------------------------------------

def selective_scan_step(A_log: np.ndarray, B: np.ndarray, C: np.ndarray,
                        D: np.ndarray, x_t: np.ndarray,
                        h_prev: np.ndarray, dt: np.ndarray,
                        dt_min: float = 0.001, dt_max: float = 0.1
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Single SSM step for streaming inference (numpy, no autograd).
    A_log : (N,)    — log of decay rates
    B     : (N,)    — input matrix row
    C     : (N,)    — output matrix row
    D     : (1,)    — skip connection
    x_t   : scalar or (1,)
    h_prev: (N,)    — previous hidden state
    dt    : scalar  — raw dt before softplus

    Returns (y, h_new): y is scalar, h_new is (N,)
    """
    # softplus then clamp
    dt_sp = np.log1p(np.exp(dt))
    dt_clamped = np.clip(dt_sp, dt_min, dt_max)

    # Zero-order hold discretization
    A_bar = np.exp(-np.exp(A_log) * dt_clamped)          # (N,)
    B_bar = (1.0 - A_bar) * B                             # (N,)

    h_new = A_bar * h_prev + B_bar * x_t                   # (N,)
    y = float(np.dot(C, h_new) + float(np.asarray(D).flat[0]) * float(np.asarray(x_t).flat[0]))
    return y, h_new


# ---------------------------------------------------------------------------
# 4. matrix_exp_trace  (NOTEARS DAG constraint)
# ---------------------------------------------------------------------------

def matrix_exp_trace(W: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Compute tr(expm(W)) and its gradient d/dW tr(expm(W)) = expm(W).T
    W: (n, n) real numpy array
    Returns (scalar_trace, grad_matrix)
    """
    eW = expm(W)
    trace = float(np.trace(eW))
    grad = eW.T  # d tr(expm(W)) / dW_ij = expm(W)_ji
    return trace, grad


# ---------------------------------------------------------------------------
# 5. log_matrix_exp  (power series with Tensor grad support)
# ---------------------------------------------------------------------------

def log_matrix_exp(W: Tensor, max_terms: int = 10) -> Tensor:
    """
    Approximate matrix exponential via truncated power series: sum_{k=0}^{max_terms} W^k / k!
    Returns Tensor with gradient support.
    W: square Tensor (n, n)
    """
    n = W.data.shape[0]
    result_data = np.eye(n, dtype=np.float64)
    W_k_data = np.eye(n, dtype=np.float64)
    factorial = 1.0
    # Store W^k for backward
    powers = [np.eye(n, dtype=np.float64)]  # W^0

    for k in range(1, max_terms + 1):
        W_k_data = W_k_data @ W.data
        factorial *= k
        result_data = result_data + W_k_data / factorial
        powers.append(W_k_data.copy())

    out = Tensor(result_data, requires_grad=W.requires_grad,
                 _children=(W,), _op="log_matrix_exp")

    def _back():
        if W.requires_grad:
            g = out.grad if out.grad is not None else np.ones((n, n))
            # d/dW sum_{k=0}^{K} W^k/k! = sum_{k=1}^{K} sum_{j=0}^{k-1} W^j (g) W^{k-1-j} / k!
            # Uses: d/dW tr(A W^k B) involves chain rule over matrix product
            dW = np.zeros((n, n), dtype=np.float64)
            factorial = 1.0
            for k in range(1, max_terms + 1):
                factorial *= k
                for j in range(k):
                    # W^j @ g @ W^{k-1-j}
                    left = powers[j]             # W^j
                    right = powers[k - 1 - j]   # W^{k-1-j}
                    dW += left.T @ g @ right.T / factorial
            W.grad = W.grad + dW if W.grad is not None else dW

    out._backward = _back
    return out


# ---------------------------------------------------------------------------
# 6. top_k_sparse  (hard mask with straight-through estimator)
# ---------------------------------------------------------------------------

def top_k_sparse(scores: Tensor, k: int) -> Tensor:
    """
    Hard top-k mask along last axis.
    Straight-through estimator: forward uses hard mask, backward passes gradient through.
    scores: (..., D)
    Returns Tensor mask of same shape, 1 for top-k, 0 elsewhere.
    """
    data = scores.data
    thresh_idx = np.argsort(data, axis=-1)[..., -k]  # (...,) index of k-th largest
    thresh = np.take_along_axis(data, thresh_idx[..., None], axis=-1)
    hard_mask = (data >= thresh).astype(np.float64)
    # Normalize ties: if more than k are tied at threshold, keep exactly k
    # by random tie-breaking (deterministic: prefer lower indices for ties)
    counts = hard_mask.sum(axis=-1, keepdims=True)
    # Where counts > k due to ties, zero out excess (keep leftmost)
    # Build adjusted mask
    adjusted = np.zeros_like(hard_mask)
    flat_scores = data.reshape(-1, data.shape[-1])
    flat_adj = adjusted.reshape(-1, data.shape[-1])
    for i in range(flat_scores.shape[0]):
        top_idx = np.argpartition(flat_scores[i], -k)[-k:]
        flat_adj[i, top_idx] = 1.0
    adjusted = flat_adj.reshape(data.shape)

    out = Tensor(adjusted, requires_grad=scores.requires_grad,
                 _children=(scores,), _op="top_k_sparse")

    def _back():
        if scores.requires_grad:
            g = out.grad if out.grad is not None else np.ones(adjusted.shape)
            # Straight-through: pass gradient as-is
            scores.grad = scores.grad + g if scores.grad is not None else g.copy()

    out._backward = _back
    return out


# ---------------------------------------------------------------------------
# 7. differentiable_topk  (soft top-k with straight-through)
# ---------------------------------------------------------------------------

def differentiable_topk(scores: Tensor, k: int,
                        temperature: float = 1.0) -> Tuple[Tensor, Tensor]:
    """
    Soft top-k selection.
    soft_mask: differentiable approximation via temperature-scaled sorting
    hard_mask: hard 0/1 mask (straight-through for backward)

    Algorithm: perturb scores by temperature, rank, produce soft weights via
    normalised differences from sorted order.

    scores: (..., D)
    Returns (soft_mask, hard_mask)
    """
    data = scores.data.copy()
    shape = data.shape
    D = shape[-1]
    flat = data.reshape(-1, D)  # (M, D)

    # Hard mask via argpartition
    hard_flat = np.zeros_like(flat)
    for i in range(flat.shape[0]):
        idx = np.argpartition(flat[i], -k)[-k:]
        hard_flat[i, idx] = 1.0
    hard_data = hard_flat.reshape(shape)

    # Soft mask: scaled softmax over top-k positions using temperature
    # s_soft[j] = exp(score[j]/T) / sum_top_k exp(score[i]/T)  for j in top-k, else 0
    soft_flat = np.zeros_like(flat)
    for i in range(flat.shape[0]):
        topk_idx = np.where(hard_flat[i] > 0)[0]
        logits = flat[i, topk_idx] / temperature
        logits -= logits.max()
        exp_l = np.exp(logits)
        soft_flat[i, topk_idx] = exp_l / exp_l.sum() * k  # scale so sum = k
    soft_data = soft_flat.reshape(shape)

    rg = scores.requires_grad
    soft_out = Tensor(soft_data, requires_grad=rg, _children=(scores,), _op="diffTopk_soft")
    hard_out = Tensor(hard_data, requires_grad=rg, _children=(scores,), _op="diffTopk_hard")

    _called = [False]

    def _back():
        if _called[0]:
            return
        _called[0] = True
        if not scores.requires_grad:
            return

        gs = soft_out.grad if soft_out.grad is not None else np.zeros(soft_data.shape)
        gh = hard_out.grad if hard_out.grad is not None else np.zeros(hard_data.shape)

        gs_flat = gs.reshape(-1, D)
        gh_flat = gh.reshape(-1, D)
        grad_flat = np.zeros_like(flat)

        for i in range(flat.shape[0]):
            topk_idx = np.where(hard_flat[i] > 0)[0]
            logits = flat[i, topk_idx] / temperature
            logits -= logits.max()
            exp_l = np.exp(logits)
            s = exp_l / exp_l.sum()  # softmax

            # Gradient of soft_mask w.r.t. scores (Jacobian of scaled softmax)
            # d(s_j * k) / d(score_m) = k/T * s_j * (1[j==m] - s_m)  for m in top-k
            g_topk = gs_flat[i, topk_idx]  # gradient flowing in from soft output
            # Jacobian-vector product
            jvp = (g_topk - (g_topk * s).sum()) * s * k / temperature
            grad_flat[i, topk_idx] += jvp
            # Straight-through for hard mask: pass gradient directly
            grad_flat[i] += gh_flat[i]

        contrib = grad_flat.reshape(shape)
        scores.grad = scores.grad + contrib if scores.grad is not None else contrib

    soft_out._backward = _back
    hard_out._backward = _back

    return soft_out, hard_out
