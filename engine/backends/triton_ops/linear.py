"""
Triton fused linear layer: y = x @ W.T + b with optional activation.
"""
import numpy as np

try:
    import torch
    import triton
    import triton.language as tl
    _TRITON = True
except ImportError:
    _TRITON = False


if _TRITON:
    @triton.jit
    def _linear_fwd_kernel(
        X_ptr, W_ptr, B_ptr, Out_ptr,
        M, N, K,
        has_bias: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Tiled GEMM: Out = X @ W.T  (+bias)"""
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        m_range = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        n_range = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

        for k in range(0, K, BLOCK_K):
            k_range = k + tl.arange(0, BLOCK_K)
            mask_mk = (m_range[:, None] < M) & (k_range[None, :] < K)
            mask_nk = (n_range[:, None] < N) & (k_range[None, :] < K)

            x_blk = tl.load(X_ptr + m_range[:, None] * K + k_range[None, :],
                             mask=mask_mk, other=0.0)
            w_blk = tl.load(W_ptr + n_range[:, None] * K + k_range[None, :],
                             mask=mask_nk, other=0.0)

            acc += tl.dot(x_blk, tl.trans(w_blk))

        if has_bias:
            b_vec = tl.load(B_ptr + n_range, mask=n_range < N, other=0.0)
            acc += b_vec[None, :]

        mask_out = (m_range[:, None] < M) & (n_range[None, :] < N)
        tl.store(Out_ptr + m_range[:, None] * N + n_range[None, :], acc, mask=mask_out)

    def linear_triton(x_np, w_np, b_np=None):
        """
        x_np: (..., K)  — any leading dims flattened to M
        w_np: (N, K)
        b_np: (N,) or None
        Returns: (..., N)
        """
        shape = x_np.shape
        M = int(np.prod(shape[:-1]))
        K = shape[-1]
        N = w_np.shape[0]

        BLOCK_M = min(64, triton.next_power_of_2(M))
        BLOCK_N = min(64, triton.next_power_of_2(N))
        BLOCK_K = min(32, triton.next_power_of_2(K))

        x = torch.from_numpy(x_np.reshape(M, K)).cuda().float().contiguous()
        w = torch.from_numpy(w_np).cuda().float().contiguous()
        out = torch.empty(M, N, device="cuda", dtype=torch.float32)

        has_bias = b_np is not None
        b = torch.from_numpy(b_np).cuda().float() if has_bias else torch.empty(0, device="cuda")

        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        _linear_fwd_kernel[grid](
            x, w, b, out,
            M, N, K,
            has_bias=has_bias,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )
        return out.cpu().numpy().reshape(*shape[:-1], N)

else:
    def linear_triton(x_np, w_np, b_np=None):
        out = x_np @ w_np.T
        if b_np is not None:
            out = out + b_np
        return out


def get_linear_triton():
    if _TRITON:
        return linear_triton
    return None
