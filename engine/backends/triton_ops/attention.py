"""
Triton FlashAttention-style kernel for VULGARIS CausalAttention.
Falls back to numpy implementation if triton/torch unavailable.
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
    def _flash_attn_fwd_kernel(
        Q_ptr, K_ptr, V_ptr, Out_ptr,
        Lse_ptr,
        B, H, T, D_head,
        scale,
        s_qb, s_qh, s_qt, s_qd,
        BLOCK_T: tl.constexpr,
        BLOCK_D: tl.constexpr,
        CAUSAL:  tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_h = tl.program_id(1)
        pid_t = tl.program_id(2)

        t_start = pid_t * BLOCK_T
        t_range = t_start + tl.arange(0, BLOCK_T)
        d_range = tl.arange(0, BLOCK_D)

        mask_t = t_range < T
        mask_d = d_range < D_head

        # Load Q block: (BLOCK_T, BLOCK_D)
        Q = tl.load(
            Q_ptr + pid_b * s_qb + pid_h * s_qh +
            t_range[:, None] * s_qt + d_range[None, :] * s_qd,
            mask=mask_t[:, None] & mask_d[None, :], other=0.0
        )

        acc = tl.zeros([BLOCK_T, BLOCK_D], dtype=tl.float32)
        lse = tl.full([BLOCK_T], float("-inf"), dtype=tl.float32)
        m   = tl.full([BLOCK_T], float("-inf"), dtype=tl.float32)

        for j in range(0, T, BLOCK_T):
            kv_range = j + tl.arange(0, BLOCK_T)
            mask_kv = kv_range < T

            if CAUSAL:
                causal_mask = kv_range[None, :] <= t_range[:, None]
            else:
                causal_mask = tl.full([BLOCK_T, BLOCK_T], True, dtype=tl.int1)

            K_blk = tl.load(
                K_ptr + pid_b * s_qb + pid_h * s_qh +
                kv_range[None, :] * s_qt + d_range[:, None] * s_qd,
                mask=mask_kv[None, :] & mask_d[:, None], other=0.0
            )
            V_blk = tl.load(
                V_ptr + pid_b * s_qb + pid_h * s_qh +
                kv_range[:, None] * s_qt + d_range[None, :] * s_qd,
                mask=mask_kv[:, None] & mask_d[None, :], other=0.0
            )

            S = tl.dot(Q, K_blk) * scale
            S = tl.where(causal_mask & mask_t[:, None] & mask_kv[None, :], S, float("-inf"))

            m_new = tl.maximum(m, tl.max(S, axis=1))
            alpha  = tl.exp(m - m_new)
            P      = tl.exp(S - m_new[:, None])

            acc  = acc * alpha[:, None] + tl.dot(P, V_blk)
            lse  = lse * alpha + tl.sum(P, axis=1)
            m    = m_new

        # Normalise
        acc = acc / (lse[:, None] + 1e-6)

        tl.store(
            Out_ptr + pid_b * s_qb + pid_h * s_qh +
            t_range[:, None] * s_qt + d_range[None, :] * s_qd,
            acc.to(tl.float16),
            mask=mask_t[:, None] & mask_d[None, :]
        )

    def flash_attention_triton(q_np, k_np, v_np, causal=True):
        """
        q, k, v: (B, H, T, D_head) float32 numpy
        Returns: (B, H, T, D_head) float32 numpy
        """
        B, H, T, D_head = q_np.shape
        scale = D_head ** -0.5
        BLOCK_T = min(64, triton.next_power_of_2(T))
        BLOCK_D = min(64, triton.next_power_of_2(D_head))

        q = torch.from_numpy(q_np).cuda().to(torch.float16).contiguous()
        k = torch.from_numpy(k_np).cuda().to(torch.float16).contiguous()
        v = torch.from_numpy(v_np).cuda().to(torch.float16).contiguous()
        out = torch.empty_like(q)
        lse = torch.empty(B, H, T, device="cuda", dtype=torch.float32)

        grid = (B, H, triton.cdiv(T, BLOCK_T))
        _flash_attn_fwd_kernel[grid](
            q, k, v, out, lse,
            B, H, T, D_head, scale,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            BLOCK_T=BLOCK_T, BLOCK_D=BLOCK_D, CAUSAL=causal,
        )
        return out.float().cpu().numpy()

else:
    def flash_attention_triton(q_np, k_np, v_np, causal=True):
        return None   # caller falls back to numpy


def get_flash_attention():
    """Returns flash_attention_triton if available, else None."""
    if _TRITON:
        return flash_attention_triton
    return None
