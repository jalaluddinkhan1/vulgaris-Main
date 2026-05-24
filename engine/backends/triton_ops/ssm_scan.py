"""
Triton GPU kernel for SSM linear recurrence.

Strategy: parallelise over (batch, d_block) — each GPU thread block owns
one (batch item, D-slice) and runs a fast sequential scan over T in SRAM.
This gives near-optimal memory bandwidth utilisation.
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
    def _ssm_fwd_kernel(
        a_ptr, b_ptr, out_ptr,
        B, T, D,
        s_ab, s_at, s_ad,
        BLOCK_D: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_d = tl.program_id(1)

        d_off = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        mask = d_off < D

        h = tl.zeros([BLOCK_D], dtype=tl.float32)

        for t in range(T):
            off = pid_b * s_ab + t * s_at + d_off * s_ad
            a = tl.load(a_ptr + off, mask=mask, other=1.0)
            b = tl.load(b_ptr + off, mask=mask, other=0.0)
            h = a * h + b
            tl.store(out_ptr + off, h, mask=mask)

    @triton.jit
    def _ssm_bwd_kernel(
        a_ptr, h_ptr, grad_h_ptr,
        grad_a_ptr, grad_b_ptr,
        B, T, D,
        s_ab, s_at, s_ad,
        BLOCK_D: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_d = tl.program_id(1)

        d_off = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        mask = d_off < D

        lam = tl.zeros([BLOCK_D], dtype=tl.float32)

        for t in range(T - 1, -1, -1):
            off = pid_b * s_ab + t * s_at + d_off * s_ad

            dh = tl.load(grad_h_ptr + off, mask=mask, other=0.0)
            a  = tl.load(a_ptr       + off, mask=mask, other=1.0)

            lam = lam + dh
            tl.store(grad_b_ptr + off, lam, mask=mask)

            if t > 0:
                h_prev_off = pid_b * s_ab + (t - 1) * s_at + d_off * s_ad
                h_prev = tl.load(h_ptr + h_prev_off, mask=mask, other=0.0)
            else:
                h_prev = tl.zeros([BLOCK_D], dtype=tl.float32)

            tl.store(grad_a_ptr + off, lam * h_prev, mask=mask)
            lam = lam * a

    def ssm_scan_triton(a_np, b_np, h_init=None):
        B, T, D = a_np.shape
        BLOCK_D = min(64, triton.next_power_of_2(D))

        a = torch.from_numpy(a_np).cuda().contiguous()
        b = torch.from_numpy(b_np).cuda().contiguous()
        out = torch.empty_like(a)

        if h_init is not None:
            b = b.clone()
            b[:, 0, :] += a[:, 0, :] * torch.from_numpy(h_init).cuda()

        grid = (B, triton.cdiv(D, BLOCK_D))
        _ssm_fwd_kernel[grid](
            a, b, out,
            B, T, D,
            a.stride(0), a.stride(1), a.stride(2),
            BLOCK_D=BLOCK_D,
        )
        return out.cpu().numpy()

else:
    def ssm_scan_triton(a_np, b_np, h_init=None):
        from engine.parallel_scan import _numpy_scan
        return _numpy_scan(a_np, b_np, h_init)
