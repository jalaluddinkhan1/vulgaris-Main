from __future__ import annotations

import numpy as np

from engine.tensor import Tensor, Parameter, zeros
from engine.module import Module
from engine.layers import Linear
from config import HTDConfig


class HTDLevel(Module):
    """
    Single SSM level with ZOH discretisation.
    dt is constrained to be near the level's time constant tau.

    State: h_t = A_bar * h_{t-1} + (1 - A_bar) * B_proj(x) + bias
    Output: y_t = C_proj(h_t)
    """

    def __init__(self, d_model: int, state_dim: int, tau: float):
        super().__init__()
        self.d_model = d_model
        self.state_dim = state_dim
        self.tau = tau

        # Diagonal SSM decay: log_A initialised to log(0.5) -> A ~ 0.5
        self.log_A = Parameter(
            np.full(state_dim, np.log(0.5), dtype=np.float64), name="log_A"
        )
        self.B_proj = Linear(d_model, state_dim)
        self.C_proj = Linear(state_dim, d_model)
        self.dt_proj = Linear(d_model, 1)

    def forward(self, x: Tensor, h_prev: Tensor | None = None,
                bias: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """
        x     : (batch, T, d_model) or (batch, d_model) for single step
        h_prev: (batch, state_dim) or None  [zeros if None]
        bias  : (batch, state_dim) optional slow->fast feedback, broadcast over T

        Returns (y, h_last):
            y      : same leading dims as x but last dim = d_model
            h_last : (batch, state_dim)
        """
        squeeze_T = x.ndim == 2
        if squeeze_T:
            x = x.unsqueeze(1)  # (batch, 1, d_model)

        B_sz, T, D = x.shape

        if h_prev is None:
            h_prev = zeros((B_sz, self.state_dim))

        # dt: (batch, T, 1) -> constrained around tau
        dt_raw = self.dt_proj(x)                    # (batch, T, 1)
        dt = dt_raw.sigmoid() * (2.0 * self.tau)    # dt in [0, 2*tau]

        # A_bar = exp(-exp(log_A) * dt)  — broadcast state_dim
        # exp(log_A): (state_dim,) ; dt: (batch, T, 1)
        A_decay = (self.log_A.exp()).reshape(1, 1, self.state_dim)  # (1,1,N)
        A_bar = (A_decay * dt * (-1.0)).exp()                       # (batch, T, N)

        B_in = self.B_proj(x)   # (batch, T, state_dim)

        # b_prime = (1 - A_bar) * B_in  — effective input after gating
        # Keep as Tensor so its backward flows to A_bar and B_in
        one_minus_a = Tensor(
            (1.0 - A_bar.data).astype(np.float32),
            requires_grad=A_bar.requires_grad,
            _children=(A_bar,), _op="one_minus_a"
        )
        _A_bar_ref = A_bar

        def _oma_back():
            if _A_bar_ref.requires_grad and one_minus_a.grad is not None:
                contrib = -one_minus_a.grad.astype(_A_bar_ref.data.dtype)
                _A_bar_ref.grad = (_A_bar_ref.grad + contrib
                                   if _A_bar_ref.grad is not None else contrib)

        one_minus_a._backward = _oma_back
        b_prime = one_minus_a * B_in    # (batch, T, state_dim) — Tensor

        # Add cross-level bias broadcast over T
        if bias is not None:
            bias_np = np.broadcast_to(
                bias.data[:, None, :], b_prime.data.shape).copy().astype(np.float32)
            bias_t = Tensor(bias_np, requires_grad=bias.requires_grad,
                            _children=(bias,), _op="bias_broadcast")
            _bias = bias; _bias_t = bias_t; _T = T

            def _bias_back():
                if _bias.requires_grad and _bias_t.grad is not None:
                    _bias.grad = ((_bias.grad + _bias_t.grad.sum(axis=1))
                                  if _bias.grad is not None
                                  else _bias_t.grad.sum(axis=1))

            bias_t._backward = _bias_back
            b_prime = b_prime + bias_t

        # Parallel scan: h[t] = A_bar[t]*h[t-1] + b_prime[t]
        from engine.parallel_scan import parallel_scan_ssm, parallel_scan_ssm_backward

        a_np = A_bar.data.astype(np.float32)
        bp_np = b_prime.data.astype(np.float32)
        h_init_np = h_prev.data.astype(np.float32)

        h_np = parallel_scan_ssm(a_np, bp_np, h_init_np)  # (batch, T, state_dim)

        h_scan = Tensor(
            h_np,
            requires_grad=A_bar.requires_grad or b_prime.requires_grad,
            _children=(A_bar, b_prime),
            _op="htd_parallel_scan"
        )
        _A_bar, _b_prime, _a_np, _h_np = A_bar, b_prime, a_np, h_np

        def _htd_scan_back():
            if h_scan.grad is None:
                return
            grad_a, grad_bp = parallel_scan_ssm_backward(
                _a_np, _h_np, h_scan.grad.astype(np.float32))
            if _A_bar.requires_grad:
                _A_bar.grad = (_A_bar.grad + grad_a
                               if _A_bar.grad is not None else grad_a)
            if _b_prime.requires_grad:
                _b_prime.grad = (_b_prime.grad + grad_bp
                                 if _b_prime.grad is not None else grad_bp)

        h_scan._backward = _htd_scan_back

        # Vectorised C projection over all timesteps at once
        from engine.tensor import cat as tcat
        B_sz2, T2, N = h_np.shape
        h_flat = h_scan.reshape(B_sz2 * T2, N)
        y_flat = self.C_proj(h_flat)           # (batch*T, d_model)
        y = y_flat.reshape(B_sz2, T2, -1)      # (batch, T, d_model)

        # h_last: final hidden state with gradient connection
        h_last_np = h_np[:, -1, :].astype(np.float64)
        h_last = Tensor(h_last_np, requires_grad=h_scan.requires_grad,
                        _children=(h_scan,), _op="htd_h_last")
        _h_scan = h_scan

        def _htd_h_last_back():
            if _h_scan.requires_grad and h_last.grad is not None:
                contrib = np.zeros_like(_h_scan.data)
                contrib[:, -1, :] = h_last.grad
                _h_scan.grad = (_h_scan.grad + contrib
                                if _h_scan.grad is not None else contrib)

        h_last._backward = _htd_h_last_back

        if squeeze_T:
            y = y.squeeze(1)

        return y, h_last


class HierarchicalTimescaleDecomposition(Module):
    """
    n_levels nested SSMs at different time constants.
    Level i subsamples input by factor 2^i and communicates via bottleneck layers.

    Fast->Slow: slow input augmented by bottleneck_down(h_fast)
    Slow->Fast: fast receives additive bias from bottleneck_up(h_slow)
    """

    def __init__(self, d_model: int, state_dim: int, config: HTDConfig):
        super().__init__()
        self.d_model = d_model
        self.state_dim = state_dim
        self.n_levels = config.n_levels
        self.taus = config.time_constants

        # SSM level modules — stored as numbered attributes for Module registration
        for i, tau in enumerate(self.taus):
            setattr(self, f"level_{i}", HTDLevel(d_model, state_dim, tau))

        # Bottleneck projections: fast(level i) -> slow(level i+1) input
        for i in range(self.n_levels - 1):
            setattr(self, f"bottleneck_down_{i}",
                    Linear(state_dim, d_model))

        # Slow(level i+1) -> fast(level i) bias feedback
        for i in range(self.n_levels - 1):
            setattr(self, f"bottleneck_up_{i}",
                    Linear(state_dim, state_dim))

        # Output: concatenate d_model from each level, project back to d_model
        self.output_proj = Linear(d_model * self.n_levels, d_model)

    def _get_level(self, i: int) -> HTDLevel:
        return getattr(self, f"level_{i}")

    def _get_bn_down(self, i: int) -> Linear:
        return getattr(self, f"bottleneck_down_{i}")

    def _get_bn_up(self, i: int) -> Linear:
        return getattr(self, f"bottleneck_up_{i}")

    def forward(self, x: Tensor,
                states: list[Tensor | None] | None = None
                ) -> tuple[Tensor, list[Tensor]]:
        """
        x     : (batch, T, d_model)
        states: list of (batch, state_dim) h_prev per level, or None

        Returns (output, new_states):
            output    : (batch, T, d_model)
            new_states: list of (batch, state_dim), one per level
        """
        B, T, D = x.shape

        if states is None:
            states = [None] * self.n_levels

        # ----------------------------------------------------------------
        # Bottom-up pass: compute each level's output
        # Level i operates on x subsampled by stride 2^i
        # Fast->Slow coupling adds bottleneck_down(h_fast) to slow input
        # ----------------------------------------------------------------
        level_inputs: list[Tensor | None] = [None] * self.n_levels
        level_inputs[0] = x  # level 0 gets full-rate input

        # Pre-compute subsampled inputs (before coupling; coupling added below)
        for i in range(1, self.n_levels):
            stride = 2 ** i
            # Subsample: take every stride-th timestep
            indices = list(range(0, T, stride))
            if len(indices) == 0:
                indices = [0]
            # Build subsampled tensor
            x_sub_np = x.data[:, indices, :]           # (B, T_i, D)
            level_inputs[i] = Tensor(
                x_sub_np,
                requires_grad=x.requires_grad,
                _children=(x,),
                _op=f"subsample_{i}"
            )
            # Attach backward for subsampling
            _i = i
            _indices = indices
            _T = T
            _sub = level_inputs[i]
            _x = x

            def _sub_back(sub=_sub, inp=_x, idx=_indices, t_full=_T):
                if inp.requires_grad and sub.grad is not None:
                    contrib = np.zeros_like(inp.data)
                    np.add.at(contrib, (slice(None), idx, slice(None)), sub.grad)
                    inp.grad = inp.grad + contrib if inp.grad is not None else contrib

            level_inputs[_i]._backward = _sub_back

        # Now run levels with coupling
        # We do two passes: first collect h_last from each level independently,
        # then on a second pass inject the cross-level feedback.
        # For efficiency we do a single pass: process level 0 first, use its
        # hidden state to augment level 1, and so on (causal).

        level_outputs: list[Tensor | None] = [None] * self.n_levels
        new_states: list[Tensor | None] = [None] * self.n_levels
        level_h_last: list[Tensor | None] = [None] * self.n_levels

        # Forward sweep: level 0 -> n_levels-1
        for i in range(self.n_levels):
            xi = level_inputs[i]   # (B, T_i, D)
            h_prev = states[i]

            # Fast->Slow: augment xi with bottleneck_down of previous (faster) level
            if i > 0 and level_h_last[i - 1] is not None:
                aug = self._get_bn_down(i - 1)(level_h_last[i - 1])  # (B, D)
                # Broadcast aug over time dimension of xi
                T_i = xi.shape[1]
                aug_expanded = aug.unsqueeze(1)  # (B, 1, D)
                aug_np = np.broadcast_to(aug_expanded.data, (B, T_i, D)).copy()
                aug_broad = Tensor(
                    aug_np,
                    requires_grad=aug.requires_grad,
                    _children=(aug,),
                    _op="broadcast_aug"
                )
                _aug = aug
                _aug_broad = aug_broad
                _T_i = T_i

                def _aug_back(ab=_aug_broad, a=_aug, ti=_T_i):
                    if a.requires_grad and ab.grad is not None:
                        contrib = ab.grad.sum(axis=1)   # (B, D)
                        a.grad = a.grad + contrib if a.grad is not None else contrib

                aug_broad._backward = _aug_back
                xi = xi + aug_broad

            # Slow->Fast: feedback bias from slower (coarser) level
            slow_bias = None
            if i < self.n_levels - 1 and level_h_last[i + 1] is not None:
                slow_bias = self._get_bn_up(i)(level_h_last[i + 1])  # (B, state_dim)

            yi, h_last_i = self._get_level(i).forward(xi, h_prev, slow_bias)
            level_outputs[i] = yi           # (B, T_i, D)
            level_h_last[i] = h_last_i      # (B, state_dim)
            new_states[i] = h_last_i

        # ----------------------------------------------------------------
        # Upsample all levels to T and concatenate
        # ----------------------------------------------------------------
        upsampled: list[Tensor] = []
        for i in range(self.n_levels):
            yi = level_outputs[i]    # (B, T_i, D)
            T_i = yi.shape[1]

            if T_i == T:
                upsampled.append(yi)
            else:
                # Nearest-neighbour upsample: repeat each timestep by (T // T_i)
                repeat = T // T_i
                # np.repeat along axis=1
                yi_up_np = np.repeat(yi.data, repeat, axis=1)  # (B, T_i*repeat, D)
                # Crop or pad to exactly T
                if yi_up_np.shape[1] > T:
                    yi_up_np = yi_up_np[:, :T, :]
                elif yi_up_np.shape[1] < T:
                    pad_len = T - yi_up_np.shape[1]
                    yi_up_np = np.pad(yi_up_np, ((0, 0), (0, pad_len), (0, 0)),
                                      mode="edge")

                yi_up = Tensor(
                    yi_up_np,
                    requires_grad=yi.requires_grad,
                    _children=(yi,),
                    _op=f"upsample_{i}"
                )
                _yi = yi
                _yi_up = yi_up
                _repeat = repeat
                _T_i2 = T_i

                def _up_back(up=_yi_up, src=_yi, r=_repeat, ti=_T_i2):
                    if src.requires_grad and up.grad is not None:
                        # Sum gradients from repeated positions back to source
                        g = up.grad[:, :ti * r, :]  # trim to clean repeat region
                        # reshape (B, T_i, r, D) then sum over r
                        g_folded = g.reshape(g.shape[0], ti, r, g.shape[2])
                        contrib = g_folded.sum(axis=2)   # (B, T_i, D)
                        src.grad = src.grad + contrib if src.grad is not None else contrib

                yi_up._backward = _up_back
                upsampled.append(yi_up)

        # Concatenate along feature dim: (B, T, D * n_levels)
        from engine.tensor import cat as tcat
        concat = tcat(upsampled, axis=2)  # (B, T, D * n_levels)

        output = self.output_proj(concat)  # (B, T, D)

        return output, new_states
