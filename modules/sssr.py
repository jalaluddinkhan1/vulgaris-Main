from __future__ import annotations

import numpy as np
from collections import OrderedDict

from engine.tensor import Tensor, Parameter, zeros, ones, cat
from engine.module import Module
from engine.layers import Linear, RMSNorm, Conv1d
from config import SSSRConfig


class SSSRHead(Module):
    """Single SSM head with Hebbian online adaptation."""

    def __init__(self, d_model: int, state_dim: int,
                 dt_min: float, dt_max: float,
                 hebbian_lr: float, stability_eps: float):
        super().__init__()
        self.d_model = d_model
        self.state_dim = state_dim
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.hebbian_lr = hebbian_lr
        self.stability_eps = stability_eps

        # HiPPO-LegS-inspired timescale hierarchy: state 0 = long memory (slow decay),
        # state N-1 = short memory (fast decay). Decay rates spaced log-uniformly
        # so each state resolves a distinct temporal scale.
        log_a_init = np.linspace(-4.0, -0.2, state_dim, dtype=np.float32)
        log_a_init += np.random.randn(state_dim) * 0.01
        self.log_A = Parameter(log_a_init, name="log_A")

        self.B_proj = Linear(d_model, state_dim, bias=False)
        self.C_proj = Linear(d_model, state_dim, bias=False)

        # D skip-connection, one scalar per output channel (d_model)
        self.D = Parameter(np.ones(d_model, dtype=np.float32), name="D")

        # dt projection + bias
        self.dt_proj = Linear(d_model, 1, bias=True)
        # initialise dt_proj bias so softplus output lands near dt_min
        # softplus^{-1}(dt_min) = log(exp(dt_min) - 1)
        dt_init_bias = np.log(np.exp(dt_min) - 1.0 + 1e-8)
        self.dt_proj.bias.data[:] = dt_init_bias
        self.dt_bias = Parameter(np.zeros(1, dtype=np.float32), name="dt_bias")

    def _compute_dt(self, x: Tensor) -> Tensor:
        """
        x: (batch, T, d_model) or (batch, d_model)
        Returns dt: (batch, T, 1) or (batch, 1), clamped to [dt_min, dt_max]
        """
        raw = self.dt_proj(x) + self.dt_bias  # (..., 1)
        # softplus
        sp_data = np.log1p(np.exp(np.clip(raw.data, -20, 20)))
        sp = Tensor(sp_data, requires_grad=raw.requires_grad,
                    _children=(raw,), _op="softplus")

        def _sp_back():
            if raw.requires_grad and sp.grad is not None:
                sig = 1.0 / (1.0 + np.exp(-np.clip(raw.data, -500, 500)))
                raw.grad = (raw.grad + sp.grad * sig
                            if raw.grad is not None else sp.grad * sig)

        sp._backward = _sp_back
        return sp.clip(self.dt_min, self.dt_max)

    def forward(self, x: Tensor, h_prev: Tensor | None = None,
                dt: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """
        x    : (batch, T, d_model)
        dt   : optional (batch, T) or (batch, T, 1) — external timestep deltas for
               irregular-rate sensors. When None, dt is learned from x via dt_proj.
        Returns (y, h_last):
            y      : (batch, T, state_dim)  — raw SSM output, projected back externally
            h_last : (batch, state_dim)
        """
        B, T, D = x.shape

        if h_prev is None:
            h_prev = zeros((B, self.state_dim))

        if dt is not None:
            # External dt: ensure shape (B, T, 1) and clamp to [dt_min, dt_max]
            if dt.data.ndim == 2:
                dt = Tensor(dt.data[:, :, None], requires_grad=dt.requires_grad,
                            _children=(dt,), _op="dt_unsqueeze")
            dt = dt.clip(self.dt_min, self.dt_max)
        else:
            dt = self._compute_dt(x)   # (B, T, 1)
        B_t = self.B_proj(x)           # (B, T, state_dim)
        C_t = self.C_proj(x)           # (B, T, state_dim)

        # A_t = exp(-exp(log_A) * dt)  — ZOH discretisation
        # exp(log_A): (state_dim,)  ->  (1, 1, state_dim)
        decay = self.log_A.exp().reshape(1, 1, self.state_dim)  # always positive
        A_t = (decay * dt * (-1.0)).exp()                        # (B, T, state_dim)

        # Parallel scan — replaces sequential O(T) loop with O(T log T) work
        from engine.parallel_scan import parallel_scan_ssm, parallel_scan_ssm_backward

        a_np = A_t.data.astype(np.float32)
        b_np = B_t.data.astype(np.float32)
        h_init_np = h_prev.data.astype(np.float32)

        h_np = parallel_scan_ssm(a_np, b_np, h_init_np)  # (B, T, state_dim)

        # Wrap as Tensor so downstream ops stay in the autograd graph
        h_all = Tensor(
            h_np,
            requires_grad=A_t.requires_grad or B_t.requires_grad,
            _children=(A_t, B_t),
            _op="parallel_scan"
        )
        _A_t, _B_t, _a_np, _h_np = A_t, B_t, a_np, h_np

        def _scan_back():
            if h_all.grad is None:
                return
            grad_a, grad_b = parallel_scan_ssm_backward(_a_np, _h_np,
                                                         h_all.grad.astype(np.float32))
            if _A_t.requires_grad:
                _A_t.grad = _A_t.grad + grad_a if _A_t.grad is not None else grad_a
            if _B_t.requires_grad:
                _B_t.grad = _B_t.grad + grad_b if _B_t.grad is not None else grad_b

        h_all._backward = _scan_back

        # Vectorised output — no per-timestep Python loop
        y = (C_t * h_all).sum(axis=-1, keepdims=True)          # (B, T, 1)
        skip = (self.D * x).sum(axis=-1, keepdims=True)         # (B, T, 1)
        y = y + skip

        # h_last: last timestep, gradient-connected
        h_last_np = h_np[:, -1, :].astype(np.float32)
        h_last = Tensor(h_last_np, requires_grad=h_all.requires_grad,
                        _children=(h_all,), _op="h_last_slice")
        _h_all = h_all

        def _h_last_back():
            if _h_all.requires_grad and h_last.grad is not None:
                contrib = np.zeros_like(_h_all.data)
                contrib[:, -1, :] = h_last.grad
                _h_all.grad = (_h_all.grad + contrib
                               if _h_all.grad is not None else contrib)

        h_last._backward = _h_last_back

        # Hebbian update during training (numpy, not autograd)
        if self.training:
            self.hebbian_update(h_np)

        return y, h_last

    def step(self, x_t: Tensor, h_prev: Tensor,
             dt: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """
        Single-step inference.
        x_t   : (batch, d_model)
        h_prev: (batch, state_dim)
        dt    : optional (batch,) or scalar — external timestep delta.
        Returns (y_t, h_new):
            y_t  : (batch, 1)
            h_new: (batch, state_dim)
        """
        x_t_3d = x_t.unsqueeze(1)                # (batch, 1, d_model)
        if dt is not None:
            dt_data = np.asarray(dt.data if isinstance(dt, Tensor) else dt,
                                 dtype=np.float32)
            dt_data = np.clip(dt_data.reshape(-1, 1), self.dt_min, self.dt_max)
            dt_2d = Tensor(dt_data, requires_grad=False)
        else:
            dt_raw = self._compute_dt(x_t_3d)     # (batch, 1, 1)
            dt_2d = dt_raw.squeeze(1)              # (batch, 1)

        B_t = self.B_proj(x_t)                    # (batch, state_dim)
        C_t = self.C_proj(x_t)                    # (batch, state_dim)

        decay = self.log_A.exp().reshape(1, self.state_dim)
        A_t = (decay * dt_2d * (-1.0)).exp()      # (batch, state_dim)

        h_new = A_t * h_prev + B_t
        y_t = (C_t * h_new).sum(axis=-1, keepdims=True)  # (batch, 1)
        skip = (self.D * x_t).sum(axis=-1, keepdims=True)
        y_t = y_t + skip

        return y_t, h_new

    def hebbian_update(self, h_states: np.ndarray):
        """
        h_states: (batch, T, state_dim)
        ΔA_log = η * mean_{batch,time}(h_t ⊙ h_{t-1} - h_t^2)
        Clamp log_A to [-5, 0].
        """
        if h_states.shape[1] < 2:
            return
        h_prev_np = h_states[:, :-1, :]   # (B, T-1, N)
        h_curr_np = h_states[:, 1:, :]    # (B, T-1, N)
        delta = (h_curr_np * h_prev_np - h_curr_np ** 2).mean(axis=(0, 1))  # (N,)
        self.log_A.data += self.hebbian_lr * delta
        np.clip(self.log_A.data, -5.0, 0.0, out=self.log_A.data)


class SelectiveSSR(Module):
    """Full SSSR block: multi-head SSM with expansion, causal conv, gating, and skip."""

    def __init__(self, d_model: int, config: SSSRConfig):
        super().__init__()
        self.d_model = d_model
        self.d_inner = config.d_inner
        self.n_heads = config.n_heads
        self.state_dim = config.state_dim
        self.dt_min = config.dt_min
        self.dt_max = config.dt_max
        self.routing_k = float(getattr(config, "routing_k", 1.0))

        # Expansion branch
        self.x_proj = Linear(d_model, config.d_inner)
        # Gating branch
        self.z_proj = Linear(d_model, d_model)
        # MoD routing: scalar score per token; None when routing_k == 1.0 (disabled)
        self.routing_proj = Linear(d_model, 1, bias=False) if self.routing_k < 1.0 else None

        # Causal depthwise conv on expansion: kernel=4, causal padding = kernel-1 on left
        self.conv_kernel = 4
        self.depthwise_conv = Conv1d(
            in_channels=config.d_inner,
            out_channels=config.d_inner,
            kernel_size=self.conv_kernel,
            padding=0,      # we do manual left padding
            groups=config.d_inner,
            bias=True,
        )

        # n_heads independent SSM heads; each head gets d_inner features
        # and produces 1-dim output; we project d_inner channels through heads
        # head_dim = d_inner // n_heads  (features per head)
        assert config.d_inner % config.n_heads == 0, \
            "d_inner must be divisible by n_heads"
        self.head_dim = config.d_inner // config.n_heads
        head_state = max(1, config.state_dim // config.n_heads)

        for i in range(config.n_heads):
            setattr(self, f"head_{i}",
                    SSSRHead(self.head_dim, head_state,
                             config.dt_min, config.dt_max,
                             config.hebbian_lr, config.stability_eps))

        # Project concatenated head outputs (n_heads scalar outputs per timestep)
        # Each head produces 1 scalar -> total n_heads per timestep
        self.y_proj = Linear(config.n_heads, d_model)

        # Skip connection
        self.skip_proj = Linear(d_model, d_model)

        # Output norm
        self.norm = RMSNorm(d_model)

        # Streaming conv cache: (B, d_inner, kernel_size-1) — populated on first step()
        self._conv_buf: np.ndarray | None = None

    def _get_head(self, i: int) -> SSSRHead:
        return getattr(self, f"head_{i}")

    def reset_conv_cache(self) -> None:
        """Clear the streaming causal-conv history buffer. Call between sequences."""
        self._conv_buf = None

    def _conv_step(self, u_t: Tensor) -> Tensor:
        """
        Single-step causal conv using a persistent ring buffer instead of
        zero-padding. Maintains exact output as full _causal_conv for streaming.

        u_t : (B, d_inner)
        Returns (B, d_inner)
        """
        B, C = u_t.data.shape
        pad = self.conv_kernel - 1

        if self._conv_buf is None or self._conv_buf.shape[0] != B:
            self._conv_buf = np.zeros((B, C, pad), dtype=np.float32)

        # Build (B, C, kernel_size) window: [history | current]
        window = np.concatenate([self._conv_buf, u_t.data[:, :, None]], axis=2)

        # Roll buffer: drop oldest sample
        self._conv_buf = window[:, :, 1:].copy()

        # Wrap window as Tensor for autograd linkage
        win_t = Tensor(window, requires_grad=u_t.requires_grad,
                       _children=(u_t,), _op="conv_window")
        _u_t = u_t

        def _win_back():
            if _u_t.requires_grad and win_t.grad is not None:
                g = win_t.grad[:, :, -1:]
                _u_t.grad = (_u_t.grad + g[:, :, 0]
                             if _u_t.grad is not None else g[:, :, 0].copy())

        win_t._backward = _win_back

        # Apply depthwise conv: (B, C, kernel_size) -> (B, C, 1)
        conv_out = self.depthwise_conv(win_t)          # (B, C, 1)
        out_np = conv_out.data[:, :, 0]
        out = Tensor(out_np, requires_grad=conv_out.requires_grad,
                     _children=(conv_out,), _op="conv_step_squeeze")
        _co = conv_out

        def _sq_back():
            if _co.requires_grad and out.grad is not None:
                contrib = out.grad[:, :, None]
                _co.grad = (_co.grad + contrib if _co.grad is not None else contrib.copy())

        out._backward = _sq_back
        return out  # (B, d_inner)

    def _causal_conv(self, u: Tensor) -> Tensor:
        """
        u: (B, T, d_inner)
        Applies depthwise conv1d with causal (left) padding.
        Returns (B, T, d_inner).
        """
        B, T, C = u.shape
        # Conv1d expects (B, C, L); transpose
        u_t = u.transpose((0, 2, 1))    # (B, d_inner, T)

        # Manual left padding: pad kernel_size-1 zeros on the left
        pad = self.conv_kernel - 1
        pad_data = np.zeros((B, C, pad), dtype=np.float32)
        pad_t = Tensor(pad_data, requires_grad=False)

        # Concatenate [pad | u_t] along time axis
        u_padded_data = np.concatenate([pad_t.data, u_t.data], axis=2)  # (B, C, T+pad)
        u_padded = Tensor(
            u_padded_data,
            requires_grad=u_t.requires_grad,
            _children=(u_t,),
            _op="causal_pad"
        )

        def _pad_back():
            if u_t.requires_grad and u_padded.grad is not None:
                # Gradient only flows to the right (non-pad) portion
                u_t.grad = (u_t.grad + u_padded.grad[:, :, pad:]
                            if u_t.grad is not None
                            else u_padded.grad[:, :, pad:].copy())

        u_padded._backward = _pad_back

        # Apply conv (no padding inside Conv1d) -> output length = T
        conv_out = self.depthwise_conv(u_padded)   # (B, d_inner, T)

        # Transpose back: (B, T, d_inner)
        out = conv_out.transpose((0, 2, 1))
        return out

    def forward(self, x: Tensor, h_states: list | None = None,
                dt: Tensor | None = None) -> tuple[Tensor, list]:
        """
        x  : (batch, T, d_model)
        dt : optional (batch, T) — external per-step timestep deltas for
             irregular-rate sensors; passed through to each SSM head.
        Returns (output, h_states_new) where output: (batch, T, d_model)
        """
        B, T, D = x.shape

        if h_states is None:
            h_states = [None] * self.n_heads

        # ── MoD routing (Mixture of Depths) ──────────────────────────────
        # Compute per-token routing scores and select top-k tokens to process
        # through SSM heads; unselected tokens pass through as identity.
        # Skipped when routing_k == 1.0 (default) for zero overhead.
        route_mask = None   # (B, T) hard mask or None
        if self.routing_proj is not None and self.routing_k < 1.0:
            from engine.ops import differentiable_topk
            scores = self.routing_proj(x)          # (B, T, 1)
            scores_2d = Tensor(
                scores.data[:, :, 0],
                requires_grad=scores.requires_grad,
                _children=(scores,),
                _op="route_squeeze"
            )
            _sc3, _sc2 = scores, scores_2d

            def _sq_back():
                if _sc3.requires_grad and _sc2.grad is not None:
                    contrib = _sc2.grad[:, :, None]
                    _sc3.grad = _sc3.grad + contrib if _sc3.grad is not None else contrib

            scores_2d._backward = _sq_back

            k_tokens = max(1, int(self.routing_k * T))
            # Flatten (B, T) → process jointly so routing is per-batch-item
            # soft_mask: (B, T) differentiable weights; hard_mask: (B, T) 0/1
            soft_mask, hard_mask = differentiable_topk(scores_2d, k=k_tokens)
            route_mask = hard_mask.data   # (B, T)

        # Expansion and causal conv
        u = self.x_proj(x)               # (B, T, d_inner)
        u = self._causal_conv(u)          # (B, T, d_inner)

        # Gating branch
        z = self.z_proj(x)               # (B, T, d_model)

        # Multi-head SSM: split u into n_heads chunks along feature dim
        head_outs = []
        h_states_new = []

        for i in range(self.n_heads):
            head = self._get_head(i)
            start = i * self.head_dim
            end = start + self.head_dim
            u_i_data = u.data[:, :, start:end]
            u_i = Tensor(u_i_data, requires_grad=u.requires_grad,
                         _children=(u,), _op=f"head_slice_{i}")

            _start, _end, _u, _u_i = start, end, u, u_i

            def _slice_back(ui=_u_i, parent=_u, s=_start, e=_end):
                if parent.requires_grad and ui.grad is not None:
                    if parent.grad is None:
                        parent.grad = np.zeros_like(parent.data)
                    parent.grad[:, :, s:e] += ui.grad

            u_i._backward = _slice_back

            y_i, h_new_i = head(u_i, h_states[i], dt=dt)   # y_i: (B, T, 1)
            head_outs.append(y_i)
            h_states_new.append(h_new_i)

        # Concatenate head outputs: (B, T, n_heads)
        y_heads = cat(head_outs, axis=-1)   # (B, T, n_heads)

        # Project to d_model
        y_ssm = self.y_proj(y_heads)        # (B, T, d_model)

        # Gated output: y_ssm * silu(z)
        gated = y_ssm * z.silu()            # (B, T, d_model)

        # Apply MoD mask: zero out SSM contribution for unrouted tokens
        # so they rely entirely on the skip connection
        if route_mask is not None:
            mask_3d = route_mask[:, :, None]   # (B, T, 1) broadcast over d_model
            gated_data = gated.data * mask_3d
            gated = Tensor(
                gated_data,
                requires_grad=gated.requires_grad,
                _children=(gated,),
                _op="mod_mask"
            )
            _gated_orig = gated._prev[0]
            _mask_3d = mask_3d

            def _mod_back():
                if _gated_orig.requires_grad and gated.grad is not None:
                    _gated_orig.grad = (
                        _gated_orig.grad + gated.grad * _mask_3d
                        if _gated_orig.grad is not None
                        else gated.grad * _mask_3d
                    )

            gated._backward = _mod_back

        # Skip connection
        skip = self.skip_proj(x)            # (B, T, d_model)

        out = self.norm(gated + skip)
        return out, h_states_new

    def step(self, x_t: Tensor, h_states: list,
             dt: Tensor | None = None) -> tuple[Tensor, list]:
        """
        Single-step streaming inference.
        x_t    : (batch, d_model)
        h_states: list of h_prev per head, each (batch, head_state_dim)
        dt     : optional (batch,) — external timestep delta for this step.
        Returns (output_t, h_states_new)  output_t: (batch, d_model)
        """
        # Expand
        u_t = self.x_proj(x_t)            # (B, d_inner)

        # Use the streaming conv cache — correct history vs. zero-padding
        u_conv = self._conv_step(u_t)      # (B, d_inner)

        # Gating
        z_t = self.z_proj(x_t)            # (B, d_model)

        # Multi-head SSM step
        head_outs = []
        h_states_new = []

        for i in range(self.n_heads):
            head = self._get_head(i)
            start = i * self.head_dim
            end = start + self.head_dim
            u_i_data = u_conv.data[:, start:end]
            u_i = Tensor(u_i_data, requires_grad=u_conv.requires_grad,
                         _children=(u_conv,), _op=f"step_slice_{i}")

            _u_conv, _u_i, _s, _e = u_conv, u_i, start, end

            def _back(ui=_u_i, p=_u_conv, s=_s, e=_e):
                if p.requires_grad and ui.grad is not None:
                    if p.grad is None:
                        p.grad = np.zeros_like(p.data)
                    p.grad[:, s:e] += ui.grad

            u_i._backward = _back

            y_i, h_new_i = head.step(u_i, h_states[i], dt=dt)   # y_i: (B, 1)
            head_outs.append(y_i)
            h_states_new.append(h_new_i)

        # (B, n_heads)
        y_heads = cat(head_outs, axis=-1)

        # Project
        y_ssm = self.y_proj(y_heads)       # (B, d_model)

        gated = y_ssm * z_t.silu()
        skip = self.skip_proj(x_t)
        out = self.norm(gated + skip)

        return out, h_states_new
