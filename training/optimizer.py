import numpy as np
from typing import List, Tuple

from engine.tensor import Parameter


class SpectralAdamW:
    """AdamW optimizer with spectral norm clipping on weight matrices."""

    def __init__(
        self,
        params: List[Parameter],
        lr: float = 3e-4,
        betas: Tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        spectral_clip: float = 2.0,
        grad_clip: float = 1.0,
    ):
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.spectral_clip = spectral_clip
        self.grad_clip = grad_clip

        # Support both flat list and param-group list of dicts
        params_list = list(params)
        if params_list and isinstance(params_list[0], dict):
            self.param_groups: list = params_list
        else:
            self.param_groups = [{"params": params_list, "lr": lr,
                                   "weight_decay": weight_decay}]
        self.params = [p for g in self.param_groups for p in g["params"]]
        # Map id(param) -> group index for per-group lr/wd lookup
        self._param_group_idx: dict = {}
        for gi, g in enumerate(self.param_groups):
            for p in g["params"]:
                self._param_group_idx[id(p)] = gi

        self._t: int = 0

        # Moment buffers, indexed by id(param)
        self._m: dict = {id(p): np.zeros_like(p.data) for p in self.params}
        self._v: dict = {id(p): np.zeros_like(p.data) for p in self.params}

        # Power-iteration singular vectors for spectral clipping (2D params only)
        # Keyed by id(param): (u_vec, v_vec)
        self._sv: dict = {}
        for p in self.params:
            if p.data.ndim == 2:
                out_f, in_f = p.data.shape
                u = np.random.randn(out_f)
                u /= (np.linalg.norm(u) + 1e-12)
                v = np.random.randn(in_f)
                v /= (np.linalg.norm(v) + 1e-12)
                self._sv[id(p)] = (u.copy(), v.copy())

    # ──────────────────────────────────────────────────────────────────────

    def zero_grad(self):
        """Clear all parameter gradients."""
        for p in self.params:
            p.grad = None

    # ──────────────────────────────────────────────────────────────────────

    def _global_grad_clip(self):
        """Clip all gradients so total L2 norm <= grad_clip."""
        total_norm_sq = 0.0
        for p in self.params:
            if p.grad is not None:
                total_norm_sq += float(np.sum(p.grad ** 2))
        total_norm = np.sqrt(total_norm_sq)
        if total_norm > self.grad_clip and total_norm > 1e-12:
            scale = self.grad_clip / total_norm
            for p in self.params:
                if p.grad is not None:
                    p.grad = p.grad * scale

    # ──────────────────────────────────────────────────────────────────────

    def _spectral_clip_param(self, param: Parameter):
        """
        3-step power iteration to estimate σ_max(W).
        If σ_max > spectral_clip, scale W in-place.
        Only for 2-D weight matrices.
        """
        if param.data.ndim != 2:
            return

        pid = id(param)
        W = param.data   # (out, in)
        u, v = self._sv.get(pid, (None, None))
        if u is None:
            out_f, in_f = W.shape
            u = np.random.randn(out_f)
            u /= (np.linalg.norm(u) + 1e-12)
            v = np.random.randn(in_f)
            v /= (np.linalg.norm(v) + 1e-12)

        for _ in range(3):
            v_new = W.T @ u
            v_new_norm = np.linalg.norm(v_new) + 1e-12
            v = v_new / v_new_norm

            u_new = W @ v
            u_new_norm = np.linalg.norm(u_new) + 1e-12
            u = u_new / u_new_norm

        sigma_max = float(u @ W @ v)

        # Store updated singular vectors
        self._sv[pid] = (u.copy(), v.copy())

        if sigma_max > self.spectral_clip and sigma_max > 1e-12:
            param.data *= self.spectral_clip / sigma_max

    # ──────────────────────────────────────────────────────────────────────

    def step(self):
        """
        1. Global gradient clipping
        2. AdamW parameter update
        3. Spectral norm clipping on 2-D matrices
        """
        self._global_grad_clip()

        self._t += 1
        beta1, beta2 = self.betas
        t = self._t

        # Bias correction factors
        bc1 = 1.0 - beta1 ** t
        bc2 = 1.0 - beta2 ** t

        for p in self.params:
            if p.grad is None:
                continue
            if not p.requires_grad:
                continue

            g = p.grad
            pid = id(p)

            # Moment updates
            m = beta1 * self._m[pid] + (1.0 - beta1) * g
            v = beta2 * self._v[pid] + (1.0 - beta2) * (g * g)
            self._m[pid] = m
            self._v[pid] = v

            m_hat = m / bc1
            v_hat = v / bc2

            # AdamW step: gradient update + decoupled weight decay (per-group lr/wd)
            gi = self._param_group_idx.get(pid, 0)
            g_lr = self.param_groups[gi].get("lr", self.lr)
            g_wd = self.param_groups[gi].get("weight_decay", self.weight_decay)
            update = m_hat / (np.sqrt(v_hat) + self.eps)
            p.data -= g_lr * update
            p.data -= g_lr * g_wd * p.data

        # Spectral norm clipping
        for p in self.params:
            if p.data.ndim == 2 and p.requires_grad:
                self._spectral_clip_param(p)

    # ──────────────────────────────────────────────────────────────────────

    def state_dict(self) -> dict:
        """Return optimizer state for checkpointing."""
        return {
            "t": self._t,
            "lr": self.lr,
            "betas": list(self.betas),
            "eps": self.eps,
            "weight_decay": self.weight_decay,
            "spectral_clip": self.spectral_clip,
            "grad_clip": self.grad_clip,
            "m": {str(i): v.copy() for i, v in enumerate(self._m.values())},
            "v": {str(i): v.copy() for i, v in enumerate(self._v.values())},
        }

    def load_state_dict(self, state: dict):
        """Restore optimizer state from checkpoint."""
        self._t = int(state.get("t", 0))
        self.lr = float(state.get("lr", self.lr))
        betas = state.get("betas", list(self.betas))
        self.betas = (float(betas[0]), float(betas[1]))
        self.eps = float(state.get("eps", self.eps))
        self.weight_decay = float(state.get("weight_decay", self.weight_decay))
        self.spectral_clip = float(state.get("spectral_clip", self.spectral_clip))
        self.grad_clip = float(state.get("grad_clip", self.grad_clip))

        m_dict = state.get("m", {})
        v_dict = state.get("v", {})
        for i, pid in enumerate(list(self._m.keys())):
            key = str(i)
            if key in m_dict:
                self._m[pid] = np.asarray(m_dict[key], dtype=np.float64)
            if key in v_dict:
                self._v[pid] = np.asarray(v_dict[key], dtype=np.float64)


# ──────────────────────────────────────────────────────────────────────────────


class CosineSchedule:
    """Cosine annealing with linear warmup."""

    def __init__(
        self,
        optimizer: SpectralAdamW,
        warmup_steps: int,
        max_steps: int,
        min_lr: float,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr = min_lr
        self._base_lr: float = optimizer.lr
        self._step_count: int = 0

    def get_lr(self, step: int) -> float:
        """Linear warmup then cosine decay from base_lr to min_lr."""
        if step < self.warmup_steps:
            # Linear warmup
            return self._base_lr * (step + 1) / max(self.warmup_steps, 1)

        # Cosine decay
        progress = (step - self.warmup_steps) / max(
            self.max_steps - self.warmup_steps, 1
        )
        progress = min(progress, 1.0)
        cosine_factor = 0.5 * (1.0 + np.cos(np.pi * progress))
        return self.min_lr + (self._base_lr - self.min_lr) * cosine_factor

    def step(self):
        """Advance schedule by one step and update optimizer lr."""
        self._step_count += 1
        new_lr = self.get_lr(self._step_count)
        self.optimizer.lr = new_lr
