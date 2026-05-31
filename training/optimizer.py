from __future__ import annotations

import numpy as np

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


class MuonOptimizer:
    """
    Muon: Momentum + Newton-Schulz orthogonalization for 2-D weight matrices.

    For 2-D params:  apply Nesterov momentum, then orthogonalize the update
    via 5 steps of Newton-Schulz iteration before scaling and applying it.

    For 1-D params (bias, scale, etc.): fall back to standard AdamW.

    Newton-Schulz coefficients: a=3.4445, b=-4.7750, c=2.0315
      X ← a·X + (b·A + c·A²)·X,  where A = X·Xᵀ  (5 iterations)

    Reference: Kosson et al. 2024 — "Muon: Momentum + Orthogonalization"
    """

    _NS_A = 3.4445
    _NS_B = -4.7750
    _NS_C = 2.0315

    def __init__(
        self,
        params: List[Parameter],
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        # AdamW fallback for 1-D params
        adam_lr: float = 3e-4,
        adam_betas: Tuple[float, float] = (0.9, 0.95),
        adam_eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov
        self.ns_steps = ns_steps
        self.adam_lr = adam_lr
        self.adam_betas = adam_betas
        self.adam_eps = adam_eps
        self.weight_decay = weight_decay

        params_list = list(params)
        self._2d_params: List[Parameter] = [p for p in params_list if p.data.ndim == 2]
        self._1d_params: List[Parameter] = [p for p in params_list if p.data.ndim != 2]

        # Momentum buffers for 2-D params
        self._buf: dict = {id(p): np.zeros_like(p.data) for p in self._2d_params}

        # AdamW state for 1-D params
        self._m: dict = {id(p): np.zeros_like(p.data) for p in self._1d_params}
        self._v: dict = {id(p): np.zeros_like(p.data) for p in self._1d_params}
        self._t: int = 0

    # ──────────────────────────────────────────────────────────────────────

    def zero_grad(self):
        for p in self._2d_params + self._1d_params:
            p.grad = None

    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _ns_orthogonalize(G: np.ndarray, steps: int = 5) -> np.ndarray:
        """Newton-Schulz orthogonalization: map G towards the nearest orthogonal matrix."""
        a, b, c = MuonOptimizer._NS_A, MuonOptimizer._NS_B, MuonOptimizer._NS_C
        # Normalize to unit spectral norm before iteration
        norm = np.linalg.norm(G) + 1e-12
        X = G / norm
        for _ in range(steps):
            A = X @ X.T          # (out, out)
            X = a * X + (b * A + c * (A @ A)) @ X
        return X

    # ──────────────────────────────────────────────────────────────────────

    def step(self):
        self._t += 1

        # --- 2-D params: Muon update ---
        for p in self._2d_params:
            if p.grad is None or not p.requires_grad:
                continue
            g = p.grad
            pid = id(p)
            buf = self._buf[pid]

            # Nesterov momentum
            buf[:] = self.momentum * buf + g
            if self.nesterov:
                effective_g = g + self.momentum * buf
            else:
                effective_g = buf

            # Orthogonalize update
            orth = self._ns_orthogonalize(effective_g, steps=self.ns_steps)

            # Scale to match RMS of raw gradient
            rms_g = float(np.sqrt(np.mean(g ** 2)) + 1e-12)
            rms_o = float(np.sqrt(np.mean(orth ** 2)) + 1e-12)
            update = orth * (rms_g / rms_o)

            p.data -= self.lr * update
            p.data -= self.lr * self.weight_decay * p.data

        # --- 1-D params: AdamW fallback ---
        beta1, beta2 = self.adam_betas
        bc1 = 1.0 - beta1 ** self._t
        bc2 = 1.0 - beta2 ** self._t

        for p in self._1d_params:
            if p.grad is None or not p.requires_grad:
                continue
            g = p.grad
            pid = id(p)
            m = beta1 * self._m[pid] + (1.0 - beta1) * g
            v = beta2 * self._v[pid] + (1.0 - beta2) * (g * g)
            self._m[pid] = m
            self._v[pid] = v
            update = (m / bc1) / (np.sqrt(v / bc2) + self.adam_eps)
            p.data -= self.adam_lr * update
            p.data -= self.adam_lr * self.weight_decay * p.data

    # ──────────────────────────────────────────────────────────────────────

    def state_dict(self) -> dict:
        return {
            "t": self._t,
            "buf": {str(i): v.copy() for i, v in enumerate(self._buf.values())},
            "m":   {str(i): v.copy() for i, v in enumerate(self._m.values())},
            "v":   {str(i): v.copy() for i, v in enumerate(self._v.values())},
        }

    def load_state_dict(self, state: dict):
        self._t = int(state.get("t", 0))
        for i, pid in enumerate(list(self._buf.keys())):
            key = str(i)
            if key in state.get("buf", {}):
                self._buf[pid] = np.asarray(state["buf"][key], dtype=np.float64)
        for i, pid in enumerate(list(self._m.keys())):
            key = str(i)
            if key in state.get("m", {}):
                self._m[pid] = np.asarray(state["m"][key], dtype=np.float64)
            if key in state.get("v", {}):
                self._v[pid] = np.asarray(state["v"][key], dtype=np.float64)


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


# ──────────────────────────────────────────────────────────────────────────────


class PCGrad:
    """
    PCGrad: gradient surgery for multi-task / multi-loss training.

    For each pair of per-loss gradient vectors, if their cosine similarity
    is negative (conflicting), the conflicting component is projected out.
    The corrected per-loss gradients are then summed and written back to
    each parameter's `.grad`.

    Reference: Yu et al., 2020 — "Gradient Surgery for Multi-Task Learning"

    Usage
    -----
        pcgrad = PCGrad(optimizer)

        losses = [task_loss, dag_loss, conformal_loss, ...]
        pcgrad.step(losses)   # zero_grad + backward + project + optimizer.step
    """

    def __init__(self, optimizer: SpectralAdamW):
        self.optimizer = optimizer

    def step(self, losses: List) -> None:
        """
        losses : list of scalar Tensor values (one per loss term).
        Performs:
          1. Backward pass for each loss independently.
          2. PCGrad projection across pairs.
          3. Accumulates corrected gradients.
          4. Calls optimizer.step().
        """
        params = self.optimizer.params
        n_losses = len(losses)

        # Collect per-loss gradients: list of {pid: grad_array}
        per_loss_grads: List[dict] = []
        for loss in losses:
            self.optimizer.zero_grad()
            loss.backward()
            grads = {}
            for p in params:
                if p.grad is not None and p.requires_grad:
                    grads[id(p)] = p.grad.copy()
            per_loss_grads.append(grads)

        # Project conflicting gradient pairs
        projected = [
            {pid: g.copy() for pid, g in grads.items()}
            for grads in per_loss_grads
        ]

        for i in range(n_losses):
            for j in range(n_losses):
                if i == j:
                    continue
                for p in params:
                    pid = id(p)
                    gi = projected[i].get(pid)
                    gj = per_loss_grads[j].get(pid)
                    if gi is None or gj is None:
                        continue
                    gi_flat = gi.ravel()
                    gj_flat = gj.ravel()
                    denom = float(np.dot(gj_flat, gj_flat)) + 1e-12
                    cos_sim = float(np.dot(gi_flat, gj_flat)) / (
                        np.linalg.norm(gi_flat) * np.linalg.norm(gj_flat) + 1e-12
                    )
                    if cos_sim < 0.0:
                        # Remove the component of gi along gj
                        projection = (float(np.dot(gi_flat, gj_flat)) / denom) * gj
                        projected[i][pid] = gi - projection.reshape(gi.shape)

        # Accumulate corrected gradients into each parameter
        self.optimizer.zero_grad()
        for i, grads in enumerate(projected):
            for p in params:
                pid = id(p)
                if pid in grads:
                    if p.grad is None:
                        p.grad = grads[pid].copy()
                    else:
                        p.grad = p.grad + grads[pid]

        self.optimizer.step()


# ──────────────────────────────────────────────────────────────────────────────


class StagedLossSchedule:
    """
    Staged loss warmup: activates auxiliary loss terms gradually to prevent
    random-feature gradients from dominating the main task signal early in
    training.

    Stage layout (default, tunable via `stages` argument):
      step 0  …  warmup_task-1        : task + temporal only
      step warmup_task  …  warmup_dag : + DAG + InfoNCE
      step warmup_dag   …  warmup_aux : + conformal + memory
      step warmup_aux+  …             : all 8 losses active

    Usage
    -----
        schedule = StagedLossSchedule(warmup_task=200, warmup_dag=500,
                                       warmup_aux=1000)
        # In training loop:
        weights = schedule.get_weights(global_step)
        total_loss = sum(w * L for w, L in zip(weights, losses))
        # OR use with PCGrad:
        active = schedule.filter_losses(global_step, losses)
        pcgrad.step(active)

    Loss index convention (matches training loop):
      0: task          1: temporal
      2: DAG           3: InfoNCE
      4: conformal     5: memory
      6: EWC           7: CBF
    """

    _DEFAULT_STAGES = {
        # (loss_index, step_at_which_it_becomes_active)
        0: 0,    # task — always active
        1: 0,    # temporal — always active
        2: None, # DAG — set from warmup_dag
        3: None, # InfoNCE
        4: None, # conformal — set from warmup_aux
        5: None, # memory
        6: None, # EWC
        7: None, # CBF
    }

    def __init__(
        self,
        warmup_task: int = 0,
        warmup_dag: int = 500,
        warmup_aux: int = 1000,
        n_losses: int = 8,
    ):
        self.n_losses = n_losses
        self._activation = list(range(n_losses))   # default: all at step 0

        stage_map = {
            0: warmup_task, 1: warmup_task,
            2: warmup_dag,  3: warmup_dag,
            4: warmup_aux,  5: warmup_aux,
            6: warmup_aux,  7: warmup_aux,
        }
        self._activation = [stage_map.get(i, 0) for i in range(n_losses)]

    def is_active(self, loss_idx: int, step: int) -> bool:
        if loss_idx >= self.n_losses:
            return False
        return step >= self._activation[loss_idx]

    def get_weights(self, step: int) -> List[float]:
        """Return a list of 0.0/1.0 weights, one per loss term."""
        return [1.0 if self.is_active(i, step) else 0.0
                for i in range(self.n_losses)]

    def filter_losses(self, step: int, losses: List) -> List:
        """Return only the active subset of losses for PCGrad."""
        return [L for i, L in enumerate(losses) if self.is_active(i, step)]
