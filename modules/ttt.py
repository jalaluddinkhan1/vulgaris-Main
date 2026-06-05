from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from engine.tensor import Tensor, Parameter
from engine.module import Module
from engine.layers import Linear, RMSNorm


@dataclass
class TTTConfig:
    """Configuration for Test-Time Training."""
    n_steps: int = 3            # inner-loop gradient steps per call
    lr: float = 1e-3            # inner-loop Adam learning rate
    mask_ratio: float = 0.3     # fraction of input channels masked for aux task
    persist: bool = False       # keep adapted weights after each call (default: restore)
    adapt_scope: str = "norms"  # "norms" | "norms+proj" | "all"
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8


class TestTimeTrainer(Module):
    """
    Test-Time Training (TTT) for VULGARIS.

    At inference time, runs K gradient steps on a masked channel reconstruction
    auxiliary task using the current batch before executing the main forward pass.
    This adapts the model to local distribution shift without any labelled data.

    Auxiliary task
    --------------
    Given x ∈ ℝ^{B × C × T}:
      1. Randomly zero mask_ratio × C input channels
      2. Run model forward → latent z ∈ ℝ^{B × T × D}  (already a Tensor in graph)
      3. Mean-pool z along time → z_pool ∈ ℝ^{B × D}
      4. recon_head(z_pool) → pred ∈ ℝ^{B × C}
      5. Target: channel-wise mean of original x over time
      6. MSE loss only on the masked channel positions
      7. Backward → update only the adaptable parameter subset

    Adaptable parameters (adapt_scope)
    -----------------------------------
    "norms"       — 8 RMSNorm scale vectors from the pre-norm layers (< 0.5% of params)
    "norms+proj"  — norms + recon_head weights
    "all"         — entire model (use with persist=False only)

    Usage
    -----
        ttt = TestTimeTrainer(model, TTTConfig(n_steps=3))
        prediction, aux = ttt.adapt_and_forward(x)
        # equivalent to: prediction, aux = ttt(x)
    """

    def __init__(self, model: Module, config: Optional[TTTConfig] = None):
        super().__init__()
        self.config = config or TTTConfig()
        self.model = model

        in_channels = getattr(model.config, "input_dim", 1)
        d_model = getattr(model, "d_model", None) or getattr(model.config, "d_model", 256)

        # Lightweight reconstruction head: latent → original channel space
        self.recon_head = Linear(d_model, in_channels)

        # Collect which parameters to adapt
        self._adaptable: List[Parameter] = self._collect_adaptable()

        # Adam state for adaptable params only
        self._m = {id(p): np.zeros_like(p.data, dtype=np.float32)
                   for p in self._adaptable}
        self._v = {id(p): np.zeros_like(p.data, dtype=np.float32)
                   for p in self._adaptable}
        self._t: int = 0

        # Snapshot storage for restoration
        self._snapshots: dict = {}

        # Stats exposed to callers
        self.last_aux_loss: float = 0.0
        self.total_adapt_steps: int = 0

    # ──────────────────────────────────────────────────────────────────────

    def _collect_adaptable(self) -> List[Parameter]:
        """Return deduplicated list of parameters to update during TTT."""
        scope = self.config.adapt_scope
        seen: dict[int, Parameter] = {}

        def _add(p: Parameter):
            if p is not None and isinstance(p, Parameter):
                seen[id(p)] = p

        # Always include recon_head (needs to learn the reconstruction mapping)
        _add(self.recon_head.weight)
        if self.recon_head.bias is not None:
            _add(self.recon_head.bias)

        m = self.model
        if scope in ("norms", "norms+proj"):
            for attr in ("norm_htd", "norm_sssr", "norm_attn", "norm_icl",
                         "norm_dah", "norm_rmc", "norm_crg", "norm_hmb"):
                norm = getattr(m, attr, None)
                if isinstance(norm, RMSNorm):
                    _add(norm.weight)

        if scope == "norms+proj":
            proj = getattr(m, "proj", None) or getattr(m, "output_head", None)
            if proj is not None:
                w = getattr(proj, "weight", None)
                b = getattr(proj, "bias", None)
                _add(w)
                _add(b)

        if scope == "all":
            for p in m.parameters():
                _add(p)

        return list(seen.values())

    # ──────────────────────────────────────────────────────────────────────

    def _snapshot(self):
        """Save current data of all adaptable parameters."""
        self._snapshots = {id(p): p.data.copy() for p in self._adaptable}

    def _restore(self):
        """Restore adaptable parameters to their last snapshot."""
        for p in self._adaptable:
            saved = self._snapshots.get(id(p))
            if saved is not None:
                p.data = saved.copy()

    def _zero_grad(self):
        for p in self._adaptable:
            p.grad = None

    def _adam_step(self):
        """Adam update on adaptable params only."""
        cfg = self.config
        self._t += 1
        bc1 = 1.0 - cfg.beta1 ** self._t
        bc2 = 1.0 - cfg.beta2 ** self._t

        for p in self._adaptable:
            if p.grad is None:
                continue
            g = np.asarray(p.grad, dtype=np.float32)
            pid = id(p)
            self._m[pid] = cfg.beta1 * self._m[pid] + (1.0 - cfg.beta1) * g
            self._v[pid] = cfg.beta2 * self._v[pid] + (1.0 - cfg.beta2) * (g * g)
            m_hat = self._m[pid] / bc1
            v_hat = self._v[pid] / bc2
            p.data = p.data - cfg.lr * m_hat / (np.sqrt(v_hat) + cfg.eps)
            p.grad = None

    # ──────────────────────────────────────────────────────────────────────

    def _aux_loss(self, x: Tensor) -> Tensor:
        """
        Masked channel reconstruction loss.

        Masks a random subset of input channels, runs the model, mean-pools the
        latent, decodes back to channel space, and returns MSE on masked positions.

        The loss graph flows through:
            MSE → recon_head → z_pool (mean of z) → z (Tensor) → norm scales
        """
        x_np = x.data  # (B, C, T)
        B, C, T = x_np.shape
        n_mask = max(1, int(C * self.config.mask_ratio))
        mask_idx = np.random.choice(C, size=n_mask, replace=False)

        # Build masked input (no grad needed on this path)
        x_masked_np = x_np.copy()
        x_masked_np[:, mask_idx, :] = 0.0
        x_masked = Tensor(x_masked_np.astype(np.float32), requires_grad=False)

        # Forward pass — z is a Tensor connected to the norm graph.
        # Coupling note: this relies on Vulgaris.forward() storing
        # aux["h_states"] = z (the full latent sequence Tensor).
        # If the model does not set this key, aux_loss returns 0 gracefully.
        _, aux = self.model(x_masked)
        z = aux.get("h_states")      # Tensor (B, T, D) — set by Vulgaris.forward()

        if z is None or not isinstance(z, Tensor):
            # Graceful fallback: h_states not available (non-Vulgaris model or
            # stripped aux dict). Return zero loss so adapt_and_forward still runs.
            return Tensor(np.array([0.0], dtype=np.float32), requires_grad=False)

        # Mean-pool over time with proper backward
        z_pool_np = z.data.mean(axis=1).astype(np.float32)  # (B, D)
        _z = z
        _T = T

        z_pool = Tensor(
            z_pool_np,
            requires_grad=z.requires_grad,
            _children=(z,),
            _op="ttt_pool",
        )

        def _pool_back():
            if _z.requires_grad and z_pool.grad is not None:
                g = (z_pool.grad[:, np.newaxis, :] / _T).astype(np.float32)
                broadcast = np.broadcast_to(g, _z.data.shape).copy()
                _z.grad = (_z.grad + broadcast if _z.grad is not None
                           else broadcast)

        z_pool._backward = _pool_back

        # Decode latent → channel predictions: (B, C)
        pred = self.recon_head(z_pool)

        # Target: per-channel mean over time of the *original* (unmasked) signal
        target_np = x_np.mean(axis=2).astype(np.float32)   # (B, C)
        target = Tensor(target_np, requires_grad=False)

        # Loss mask: 1/(B*n_mask) on masked channels, 0 elsewhere — vectorised MSE
        loss_w = np.zeros((B, C), dtype=np.float32)
        loss_w[:, mask_idx] = 1.0 / (B * n_mask)
        loss_w_t = Tensor(loss_w, requires_grad=False)

        diff = pred - target
        loss = (diff * diff * loss_w_t).sum()
        return loss

    # ──────────────────────────────────────────────────────────────────────

    def adapt_and_forward(
        self,
        x: Tensor,
        **forward_kwargs,
    ) -> tuple[Tensor, dict]:
        """
        Run TTT inner loop, then execute the main model forward pass.

        Steps
        -----
        1. Snapshot adaptable params (for restoration if persist=False)
        2. K inner-loop steps: aux_loss → backward → Adam on adaptable params
        3. Main model forward with the (now adapted) parameters
        4. If persist=False: restore params to snapshot
        5. Return (prediction, aux_dict)
        """
        self._snapshot()
        total_loss = 0.0

        for _ in range(self.config.n_steps):
            self._zero_grad()
            loss = self._aux_loss(x)
            if float(loss.data) > 0.0:
                loss.backward()
            self._adam_step()
            total_loss += float(loss.data)
            self.total_adapt_steps += 1

        self.last_aux_loss = total_loss / max(self.config.n_steps, 1)

        # Main forward with adapted parameters
        pred, aux = self.model(x, **forward_kwargs)
        aux["ttt_aux_loss"] = self.last_aux_loss
        aux["ttt_steps"] = self.config.n_steps
        aux["ttt_persist"] = self.config.persist

        if not self.config.persist:
            self._restore()

        return pred, aux

    # ──────────────────────────────────────────────────────────────────────

    def reset_adam_state(self):
        """
        Reset Adam momentum buffers.
        Call between independent sequences to prevent momentum bleed-over.
        """
        self._t = 0
        for p in self._adaptable:
            pid = id(p)
            self._m[pid] = np.zeros_like(p.data, dtype=np.float32)
            self._v[pid] = np.zeros_like(p.data, dtype=np.float32)

    def stats(self) -> dict:
        return {
            "n_adaptable_params": sum(p.data.size for p in self._adaptable),
            "n_adaptable_tensors": len(self._adaptable),
            "total_adapt_steps": self.total_adapt_steps,
            "last_aux_loss": self.last_aux_loss,
            "adapt_scope": self.config.adapt_scope,
            "persist": self.config.persist,
        }

    def forward(self, x: Tensor, **kwargs) -> tuple[Tensor, dict]:
        return self.adapt_and_forward(x, **kwargs)
