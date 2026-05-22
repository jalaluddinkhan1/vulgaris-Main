"""Self-supervised pretraining for VULGARIS: masked reconstruction + temporal contrastive + forecasting."""
import numpy as np
from typing import Optional, Tuple

from engine.tensor import Tensor
from engine.layers import Linear
from engine.module import Module


class MaskedReconstructionHead(Module):
    """Projects latent back to input channel space for masked reconstruction loss."""

    def __init__(self, d_model: int, in_channels: int):
        super().__init__()
        self.proj = Linear(d_model, in_channels)

    def forward(self, z: Tensor) -> Tensor:
        # z: (B, T, d_model) → (B, T, in_channels)
        B, T, D = z.data.shape
        z_flat = z.reshape(B * T, D)
        out = self.proj(z_flat)
        return out.reshape(B, T, -1)


class SelfSupervisedTrainer:
    """
    Two pretraining objectives:
      1. Masked reconstruction: randomly mask 15-30% of timesteps, reconstruct with MSE.
      2. Temporal contrastive: InfoNCE between anchor and positive (nearby) timesteps.

    Usage:
        trainer = SelfSupervisedTrainer(model, in_channels=9, d_model=64)
        loss = trainer.pretrain_step(x_np)
    """

    def __init__(
        self,
        model,
        in_channels: int,
        d_model: int,
        mask_ratio_min: float = 0.15,
        mask_ratio_max: float = 0.30,
        contrastive_temp: float = 0.07,
        contrastive_pos_window: int = 5,
        lr: float = 1e-3,
    ):
        self.model = model
        self.in_channels = in_channels
        self.d_model = d_model
        self.mask_ratio_min = mask_ratio_min
        self.mask_ratio_max = mask_ratio_max
        self.contrastive_temp = contrastive_temp
        self.contrastive_pos_window = contrastive_pos_window

        self.recon_head = MaskedReconstructionHead(d_model, in_channels)

        # Collect all parameters for simple SGD
        self._params = list(model.parameters()) + list(self.recon_head.parameters())
        self.lr = lr
        self._step = 0

    def _zero_grad(self):
        for p in self._params:
            p.grad = None

    def _sgd_step(self):
        for p in self._params:
            if p.grad is not None:
                p.data -= self.lr * p.grad

    def _make_mask(self, B: int, T: int) -> np.ndarray:
        """Boolean mask (B, T): True = masked position."""
        ratio = np.random.uniform(self.mask_ratio_min, self.mask_ratio_max)
        n_mask = max(1, int(T * ratio))
        mask = np.zeros((B, T), dtype=bool)
        for b in range(B):
            idx = np.random.choice(T, n_mask, replace=False)
            mask[b, idx] = True
        return mask

    def masked_reconstruction_loss(self, x_np: np.ndarray) -> Tuple[float, Tensor]:
        """
        x_np: (B, C, T) float32
        Returns (scalar_loss, loss_tensor)
        """
        B, C, T = x_np.shape
        mask = self._make_mask(B, T)  # (B, T) bool

        # Zero out masked positions in input
        x_masked = x_np.copy()
        for b in range(B):
            x_masked[b, :, mask[b]] = 0.0

        x_t = Tensor(x_masked.astype(np.float32), requires_grad=False)
        _, aux = self.model(x_t, domain_idx=0)
        z = aux.get("h_states")  # (B, T, d_model)

        if z is None:
            return 0.0, Tensor(np.array([[0.0]], dtype=np.float32))

        # Reconstruct: (B, T, C)
        recon = self.recon_head(z)

        # x_np transposed: (B, T, C)
        x_target = x_np.transpose(0, 2, 1)  # (B, T, C)

        # MSE only on masked positions
        mask_expanded = mask[:, :, None]                          # (B, T, 1)
        recon_np = recon.data                                     # (B, T, C)
        diff_np = recon_np - x_target                             # (B, T, C)
        sq_np = diff_np ** 2                                      # (B, T, C)
        n_masked = int(mask.sum())
        if n_masked == 0:
            return 0.0, Tensor(np.array([[0.0]], dtype=np.float32))
        loss_scalar = float((sq_np * mask_expanded).sum()) / (n_masked * x_target.shape[2])

        # Wrap as Tensor so backward can reach recon
        loss_t = Tensor(
            np.array([[loss_scalar]], dtype=np.float32),
            requires_grad=recon.requires_grad,
            _children=(recon,),
            _op="masked_mse"
        )
        _recon = recon
        _mask_exp = mask_expanded.astype(np.float32)
        _n_denom = float(n_masked * x_target.shape[2])

        def _masked_mse_back():
            if _recon.requires_grad and loss_t.grad is not None:
                g = float(loss_t.grad.sum())
                grad_recon = 2.0 * (recon_np - x_target) * _mask_exp / _n_denom * g
                _recon.grad = _recon.grad + grad_recon if _recon.grad is not None else grad_recon

        loss_t._backward = _masked_mse_back
        return loss_scalar, loss_t

    def temporal_contrastive_loss(self, x_np: np.ndarray):
        """
        InfoNCE between anchor (t) and a nearby positive (t+delta).
        Returns (scalar_float, loss_Tensor) so gradients can flow back.
        """
        from engine.tensor import Tensor
        B, C, T = x_np.shape
        if T < self.contrastive_pos_window + 2:
            return 0.0, Tensor(np.array([[0.0]], dtype=np.float32))

        x_t = Tensor(x_np.astype(np.float32), requires_grad=False)
        _, aux = self.model(x_t, domain_idx=0)
        z = aux.get("h_states")  # (B, T, d_model) — Tensor, requires_grad=True through params
        if z is None or not z.requires_grad:
            return 0.0, Tensor(np.array([[0.0]], dtype=np.float32))

        z_np = z.data  # (B, T, d_model)

        # Sample one (anchor, positive) pair per batch item
        t_anchors = np.random.randint(0, T - self.contrastive_pos_window - 1, size=B)
        deltas    = np.random.randint(1, self.contrastive_pos_window + 1, size=B)
        t_pos_arr = t_anchors + deltas

        # Normalize all embeddings: (B, T, d_model)
        eps = 1e-8
        norms = np.linalg.norm(z_np, axis=-1, keepdims=True) + eps
        z_norm = z_np / norms   # (B, T, d_model)

        # Build anchor embeddings (B, d_model)
        anchors_np = np.array([z_norm[b, t_anchors[b]] for b in range(B)])

        # similarity scores: anchor vs all T timesteps  (B, T)
        # sim[b, t] = dot(anchor[b], z_norm[b, t])
        sim_np = np.einsum('bd,btd->bt', anchors_np, z_norm) / self.contrastive_temp

        # InfoNCE: -log(exp(sim_pos) / sum_t exp(sim))  per batch item
        sim_pos_np  = np.array([sim_np[b, t_pos_arr[b]] for b in range(B)])  # (B,)
        log_sum_exp = np.log(np.sum(np.exp(sim_np - sim_np.max(axis=-1, keepdims=True)),
                                    axis=-1) + eps) + sim_np.max(axis=-1)       # (B,)
        per_item    = -(sim_pos_np - log_sum_exp)                               # (B,)
        loss_scalar = float(per_item.mean())

        # Wrap as Tensor with gradient flowing back through z
        loss_t = Tensor(
            np.array([[loss_scalar]], dtype=np.float32),
            requires_grad=z.requires_grad,
            _children=(z,),
            _op="infonce"
        )

        _z = z; _sim_np = sim_np; _z_norm = z_norm; _norms = norms
        _anchors_np = anchors_np; _t_anchors = t_anchors; _t_pos_arr = t_pos_arr
        _temp = self.contrastive_temp

        def _infonce_back():
            if not _z.requires_grad or loss_t.grad is None:
                return
            g = float(loss_t.grad.sum()) / B
            # softmax over time for each batch item
            shifted = _sim_np - _sim_np.max(axis=-1, keepdims=True)
            exp_sim = np.exp(shifted)
            softmax_t = exp_sim / (exp_sim.sum(axis=-1, keepdims=True) + 1e-8)  # (B, T)

            # d(loss)/d(sim[b,t]) = softmax[b,t] - 1{t==t_pos[b]}
            d_sim = softmax_t.copy()  # (B, T)
            for b in range(B):
                d_sim[b, _t_pos_arr[b]] -= 1.0

            # d(sim[b,t])/d(z_norm[b,t2,d]) = anchor[b,d]/temp * 1{t2==t}
            # We need d(loss)/d(z.data[b, t, d])
            # sim[b,t] = sum_d anchor[b,d] * z_norm[b,t,d] / temp
            # so d(sim[b,t])/d(z[b,t,d]) ≈ anchor[b,d] / (norms[b,t]*temp)  (approx, ignore norm grad)
            contrib = np.zeros_like(_z.data)
            for b in range(B):
                # gradient from z_norm contribution as "keys"
                contrib[b] += (d_sim[b, :, None] * _anchors_np[b:b+1, :]) / (_norms[b] * _temp)
                # gradient from anchor contribution (anchor at t_anchors[b])
                d_anchor = (d_sim[b] @ _z_norm[b]) / _temp    # (d_model,)
                contrib[b, _t_anchors[b]] += d_anchor / (_norms[b, _t_anchors[b], 0] + 1e-8)

            contrib *= g
            _z.grad = _z.grad + contrib if _z.grad is not None else contrib

        loss_t._backward = _infonce_back
        return loss_scalar, loss_t

    def pretrain_step(self, x_np: np.ndarray) -> dict:
        """
        Run one pretraining step combining both objectives.
        x_np: (B, C, T) numpy float32
        Returns dict of losses.
        """
        self.model.train()
        self._zero_grad()

        recon_loss, recon_tensor = self.masked_reconstruction_loss(x_np)
        contrastive_loss = self.temporal_contrastive_loss(x_np)

        # Simple gradient step on reconstruction (contrastive is numpy-only)
        # For a full implementation, contrastive would also flow gradients
        if recon_loss > 0 and recon_tensor.data.item() > 0:
            recon_tensor.backward()
            self._sgd_step()

        self._step += 1

        return {
            "recon_loss": recon_loss,
            "contrastive_loss": contrastive_loss,
            "total_pretrain_loss": recon_loss + contrastive_loss,
            "step": self._step,
        }
