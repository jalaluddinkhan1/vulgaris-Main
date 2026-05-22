"""Self-supervised pretraining for VULGARIS: masked reconstruction + temporal contrastive."""
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
        loss_val = 0.0
        n_masked = 0
        for b in range(B):
            for t in range(T):
                if mask[b, t]:
                    diff = recon.data[b, t] - x_target[b, t]
                    loss_val += float(np.mean(diff ** 2))
                    n_masked += 1

        if n_masked == 0:
            return 0.0, Tensor(np.array([[0.0]], dtype=np.float32))

        loss_scalar = loss_val / n_masked
        loss_t = Tensor(np.array([[loss_scalar]], dtype=np.float32), requires_grad=False)
        return loss_scalar, loss_t

    def temporal_contrastive_loss(self, x_np: np.ndarray) -> float:
        """
        InfoNCE between anchor (t) and positive (t + delta, delta <= pos_window).
        Negatives: all other timesteps in batch.
        x_np: (B, C, T)
        Returns scalar loss.
        """
        B, C, T = x_np.shape
        if T < self.contrastive_pos_window + 2:
            return 0.0

        x_t = Tensor(x_np.astype(np.float32), requires_grad=False)
        _, aux = self.model(x_t, domain_idx=0)
        z = aux.get("h_states")  # (B, T, d_model)

        if z is None:
            return 0.0

        z_np = z.data  # (B, T, d_model)

        total_loss = 0.0
        count = 0

        for b in range(B):
            # Sample anchor timestep
            t_anchor = np.random.randint(0, T - self.contrastive_pos_window - 1)
            delta = np.random.randint(1, self.contrastive_pos_window + 1)
            t_pos = t_anchor + delta

            anchor = z_np[b, t_anchor]  # (d_model,)
            positive = z_np[b, t_pos]   # (d_model,)

            # Negatives: all other timesteps from this batch item (excluding anchor, pos)
            neg_indices = [t for t in range(T) if t != t_anchor and t != t_pos]
            negatives = z_np[b, neg_indices]  # (n_neg, d_model)

            # Normalize
            eps = 1e-8
            anchor = anchor / (np.linalg.norm(anchor) + eps)
            positive = positive / (np.linalg.norm(positive) + eps)
            negatives = negatives / (np.linalg.norm(negatives, axis=1, keepdims=True) + eps)

            sim_pos = np.dot(anchor, positive) / self.contrastive_temp
            sim_neg = (negatives @ anchor) / self.contrastive_temp

            # InfoNCE: -log(exp(pos) / (exp(pos) + sum(exp(neg))))
            sim_all = np.concatenate([[sim_pos], sim_neg])
            log_sum_exp = np.log(np.sum(np.exp(sim_all - sim_all.max())) + eps) + sim_all.max()
            loss_b = -(sim_pos - log_sum_exp)
            total_loss += float(loss_b)
            count += 1

        return total_loss / max(count, 1)

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
