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


class ForecastHead(Module):
    """
    Projects the last latent step to H forecast steps across C channels.
    Takes z (B, T, d_model) → predictions (B, H, C).
    """

    def __init__(self, d_model: int, in_channels: int, horizon: int):
        super().__init__()
        self.horizon = horizon
        self.in_channels = in_channels
        self.proj = Linear(d_model, horizon * in_channels)

    def forward(self, z: Tensor) -> Tensor:
        # Use last timestep as the summary vector
        B, T, D = z.data.shape
        z_last = z.reshape(B * T, D)

        # Gradient-connected slice of last timestep
        last_np = z.data[:, -1, :]   # (B, D)
        last_t = Tensor(
            last_np.astype(np.float32),
            requires_grad=z.requires_grad,
            _children=(z,),
            _op="last_step_slice"
        )
        _z_ref = z
        def _last_back():
            if _z_ref.requires_grad and last_t.grad is not None:
                contrib = np.zeros_like(_z_ref.data)
                contrib[:, -1, :] = last_t.grad
                _z_ref.grad = _z_ref.grad + contrib if _z_ref.grad is not None else contrib
        last_t._backward = _last_back

        out = self.proj(last_t)       # (B, H*C)
        return out.reshape(B, self.horizon, self.in_channels)


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
        forecast_horizon: int = 8,
        lr: float = 1e-3,
    ):
        self.model = model
        self.in_channels = in_channels
        self.d_model = d_model
        self.mask_ratio_min = mask_ratio_min
        self.mask_ratio_max = mask_ratio_max
        self.contrastive_temp = contrastive_temp
        self.contrastive_pos_window = contrastive_pos_window
        self.forecast_horizon = forecast_horizon

        self.recon_head = MaskedReconstructionHead(d_model, in_channels)
        self.forecast_head = ForecastHead(d_model, in_channels, forecast_horizon)

        # Collect all parameters for optimizer
        self._params = (
            list(model.parameters())
            + list(self.recon_head.parameters())
            + list(self.forecast_head.parameters())
        )
        self.lr = lr
        self._step = 0
        self._optimizer = None

    def _zero_grad(self):
        for p in self._params:
            p.grad = None

    def _opt_step(self):
        if self._optimizer is None:
            from training.optimizer import SpectralAdamW
            self._optimizer = SpectralAdamW(self._params, lr=self.lr)
        self._optimizer.step()

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
        Run one pretraining step combining masked reconstruction + temporal contrastive.
        x_np: (B, C, T) numpy float32
        Returns dict of losses.
        """
        self.model.train()
        self._zero_grad()

        recon_scalar, recon_t = self.masked_reconstruction_loss(x_np)
        cont_scalar, cont_t = self.temporal_contrastive_loss(x_np)

        total = recon_scalar + cont_scalar

        # Backward through both objectives
        if recon_scalar > 0:
            recon_t.backward()
        if cont_scalar > 0:
            cont_t.backward()

        if total > 0:
            self._opt_step()

        self._step += 1
        return {
            "recon_loss": recon_scalar,
            "contrastive_loss": cont_scalar,
            "total_pretrain_loss": total,
            "step": self._step,
        }

    def forecast_pretrain_step(self, x_np: np.ndarray, horizon: Optional[int] = None) -> dict:
        """
        Forecasting pretraining: given context [0, T-H), predict [T-H, T).

        x_np  : (B, C, T) float32 — full window including the target horizon
        horizon: if None, uses self.forecast_horizon
        Returns dict with 'forecast_loss' and 'step'.
        """
        H = horizon if horizon is not None else self.forecast_horizon
        B, C, T = x_np.shape
        if T <= H:
            return {"forecast_loss": 0.0, "step": self._step}

        T_ctx = T - H
        x_ctx = x_np[:, :, :T_ctx]          # (B, C, T_ctx) — input to encoder
        x_tgt = x_np[:, :, T_ctx:]          # (B, C, H)     — ground truth

        self.model.train()
        self._zero_grad()

        x_t = Tensor(x_ctx.astype(np.float32), requires_grad=False)
        _, aux = self.model(x_t, domain_idx=0)
        z = aux.get("h_states")              # (B, T_ctx, d_model)

        if z is None:
            return {"forecast_loss": 0.0, "step": self._step}

        # Temporarily resize forecast head if horizon changed at runtime
        if H != self.forecast_head.horizon:
            from engine.layers import Linear as _Lin
            self.forecast_head.horizon = H
            self.forecast_head.proj = _Lin(self.d_model, H * self.in_channels)
            # refresh param list
            self._params = (
                list(self.model.parameters())
                + list(self.recon_head.parameters())
                + list(self.forecast_head.parameters())
            )
            self._optimizer = None

        pred = self.forecast_head(z)         # (B, H, C)

        # Target: (B, H, C)
        x_tgt_t = x_tgt.transpose(0, 2, 1)  # (B, H, C)
        pred_np = pred.data
        diff_np = pred_np - x_tgt_t
        loss_scalar = float((diff_np ** 2).mean())

        loss_t = Tensor(
            np.array([[loss_scalar]], dtype=np.float32),
            requires_grad=pred.requires_grad,
            _children=(pred,),
            _op="forecast_mse"
        )
        _pred = pred
        _denom = float(B * H * C)

        def _fcast_back():
            if _pred.requires_grad and loss_t.grad is not None:
                g = float(loss_t.grad.sum())
                _pred.grad = (
                    (_pred.grad + 2.0 * diff_np / _denom * g)
                    if _pred.grad is not None
                    else (2.0 * diff_np / _denom * g).astype(np.float32)
                )

        loss_t._backward = _fcast_back

        if loss_scalar > 0:
            loss_t.backward()
            self._opt_step()

        self._step += 1
        return {"forecast_loss": loss_scalar, "step": self._step}

    def channel_correlation_step(self, x_np: np.ndarray) -> dict:
        """
        Channel correlation pretraining: predict which channel pairs are correlated.

        For each batch item, compute the empirical correlation matrix C (C×C),
        then train the model to reconstruct C from the mean latent z̄ (d_model).
        This forces the encoder to capture inter-channel relationships.

        x_np: (B, C, T) float32
        Returns dict with 'corr_loss'.
        """
        B, C_ch, T = x_np.shape

        # Empirical correlation targets: (B, C, C)
        targets = np.zeros((B, C_ch, C_ch), dtype=np.float32)
        for b in range(B):
            x_b = x_np[b]                            # (C, T)
            mu = x_b.mean(axis=1, keepdims=True)
            x_c = x_b - mu
            std = x_c.std(axis=1, keepdims=True) + 1e-8
            x_n = x_c / std
            targets[b] = (x_n @ x_n.T) / T          # (C, C)

        self.model.train()
        self._zero_grad()

        x_t = Tensor(x_np.astype(np.float32), requires_grad=False)
        _, aux = self.model(x_t, domain_idx=0)
        z = aux.get("h_states")                      # (B, T, d_model)

        if z is None:
            return {"corr_loss": 0.0}

        # Mean-pool latent: (B, d_model)
        z_mean_np = z.data.mean(axis=1)
        z_mean = Tensor(
            z_mean_np.astype(np.float32),
            requires_grad=z.requires_grad,
            _children=(z,),
            _op="corr_pool"
        )
        _z_ref = z; _T = T

        def _corr_pool_back():
            if _z_ref.requires_grad and z_mean.grad is not None:
                contrib = np.broadcast_to(z_mean.grad[:, None, :] / _T, _z_ref.data.shape).copy()
                _z_ref.grad = _z_ref.grad + contrib if _z_ref.grad is not None else contrib

        z_mean._backward = _corr_pool_back

        # Project to C*C with a lazily-created head
        n_pairs = C_ch * C_ch
        if not hasattr(self, '_corr_proj') or self._corr_proj.weight.data.shape != (n_pairs, self.d_model):
            from engine.layers import Linear as _Lin
            self._corr_proj = _Lin(self.d_model, n_pairs)
            self._params += list(self._corr_proj.parameters())
            self._optimizer = None

        pred_flat = self._corr_proj(z_mean)          # (B, C*C)
        pred_corr = pred_flat.reshape(B, C_ch, C_ch) # (B, C, C)

        diff = pred_corr.data - targets
        loss_scalar = float((diff ** 2).mean())

        loss_t = Tensor(
            np.array([[loss_scalar]], dtype=np.float32),
            requires_grad=pred_flat.requires_grad,
            _children=(pred_flat,),
            _op="corr_mse"
        )
        _pf = pred_flat; _diff = diff; _denom = float(B * C_ch * C_ch)

        def _corr_back():
            if _pf.requires_grad and loss_t.grad is not None:
                g = float(loss_t.grad.sum())
                grad_flat = (2.0 * _diff / _denom * g).reshape(B, n_pairs).astype(np.float32)
                _pf.grad = _pf.grad + grad_flat if _pf.grad is not None else grad_flat

        loss_t._backward = _corr_back

        if loss_scalar > 0:
            loss_t.backward()
            self._opt_step()

        return {"corr_loss": loss_scalar}
