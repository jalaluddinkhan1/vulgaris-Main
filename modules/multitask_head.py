"""Multi-task output head for VULGARIS."""
import numpy as np
from typing import Tuple

from engine.tensor import Tensor
from engine.layers import Linear, RMSNorm
from engine.module import Module


class MultiTaskHead(Module):
    """
    Four simultaneous outputs from latent representation:
      - forecast:      (B, H, C)    future H steps over C channels
      - anomaly_score: (B, 1)       scalar anomaly probability
      - class_probs:   (B, n_classes) softmax classification
      - uncertainty:   (B, 1)       aleatoric uncertainty estimate

    Usage:
        head = MultiTaskHead(d_model=64, n_classes=5, forecast_horizon=10, n_channels=9)
        forecast, anomaly, class_probs, uncertainty = head(z)  # z: (B, T, d_model)
    """

    def __init__(
        self,
        d_model: int,
        n_classes: int,
        forecast_horizon: int = 10,
        n_channels: int = 1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_classes = n_classes
        self.forecast_horizon = forecast_horizon
        self.n_channels = n_channels

        # Shared backbone norm
        self.norm = RMSNorm(d_model)

        # Forecast head: pooled latent → (H * C) → reshape
        self.forecast_proj = Linear(d_model, forecast_horizon * n_channels)

        # Anomaly head: single sigmoid output
        self.anomaly_proj = Linear(d_model, 1)

        # Classification head
        self.class_proj = Linear(d_model, n_classes)

        # Uncertainty head: log-variance output
        self.uncertainty_proj = Linear(d_model, 1)

    def forward(self, z: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        z: (B, T, d_model)
        Returns: (forecast, anomaly_score, class_probs, uncertainty)
        """
        # Pool over time: mean of last 25% of timesteps (recent context)
        B, T, D = z.data.shape
        recent_start = max(0, T - max(1, T // 4))

        # Mean pool recent timesteps
        T_recent = z.data.shape[1] - recent_start
        z_pool_np = z.data[:, recent_start:, :].mean(axis=1)   # (B, d_model)
        z_pooled = Tensor(
            z_pool_np,
            requires_grad=z.requires_grad,
            _children=(z,),
            _op="recent_mean_pool"
        )
        _z_ref = z
        _recent_start = recent_start
        _T_recent = T_recent

        def _pool_back():
            if _z_ref.requires_grad and z_pooled.grad is not None:
                contrib = np.zeros_like(_z_ref.data)
                contrib[:, _recent_start:, :] = z_pooled.grad[:, None, :] / _T_recent
                _z_ref.grad = _z_ref.grad + contrib if _z_ref.grad is not None else contrib

        z_pooled._backward = _pool_back

        # Normalize
        z_n = self.norm(z_pooled)

        # Forecast
        fc = self.forecast_proj(z_n)  # (B, H*C)
        H, C = self.forecast_horizon, self.n_channels
        fc_np = fc.data.reshape(B, H, C)
        forecast = Tensor(fc_np, requires_grad=fc.requires_grad, _children=(fc,), _op="fc_reshape")

        def _fc_back():
            if fc.requires_grad and forecast.grad is not None:
                fc.grad = fc.grad + forecast.grad.reshape(B, H * C) if fc.grad is not None else forecast.grad.reshape(B, H * C)
        forecast._backward = _fc_back

        # Anomaly: sigmoid
        a_raw = self.anomaly_proj(z_n)  # (B, 1)
        anomaly_np = 1.0 / (1.0 + np.exp(-np.clip(a_raw.data, -30, 30)))
        anomaly_score = Tensor(anomaly_np, requires_grad=a_raw.requires_grad, _children=(a_raw,), _op="sigmoid_anomaly")

        def _anomaly_back():
            if a_raw.requires_grad and anomaly_score.grad is not None:
                d = anomaly_np * (1.0 - anomaly_np)
                a_raw.grad = a_raw.grad + anomaly_score.grad * d if a_raw.grad is not None else anomaly_score.grad * d
        anomaly_score._backward = _anomaly_back

        # Classification: softmax
        c_raw = self.class_proj(z_n)  # (B, n_classes)
        class_probs = c_raw.softmax(axis=-1)

        # Uncertainty: softplus of log-variance
        u_raw = self.uncertainty_proj(z_n)  # (B, 1)
        u_np = np.log1p(np.exp(np.clip(u_raw.data, -20, 20)))  # softplus
        uncertainty = Tensor(u_np, requires_grad=u_raw.requires_grad, _children=(u_raw,), _op="softplus_unc")

        def _unc_back():
            if u_raw.requires_grad and uncertainty.grad is not None:
                d = 1.0 / (1.0 + np.exp(-np.clip(u_raw.data, -20, 20)))
                u_raw.grad = u_raw.grad + uncertainty.grad * d if u_raw.grad is not None else uncertainty.grad * d
        uncertainty._backward = _unc_back

        return forecast, anomaly_score, class_probs, uncertainty
