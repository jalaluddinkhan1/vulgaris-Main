from __future__ import annotations

import numpy as np

from engine.tensor import Tensor
from engine.module import Module
from engine.layers import Linear, RMSNorm


class WeibullHead(Module):
    """
    Weibull Time-to-Failure prediction head for OT / predictive maintenance.

    Parameterises a Weibull distribution over failure time from the model's
    final hidden state.  Outputs scale λ and shape k, from which:

        Mean TTF  = λ · Γ(1 + 1/k)
        Median TTF = λ · (ln 2)^(1/k)
        Std TTF   = λ · sqrt(Γ(1 + 2/k) − Γ(1 + 1/k)²)

    Shape k interpretation
    ----------------------
    k < 1 : infant mortality / early-life failure
    k = 1 : constant failure rate (exponential distribution)
    k > 1 : wear-out failure (most mechanical components: k ≈ 2–4)

    Loss
    ----
    Supports both uncensored (failure observed) and right-censored
    (unit still running) samples:

        uncensored: L = -log f(t) = -log k + log λ − (k−1)·log(t/λ) + (t/λ)^k
        censored  : L = -log S(t) = (t/λ)^k

    Usage
    -----
        head = WeibullHead(d_model=256)
        log_lambda, log_k = head(h_last)         # h_last: (B, d_model)
        ttf_mean, ttf_std = head.predict_ttf(h_last)

        # Training
        loss = head.nll_loss(h_last, t_obs, censored_mask)
    """

    def __init__(self, d_model: int, hidden_dim: int | None = None):
        super().__init__()
        hd = hidden_dim or d_model // 2
        self.fc1        = Linear(d_model, hd)
        self.norm       = RMSNorm(hd)
        self.log_scale  = Linear(hd, 1)   # log(λ) — log of Weibull scale
        self.log_shape  = Linear(hd, 1)   # log(k)  — log of Weibull shape

    # ──────────────────────────────────────────────────────────────────────

    def forward(self, h: Tensor) -> tuple[Tensor, Tensor]:
        """
        h : (B, d_model) — last-timestep latent (z[:, -1, :])

        Returns
        -------
        log_lambda : (B, 1) Tensor — log scale parameter
        log_k      : (B, 1) Tensor — log shape parameter
        """
        shared     = self.norm(self.fc1(h).silu())
        log_lambda = self.log_scale(shared)   # (B, 1), unrestricted → λ = exp(.)
        log_k      = self.log_shape(shared)   # (B, 1), unrestricted → k = exp(.)
        return log_lambda, log_k

    # ──────────────────────────────────────────────────────────────────────

    def predict_ttf(
        self, h: Tensor
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns mean, median, std, and (λ, k) for inspection.

        All outputs are (B, 1) numpy arrays in the same time units as training.
        """
        from scipy.special import gamma as _gamma
        log_lambda, log_k = self.forward(h)
        lam = np.exp(log_lambda.data)   # (B, 1)
        k   = np.exp(log_k.data)        # (B, 1)

        mean_ttf   = lam * _gamma(1.0 + 1.0 / k)
        median_ttf = lam * (np.log(2) ** (1.0 / k))
        second_mom = lam ** 2 * _gamma(1.0 + 2.0 / k)
        std_ttf    = np.sqrt(np.maximum(second_mom - mean_ttf ** 2, 0.0))

        return mean_ttf, median_ttf, std_ttf, (lam, k)

    # ──────────────────────────────────────────────────────────────────────

    def nll_loss(
        self,
        h: Tensor,
        t_obs: np.ndarray,
        censored: np.ndarray | None = None,
    ) -> Tensor:
        """
        Weibull negative log-likelihood supporting right-censored observations.

        Parameters
        ----------
        h          : (B, d_model) — last-timestep latent
        t_obs      : (B,) or (B,1) — observed time (time-to-failure or
                     last-seen time for censored samples)
        censored   : (B,) bool — True = unit still running (right-censored).
                     None means all samples are uncensored.

        Returns
        -------
        Scalar Tensor — mean NLL over the batch.
        """
        log_lambda, log_k = self.forward(h)           # (B, 1) each

        t = np.asarray(t_obs, dtype=np.float32).reshape(-1, 1)
        t = np.maximum(t, 1e-6)                        # numerical safety

        # Convert to numpy for loss (backward wired through log_lambda, log_k)
        ll = log_lambda.data                           # (B, 1)
        lk = log_k.data                               # (B, 1)
        lam = np.exp(ll)
        k   = np.exp(lk)

        # Uncensored NLL: -log f(t) = log λ − log k − (k−1)log(t/λ) + (t/λ)^k
        t_over_lam = t / lam
        nll_uncensored = (ll - lk
                          - (k - 1.0) * np.log(t_over_lam + 1e-8)
                          + t_over_lam ** k)

        if censored is not None:
            c = np.asarray(censored, dtype=np.float32).reshape(-1, 1)
            # Censored NLL: -log S(t) = (t/λ)^k
            nll_censored = t_over_lam ** k
            nll_np = c * nll_censored + (1.0 - c) * nll_uncensored
        else:
            nll_np = nll_uncensored

        nll_mean = float(nll_np.mean())
        loss = Tensor(
            np.array([[nll_mean]], dtype=np.float32),
            requires_grad=log_lambda.requires_grad or log_k.requires_grad,
            _children=(log_lambda, log_k),
            _op="weibull_nll",
        )
        _ll, _lk = log_lambda, log_k
        _t, _lam, _k = t, lam, k
        _t_ol = t_over_lam
        _cens  = (np.asarray(censored, dtype=np.float32).reshape(-1, 1)
                  if censored is not None else None)
        _B = h.data.shape[0]

        def _weibull_back():
            if loss.grad is None:
                return
            g = float(loss.grad.sum()) / _B

            # d(NLL_uncens)/d(log_lambda):
            #   d/d(ll) [ ll - lk - (k-1)*log(t/lam) + (t/lam)^k ]
            #   = 1 + (k-1) + k*(t/lam)^k        (via chain rule through lam=exp(ll))
            #   = k + k*(t/lam)^k - 1 + 1 = k*(1 + (t/lam)^k)
            dll_unc = _k * (1.0 + _t_ol ** _k)

            # d(NLL_uncens)/d(log_k):
            #   d/d(lk) [ -lk - (k-1)*log(t/lam) + (t/lam)^k ]
            #   = -1 - k*log(t/lam) + k*log(t/lam)*(t/lam)^k    (chain through k=exp(lk))
            #   = -1 + k*log(t/lam)*((t/lam)^k - 1)
            log_tol = np.log(_t_ol + 1e-8)
            dlk_unc = -1.0 + _k * log_tol * (_t_ol ** _k - 1.0)

            if _cens is not None:
                # d(NLL_cens)/d(log_lambda) = k*(t/lam)^k
                dll_cens = _k * _t_ol ** _k
                # d(NLL_cens)/d(log_k)  = k*(t/lam)^k * log(t/lam)
                dlk_cens = _k * _t_ol ** _k * log_tol

                dll = _cens * dll_cens + (1.0 - _cens) * dll_unc
                dlk = _cens * dlk_cens + (1.0 - _cens) * dlk_unc
            else:
                dll = dll_unc
                dlk = dlk_unc

            if _ll.requires_grad:
                g_ll = (dll * g).astype(np.float32)
                _ll.grad = _ll.grad + g_ll if _ll.grad is not None else g_ll
            if _lk.requires_grad:
                g_lk = (dlk * g).astype(np.float32)
                _lk.grad = _lk.grad + g_lk if _lk.grad is not None else g_lk

        loss._backward = _weibull_back
        return loss
