import numpy as np
from typing import Dict, Tuple

from engine.tensor import Tensor
from engine.module import Module
from .conformal import NonStationaryConformal


class VulgarisLoss(Module):
    """
    Unified Variational Loss combining all module-level objectives.

    L = L_task + β*L_memory + γ*L_dag + δ*L_ewc + ε*L_conformal
      + ζ*L_cbf + η*L_temporal + θ*L_contrastive
    """

    def __init__(self, config, task: str = None):
        super().__init__()
        if task is None:
            task = 'classification' if getattr(config, 'n_classes', 0) > 0 else 'regression'
        self.task = task

        tc = config.training

        self.beta  = tc.beta_hmb           # memory compression weight
        self.gamma = tc.gamma_crg          # DAG sparsity + acyclicity weight
        self.delta = tc.delta_ewc          # EWC elastic weight consolidation
        self.eps   = tc.epsilon_conformal  # conformal coverage gap weight
        self.zeta  = 0.1                   # CBF safety constraint violations
        self.eta   = 0.01                  # temporal coherence
        self.theta = 0.01                  # CMLA contrastive alignment

        self.conformal = NonStationaryConformal(
            alpha=tc.epsilon_conformal,
            forgetting_factor=0.01,
            max_calibration_size=2000,
        )
        self.lambda_tfc  = 0.01   # Time-Frequency Consistency weight
        self.lambda_mae  = 0.1    # MAE reconstruction weight
        self.lambda_rmc  = 0.01   # Regime Mixture Core load-balancing weight

    # ──────────────────────────────────────────────────────────────────────

    def tfc_loss(self, h_time: Tensor, h_freq: Tensor,
                 temperature: float = 0.1) -> Tensor:
        """
        Time-Frequency Consistency (TF-C) contrastive loss.

        Encodes two views of the same signal — one from the time domain
        (h_time) and one from a frequency-augmented view (h_freq) — and
        maximises their cosine similarity (NT-Xent in a batch of B pairs).

        h_time, h_freq: (B, d_model)  — mean-pooled latent representations

        L_TFC = -1/B * sum_i log [ exp(sim(t_i,f_i)/T) /
                                   sum_j exp(sim(t_i,f_j)/T) ]

        Reference: Zhang et al. 2022 — "Self-Supervised Contrastive Pre-Training
        for Time Series via Time-Frequency Consistency"
        """
        B = h_time.data.shape[0]
        if B < 2:
            return Tensor(np.array([[0.0]]), requires_grad=False)

        # L2-normalise both views
        norm_t = np.linalg.norm(h_time.data, axis=-1, keepdims=True) + 1e-8
        norm_f = np.linalg.norm(h_freq.data, axis=-1, keepdims=True) + 1e-8
        z_t = h_time.data / norm_t   # (B, d_model)
        z_f = h_freq.data / norm_f   # (B, d_model)

        # Similarity matrix: (B, B) — sim(t_i, f_j)
        sim = (z_t @ z_f.T) / temperature   # (B, B)

        # NT-Xent: each diagonal is the positive pair
        sim_max = sim.max(axis=-1, keepdims=True)
        exp_sim = np.exp(sim - sim_max)
        log_sum_exp = np.log(exp_sim.sum(axis=-1, keepdims=True) + 1e-12) + sim_max
        diag_sim = sim[np.arange(B), np.arange(B)]
        loss_val = float((-diag_sim + log_sum_exp.squeeze(-1)).mean())

        tfc_t = Tensor(
            np.array([[loss_val]], dtype=np.float64),
            requires_grad=h_time.requires_grad or h_freq.requires_grad,
            _children=(h_time, h_freq),
            _op="tfc_loss"
        )

        _h_t, _h_f = h_time, h_freq
        _z_t, _z_f = z_t, z_f
        _sim, _exp_sim, _log_sum_exp, _B = sim, exp_sim, log_sum_exp, B
        _norm_t, _norm_f = norm_t, norm_f
        _temperature = temperature

        def _tfc_back():
            if tfc_t.grad is None:
                return
            g_scale = float(tfc_t.grad.sum()) / _B

            # d_loss / d_sim[i,j] = (-1[i==j] + softmax[i,j]) / B
            softmax = _exp_sim / (_exp_sim.sum(axis=-1, keepdims=True) + 1e-12)
            d_sim = softmax.copy()
            d_sim[np.arange(_B), np.arange(_B)] -= 1.0
            d_sim = d_sim * g_scale / _temperature  # (B, B)

            # Gradient of sim = z_t @ z_f.T w.r.t. z_t and z_f (normalised)
            d_z_t = d_sim @ _z_f       # (B, d_model)
            d_z_f = d_sim.T @ _z_t    # (B, d_model)

            # Chain through L2-normalisation: d/dx (x/||x||) = (I - xx^T/||x||^2)/||x||
            def _norm_grad(x_raw, z_norm, d_z, norms):
                n2 = (norms ** 2)
                # d_x = (d_z - (d_z * z_norm).sum(-1, keepdims=True) * z_norm) / norms
                proj = (d_z * z_norm).sum(axis=-1, keepdims=True) * z_norm
                return (d_z - proj) / norms

            if _h_t.requires_grad:
                d_ht = _norm_grad(_h_t.data, _z_t, d_z_t, _norm_t)
                _h_t.grad = _h_t.grad + d_ht if _h_t.grad is not None else d_ht
            if _h_f.requires_grad:
                d_hf = _norm_grad(_h_f.data, _z_f, d_z_f, _norm_f)
                _h_f.grad = _h_f.grad + d_hf if _h_f.grad is not None else d_hf

        tfc_t._backward = _tfc_back
        return tfc_t

    # ──────────────────────────────────────────────────────────────────────

    def task_loss(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """MSE for regression; cross-entropy for classification."""
        if self.task == 'classification':
            batch = y_pred.shape[0]
            y_idx = y_true.data.astype(int).flatten()
            probs = np.clip(y_pred.data, 1e-12, 1.0)
            ce_vals = -np.log(probs[np.arange(batch), y_idx])
            ce_scalar = float(ce_vals.mean())

            ce_t = Tensor(
                np.array([[ce_scalar]], dtype=np.float64),
                requires_grad=y_pred.requires_grad,
                _children=(y_pred,),
                _op="cross_entropy"
            )

            _probs = probs.copy()
            _y_idx = y_idx.copy()
            _batch = batch

            def _ce_back():
                if y_pred.requires_grad and ce_t.grad is not None:
                    g_scalar = float(ce_t.grad.sum())
                    # Gradient of cross-entropy through softmax output:
                    # d(-log p_c)/d p_j = -1/p_c if j==c, 0 otherwise
                    g = np.zeros_like(_probs)
                    g[np.arange(_batch), _y_idx] = -1.0 / (
                        _probs[np.arange(_batch), _y_idx] + 1e-12
                    )
                    g = g / _batch * g_scalar
                    y_pred.grad = y_pred.grad + g if y_pred.grad is not None else g

            ce_t._backward = _ce_back
            return ce_t

        else:
            # Regression: MSE
            diff = y_pred - y_true
            return (diff * diff).mean()

    # ──────────────────────────────────────────────────────────────────────

    def temporal_coherence_loss(self, h_states: Tensor) -> Tensor:
        """
        Penalise large jumps in hidden state: mean ||h_t - h_{t-1}||^2.
        h_states: (batch, T, d_model)
        """
        T = h_states.shape[1]
        if T < 2:
            return Tensor(np.array([[0.0]]), requires_grad=False)

        # Extract consecutive pairs via numpy slice, build Tensors with grad linkage
        h_curr_data = h_states.data[:, 1:, :]
        h_prev_data = h_states.data[:, :-1, :]

        h_curr = Tensor(
            h_curr_data,
            requires_grad=h_states.requires_grad,
            _children=(h_states,),
            _op="htc_curr"
        )
        h_prev = Tensor(
            h_prev_data,
            requires_grad=h_states.requires_grad,
            _children=(h_states,),
            _op="htc_prev"
        )

        _hs = h_states

        def _curr_back():
            if _hs.requires_grad and h_curr.grad is not None:
                contrib = np.zeros_like(_hs.data)
                contrib[:, 1:, :] += h_curr.grad
                _hs.grad = _hs.grad + contrib if _hs.grad is not None else contrib

        def _prev_back():
            if _hs.requires_grad and h_prev.grad is not None:
                contrib = np.zeros_like(_hs.data)
                contrib[:, :-1, :] += h_prev.grad
                _hs.grad = _hs.grad + contrib if _hs.grad is not None else contrib

        h_curr._backward = _curr_back
        h_prev._backward = _prev_back

        diff = h_curr - h_prev
        return (diff * diff).mean()

    # ──────────────────────────────────────────────────────────────────────

    def forward(
        self, y_pred: Tensor, y_true: Tensor, aux: dict
    ) -> Tuple[Tensor, dict]:
        """
        Compute unified variational loss.

        aux keys:
            dag_penalty      : float
            memory_loss      : float
            cbf_loss         : float
            h_states         : Tensor (B, T, d_model)
            cmla_loss_tensor : Tensor scalar
            ewc_loss         : float or Tensor (optional)
            y_pred_sigma     : np.ndarray (optional)

        Returns (total_loss Tensor, components dict of floats for logging).
        """
        components: dict = {}

        # ── Task loss ────────────────────────────────────────────────────
        l_task = self.task_loss(y_pred, y_true)
        components["task_loss"] = float(l_task.data.sum())

        # ── Memory (HMB) ─────────────────────────────────────────────────
        mem_val = float(aux.get("memory_loss", 0.0))
        l_mem = Tensor(np.array([[mem_val]]), requires_grad=False)
        components["memory_loss"] = mem_val

        # ── DAG penalty (CRG) ─────────────────────────────────────────────
        dag_val = float(aux.get("dag_penalty", 0.0))
        l_dag = Tensor(np.array([[dag_val]]), requires_grad=False)
        components["dag_penalty"] = dag_val

        # ── EWC (SHCAL) ──────────────────────────────────────────────────
        ewc_raw = aux.get("ewc_loss", 0.0)
        if isinstance(ewc_raw, Tensor):
            l_ewc = ewc_raw
            components["ewc_loss"] = float(ewc_raw.data.sum())
        else:
            l_ewc = Tensor(np.array([[float(ewc_raw)]]), requires_grad=False)
            components["ewc_loss"] = float(ewc_raw)

        # ── Conformal coverage gap ────────────────────────────────────────
        sigma_np = aux.get("y_pred_sigma", None)
        if sigma_np is not None:
            yp = y_pred.data.flatten()
            yt = y_true.data.flatten()
            sig = np.asarray(sigma_np, dtype=np.float64).flatten()
            n = min(len(yp), len(yt), len(sig))
            cov_gap = self.conformal.coverage_loss(yp[:n], yt[:n], sig[:n])
        else:
            cov_gap = 0.0
        l_conf = Tensor(np.array([[cov_gap]]), requires_grad=False)
        components["conformal_loss"] = cov_gap

        # ── CBF safety ───────────────────────────────────────────────────
        cbf_val = float(aux.get("cbf_loss", 0.0))
        l_cbf = Tensor(np.array([[cbf_val]]), requires_grad=False)
        components["cbf_loss"] = cbf_val

        # ── Temporal coherence (HTD) ──────────────────────────────────────
        h_states = aux.get("h_states", None)
        if h_states is not None and isinstance(h_states, Tensor) and h_states.ndim == 3:
            l_temporal = self.temporal_coherence_loss(h_states)
        else:
            l_temporal = Tensor(np.array([[0.0]]), requires_grad=False)
        components["temporal_loss"] = float(l_temporal.data.sum())

        # ── CMLA contrastive ─────────────────────────────────────────────
        cmla_t = aux.get("cmla_loss_tensor", None)
        if cmla_t is not None and isinstance(cmla_t, Tensor):
            l_contrastive = cmla_t
            components["cmla_loss"] = float(cmla_t.data.sum())
        else:
            l_contrastive = Tensor(np.array([[0.0]]), requires_grad=False)
            components["cmla_loss"] = 0.0

        # ── TF-C: Time-Frequency Consistency ─────────────────────────────
        # aux["tfc_pair"] = (h_time, h_freq) — mean-pooled latents from
        # time-domain and frequency-augmented views of the same batch.
        # Computed in the training pipeline before the forward pass.
        tfc_pair = aux.get("tfc_pair", None)
        if tfc_pair is not None:
            h_time, h_freq = tfc_pair
            l_tfc = self.tfc_loss(h_time, h_freq)
            components["tfc_loss"] = float(l_tfc.data.sum())
        else:
            l_tfc = Tensor(np.array([[0.0]]), requires_grad=False)
            components["tfc_loss"] = 0.0

        # ── MAE reconstruction ────────────────────────────────────────────
        # aux["mae_loss"] = scalar Tensor from model.mae_forward().
        mae_raw = aux.get("mae_loss", None)
        if mae_raw is not None and isinstance(mae_raw, Tensor):
            l_mae = mae_raw
            components["mae_loss"] = float(mae_raw.data.sum())
        else:
            l_mae = Tensor(np.array([[0.0]]), requires_grad=False)
            components["mae_loss"] = 0.0

        # ── RMC load-balancing ────────────────────────────────────────────
        # aux["rmc_balance_loss"] written by Vulgaris.forward().
        rmc_raw = aux.get("rmc_balance_loss", None)
        if isinstance(rmc_raw, (int, float)):
            l_rmc = Tensor(np.array([[float(rmc_raw)]], dtype=np.float64),
                           requires_grad=False)
            components["rmc_balance_loss"] = float(rmc_raw)
        else:
            l_rmc = Tensor(np.array([[0.0]]), requires_grad=False)
            components["rmc_balance_loss"] = 0.0

        # ── Total ─────────────────────────────────────────────────────────
        total = (
            l_task
            + l_mem         * self.beta
            + l_dag         * self.gamma
            + l_ewc         * self.delta
            + l_conf        * self.eps
            + l_cbf         * self.zeta
            + l_temporal    * self.eta
            + l_contrastive * self.theta
            + l_tfc         * self.lambda_tfc
            + l_mae         * self.lambda_mae
            + l_rmc         * self.lambda_rmc
        )
        components["total_loss"] = float(total.data.sum())

        return total, components

    # ──────────────────────────────────────────────────────────────────────

    def update_conformal(
        self, y_pred_np: np.ndarray, y_true_np: np.ndarray, sigma_np: np.ndarray
    ):
        """Pass new observations to the conformal predictor for calibration update."""
        self.conformal.update(y_pred_np, y_true_np, sigma_np)

    def predict_interval(self, y_pred_np: np.ndarray, sigma_np: np.ndarray):
        """Returns (lower, upper) prediction interval arrays."""
        return self.conformal.predict_interval(y_pred_np, sigma_np)
