from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np

from engine.tensor import Tensor, Parameter, zeros
from engine.module import Module
from engine.layers import Linear, RMSNorm

if TYPE_CHECKING:
    from config import CMLAConfig


class ModalityEncoder(Module):
    """
    Two-layer MLP encoder for a single modality: d_in -> d_model.
    Architecture: Linear -> SiLU -> Linear -> RMSNorm
    Produces L2-normalised embeddings.
    """

    def __init__(self, d_in: int, d_model: int):
        super().__init__()
        d_hidden = max(d_in * 2, d_model)
        self.fc1 = Linear(d_in, d_hidden)
        self.fc2 = Linear(d_hidden, d_model)
        self.norm = RMSNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        """
        x: (..., d_in)
        Returns: (..., d_model)  L2-normalised
        """
        h = self.fc1(x).silu()
        h = self.fc2(h)
        h = self.norm(h)
        # L2 normalise along last dim
        n = h.norm(p=2, axis=-1, keepdims=True)  # (..., 1)
        z = h / (n + 1e-8)
        return z


class CrossModalLatentAlignment(Module):
    """
    Projects M heterogeneous sensor modalities into a shared latent space.
    Uses InfoNCE contrastive loss and inverse-variance weighted fusion.

    Maintains a running alignment matrix Omega[i,j] = cosine_sim(E_i, E_j).
    """

    def __init__(self, modality_dims: list[int], d_model: int,
                 temperature: float = 0.07,
                 config: "CMLAConfig | None" = None):
        """
        config : optional CMLAConfig — when provided, its contrastive_temp and
                 contrastive_weight fields override the temperature argument.
        """
        super().__init__()
        self.n_modalities = len(modality_dims)
        self.d_model = d_model
        # CMLAConfig takes precedence over the bare temperature kwarg
        if config is not None:
            self.temperature          = config.contrastive_temp
            self.contrastive_weight   = config.contrastive_weight
        else:
            self.temperature          = temperature
            self.contrastive_weight   = 0.1

        # One encoder per modality
        for m, d_in in enumerate(modality_dims):
            setattr(self, f"encoder_{m}", ModalityEncoder(d_in, d_model))

        # Running alignment matrix (numpy, not a Parameter — diagnostics only)
        self.alignment_matrix = np.zeros(
            (self.n_modalities, self.n_modalities), dtype=np.float64
        )
        self._align_momentum = 0.99
        self._align_count = 0

    def _get_encoder(self, m: int) -> ModalityEncoder:
        return getattr(self, f"encoder_{m}")

    # ------------------------------------------------------------------
    def encode(self, modality_idx: int, x: Tensor) -> Tensor:
        """
        Encode one modality.
        x: (batch, T, d_in) or (batch, d_in)
        Returns normalised embedding z: same shape with last dim = d_model
        """
        return self._get_encoder(modality_idx)(x)

    # ------------------------------------------------------------------
    def _update_alignment(self, embeddings: list[Tensor]):
        """Update running cosine-similarity alignment matrix (numpy side)."""
        M = self.n_modalities
        # Compute mean embedding per modality over batch*T
        means = []
        for z in embeddings:
            z_np = z.data.reshape(-1, self.d_model)   # (B*T, D)
            mu = z_np.mean(axis=0)                     # (D,)
            norm = np.linalg.norm(mu) + 1e-8
            means.append(mu / norm)

        new_align = np.zeros((M, M), dtype=np.float64)
        for i in range(M):
            for j in range(M):
                new_align[i, j] = float(np.dot(means[i], means[j]))

        alpha = self._align_momentum
        self.alignment_matrix = alpha * self.alignment_matrix + (1.0 - alpha) * new_align
        self._align_count += 1

    # ------------------------------------------------------------------
    def fuse(self, embeddings: list[Tensor]) -> tuple[Tensor, list[Tensor]]:
        """
        Inverse-variance weighted fusion of modality embeddings.

        embeddings[m]: (batch, T, d_model)

        Uncertainty per modality:
            u_m = ||z_m - z_mean||^2 / (D * sigma^2_base)
        where z_mean is the simple mean over modalities.

        Returns (z_fused, uncertainties)
            z_fused     : (batch, T, d_model)
            uncertainties: list of M Tensor scalars per batch*T (as (B,T) tensors)
        """
        M = self.n_modalities
        D = self.d_model
        sigma2_base = 1.0  # normalisation constant

        # Stack embeddings: (M, B, T, D)
        z_stack_np = np.stack([z.data for z in embeddings], axis=0)  # (M,B,T,D)
        z_mean_np = z_stack_np.mean(axis=0)                           # (B,T,D)

        uncertainties_np = []
        weights_np = []
        for m in range(M):
            diff = z_stack_np[m] - z_mean_np   # (B,T,D)
            u_m = (diff ** 2).sum(axis=-1) / (D * sigma2_base)  # (B,T)
            u_m = np.clip(u_m, 1e-6, None)
            uncertainties_np.append(u_m)
            weights_np.append(1.0 / u_m)       # inverse-variance weight (B,T)

        w_sum = sum(weights_np)   # (B,T)
        # Normalised weights per modality: (B,T)
        norm_weights = [w / w_sum for w in weights_np]

        # Weighted sum of embeddings
        # z_fused: (B,T,D) = sum_m w_m[:,:,None] * z_m
        fused_np = sum(norm_weights[m][:, :, None] * z_stack_np[m]
                       for m in range(M))   # (B,T,D)

        # Build Tensor with gradient support through embeddings
        rg = any(z.requires_grad for z in embeddings)
        fused = Tensor(
            fused_np,
            requires_grad=rg,
            _children=tuple(embeddings),
            _op="cmla_fuse"
        )

        # Backward: dL/dz_m = dL/dfused * w_m_norm (simplified — treat weights as const)
        _norm_weights = [nw.copy() for nw in norm_weights]

        def _fuse_back():
            if fused.grad is None:
                return
            g = fused.grad   # (B,T,D)
            for m_idx, z_m in enumerate(embeddings):
                if z_m.requires_grad:
                    # grad w.r.t. z_m treating weights as constant
                    contrib = g * _norm_weights[m_idx][:, :, None]
                    z_m.grad = z_m.grad + contrib if z_m.grad is not None else contrib

        fused._backward = _fuse_back

        # Wrap uncertainties as Tensors (no grad — diagnostic only)
        unc_tensors = [Tensor(u, requires_grad=False) for u in uncertainties_np]

        return fused, unc_tensors

    # ------------------------------------------------------------------
    def contrastive_loss(self, embeddings: list[Tensor]) -> Tensor:
        """
        InfoNCE loss across modalities.

        embeddings[m]: (B, T, D)  L2-normalised

        Treats each (modality, timestep) pair as an anchor.
        Positive: same timestep t, any different modality m' != m.
        Negatives: all (modality, timestep) pairs except anchor itself.

        We flatten to (M*B*T, D) and compute the full N x N similarity
        matrix, then extract the loss.

        For a batch of B samples and M modalities:
            N = M * B * T  (total instances)
            sim[i,j] = dot(z_i, z_j) / temperature
            For instance (m, b, t):
                positives = instances (m', b, t) for m' != m  [same b,t different m]
                loss_i = -log [sum_pos exp(sim[i,pos]) / sum_{j!=i} exp(sim[i,j])]

        We use the NT-Xent (multiple-positive) formulation.

        Returns scalar loss Tensor.
        """
        M = self.n_modalities
        B, T, D = embeddings[0].shape

        # ── Timestep sub-sampling to prevent OOM ─────────────────────────
        # Full N = M*B*T.  At T=1024, B=32, M=4 → N=131,072 → sim is 137 GB.
        # Sub-sample k timesteps per modality per batch item.
        # k=64 is sufficient for InfoNCE to converge (Liu et al. 2021).
        max_t = getattr(self, "max_contrastive_timesteps", 64)
        if T > max_t:
            t_idx = np.random.choice(T, size=max_t, replace=False)
            z_sub = [z.data[:, t_idx, :]   for z in embeddings]   # each (B, max_t, D)
            T_use = max_t
        else:
            z_sub = [z.data for z in embeddings]
            T_use = T

        # Flatten: (M*B*T_use, D)
        z_list_np = [zs.reshape(B * T_use, D) for zs in z_sub]
        z_all_np  = np.concatenate(z_list_np, axis=0)             # (N, D)
        N  = M * B * T_use
        BT = B * T_use

        # Similarity matrix: (N, N) — safe at T_use=64, B=32, M=4: N=8192, ~0.5 GB
        sim_np = (z_all_np @ z_all_np.T) / self.temperature       # (N, N)

        # Build positive mask: instance i=(m,bt) is positive with j=(m',bt) for m'!=m, same bt
        # index i = m * BT + bt  where bt in [0, BT)
        BT = B * T_use   # use sub-sampled length
        pos_mask = np.zeros((N, N), dtype=np.float64)
        for m_i in range(M):
            for m_j in range(M):
                if m_i == m_j:
                    continue
                # All bt indices on diagonal blocks
                for bt in range(BT):
                    i = m_i * BT + bt
                    j = m_j * BT + bt
                    pos_mask[i, j] = 1.0

        # Self-mask: exclude diagonal
        self_mask = np.eye(N, dtype=np.float64)

        # Numerically stable logsumexp for denominator (all j != i)
        # denom_mask: (N, N)  1 where j != i
        denom_mask = 1.0 - self_mask   # (N, N)

        # Shift for numerical stability
        sim_max = sim_np.max(axis=1, keepdims=True)  # (N, 1)
        exp_sim = np.exp(sim_np - sim_max)            # (N, N)

        # For each anchor i: sum over positives and all non-self
        sum_pos = (exp_sim * pos_mask).sum(axis=1)              # (N,)
        sum_denom = (exp_sim * denom_mask).sum(axis=1)          # (N,)
        n_pos = pos_mask.sum(axis=1)                            # (N,) number of positives per anchor

        # Loss per anchor (only where n_pos > 0)
        valid = n_pos > 0
        # log( sum_pos / sum_denom ) = log(sum_pos) - log(sum_denom)
        loss_per = -np.log(np.clip(sum_pos, 1e-12, None) / np.clip(sum_denom, 1e-12, None))
        loss_np = loss_per[valid].mean() if valid.any() else np.float64(0.0)

        rg = any(z.requires_grad for z in embeddings)
        loss_t = Tensor(
            np.array([[loss_np]]),
            requires_grad=rg,
            _children=tuple(embeddings),
            _op="infonce"
        )

        # Backward through InfoNCE
        _z_all    = z_all_np.copy()
        _sim      = sim_np.copy()
        _exp_sim  = exp_sim.copy()
        _pos_mask = pos_mask.copy()
        _denom_mask = denom_mask.copy()
        _sum_pos   = sum_pos.copy()
        _sum_denom = sum_denom.copy()
        _n_pos     = n_pos.copy()
        _valid     = valid.copy()
        _N_valid   = valid.sum()
        _BT        = BT      # B * T_use (sub-sampled)
        _T_use     = T_use
        _T_full    = T       # original full T for scatter-back
        _t_idx     = t_idx if T > max_t else None   # sub-sample indices or None
        _M = M
        _N = N

        def _infonce_back():
            if not rg or loss_t.grad is None:
                return
            g_scalar = float(loss_t.grad.sum())

            # Gradient of loss w.r.t. sim[i,j]
            # loss = (1/N_valid) * sum_i [ -log(sum_pos_i) + log(sum_denom_i) ]
            # d loss / d sim[i,j]:
            #   If j is positive for i (pos_mask[i,j]=1):
            #       d/d sim[i,j] of -log(sum_pos) = -exp(sim[i,j]) / sum_pos[i]
            #   For denominator term:
            #       d/d sim[i,j] of log(sum_denom) = exp(sim[i,j]) / sum_denom[i]  if j!=i
            # Combined:
            #   dL/d sim[i,j] = (1/N_valid) * [
            #       - pos_mask[i,j] * exp_sim[i,j] / sum_pos[i]     (pos numerator)
            #       + denom_mask[i,j] * exp_sim[i,j] / sum_denom[i] (denominator)
            #   ]  for valid anchors i

            n_valid = max(_N_valid, 1)
            # Vectorised (N, N) gradient — replaces O(N²) Python loop.
            # Only valid rows contribute; invalid rows are zeroed via valid_mask.
            valid_mask = _valid[:, None].astype(np.float64)   # (N, 1)
            d_sim = (
                - _pos_mask   * _exp_sim / (_sum_pos[:, None]   + 1e-12)
                + _denom_mask * _exp_sim / (_sum_denom[:, None] + 1e-12)
            ) * valid_mask * (g_scalar / n_valid)

            # sim = z_all @ z_all.T / temperature
            # d loss / d z_all = (d_sim + d_sim.T) @ z_all / temperature
            # (symmetric because sim is symmetric)
            d_z_all = (d_sim + d_sim.T) @ _z_all / self.temperature  # (N, D)

            # Distribute gradients back to embeddings
            # If timesteps were sub-sampled, scatter gradients back to full T
            for m_idx, z_m in enumerate(embeddings):
                if z_m.requires_grad:
                    g_sub = d_z_all[m_idx * _BT: (m_idx + 1) * _BT, :]  # (B*T_use, D)
                    B_loc = z_m.shape[0]
                    D_loc = z_m.shape[-1]
                    if _t_idx is not None:
                        # Scatter into full (B, T, D) gradient tensor
                        g_full = np.zeros((B_loc, _T_full, D_loc), dtype=np.float32)
                        g_full[:, _t_idx, :] = g_sub.reshape(B_loc, _T_use, D_loc)
                        g_slice = g_full
                    else:
                        g_slice = g_sub.reshape(z_m.shape)
                    z_m.grad = z_m.grad + g_slice if z_m.grad is not None else g_slice

        loss_t._backward = _infonce_back

        return loss_t

    # ------------------------------------------------------------------
    def forward(self, inputs: list[Tensor]) -> tuple[Tensor, Tensor, list[Tensor]]:
        """
        inputs[m]: (batch, T, d_m)  for modality m

        Returns (fused, loss, uncertainties):
            fused        : (batch, T, d_model)
            loss         : scalar Tensor  (InfoNCE alignment loss)
            uncertainties: List[Tensor] of shape (batch, T) per modality
        """
        # Encode each modality
        embeddings = [self.encode(m, inputs[m]) for m in range(self.n_modalities)]

        # Compute contrastive loss
        loss = self.contrastive_loss(embeddings)

        # Update running alignment matrix (numpy, no grad)
        self._update_alignment(embeddings)

        # Fuse
        fused, uncertainties = self.fuse(embeddings)

        return fused, loss, uncertainties
