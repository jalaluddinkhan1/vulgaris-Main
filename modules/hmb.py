import numpy as np
from collections import deque
from typing import Dict, List, Optional, Tuple

from engine.tensor import Tensor, Parameter
from engine.module import Module
from engine.layers import Linear
from config import HMBConfig


class MemoryVAE(Module):
    """Compress and reconstruct memory entries via a VAE bottleneck."""

    def __init__(self, embed_dim: int, compress_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.compress_dim = compress_dim
        mid = max(embed_dim // 2, compress_dim)

        # Encoder: D -> D/2 -> compress_dim*2  (outputs mu || logvar)
        self.enc1 = Linear(embed_dim, mid)
        self.enc2 = Linear(mid, compress_dim * 2)

        # Decoder: compress_dim -> D/2 -> D
        self.dec1 = Linear(compress_dim, mid)
        self.dec2 = Linear(mid, embed_dim)

    def encode(self, h: Tensor) -> Tuple[Tensor, Tensor]:
        """
        h: (..., embed_dim)
        Returns (mu, logvar) each (..., compress_dim).
        """
        x = self.enc1(h).relu()
        out = self.enc2(x)                       # (..., compress_dim*2)
        # split along last dim
        c = self.compress_dim
        mu_data = out.data[..., :c]
        lv_data = out.data[..., c:]

        mu = Tensor(mu_data, requires_grad=out.requires_grad,
                    _children=(out,), _op="vae_mu")
        lv = Tensor(lv_data, requires_grad=out.requires_grad,
                    _children=(out,), _op="vae_lv")

        def _mu_back():
            if out.requires_grad and mu.grad is not None:
                g = np.zeros_like(out.data)
                g[..., :c] = mu.grad
                out.grad = out.grad + g if out.grad is not None else g

        def _lv_back():
            if out.requires_grad and lv.grad is not None:
                g = np.zeros_like(out.data)
                g[..., c:] = lv.grad
                out.grad = out.grad + g if out.grad is not None else g

        mu._backward = _mu_back
        lv._backward = _lv_back
        return mu, lv

    def decode(self, z: Tensor) -> Tensor:
        """z: (..., compress_dim) -> (..., embed_dim)"""
        x = self.dec1(z).relu()
        return self.dec2(x)

    def vae_loss(self, h: Tensor) -> Tuple[Tensor, Tensor]:
        """
        h: (..., embed_dim)
        Returns (reconstruction_loss, kl_loss) both scalar Tensors.
        """
        mu, logvar = self.encode(h)

        # Reparameterise: z = mu + eps * exp(0.5 * logvar)
        eps_data = np.random.randn(*mu.shape)
        eps = Tensor(eps_data, requires_grad=False)
        std = (logvar * 0.5).exp()
        z = mu + eps * std

        h_recon = self.decode(z)

        # Reconstruction: MSE
        diff = h_recon - h
        recon_loss = (diff * diff).mean()

        # KL: -0.5 * mean(1 + logvar - mu^2 - exp(logvar))
        kl_elem = (logvar.exp() + mu * mu - logvar) * (-0.5) + Tensor(
            np.full(logvar.shape, 0.5))
        kl_loss = kl_elem.mean()

        return recon_loss, kl_loss


class HierarchicalMemoryBank(Module):
    """
    Event-driven hierarchical memory with working buffer, archive, and offline consolidation.
    """

    def __init__(self, config: HMBConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.embed_dim
        self.compress_dim = config.compress_dim
        self.buffer_size = config.buffer_size
        self.archive_size = config.archive_size
        self.surprise_threshold = config.surprise_threshold
        self.consolidation_interval = config.consolidation_interval

        self.vae = MemoryVAE(config.embed_dim, config.compress_dim)

        # Projection to align retrieved context with model dim (identity if same)
        self.context_proj = Linear(config.embed_dim, config.embed_dim)

        # Ring buffer: entries are (h_np: ndarray(embed_dim,), t: int, uncertainty: float)
        object.__setattr__(self, "working_buffer", deque(maxlen=config.buffer_size))

        # Archive: tag -> dict with h_z, uncertainty, timestamp, access_count
        object.__setattr__(self, "archive", {})

        # Running variance for surprise normalisation (EMA)
        object.__setattr__(self, "running_var", 1.0)
        object.__setattr__(self, "step", 0)
        object.__setattr__(self, "_surprise_ema_alpha", 0.99)
        object.__setattr__(self, "_total_surprise", 0.0)
        object.__setattr__(self, "_surprise_count", 0)

    # ------------------------------------------------------------------
    def compute_surprise(self, h_actual: np.ndarray,
                         h_predicted: np.ndarray) -> np.ndarray:
        """
        h_actual, h_predicted: (batch, state_dim) numpy
        Returns (batch,) surprise scores.
        Updates running_var via EMA.
        """
        diff = h_actual - h_predicted                         # (batch, D)
        sq_err = (diff ** 2).sum(axis=-1)                    # (batch,)
        surprise = sq_err / (2.0 * self.running_var + 1e-8)  # (batch,)

        # EMA update of variance using mean squared error
        batch_var = float(sq_err.mean())
        alpha = self._surprise_ema_alpha
        new_var = alpha * self.running_var + (1.0 - alpha) * batch_var
        object.__setattr__(self, "running_var", max(new_var, 1e-6))

        # Track for stats
        object.__setattr__(self, "_total_surprise",
                           self._total_surprise + float(surprise.mean()))
        object.__setattr__(self, "_surprise_count", self._surprise_count + 1)

        return surprise

    # ------------------------------------------------------------------
    def write(self, h: Tensor, timestamp: int,
              h_predicted: Optional[Tensor] = None):
        """
        h: (batch, T, d_model) Tensor
        Adds averaged representation to working buffer.
        If surprise > threshold, also archives immediately.
        """
        h_np = h.data                        # (B, T, D)
        B, T, D = h_np.shape

        for b in range(B):
            h_b = h_np[b]                    # (T, D)
            h_mean = h_b.mean(axis=0)        # (D,)

            uncertainty = 0.0
            surprise = 0.0
            if h_predicted is not None:
                pred_np = h_predicted.data
                if pred_np.ndim == 3:
                    pred_b = pred_np[b].mean(axis=0)
                else:
                    pred_b = pred_np[b] if pred_np.ndim == 2 else pred_np
                surp_arr = self.compute_surprise(
                    h_mean[None, :], pred_b[None, :])
                surprise = float(surp_arr[0])
                uncertainty = surprise

            self.working_buffer.append((h_mean.copy(), timestamp, uncertainty))

            # High-surprise event: write to archive immediately
            if surprise > self.surprise_threshold:
                tag = f"event_{timestamp}_{b}"
                self._archive_entry(h_mean, uncertainty, timestamp, tag)

    # ------------------------------------------------------------------
    def _archive_entry(self, h_np: np.ndarray, uncertainty: float,
                       timestamp: int, tag: str):
        """Compress h_np via VAE and store in archive, evicting LRU if full."""
        h_t = Tensor(h_np[None, :], requires_grad=False)      # (1, D)
        mu, logvar = self.vae.encode(h_t)
        h_z = mu.data[0].copy()                                # (compress_dim,)

        archive = self.archive
        if len(archive) >= self.archive_size:
            # Evict least recently accessed (smallest timestamp + lowest access_count)
            lru_tag = min(
                archive.keys(),
                key=lambda k: archive[k]["timestamp"] + archive[k]["access_count"]
            )
            del archive[lru_tag]

        archive[tag] = {
            "h_z": h_z,
            "uncertainty": uncertainty,
            "timestamp": timestamp,
            "access_count": 0,
        }

    # ------------------------------------------------------------------
    def retrieve(self, h_query: Tensor, top_k: int = 8
                 ) -> Tuple[Tensor, Tensor]:
        """
        h_query: (batch, d_model) Tensor
        Retrieves from working buffer + archive via cosine similarity.
        Returns (retrieved_context, uncertainty_weights):
            retrieved_context  : (batch, d_model) Tensor
            uncertainty_weights: (batch,) Tensor
        """
        B, D = h_query.shape
        q_np = h_query.data                                    # (B, D)

        # Collect candidate memory vectors (numpy)
        cands_h: List[np.ndarray] = []      # each (D,) uncompressed
        cands_u: List[float] = []

        # From working buffer (stored at embed_dim)
        for (h_np, ts, unc) in self.working_buffer:
            if h_np.shape[0] == D:
                cands_h.append(h_np)
                cands_u.append(unc)

        # From archive (decompress via VAE decoder)
        archive = self.archive
        for tag, entry in archive.items():
            h_z = entry["h_z"]                               # (compress_dim,)
            z_t = Tensor(h_z[None, :], requires_grad=False)  # (1, compress_dim)
            h_dec = self.vae.decode(z_t).data[0]              # (D,)
            if h_dec.shape[0] == D:
                cands_h.append(h_dec)
                cands_u.append(entry["uncertainty"])
                # Increment access count
                entry["access_count"] += 1

        if len(cands_h) == 0:
            # No memories yet: return zeros
            zero_ctx = Tensor(np.zeros((B, D), dtype=np.float64),
                              requires_grad=False)
            zero_unc = Tensor(np.ones((B,), dtype=np.float64),
                              requires_grad=False)
            return zero_ctx, zero_unc

        cands_np = np.stack(cands_h, axis=0)                 # (M, D)
        cands_u_np = np.array(cands_u, dtype=np.float64)     # (M,)
        M = cands_np.shape[0]

        # Limit to top_k candidates (by cosine sim to first batch element as proxy)
        if M > top_k * 4:
            q0 = q_np[0]
            q0_norm = q0 / (np.linalg.norm(q0) + 1e-8)
            c_norms = np.linalg.norm(cands_np, axis=1, keepdims=True) + 1e-8
            cos = cands_np / c_norms @ q0_norm                     # (M,)
            idx = np.argpartition(cos, -min(top_k * 4, M))[-min(top_k * 4, M):]
            cands_np = cands_np[idx]
            cands_u_np = cands_u_np[idx]
            M = cands_np.shape[0]

        # Per-query cosine similarity: (B, M)
        q_norm = q_np / (np.linalg.norm(q_np, axis=1, keepdims=True) + 1e-8)
        c_norm = cands_np / (np.linalg.norm(cands_np, axis=1, keepdims=True) + 1e-8)
        sims = q_norm @ c_norm.T                              # (B, M)

        # Uncertainty weighting: w_k = 1 / (u_k + 1e-4)
        unc_w = 1.0 / (cands_u_np + 1e-4)                   # (M,)

        # Temperature-scaled softmax over memory slots
        tau = 0.1
        logits = sims / tau + np.log(unc_w + 1e-12)[None, :]  # (B, M)
        logits -= logits.max(axis=1, keepdims=True)
        exp_l = np.exp(logits)
        attn = exp_l / (exp_l.sum(axis=1, keepdims=True) + 1e-8)  # (B, M)

        # Weighted sum of memory vectors
        ctx_np = attn @ cands_np                              # (B, D)
        avg_unc_np = attn @ unc_w                             # (B,)

        # Build Tensor with gradient through h_query
        ctx_t = Tensor(ctx_np, requires_grad=h_query.requires_grad,
                       _children=(h_query,), _op="mem_retrieve")

        _q_np = q_np.copy()
        _cands = cands_np.copy()
        _attn = attn.copy()
        _tau = tau

        def _retrieve_back():
            if h_query.requires_grad and ctx_t.grad is not None:
                g = ctx_t.grad                                  # (B, D)
                # d ctx / d q_np = d(attn @ cands) / d(q_norm)
                # Simplified: pass gradient through attention weights
                # d attn_bm / d logit_bm = attn_bm * (1 - attn_bm)  (diag of Jacobian)
                d_logit = (_attn * (g @ _cands.T)) - (
                    _attn * (_attn * (g @ _cands.T)).sum(axis=1, keepdims=True))
                # logits = sims/tau,  sims = q_norm @ c_norm.T
                # d sims / d q = c_norm.T,  but q_norm = q / ||q||
                d_sim = d_logit / _tau                         # (B, M)
                # d ctx / d q_raw (approximate, treating q_norm normalization as const)
                q_norm2 = _q_np / (np.linalg.norm(_q_np, axis=1, keepdims=True) + 1e-8)
                c_norm2 = _cands / (np.linalg.norm(_cands, axis=1, keepdims=True) + 1e-8)
                d_q = d_sim @ c_norm2                          # (B, D)
                # Chain through normalization: d/dx (x/||x||) ≈ I/||x||
                q_norms = np.linalg.norm(_q_np, axis=1, keepdims=True) + 1e-8
                d_q_raw = d_q / q_norms
                h_query.grad = (h_query.grad + d_q_raw
                                if h_query.grad is not None else d_q_raw)

        ctx_t._backward = _retrieve_back

        ctx_proj = self.context_proj(ctx_t)
        unc_t = Tensor(avg_unc_np, requires_grad=False)

        return ctx_proj, unc_t

    # ------------------------------------------------------------------
    def consolidate(self):
        """
        Compress working buffer entries via VAE into archive.
        Called periodically (every consolidation_interval steps).
        """
        buf = list(self.working_buffer)
        if len(buf) == 0:
            return

        # Group by rough time-bin (tag = dominant frequency proxy: bin index)
        bin_size = max(1, len(buf) // 16)
        for i in range(0, len(buf), bin_size):
            chunk = buf[i: i + bin_size]
            h_chunk = np.stack([e[0] for e in chunk], axis=0)     # (k, D)
            h_mean = h_chunk.mean(axis=0)                          # (D,)
            avg_unc = float(np.mean([e[2] for e in chunk]))
            max_ts = max(e[1] for e in chunk)
            tag = f"consolidate_{max_ts}_{i}"
            self._archive_entry(h_mean, avg_unc, max_ts, tag)

    # ------------------------------------------------------------------
    def forward(self, h: Tensor, h_predicted: Optional[Tensor] = None,
                timestamp: Optional[int] = None) -> Tuple[Tensor, Tensor]:
        """
        h        : (batch, T, d_model)
        h_predicted: (batch, T, d_model) or None
        timestamp: int or None (uses internal step counter)
        Returns (h_enriched, memory_loss):
            h_enriched  : (batch, T, d_model)  — h + retrieved context
            memory_loss : scalar Tensor         — VAE reconstruction term
        """
        B, T, D = h.shape
        object.__setattr__(self, "step", self.step + 1)
        ts = timestamp if timestamp is not None else self.step

        # Write to memory
        self.write(h, ts, h_predicted)

        # Retrieve context using mean of h over T as query
        h_mean_np = h.data.mean(axis=1)                          # (B, D)
        h_query = Tensor(h_mean_np, requires_grad=h.requires_grad,
                         _children=(h,), _op="hmb_query")

        def _query_back():
            if h.requires_grad and h_query.grad is not None:
                contrib = np.broadcast_to(
                    h_query.grad[:, None, :] / T,
                    (B, T, D)
                ).copy()
                h.grad = h.grad + contrib if h.grad is not None else contrib

        h_query._backward = _query_back

        ctx, _unc = self.retrieve(h_query)                        # (B, D)

        # Broadcast ctx over T and add to h
        ctx_data = ctx.data[:, None, :]                           # (B, 1, D)
        ctx_broad_np = np.broadcast_to(ctx_data, (B, T, D)).copy()
        ctx_broad = Tensor(ctx_broad_np, requires_grad=ctx.requires_grad,
                           _children=(ctx,), _op="ctx_broadcast")

        def _broad_back():
            if ctx.requires_grad and ctx_broad.grad is not None:
                contrib = ctx_broad.grad.sum(axis=1)              # (B, D)
                ctx.grad = ctx.grad + contrib if ctx.grad is not None else contrib

        ctx_broad._backward = _broad_back
        h_enriched = h + ctx_broad                                # (B, T, D)

        # VAE memory loss on a small sample from working buffer
        memory_loss = self._compute_memory_loss()

        # Periodic consolidation
        if self.step % self.consolidation_interval == 0:
            self.consolidate()

        return h_enriched, memory_loss

    def _compute_memory_loss(self) -> Tensor:
        """Sample a mini-batch from the working buffer and compute VAE loss."""
        buf = list(self.working_buffer)
        if len(buf) < 4:
            # Return zero loss Tensor
            zero = Tensor(np.array([[0.0]], dtype=np.float64), requires_grad=False)
            return zero

        sample_size = min(16, len(buf))
        indices = np.random.choice(len(buf), sample_size, replace=False)
        h_sample = np.stack([buf[i][0] for i in indices], axis=0)  # (k, D)
        h_t = Tensor(h_sample, requires_grad=False)

        recon_loss, kl_loss = self.vae.vae_loss(h_t)
        beta = 0.1
        return recon_loss + beta * kl_loss

    # ------------------------------------------------------------------
    def get_stats(self) -> Dict:
        avg_surp = (self._total_surprise / max(self._surprise_count, 1))
        return {
            "buffer_size": len(self.working_buffer),
            "archive_size": len(self.archive),
            "running_var": self.running_var,
            "avg_surprise": avg_surp,
            "step": self.step,
        }
