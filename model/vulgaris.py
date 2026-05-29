import numpy as np
from collections import namedtuple
from typing import Dict, List, Optional, Tuple, Union

from engine.tensor import Tensor, Parameter, zeros, ones, randn, rand, cat, stack
from engine.module import Module
from engine.layers import Linear, LayerNorm, RMSNorm, Dropout, SwiGLU, CausalAttention, RevIN

from modules.ase import AdaptiveSignalEmbedding
from modules.sssr import SelectiveSSR
from modules.crg import CausalRoutingGraph
from modules.hmb import HierarchicalMemoryBank
from modules.shcal import SHCAL
from modules.dah import DomainAdaptiveHypernetwork
from modules.ese import ExplainabilityEngine
from modules.cmla import CrossModalLatentAlignment
from modules.htd import HierarchicalTimescaleDecomposition
from modules.safety import SafetyPolicyHead
from modules.icl import InContextLearning
from modules.rmc import RegimeMixtureCore
from config import ModelConfig


VulgarisState = namedtuple('VulgarisState', ['sssr_states', 'hmb_buffer', 'htd_states', 'step'])


class WorldModelHead(Module):
    """
    Latent World Model — predicts k future latent states autoregressively.

    Uses a lightweight GRU-like recurrent cell in d_model latent space to
    roll out k steps from the current final hidden state, without touching
    the input space at all.  Returns both predicted latents and per-step
    uncertainty (variance of the prediction vs a learned prior).

    Architecture per step:
        h_t+1 = tanh(W_h · h_t + b_h)          update gate
        σ_t   = softplus(W_u · h_t + b_u)       uncertainty estimate

    The predicted latents can be:
      - Used to compute a future-aware output (blend with output_head)
      - Used as targets for a latent consistency loss during training
      - Visualised to understand what the model "expects" to happen next

    Usage
    -----
        wm = WorldModelHead(d_model=256, horizon=5)
        future_z, uncertainties = wm(z_last)
        # future_z      : (B, k, d_model)
        # uncertainties : (B, k)  — per-step predictive uncertainty
    """

    def __init__(self, d_model: int, horizon: int = 5):
        super().__init__()
        self.d_model = d_model
        self.horizon = horizon

        # Transition: h → h_next
        self.trans = Linear(d_model, d_model)
        # Uncertainty head: h → scalar uncertainty per step
        self.unc   = Linear(d_model, 1)

    def forward(self, h_last: Tensor, horizon: Optional[int] = None) -> tuple:
        """
        h_last  : (B, d_model) — final hidden state from Vulgaris forward pass
        horizon : steps to roll out (overrides self.horizon if given)

        Returns:
            future_z      : (B, k, d_model) Tensor — predicted latent trajectory
            uncertainties : (B, k) numpy array — softplus uncertainty per step
        """
        k = horizon if horizon is not None else self.horizon
        B = h_last.data.shape[0]

        steps_z   = []
        steps_unc = []
        h = h_last

        for _ in range(k):
            h_next = self.trans(h).tanh()   # (B, d_model)
            u_raw  = self.unc(h_next)       # (B, 1)
            unc    = float(np.mean(
                np.log1p(np.exp(np.clip(u_raw.data, -20, 20)))
            ))
            steps_z.append(h_next)
            steps_unc.append(unc)
            h = h_next

        # Stack into (B, k, d_model)
        stacked_np = np.stack([z.data for z in steps_z], axis=1)
        future_z = Tensor(
            stacked_np,
            requires_grad=steps_z[0].requires_grad,
            _children=tuple(steps_z),
            _op="wm_stack"
        )
        _steps = steps_z

        def _stack_back():
            if future_z.grad is None:
                return
            for i, sz in enumerate(_steps):
                if sz.requires_grad:
                    g = future_z.grad[:, i, :]
                    sz.grad = sz.grad + g if sz.grad is not None else g.copy()

        future_z._backward = _stack_back

        uncertainties = np.array(steps_unc)   # (k,) — scalar per step
        return future_z, uncertainties


class MAEDecoder(Module):
    """
    Lightweight reconstruction head for Masked Autoencoder pretraining.

    Projects latent d_model tokens back to raw input space (in_channels) so
    MSE on masked positions can be used as a self-supervised pretraining signal
    on completely unlabeled sensor data.

    Architecture: Linear(d_model, d_model) → SiLU → Linear(d_model, in_channels)
    """

    def __init__(self, d_model: int, in_channels: int):
        super().__init__()
        self.fc1 = Linear(d_model, d_model)
        self.fc2 = Linear(d_model, in_channels)

    def forward(self, z: Tensor) -> Tensor:
        """z: (B, T, d_model) → (B, T, in_channels)"""
        return self.fc2(self.fc1(z).silu())


class I2ABlend(Module):
    """
    Imagination-to-Action blending gate.

    Learns a per-sample gating coefficient β ∈ (0, 1) from the last hidden
    state that blends the reactive prediction with a one-step imagined
    prediction:
        output = (1 - β) * reactive + β * imagined

    β is close to 0 early in training (gate initialised to negative bias) so
    the model starts reactive and gradually learns when imagination helps.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.gate = Linear(d_model, 1)
        # Bias toward reactive output early in training
        self.gate.bias.data[:] = -2.0

    def forward(self, h_last: Tensor, reactive: Tensor, imagined: Tensor) -> Tensor:
        """
        h_last  : (B, d_model) — last hidden state
        reactive: (B, out_dim)
        imagined: (B, out_dim)
        Returns: (B, out_dim) blended output
        """
        beta = self.gate(h_last).sigmoid()   # (B, 1) ∈ (0, 1)
        return reactive * (Tensor(np.ones_like(beta.data)) - beta) + imagined * beta


class OutputHead(Module):
    """Regression or classification head."""

    def __init__(self, d_model: int, output_dim: int, n_classes: int = 0):
        super().__init__()
        self.d_model = d_model
        self.output_dim = output_dim
        self.n_classes = n_classes
        self.is_classifier = n_classes > 0

        out_features = n_classes if self.is_classifier else output_dim
        self.head = Linear(d_model, out_features)

    def forward(self, x: Tensor) -> Tensor:
        # x: (batch, T, d_model) — pool last timestep
        # Slice last timestep: (batch, d_model)
        last = Tensor(
            x.data[:, -1, :],
            requires_grad=x.requires_grad,
            _children=(x,),
            _op="last_timestep"
        )

        _x = x

        def _last_back():
            if _x.requires_grad and last.grad is not None:
                contrib = np.zeros_like(_x.data)
                contrib[:, -1, :] = last.grad
                _x.grad = _x.grad + contrib if _x.grad is not None else contrib

        last._backward = _last_back

        out = self.head(last)  # (batch, out_features)

        if self.is_classifier:
            return out.softmax(axis=-1)
        return out


class Vulgaris(Module):
    """
    VULGARIS — complete model integrating ASE, HTD, SSSR, CRG, HMB, DAH, ESE,
    CMLA (multi-modal), SafetyPolicyHead.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        d_model = config.ase.latent_dim
        self.d_model = d_model
        self.output_dim = config.output_dim
        self.n_classes = config.n_classes

        # ── Reversible Instance Normalisation (applied before ASE) ────────
        self.revin = RevIN(num_features=config.input_dim, eps=1e-5, affine=True)

        # ── Core modules ──────────────────────────────────────────────────
        self.ase = AdaptiveSignalEmbedding(
            in_channels=config.input_dim,
            n_filters=config.ase.n_filters,
            n_scales=config.ase.n_scales,
            filter_len=config.ase.filter_len,
            latent_dim=d_model,
        )

        self.htd = HierarchicalTimescaleDecomposition(
            d_model=d_model,
            state_dim=d_model // 4,
            config=config.htd,
        )

        self.sssr = SelectiveSSR(d_model=d_model, config=config.sssr)

        # ── Causal attention (post-SSSR, pre-CRG) ─────────────────────────
        self.attn = CausalAttention(d_model=d_model, n_heads=max(1, d_model // 64))

        # ── In-Context Learning adapter ────────────────────────────────────
        output_or_classes = config.n_classes if config.n_classes > 0 else config.output_dim
        self.icl = InContextLearning(
            d_model=d_model,
            output_dim=output_or_classes,
            n_heads=max(1, d_model // 64),
        )

        self.crg = CausalRoutingGraph(d_model=d_model, config=config.crg)

        self.hmb = HierarchicalMemoryBank(config=config.hmb)

        # DAH targets the SSSR linear layers
        target_layers = {
            "x_proj": self.sssr.x_proj,
            "z_proj": self.sssr.z_proj,
            "y_proj": self.sssr.y_proj,
            "skip_proj": self.sssr.skip_proj,
        }
        self.dah = DomainAdaptiveHypernetwork(
            target_layers=target_layers,
            config=config.dah,
        )

        self.ese = ExplainabilityEngine(
            d_model=d_model,
            n_output=config.n_classes if config.n_classes > 0 else config.output_dim,
            config=config.ese,
        )

        self.output_head = OutputHead(
            d_model=d_model,
            output_dim=config.output_dim,
            n_classes=config.n_classes,
        )

        # Optional: safety head
        self.safety = SafetyPolicyHead(
            d_model=d_model,
            action_dim=config.output_dim,
            config=config.safety,
        )

        # SHCAL wraps the SSSR linear layers for Hebbian/EWC adaptation
        self.shcal = SHCAL(
            monitored_modules=[
                self.sssr.x_proj,
                self.sssr.z_proj,
                self.sssr.y_proj,
                self.sssr.skip_proj,
            ],
            config=config.shcal,
        )

        # ── Regime Mixture Core ───────────────────────────────────────────
        rmc_experts = getattr(config, "rmc_n_experts", 4)
        self.rmc = RegimeMixtureCore(d_model=d_model, n_experts=rmc_experts)

        # ── MAE pretraining decoder ───────────────────────────────────────
        self.mae_decoder = MAEDecoder(d_model=d_model, in_channels=config.input_dim)

        # ── Imagination-to-Action blending gate ───────────────────────────
        self.i2a = I2ABlend(d_model=d_model)

        # ── World Model head ──────────────────────────────────────────────
        self.world_model = WorldModelHead(d_model=d_model, horizon=5)

        # CMLA instantiated lazily based on n_modalities > 1
        # Stored as None when single modality; pipeline must call init_cmla if needed
        object.__setattr__(self, "cmla", None)
        object.__setattr__(self, "_n_modalities", 1)
        object.__setattr__(self, "_current_domain", 0)

    # ──────────────────────────────────────────────────────────────────────
    # Multi-modal initialisation (called externally when needed)
    # ──────────────────────────────────────────────────────────────────────

    def init_cmla(self, modality_dims: List[int]):
        """Initialise CrossModalLatentAlignment for multi-modal inputs."""
        cmla = CrossModalLatentAlignment(
            modality_dims=modality_dims,
            d_model=self.d_model,
        )
        # Register as sub-module
        mods = object.__getattribute__(self, "_modules")
        mods["cmla"] = cmla
        object.__setattr__(self, "cmla", cmla)
        object.__setattr__(self, "_n_modalities", len(modality_dims))

    # ──────────────────────────────────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────────────────────────────────

    def mae_forward(
        self,
        x: Tensor,
        mask_ratio: float = 0.75,
        domain_idx: int = 0,
    ) -> Tuple[Tensor, Tensor, np.ndarray]:
        """
        Masked Autoencoder pretraining forward pass.

        Randomly masks `mask_ratio` fraction of input channels per timestep,
        encodes through ASE, decodes with MAEDecoder, returns MSE on masked
        positions only.

        Args:
            x          : (B, C, T) raw sensor input
            mask_ratio : fraction of (channel, timestep) patches to mask
            domain_idx : domain adapter index

        Returns:
            (reconstructed, mae_loss, mask_bool)
            reconstructed : (B, T, C) predicted values at all positions
            mae_loss      : scalar Tensor — MSE over masked positions
            mask_bool     : (B, C, T) bool numpy array, True = masked
        """
        B, C, T = x.data.shape

        # Build random channel-timestep mask
        mask_bool = np.random.rand(B, C, T) < mask_ratio  # True = masked out

        # Replace masked positions with zeros before encoding
        x_masked_data = x.data.copy()
        x_masked_data[mask_bool] = 0.0
        x_masked = Tensor(x_masked_data, requires_grad=x.requires_grad,
                          _children=(x,), _op="mae_mask")
        _x_orig = x

        def _mae_mask_back():
            if _x_orig.requires_grad and x_masked.grad is not None:
                # Gradient does not flow through masked positions
                g = x_masked.grad.copy()
                g[mask_bool] = 0.0
                _x_orig.grad = _x_orig.grad + g if _x_orig.grad is not None else g

        x_masked._backward = _mae_mask_back

        # Encode: RevIN + ASE
        x_norm = self.revin.normalize(x_masked)
        z = self.ase(x_norm)   # (B, T, d_model)

        # Decode
        reconstructed = self.mae_decoder(z)   # (B, T, C)

        # Target: original x transposed to (B, T, C)
        x_target = Tensor(
            x.data.transpose(0, 2, 1).astype(np.float64),
            requires_grad=False,
        )  # (B, T, C)

        # MSE only over masked positions: transpose mask to (B, T, C)
        mask_bt_c = mask_bool.transpose(0, 2, 1)   # (B, T, C)

        diff = reconstructed - x_target
        diff_sq = diff * diff

        # Zero out unmasked positions before averaging
        diff_sq_np = diff_sq.data.copy()
        diff_sq_np[~mask_bt_c] = 0.0
        n_masked = mask_bt_c.sum()
        if n_masked == 0:
            n_masked = 1

        mae_loss_val = float(diff_sq_np.sum()) / n_masked
        mae_loss = Tensor(
            np.array([[mae_loss_val]], dtype=np.float64),
            requires_grad=reconstructed.requires_grad,
            _children=(diff_sq,),
            _op="mae_loss"
        )
        _diff_sq = diff_sq
        _mask_bt_c = mask_bt_c

        def _mae_loss_back():
            if _diff_sq.requires_grad and mae_loss.grad is not None:
                g_scale = float(mae_loss.grad.sum()) / n_masked
                contrib = np.zeros_like(_diff_sq.data)
                contrib[_mask_bt_c] = g_scale
                _diff_sq.grad = (_diff_sq.grad + contrib
                                 if _diff_sq.grad is not None else contrib)

        mae_loss._backward = _mae_loss_back

        return reconstructed, mae_loss, mask_bool

    def world_model_forward(
        self,
        x: Tensor,
        domain_idx: int = 0,
        horizon: int = 5,
    ) -> dict:
        """
        World model forward: encodes x, then predicts `horizon` future latent
        states autoregressively in latent space without any external input.

        Args:
            x        : (B, C, T) raw sensor input
            domain_idx : domain adapter index
            horizon  : number of future steps to predict

        Returns dict with:
            "future_latents"   : (B, horizon, d_model) predicted latent trajectory
            "uncertainties"    : (horizon,) numpy — per-step predictive uncertainty
            "future_outputs"   : (B, horizon, output_dim) outputs from future latents
            "current_output"   : (B, output_dim) reactive output at the current step
        """
        # Encode current input
        x_norm = self.revin.normalize(x)
        z = self.ase(x_norm)        # (B, T, d_model)
        z_htd, _ = self.htd(z)
        z = z + z_htd
        z_ssm, _ = self.sssr(z)
        z = z + z_ssm
        z_attn = self.attn(z)
        z = z + z_attn

        # DAH domain adapter
        self.dah.set_domain(domain_idx)
        adapters = self.dah.get_adapters(domain_idx)
        if "skip_proj" in adapters:
            A_d, B_d = adapters["skip_proj"]
            B_z, T_z, D_z = z.data.shape
            z_flat = z.reshape(B_z * T_z, D_z)
            z_adapt = self.dah._adapter_layers["skip_proj"](z_flat, A_d, B_d)
            z = z + z_adapt.reshape(B_z, T_z, D_z)

        z_crg, _ = self.crg(z)
        z = z + z_crg
        z_hmb, _ = self.hmb(z)
        z = z + z_hmb

        # Current reactive output
        current_output = self.output_head(z)   # (B, output_dim)

        # World model rollout from last hidden state
        h_last = Tensor(z.data[:, -1, :], requires_grad=False)  # (B, d_model)
        future_z, uncertainties = self.world_model(h_last, horizon=horizon)
        # future_z: (B, horizon, d_model)

        # Project each future step to output space
        future_outputs = []
        for step_i in range(horizon):
            h_i_data = future_z.data[:, step_i, :]
            # Expand to (B, 1, d_model) for output_head which expects (..., T, d_model)
            h_i_3d = np.zeros((h_i_data.shape[0], 1, h_i_data.shape[1]),
                               dtype=np.float64)
            h_i_3d[:, 0, :] = h_i_data
            z_i = Tensor(h_i_3d, requires_grad=False)
            out_i = self.output_head(z_i)   # (B, output_dim)
            future_outputs.append(out_i.data.copy())

        future_outputs_np = np.stack(future_outputs, axis=1)  # (B, horizon, output_dim)

        return {
            "future_latents":  future_z,
            "uncertainties":   uncertainties,
            "future_outputs":  future_outputs_np,
            "current_output":  current_output,
        }

    def forward(
        self,
        x: Union[Tensor, List[Tensor]],
        domain_idx: int = 0,
        timestamps: Optional[Tensor] = None,
        use_safety: bool = False,
        mask: Optional[np.ndarray] = None,
        context: Optional[List[tuple]] = None,
        use_imagination: bool = False,
    ) -> Tuple[Tensor, dict]:
        """
        x       : (batch, in_channels, T) or List[Tensor] for multi-modal
        context : optional list of (x_ref, y_ref) pairs for zero-shot in-context
                  adaptation. Each x_ref is (B, C, T_ref) numpy or Tensor;
                  y_ref is (B, output_dim) numpy or Tensor.
                  Example::
                      context = [(x_ref1, y_ref1), (x_ref2, y_ref2)]
                      output, aux = model(x_query, context=context)
        Returns (output, aux_losses).
        """
        aux_losses: dict = {}
        cmla_loss = Tensor(np.array([[0.0]]), requires_grad=False)

        # ── Multi-modal path ──────────────────────────────────────────────
        if isinstance(x, list):
            cmla = object.__getattribute__(self, "cmla")
            if cmla is None:
                # Auto-init with inferred dims (assumes (B, C, T) per modality)
                dims = [xi.data.shape[1] for xi in x]
                self.init_cmla(dims)
                cmla = object.__getattribute__(self, "cmla")

            # Each modality: (B, C, T) → transpose to (B, T, C) for CMLA encoders
            modal_inputs = []
            for xi in x:
                if xi.ndim == 3:
                    # (B, C, T) → (B, T, C)
                    xi_t = Tensor(
                        xi.data.transpose(0, 2, 1),
                        requires_grad=xi.requires_grad,
                        _children=(xi,),
                        _op="modal_transpose"
                    )
                    _xi = xi
                    _xi_t = xi_t

                    def _mt_back(xt=_xi_t, xo=_xi):
                        if xo.requires_grad and xt.grad is not None:
                            contrib = xt.grad.transpose(0, 2, 1)
                            xo.grad = xo.grad + contrib if xo.grad is not None else contrib

                    xi_t._backward = _mt_back
                    modal_inputs.append(xi_t)
                else:
                    modal_inputs.append(xi)

            fused, cmla_loss, _ = cmla(modal_inputs)  # (B, T, d_model)
            aux_losses["cmla_loss"] = float(cmla_loss.data.sum())

            # Convert fused (B, T, d_model) back through ASE by treating it as
            # already in latent space — skip ASE, use fused directly as z
            z = fused

        else:
            # ── Single-modality path: RevIN → ASE ────────────────────────
            x_norm = self.revin.normalize(x)   # (B, C, T) instance-normalised
            z = self.ase(x_norm, timestamps, mask=mask)   # (B, T, d_model)

        # ── HTD ──────────────────────────────────────────────────────────
        z_htd, _ = self.htd(z)
        z = z + z_htd

        # ── SSSR ─────────────────────────────────────────────────────────
        z_ssm, _ = self.sssr(z)
        z = z + z_ssm

        # ── Causal Attention ─────────────────────────────────────────────
        z_attn = self.attn(z)
        z = z + z_attn

        # ── In-Context Learning (zero-shot conditioning) ──────────────────
        if context is not None and len(context) > 0:
            ctx_latents, ctx_labels = [], []
            for x_ref, y_ref in context:
                if not isinstance(x_ref, Tensor):
                    x_ref = Tensor(np.asarray(x_ref, dtype=np.float32))
                if not isinstance(y_ref, Tensor):
                    y_ref = Tensor(np.asarray(y_ref, dtype=np.float32))
                x_ref_norm = self.revin.normalize(x_ref)
                z_ref = self.ase(x_ref_norm)          # (B, T_ref, d_model)
                ctx_latents.append(z_ref)
                ctx_labels.append(y_ref)
            ctx_stack = self.icl.encode_context(ctx_latents, ctx_labels)
            z_icl = self.icl(z, ctx_stack)
            z = z + z_icl
            aux_losses["icl_active"] = True
        else:
            aux_losses["icl_active"] = False

        # ── DAH: apply domain adapter as residual on SSM output ───────────
        # skip_proj maps d_model→d_model so its adapter is shape-compatible with z
        self.dah.set_domain(domain_idx)
        adapters = self.dah.get_adapters(domain_idx)
        if "skip_proj" in adapters:
            A_d, B_d = adapters["skip_proj"]
            B_z, T_z, D_z = z.data.shape
            z_flat = z.reshape(B_z * T_z, D_z)
            z_adapt = self.dah._adapter_layers["skip_proj"](z_flat, A_d, B_d)
            z_adapt = z_adapt.reshape(B_z, T_z, D_z)
            z = z + z_adapt

        # ── CRG ──────────────────────────────────────────────────────────
        z_crg, dag_penalty = self.crg(z)
        z = z + z_crg
        aux_losses["dag_penalty"] = float(dag_penalty.data.sum())

        # ── RMC: Regime Mixture Core ──────────────────────────────────────
        z_rmc, rmc_balance = self.rmc(z)
        z = z + z_rmc
        aux_losses["rmc_balance_loss"] = float(rmc_balance.data.sum())

        # ── HMB ──────────────────────────────────────────────────────────
        z_hmb, memory_loss = self.hmb(z)
        z = z + z_hmb
        aux_losses["memory_loss"] = float(memory_loss.data.sum())

        object.__setattr__(self, "_current_domain", domain_idx)

        # ── Output head ──────────────────────────────────────────────────
        output = self.output_head(z)   # (batch, output_dim) or (batch, n_classes)

        # ── I2A: Imagination-to-Action blending ───────────────────────────
        # One-step imagined prediction: run last hidden state through SSSR for
        # one extra step to get an "imagined" next output, then gate-blend with
        # the reactive output. Beta gate starts near 0 so training is stable.
        if use_imagination and not isinstance(x, list):
            h_last_data = z.data[:, -1:, :]          # (B, 1, d_model)
            h_last_3d = Tensor(h_last_data, requires_grad=False)
            z_imag_ssm, _ = self.sssr(h_last_3d)     # (B, 1, d_model)
            # Build imagined z: copy z, replace last timestep with imagined
            z_imag_data = z.data.copy()
            z_imag_data[:, -1:, :] = z.data[:, -1:, :] + z_imag_ssm.data
            z_imag = Tensor(z_imag_data, requires_grad=False)
            imagined_out = self.output_head(z_imag)  # (B, output_dim)
            h_last_t = Tensor(z.data[:, -1, :], requires_grad=False)
            output = self.i2a(h_last_t, output, imagined_out)
            aux_losses["i2a_active"] = True
        else:
            aux_losses["i2a_active"] = False

        # ── Safety filter (optional) ──────────────────────────────────────
        if use_safety:
            # Use last timestep of z as state for safety head
            state_np = z.data[:, -1, :]   # (batch, d_model)
            state_t = Tensor(state_np, requires_grad=False)
            # Safety head returns safe action: (batch, output_dim)
            safe_output = self.safety(state_t)
            cbf_loss = self.safety.cbf_loss(state_t, state_t)
            aux_losses["cbf_loss"] = float(cbf_loss.data.sum())
            output = safe_output
        else:
            aux_losses["cbf_loss"] = 0.0

        # ── ESE: record latents for explainability (no grad) ─────────────
        h_np = z.data.copy()
        y_np = output.data.copy()
        # Only record last timestep mean for CART to avoid memory explosion
        h_record = h_np[:, -1, :]    # (batch, d_model)
        if self.training:
            self.ese.record(h_record, y_np)

        # Expose h_states for temporal coherence loss
        aux_losses["h_states"] = z    # Tensor (B, T, d_model)

        # Accumulate CMLA loss into aux for loss function
        aux_losses["cmla_loss_tensor"] = cmla_loss

        return output, aux_losses

    # ──────────────────────────────────────────────────────────────────────
    # Single-step streaming inference
    # ──────────────────────────────────────────────────────────────────────

    def step(
        self,
        x_t: Tensor,
        state: 'VulgarisState',
        domain_idx: int = 0,
    ) -> Tuple[Tensor, 'VulgarisState']:
        """
        Single-step streaming inference.
        x_t: (batch, in_channels)
        state: VulgarisState
        Returns (output_t, new_state)
        """
        # Add time dim for ASE: (batch, in_channels, 1)
        x_3d = Tensor(
            x_t.data[:, :, None],
            requires_grad=x_t.requires_grad,
            _children=(x_t,),
            _op="step_unsqueeze"
        )

        def _step_unsq_back():
            if x_t.requires_grad and x_3d.grad is not None:
                contrib = x_3d.grad[:, :, 0]
                x_t.grad = x_t.grad + contrib if x_t.grad is not None else contrib

        x_3d._backward = _step_unsq_back

        # ASE on single step
        z = self.ase(x_3d)           # (batch, 1, d_model)

        # HTD single step
        htd_states_in = state.htd_states if state.htd_states else None
        z_htd, new_htd_states = self.htd(z, htd_states_in)
        z = z + z_htd

        # SSSR single step
        sssr_states_in = state.sssr_states if state.sssr_states else None
        z_ssm, new_sssr_states = self.sssr(z, sssr_states_in)
        z = z + z_ssm

        # CRG single step
        z_crg, _ = self.crg(z)
        z = z + z_crg

        # HMB single step
        z_hmb, _ = self.hmb(z, timestamp=state.step)
        z = z + z_hmb

        # Output head
        output = self.output_head(z)   # (batch, output_dim)

        new_state = VulgarisState(
            sssr_states=new_sssr_states,
            hmb_buffer=None,   # HMB manages its own buffer internally
            htd_states=new_htd_states,
            step=state.step + 1,
        )

        return output, new_state

    # ──────────────────────────────────────────────────────────────────────
    # State management
    # ──────────────────────────────────────────────────────────────────────

    def init_state(self, batch_size: int) -> 'VulgarisState':
        """Initialize all hidden states to zeros."""
        n_sssr_heads = self.sssr.n_heads
        head_state_dim = max(1, self.config.sssr.state_dim // n_sssr_heads)
        sssr_states = [
            zeros((batch_size, head_state_dim))
            for _ in range(n_sssr_heads)
        ]

        n_htd_levels = self.config.htd.n_levels
        htd_state_dim = self.d_model // 4
        htd_states = [
            zeros((batch_size, htd_state_dim))
            for _ in range(n_htd_levels)
        ]

        return VulgarisState(
            sssr_states=sssr_states,
            hmb_buffer=None,
            htd_states=htd_states,
            step=0,
        )

    def rollout(
        self,
        x: Tensor,
        horizon: int,
        domain_idx: int = 0,
    ) -> Tensor:
        """
        Autoregressive multi-step ahead forecasting.

        Runs the context window through step() to warm up the recurrent state,
        then rolls out `horizon` steps, feeding each prediction back as the next
        input (unknown channels padded with zeros).

        Args:
            x       : (batch, in_channels, T)  — context window
            horizon : number of future steps to predict
            domain_idx : domain adapter index

        Returns:
            predictions : (batch, horizon, output_dim) as a plain Tensor (no grad)
        """
        self.eval()
        B, C, T = x.data.shape
        state = self.init_state(B)

        # Warm up state on context window
        for t in range(T):
            x_t = Tensor(x.data[:, :, t])        # (B, C)
            _, state = self.step(x_t, state, domain_idx)

        # Autoregressive rollout
        predictions = []
        last_x = Tensor(x.data[:, :, -1])        # (B, C) — seed from last context step
        for _ in range(horizon):
            output_t, state = self.step(last_x, state, domain_idx)   # (B, output_dim)
            predictions.append(output_t.data.copy())
            # Project prediction back to input space (fill known dims, zero the rest)
            next_input = np.zeros((B, C), dtype=np.float32)
            out_dim = output_t.data.shape[-1]
            next_input[:, :min(out_dim, C)] = output_t.data[:, :min(out_dim, C)]
            last_x = Tensor(next_input)

        preds_np = np.stack(predictions, axis=1)  # (B, horizon, output_dim)
        return Tensor(preds_np)

    def set_domain(self, domain_idx: int):
        """Switch domain adapter."""
        self.dah.set_domain(domain_idx)
        object.__setattr__(self, "_current_domain", domain_idx)

    def freeze_base(self):
        """Freeze all base parameters (non-adapter). Called before domain adaptation."""
        adapter_param_ids = set()
        # Collect adapter parameter ids from DAH
        for p in self.dah.parameters():
            adapter_param_ids.add(id(p))
        # Collect SHCAL recalib_scale
        adapter_param_ids.add(id(self.shcal.recalib_scale))

        # Freeze everything that is not in DAH or SHCAL recalib
        for p in self.parameters():
            if id(p) not in adapter_param_ids:
                p.requires_grad = False

    def n_base_params(self) -> int:
        """Count base (non-adapter) parameters."""
        adapter_param_ids = set(id(p) for p in self.dah.parameters())
        return sum(
            p.data.size for p in self.parameters()
            if id(p) not in adapter_param_ids
        )

    def n_adapter_params(self) -> int:
        """Count adapter parameters only."""
        return sum(p.data.size for p in self.dah.parameters())

    def save(self, path: str) -> None:
        """
        Save model weights, config, and metadata to a versioned checkpoint directory.

        Creates:
            <path>/
                weights.npz      — all model parameters (float32, no pickle)
                config.yaml      — full ModelConfig as YAML
                metadata.json    — version, timestamp, param counts

        Args:
            path: Directory path to save checkpoint into (created if missing).
        """
        import hashlib, os, json, time
        os.makedirs(path, exist_ok=True)

        # Weights
        state = self.state_dict()
        weights_path = os.path.join(path, "weights.npz")
        np.savez_compressed(weights_path, **{k: v.astype(np.float32) for k, v in state.items()})

        # SHA-256 of weights file (tamper detection)
        h = hashlib.sha256()
        with open(weights_path, "rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        weights_sha256 = h.hexdigest()

        # Config
        config_path = os.path.join(path, "config.yaml")
        self.config.to_yaml(config_path)

        # Metadata
        meta = {
            "vulgaris_version": "0.1.0",
            "format_version":   1,
            "saved_at":         time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "n_base_params":    self.n_base_params(),
            "n_adapter_params": self.n_adapter_params(),
            "d_model":          self.d_model,
            "input_dim":        self.config.input_dim,
            "output_dim":       self.config.output_dim,
            "n_classes":        self.config.n_classes,
            "weights_sha256":   weights_sha256,
        }
        with open(os.path.join(path, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Vulgaris":
        """
        Load a Vulgaris model from a versioned checkpoint directory.

        Args:
            path: Directory path containing weights.npz, config.yaml, metadata.json.

        Returns:
            Restored Vulgaris instance with loaded weights.

        Raises:
            FileNotFoundError: If path or required files are missing.
            ValueError: If checkpoint format version is incompatible.
        """
        import os, json
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Checkpoint directory not found: {path}")

        meta_path = os.path.join(path, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            fmt_ver = meta.get("format_version", 1)
            if fmt_ver > 1:
                raise ValueError(
                    f"Checkpoint format version {fmt_ver} is newer than this "
                    f"vulgaris version (supports up to format_version=1). "
                    f"Please upgrade vulgaris."
                )

        config_path = os.path.join(path, "config.yaml")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"config.yaml not found in checkpoint: {path}")

        from config import ModelConfig
        config = ModelConfig.from_yaml(config_path)
        model = cls(config)

        weights_path = os.path.join(path, "weights.npz")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"weights.npz not found in checkpoint: {path}")

        # SHA-256 integrity check (skip if hash was not stored in older checkpoints)
        if os.path.exists(meta_path):
            expected_hash = meta.get("weights_sha256")
            if expected_hash:
                import hashlib
                h = hashlib.sha256()
                with open(weights_path, "rb") as fh:
                    for chunk in iter(lambda: fh.read(65536), b""):
                        h.update(chunk)
                actual_hash = h.hexdigest()
                if actual_hash != expected_hash:
                    raise ValueError(
                        f"Checkpoint integrity check failed: weights.npz SHA-256 mismatch.\n"
                        f"  Expected : {expected_hash}\n"
                        f"  Got      : {actual_hash}\n"
                        f"The file may be corrupted or tampered with."
                    )

        data = np.load(weights_path, allow_pickle=False)
        state = {k: data[k].astype(np.float32) for k in data.files}
        model.load_state_dict(state)

        return model
