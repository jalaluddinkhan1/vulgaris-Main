import numpy as np
from collections import namedtuple
from typing import Dict, List, Optional, Tuple, Union

from engine.tensor import Tensor, Parameter, zeros, ones, randn, rand, cat, stack
from engine.module import Module
from engine.layers import Linear, LayerNorm, RMSNorm, Dropout, SwiGLU

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
from config import ModelConfig


VulgarisState = namedtuple('VulgarisState', ['sssr_states', 'hmb_buffer', 'htd_states', 'step'])


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

    def forward(
        self,
        x: Union[Tensor, List[Tensor]],
        domain_idx: int = 0,
        timestamps: Optional[Tensor] = None,
        use_safety: bool = False,
    ) -> Tuple[Tensor, dict]:
        """
        x: (batch, in_channels, T) or List[Tensor] for multi-modal
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
            # ── Single-modality path: ASE ─────────────────────────────────
            z = self.ase(x, timestamps)   # (B, T, d_model)

        # ── HTD ──────────────────────────────────────────────────────────
        z_htd, _ = self.htd(z)
        z = z + z_htd

        # ── SSSR ─────────────────────────────────────────────────────────
        z_ssm, _ = self.sssr(z)
        z = z + z_ssm

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

        # ── HMB ──────────────────────────────────────────────────────────
        z_hmb, memory_loss = self.hmb(z)
        z = z + z_hmb
        aux_losses["memory_loss"] = float(memory_loss.data.sum())

        object.__setattr__(self, "_current_domain", domain_idx)

        # ── Output head ──────────────────────────────────────────────────
        output = self.output_head(z)   # (batch, output_dim) or (batch, n_classes)

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
        import os, json, time
        os.makedirs(path, exist_ok=True)

        # Weights
        state = self.state_dict()
        weights_path = os.path.join(path, "weights.npz")
        np.savez_compressed(weights_path, **{k: v.astype(np.float32) for k, v in state.items()})

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

        data = np.load(weights_path, allow_pickle=False)
        state = {k: data[k].astype(np.float32) for k in data.files}
        model.load_state_dict(state)

        return model
