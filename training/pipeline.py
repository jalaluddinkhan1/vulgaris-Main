import os
import time
import numpy as np
from collections import deque
from typing import Dict, Iterator, Optional

from engine.tensor import Tensor
from .loss import VulgarisLoss
from .optimizer import SpectralAdamW, CosineSchedule


class TrainingPipeline:
    """
    Complete training loop with checkpointing, logging, and online adaptation.
    """

    def __init__(
        self,
        model,                    # Vulgaris instance
        config,                   # ModelConfig
        loss_fn: VulgarisLoss,
        optimizer: SpectralAdamW,
        scheduler: CosineSchedule,
    ):
        self.model = model
        self.config = config
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.step_count: int = 0
        self.best_loss: float = float('inf')

        checkpoint_dir = getattr(config.training, "checkpoint_dir", "checkpoints")
        self.checkpoint_dir: str = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Rolling metrics deques (last 100 steps)
        self.metrics: Dict[str, deque] = {
            "total_loss":     deque(maxlen=100),
            "task_loss":      deque(maxlen=100),
            "dag_penalty":    deque(maxlen=100),
            "memory_loss":    deque(maxlen=100),
            "temporal_loss":  deque(maxlen=100),
            "ewc_loss":       deque(maxlen=100),
            "cbf_loss":       deque(maxlen=100),
            "cmla_loss":      deque(maxlen=100),
            "conformal_loss": deque(maxlen=100),
        }

    # ──────────────────────────────────────────────────────────────────────

    def train_step(
        self, x: np.ndarray, y: np.ndarray, domain_idx: int = 0
    ) -> dict:
        """
        Single training step.
        x: (batch, in_channels, T) numpy
        y: (batch, output_dim) or (batch,) numpy
        Returns metrics dict.
        """
        self.model.train()
        self.optimizer.zero_grad()

        x_t = Tensor(x.astype(np.float32), requires_grad=False)
        y_t = Tensor(y.astype(np.float32), requires_grad=False)

        # Forward
        output, aux = self.model(x_t, domain_idx=domain_idx)

        # Loss
        total_loss, components = self.loss_fn(output, y_t, aux)

        # Backward
        total_loss.backward()

        # NaN/Inf guard: skip corrupt steps silently
        bad_grads = False
        for p in self.model.parameters():
            if p.grad is not None and not np.isfinite(p.grad).all():
                bad_grads = True
                break
        if bad_grads:
            self.optimizer.zero_grad()
            components["total_loss"] = float('inf')
            return components

        # Optimizer step + LR schedule
        self.optimizer.step()
        self.scheduler.step()

        # Clear DAH cache so hypernetwork is re-run each step (ensures grad flow)
        if hasattr(self.model, 'dah'):
            self.model.dah.clear_cache()

        self.step_count += 1

        # Periodic: conformal update
        # Use per-sample max-probability as scalar prediction so shapes always
        # match y_true regardless of output_dim / n_classes.
        if self.step_count % 50 == 0:
            y_pred_scalar = output.data.reshape(output.data.shape[0], -1).max(axis=1)
            y_true_scalar = y_t.data.flatten()[:output.data.shape[0]]
            sigma_scalar  = np.ones(len(y_pred_scalar)) * 0.1
            self.loss_fn.update_conformal(y_pred_scalar, y_true_scalar, sigma_scalar)

        # Record rolling metrics
        for k, v in components.items():
            if k in self.metrics:
                self.metrics[k].append(float(v))

        # Checkpoint on improvement
        total_val = components.get("total_loss", float('inf'))
        if total_val < self.best_loss:
            self.best_loss = total_val
            self.save_checkpoint(tag="best")

        return components

    # ──────────────────────────────────────────────────────────────────────

    def eval_step(
        self, x: np.ndarray, y: np.ndarray, domain_idx: int = 0
    ) -> dict:
        """
        Evaluation step (no gradient computation).
        Returns metrics + prediction intervals.
        """
        self.model.eval()

        x_t = Tensor(x.astype(np.float32), requires_grad=False)
        y_t = Tensor(y.astype(np.float32), requires_grad=False)

        output, aux = self.model(x_t, domain_idx=domain_idx)

        total_loss, components = self.loss_fn(output, y_t, aux)

        # Prediction intervals using conformal predictor
        sigma_np = np.ones_like(output.data) * 0.1
        y_pred_flat = output.data.flatten()
        sigma_flat = sigma_np.flatten()
        lower, upper = self.loss_fn.predict_interval(y_pred_flat, sigma_flat)

        components["interval_lower"] = lower.mean() if len(lower) > 0 else 0.0
        components["interval_upper"] = upper.mean() if len(upper) > 0 else 0.0
        components["interval_width"] = float(np.mean(upper - lower))
        components["conformal_coverage"] = self.loss_fn.conformal.current_coverage()

        return components

    # ──────────────────────────────────────────────────────────────────────

    def train_epoch(
        self, data_iter: Iterator, domain_idx: int = 0
    ) -> dict:
        """
        Run train_step for all batches in data_iter.
        Returns epoch-averaged metrics dict.
        """
        epoch_metrics: Dict[str, list] = {}

        for batch in data_iter:
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                x_np, y_np = batch
            else:
                raise ValueError(
                    "data_iter must yield (x, y) tuples of numpy arrays"
                )

            step_metrics = self.train_step(x_np, y_np, domain_idx=domain_idx)

            for k, v in step_metrics.items():
                if k not in epoch_metrics:
                    epoch_metrics[k] = []
                epoch_metrics[k].append(float(v))

        # Average over epoch
        averaged = {k: float(np.mean(v)) for k, v in epoch_metrics.items()}
        return averaged

    # ──────────────────────────────────────────────────────────────────────

    def online_adapt(
        self, x: np.ndarray, y: np.ndarray, domain_idx: int = 0
    ):
        """
        Lightweight online update without full backward pass.
        - SHCAL Hebbian update on SSSR linear layers
        - Conformal calibration update
        No optimizer.step (offline training gradient update is skipped).
        """
        self.model.eval()

        x_t = Tensor(x.astype(np.float32), requires_grad=False)

        # Single forward pass to get activations
        output, aux = self.model(x_t, domain_idx=domain_idx)

        # SHCAL Hebbian update: feed (pre, post) activation pairs for SSSR linears.
        # Monitored layers: [x_proj, z_proj, y_proj, skip_proj]
        # We approximate activations from h_states (B, T, d_model) by mean over T.
        h_states = aux.get("h_states", None)
        if h_states is not None and isinstance(h_states, Tensor):
            h_np = h_states.data          # (B, T, d_model)
            pre_np = h_np.mean(axis=1)    # (B, d_model) — input proxy (d_model)

            sssr = self.model.sssr
            layers = [sssr.x_proj, sssr.z_proj, sssr.y_proj, sssr.skip_proj]
            activations = {}
            for idx, layer in enumerate(layers):
                d_in  = layer.in_features
                d_out = layer.out_features
                B_sz  = pre_np.shape[0]

                # Construct a batch-mean pre vector of matching d_in
                if d_in <= pre_np.shape[1]:
                    pre_layer = pre_np[:, :d_in]
                else:
                    # Pad with zeros if layer expects more features than d_model
                    pad = np.zeros((B_sz, d_in - pre_np.shape[1]), dtype=np.float32)
                    pre_layer = np.concatenate([pre_np, pad], axis=1)

                # Post-activation proxy: forward the layer with truncated input
                # to get matching d_out shape
                post_layer = pre_layer[:, :d_out] if d_out <= d_in else np.zeros(
                    (B_sz, d_out), dtype=np.float32
                )

                activations[str(idx)] = (pre_layer, post_layer)

            self.model.shcal.hebbian_update(activations)

        # Update conformal calibration
        y_np = y.astype(np.float32)
        sigma_np = np.ones(output.data.size) * 0.1
        self.loss_fn.update_conformal(
            output.data.flatten(),
            y_np.flatten(),
            sigma_np,
        )

    # ──────────────────────────────────────────────────────────────────────

    def save_checkpoint(self, tag: str = 'latest'):
        """
        Save model state_dict, optimizer state, step_count, best_loss.
        Uses numpy .npz format (no pickle).
        """
        path = os.path.join(self.checkpoint_dir, f"checkpoint_{tag}.npz")

        model_state = self.model.state_dict()
        opt_state = self.optimizer.state_dict()

        # Flatten to numpy-serialisable format:
        # model params: {name: array}
        # optimizer: t, lr, m_i, v_i arrays
        save_dict: dict = {}

        for name, arr in model_state.items():
            save_dict[f"model__{name}"] = arr

        save_dict["opt__t"] = np.array([opt_state["t"]], dtype=np.int64)
        save_dict["opt__lr"] = np.array([opt_state["lr"]], dtype=np.float64)
        save_dict["opt__betas"] = np.array(opt_state["betas"], dtype=np.float64)
        save_dict["opt__eps"] = np.array([opt_state["eps"]], dtype=np.float64)
        save_dict["opt__weight_decay"] = np.array(
            [opt_state["weight_decay"]], dtype=np.float64
        )
        save_dict["opt__spectral_clip"] = np.array(
            [opt_state["spectral_clip"]], dtype=np.float64
        )
        save_dict["opt__grad_clip"] = np.array(
            [opt_state["grad_clip"]], dtype=np.float64
        )

        for k, v in opt_state.get("m", {}).items():
            save_dict[f"opt__m__{k}"] = np.asarray(v, dtype=np.float64)
        for k, v in opt_state.get("v", {}).items():
            save_dict[f"opt__v__{k}"] = np.asarray(v, dtype=np.float64)

        save_dict["meta__step_count"] = np.array([self.step_count], dtype=np.int64)
        save_dict["meta__best_loss"] = np.array([self.best_loss], dtype=np.float64)

        np.savez_compressed(path, **save_dict)

    # ──────────────────────────────────────────────────────────────────────

    def load_checkpoint(self, path: str):
        """Load and restore model + optimizer state from .npz checkpoint."""
        data = np.load(path, allow_pickle=False)

        # Restore model parameters
        model_state = {}
        for key in data.files:
            if key.startswith("model__"):
                param_name = key[len("model__"):]
                model_state[param_name] = data[key]
        self.model.load_state_dict(model_state)

        # Restore optimizer scalar metadata
        opt_state: dict = {}
        if "opt__t" in data:
            opt_state["t"] = int(data["opt__t"][0])
        if "opt__lr" in data:
            opt_state["lr"] = float(data["opt__lr"][0])
        if "opt__betas" in data:
            opt_state["betas"] = data["opt__betas"].tolist()
        if "opt__eps" in data:
            opt_state["eps"] = float(data["opt__eps"][0])
        if "opt__weight_decay" in data:
            opt_state["weight_decay"] = float(data["opt__weight_decay"][0])
        if "opt__spectral_clip" in data:
            opt_state["spectral_clip"] = float(data["opt__spectral_clip"][0])
        if "opt__grad_clip" in data:
            opt_state["grad_clip"] = float(data["opt__grad_clip"][0])

        m_dict = {}
        v_dict = {}
        for key in data.files:
            if key.startswith("opt__m__"):
                k = key[len("opt__m__"):]
                m_dict[k] = data[key]
            elif key.startswith("opt__v__"):
                k = key[len("opt__v__"):]
                v_dict[k] = data[key]

        opt_state["m"] = m_dict
        opt_state["v"] = v_dict
        self.optimizer.load_state_dict(opt_state)

        # Restore training metadata
        if "meta__step_count" in data:
            self.step_count = int(data["meta__step_count"][0])
        if "meta__best_loss" in data:
            self.best_loss = float(data["meta__best_loss"][0])

    # ──────────────────────────────────────────────────────────────────────

    def _log_metrics(self, metrics: dict):
        """Print formatted metrics, using rich if available else plain print."""
        try:
            from rich.console import Console
            from rich.table import Table

            console = Console()
            table = Table(show_header=True, header_style="bold cyan")
            table.add_column("Metric", style="dim")
            table.add_column("Value", justify="right")

            for k, v in sorted(metrics.items()):
                if isinstance(v, float):
                    table.add_row(k, f"{v:.6f}")
                else:
                    table.add_row(k, str(v))

            console.print(table)

        except ImportError:
            line_parts = [f"step={self.step_count}"]
            for k, v in sorted(metrics.items()):
                if isinstance(v, float):
                    line_parts.append(f"{k}={v:.6f}")
                else:
                    line_parts.append(f"{k}={v}")
            print("  |  ".join(line_parts))
