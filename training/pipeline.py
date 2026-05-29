import os
import time
import numpy as np
from collections import deque
from typing import Dict, Iterator, Optional

from engine.tensor import Tensor
from .loss import VulgarisLoss
from .optimizer import SpectralAdamW, CosineSchedule
from .distributed import (
    is_initialized, is_main_process,
    allreduce_gradients, allreduce_scalar,
)


class TimeSeriesAugment:
    """
    Lightweight time-series augmentations applied to (B, C, T) numpy arrays.
    Each augmentation fires independently with probability `aug_prob`.

    Parameters
    ----------
    noise_std          : std of additive Gaussian noise
    channel_dropout_p  : probability of zeroing an entire channel per batch item
    scale_range        : (lo, hi) uniform magnitude scaling per channel
    time_warp_max      : max timesteps to shift/roll signal (0 to disable)
    aug_prob           : probability each augmentation fires per call
    """

    def __init__(
        self,
        noise_std: float = 0.01,
        channel_dropout_p: float = 0.1,
        scale_range: tuple = (0.8, 1.2),
        time_warp_max: int = 4,
        aug_prob: float = 0.5,
        spectral_drop_p: float = 0.1,
        spectral_drop_width: int = 4,
    ):
        self.noise_std = noise_std
        self.channel_dropout_p = channel_dropout_p
        self.scale_range = scale_range
        self.time_warp_max = time_warp_max
        self.aug_prob = aug_prob
        self.spectral_drop_p = spectral_drop_p
        self.spectral_drop_width = spectral_drop_width

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """x: (B, C, T) float32.  Returns augmented copy."""
        x = x.copy()

        # Gaussian noise
        if self.noise_std > 0 and np.random.rand() < self.aug_prob:
            x += np.random.randn(*x.shape).astype(np.float32) * self.noise_std

        # Channel dropout: zero out random channels per batch item
        if self.channel_dropout_p > 0 and np.random.rand() < self.aug_prob:
            for b in range(x.shape[0]):
                drop = np.random.rand(x.shape[1]) < self.channel_dropout_p
                x[b, drop, :] = 0.0

        # Magnitude scaling per channel per batch item
        if np.random.rand() < self.aug_prob:
            lo, hi = self.scale_range
            scales = np.random.uniform(lo, hi,
                                       (x.shape[0], x.shape[1], 1)).astype(np.float32)
            x *= scales

        # Time warping: random circular roll of the time axis per batch item
        if self.time_warp_max > 0 and np.random.rand() < self.aug_prob:
            for b in range(x.shape[0]):
                shift = np.random.randint(-self.time_warp_max, self.time_warp_max + 1)
                if shift != 0:
                    x[b] = np.roll(x[b], shift, axis=-1)

        # Spectral augmentation: randomly zero out a contiguous frequency band
        # per batch item in the FFT domain. Forces the model to be robust to
        # missing harmonics — critical for vibration and power signal training.
        if self.spectral_drop_p > 0 and np.random.rand() < self.aug_prob:
            X_fft = np.fft.rfft(x, axis=-1)          # (B, C, F) complex
            n_freqs = X_fft.shape[-1]
            if n_freqs > self.spectral_drop_width:
                for b in range(x.shape[0]):
                    start = np.random.randint(0, n_freqs - self.spectral_drop_width)
                    drop_ch = np.random.rand(x.shape[1]) < self.spectral_drop_p
                    X_fft[b, drop_ch, start:start + self.spectral_drop_width] = 0.0
            x = np.fft.irfft(X_fft, n=x.shape[-1], axis=-1).astype(np.float32)

        return x

    def __repr__(self) -> str:
        return (f"TimeSeriesAugment(noise_std={self.noise_std}, "
                f"channel_dropout_p={self.channel_dropout_p}, "
                f"scale_range={self.scale_range}, aug_prob={self.aug_prob}, "
                f"spectral_drop_p={self.spectral_drop_p})")


class CurriculumSchedule:
    """
    Linearly increases the training sequence length from `t_min` to `t_max`
    over the first `warmup_frac` fraction of total training steps.

    After warmup, always returns `t_max`.

    Parameters
    ----------
    t_min        : starting sequence length (e.g. T // 4)
    t_max        : full sequence length
    total_steps  : total number of training steps planned
    warmup_frac  : fraction of total_steps used for curriculum warmup (default 0.3)

    Usage
    -----
    curriculum = CurriculumSchedule(t_min=16, t_max=64, total_steps=10000)
    seq_len = curriculum.get(current_step)   # int
    x_crop = x[:, :, :seq_len]
    """

    def __init__(
        self,
        t_min: int,
        t_max: int,
        total_steps: int,
        warmup_frac: float = 0.3,
    ):
        self.t_min = t_min
        self.t_max = t_max
        self.total_steps = total_steps
        self.warmup_steps = max(1, int(total_steps * warmup_frac))

    def get(self, step: int) -> int:
        """Return sequence length for this training step."""
        if step >= self.warmup_steps:
            return self.t_max
        progress = step / self.warmup_steps          # 0 → 1
        length = self.t_min + int((self.t_max - self.t_min) * progress)
        return max(self.t_min, min(length, self.t_max))

    def __repr__(self) -> str:
        return (f"CurriculumSchedule(t_min={self.t_min}, t_max={self.t_max}, "
                f"warmup_steps={self.warmup_steps})")


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
        augment: Optional[TimeSeriesAugment] = None,
        curriculum: Optional[CurriculumSchedule] = None,
        tfc_enabled: bool = False,
        mae_mask_ratio: float = 0.0,
    ):
        self.model = model
        self.config = config
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.step_count: int = 0
        self.best_loss: float = float('inf')
        self.augment = augment
        self.curriculum = curriculum
        # Phase 2: TF-C and MAE pretraining flags
        self.tfc_enabled = tfc_enabled
        self.mae_mask_ratio = mae_mask_ratio

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
            "tfc_loss":           deque(maxlen=100),
            "mae_loss":           deque(maxlen=100),
            "rmc_balance_loss":   deque(maxlen=100),
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

        # Curriculum: crop sequence to scheduled length
        if self.curriculum is not None:
            seq_len = self.curriculum.get(self.step_count)
            x = x[:, :, :seq_len]

        # Augmentation: apply stochastic transforms to input
        if self.augment is not None:
            x = self.augment(x)

        x_t = Tensor(x.astype(np.float32), requires_grad=False)
        y_t = Tensor(y.astype(np.float32), requires_grad=False)

        # Forward
        output, aux = self.model(x_t, domain_idx=domain_idx)

        # ── TF-C: Time-Frequency Consistency ─────────────────────────────
        # Encode the frequency-augmented view and pass the pair to the loss.
        if self.tfc_enabled:
            x_fft = x.copy()
            X_f = np.fft.rfft(x_fft, axis=-1)
            n_freqs = X_f.shape[-1]
            drop_w = max(1, n_freqs // 8)
            for b in range(x.shape[0]):
                st = np.random.randint(0, max(1, n_freqs - drop_w))
                X_f[b, :, st:st + drop_w] = 0.0
            x_freq = np.fft.irfft(X_f, n=x.shape[-1], axis=-1).astype(np.float32)
            x_freq_t = Tensor(x_freq, requires_grad=True)
            # Encode both views through ASE; mean-pool over time → (B, d_model)
            x_time_t = Tensor(x.astype(np.float32), requires_grad=True)
            x_time_norm = self.model.revin.normalize(x_time_t)
            x_freq_norm = self.model.revin.normalize(x_freq_t)
            h_time = self.model.ase(x_time_norm)   # (B, T, d_model)
            h_freq = self.model.ase(x_freq_norm)   # (B, T, d_model)
            # Mean-pool over T
            h_time_pool = Tensor(h_time.data.mean(axis=1), requires_grad=h_time.requires_grad,
                                 _children=(h_time,), _op="tfc_pool_t")
            h_freq_pool = Tensor(h_freq.data.mean(axis=1), requires_grad=h_freq.requires_grad,
                                 _children=(h_freq,), _op="tfc_pool_f")
            aux["tfc_pair"] = (h_time_pool, h_freq_pool)

        # ── MAE reconstruction ────────────────────────────────────────────
        if self.mae_mask_ratio > 0.0:
            _, mae_loss, _ = self.model.mae_forward(
                x_t, mask_ratio=self.mae_mask_ratio, domain_idx=domain_idx
            )
            aux["mae_loss"] = mae_loss

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

        # Distributed: synchronise gradients across all ranks before update
        if is_initialized():
            allreduce_gradients(self.model.parameters())

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

        # Distributed: average reported loss across ranks for consistent logging
        total_val = components.get("total_loss", float('inf'))
        if is_initialized():
            total_val = allreduce_scalar(total_val, op="mean")
            components["total_loss"] = total_val

        # Checkpoint on improvement
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

    def active_step(
        self,
        x_pool: np.ndarray,
        x_labeled: np.ndarray,
        y_labeled: np.ndarray,
        n_query: int = 10,
        strategy: str = "uncertainty",
        domain_idx: int = 0,
    ) -> dict:
        """
        Active learning step: query the most uncertain samples from an unlabeled
        pool, then run a supervised training step on the already-labeled data.

        x_pool    : (N, C, T) — unlabeled pool (will NOT be labeled here;
                                caller is responsible for obtaining labels)
        x_labeled : (B, C, T) — currently labeled training data
        y_labeled : (B, out_dim) — labels for x_labeled
        n_query   : number of pool samples to flag for labeling
        strategy  : "uncertainty", "entropy", "margin", "random"

        Returns:
            dict with "query_indices" (most uncertain) + train_step metrics.
        """
        from .active_learning import ActiveLearner

        learner = ActiveLearner(self.model, strategy=strategy)
        query_idx = learner.query(x_pool, n_query=n_query, domain_idx=domain_idx)

        # Train on the labeled data that we already have
        train_metrics = self.train_step(x_labeled, y_labeled, domain_idx=domain_idx)
        train_metrics["query_indices"] = query_idx.tolist()
        train_metrics["n_queried"] = int(len(query_idx))
        return train_metrics

    # ──────────────────────────────────────────────────────────────────────

    def save_checkpoint(self, tag: str = 'latest'):
        """
        Save model state_dict, optimizer state, step_count, best_loss.
        Uses numpy .npz format (no pickle).
        Only rank 0 writes to disk in distributed runs.
        """
        if is_initialized() and not is_main_process():
            return   # non-zero ranks skip — rank 0 holds the canonical copy

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
