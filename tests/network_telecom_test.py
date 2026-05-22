# -*- coding: utf-8 -*-
"""
VULGARIS — Network Telemetry End-to-End Test
=============================================
Generates diverse synthetic telecom data, trains the model, validates,
evaluates, and prints ASCII diagrams so you can see every step running.

Run from project root:
    python tests/network_telecom_test.py

CPU-only. No GPU required. No external data files needed.
"""

import sys
import os
import time
import math
import gc
import numpy as np
from collections import defaultdict

# ── Make sure project root is on path ────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from config import ModelConfig, ASEConfig, SSSRConfig, CRGConfig, HMBConfig
from config import SHCALConfig, DAHConfig, ESEConfig, HTDConfig, SafetyConfig, TrainingConfig
from model.vulgaris import Vulgaris
from training.loss import VulgarisLoss
from training.optimizer import SpectralAdamW, CosineSchedule
from training.pipeline import TrainingPipeline
from engine.tensor import Tensor

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

SEED        = 42
N_CHANNELS  = 9        # telemetry channels (see CHANNEL_NAMES)
WINDOW      = 50       # timesteps per sample  (0.5s at 100Hz)
N_SAMPLES   = 2400     # total windows generated
TRAIN_FRAC  = 0.70
VAL_FRAC    = 0.15     # test gets the rest (0.15)
EPOCHS      = 8
BATCH       = 16
LR          = 5e-4
N_CLASSES   = 5

CHANNEL_NAMES = [
    "latency_ms", "packet_loss_%", "cpu_load_%",
    "bandwidth_%", "jitter_ms", "queue_depth",
    "signal_strength_dBm", "temperature_C", "error_rate"
]

EVENT_NAMES = {
    0: "NORMAL",
    1: "CONGESTION",
    2: "HW_FAULT",
    3: "DDOS",
    4: "ROUTING_LOOP",
}

np.random.seed(SEED)


# ─────────────────────────────────────────────────────────────────────────────
# DIVERSE SYNTHETIC DATA GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

def _noise(shape, scale=1.0):
    return np.random.randn(*shape) * scale


def _smooth(arr, k=5):
    """Simple running-mean smoothing along time axis."""
    out = arr.copy()
    for i in range(k, len(arr)):
        out[i] = arr[i - k:i].mean()
    return out


def gen_normal(T):
    """Healthy network — low latency, low loss, moderate CPU."""
    t = np.linspace(0, 1, T)
    # Diurnal pattern: CPU higher during "business hours"
    diurnal = 0.3 * np.sin(2 * np.pi * t * 2 + np.random.uniform(0, 6))
    x = np.zeros((T, N_CHANNELS))
    x[:, 0] = _smooth(10 + diurnal * 15 + _noise((T,), 2))          # latency   10-40ms
    x[:, 1] = np.clip(_smooth(0.01 + _noise((T,), 0.01)), 0, 0.05)  # pkt loss  <5%
    x[:, 2] = np.clip(30 + diurnal * 25 + _noise((T,), 5), 10, 80)  # cpu       10-80%
    x[:, 3] = np.clip(40 + diurnal * 20 + _noise((T,), 8), 20, 85)  # bw        20-85%
    x[:, 4] = _smooth(2 + _noise((T,), 0.5))                         # jitter    1-5ms
    x[:, 5] = np.clip(_smooth(10 + _noise((T,), 3)), 2, 30)          # queue     2-30
    x[:, 6] = -70 + _noise((T,), 3)                                  # signal   -70dBm
    x[:, 7] = 45 + _noise((T,), 2)                                   # temp      45C
    x[:, 8] = np.clip(_smooth(0.001 + _noise((T,), 0.001)), 0, 0.01) # errors   tiny
    return x


def gen_congestion(T):
    """Gradual congestion — bandwidth saturates, latency rises, queue grows."""
    x = gen_normal(T)
    onset = T // 3
    ramp = np.linspace(0, 1, T - onset)
    # Bandwidth saturates
    x[onset:, 3] = np.clip(x[onset:, 3] + ramp * 55 + _noise((T - onset,), 4), 0, 100)
    # Latency rises as queues fill
    x[onset:, 0] = x[onset:, 0] + ramp * 180 + _noise((T - onset,), 15)
    # Queue depth explodes
    x[onset:, 5] = np.clip(x[onset:, 5] + ramp * 120, 0, 200)
    # Packet loss rises
    x[onset:, 1] = np.clip(x[onset:, 1] + ramp * 0.25, 0, 0.35)
    # CPU rises (scheduler overloaded)
    x[onset:, 2] = np.clip(x[onset:, 2] + ramp * 40, 0, 100)
    return x


def gen_hw_fault(T):
    """Hardware fault — temperature spike → errors → drops."""
    x = gen_normal(T)
    fault_t = T // 2
    # Temperature rises quickly (fan failure / hardware issue)
    x[fault_t:, 7] = np.clip(75 + np.linspace(0, 1, T - fault_t) * 35
                              + _noise((T - fault_t,), 3), 70, 110)
    # Error rate spikes
    x[fault_t:, 8] = np.clip(0.05 + np.linspace(0, 1, T - fault_t) * 0.3
                              + _noise((T - fault_t,), 0.03), 0, 0.4)
    # Signal strength drops (antenna/RF issue)
    x[fault_t:, 6] = -70 + np.linspace(0, 1, T - fault_t) * (-30) + _noise((T - fault_t,), 2)
    # Latency spikes irregularly
    x[fault_t:, 0] += np.random.exponential(20, T - fault_t)
    return x


def gen_ddos(T):
    """DDoS — sudden bandwidth + CPU spike, heavy packet loss."""
    x = gen_normal(T)
    start = T // 4
    burst_len = int(T * 0.6)
    burst = np.zeros(T)
    burst[start:start + burst_len] = 1.0
    # Near-full bandwidth instantly
    x[:, 3] = np.clip(x[:, 3] + burst * (95 + _noise((T,), 3)), 0, 100)
    # CPU maxes out processing flood
    x[:, 2] = np.clip(x[:, 2] + burst * (60 + _noise((T,), 5)), 0, 100)
    # Massive packet loss
    x[:, 1] = np.clip(x[:, 1] + burst * 0.6 + _noise((T,), 0.05), 0, 0.8)
    # Latency wildly variable under flood
    x[:, 0] = x[:, 0] + burst * np.random.exponential(100, T)
    # Queue instantly saturates
    x[:, 5] = np.clip(x[:, 5] + burst * 150, 0, 200)
    return x


def gen_routing_loop(T):
    """Routing loop — TTL-induced oscillating latency and jitter."""
    x = gen_normal(T)
    t = np.linspace(0, 1, T)
    # Oscillating latency (loop detection timeout pattern)
    freq = 3.0 + np.random.uniform(0, 2)
    amplitude = 80 + np.random.uniform(0, 60)
    x[:, 0] = np.clip(x[:, 0] + amplitude * np.abs(np.sin(2 * np.pi * freq * t)), 0, 400)
    # Jitter oscillates with same frequency but phase-shifted
    x[:, 4] = np.clip(x[:, 4] + 30 * np.abs(np.sin(2 * np.pi * freq * t + 1.2)), 0, 80)
    # Moderate packet loss from TTL expiry
    x[:, 1] = np.clip(x[:, 1] + 0.08 * np.abs(np.sin(2 * np.pi * freq * t)), 0, 0.3)
    # CPU elevated (route processing)
    x[:, 2] = np.clip(x[:, 2] + 20 * np.abs(np.sin(2 * np.pi * freq * t)), 0, 100)
    return x


GENERATORS = {
    0: gen_normal,
    1: gen_congestion,
    2: gen_hw_fault,
    3: gen_ddos,
    4: gen_routing_loop,
}


def build_dataset(n_samples=N_SAMPLES, window=WINDOW):
    """Generate balanced dataset across all 5 event types."""
    per_class = n_samples // N_CLASSES
    X_list, y_list = [], []

    for cls, gen_fn in GENERATORS.items():
        for _ in range(per_class):
            # Generate a longer sequence and take a random window
            total_len = window + 20
            sig = gen_fn(total_len)
            start = np.random.randint(0, 21)
            window_data = sig[start:start + window]   # (T, C)
            X_list.append(window_data)
            y_list.append(cls)

    X = np.stack(X_list)    # (N, T, C)
    y = np.array(y_list)    # (N,)

    # Shuffle
    idx = np.random.permutation(len(y))
    return X[idx], y[idx]


def normalize_dataset(X_train, X_val, X_test):
    """Fit normalization on train, apply to all splits."""
    # X shape: (N, T, C) — normalize per channel across N and T
    flat = X_train.reshape(-1, N_CHANNELS)
    mu  = flat.mean(0)
    std = flat.std(0) + 1e-8
    normalize = lambda X: (X - mu[None, None, :]) / std[None, None, :]
    return normalize(X_train), normalize(X_val), normalize(X_test), mu, std


def to_channels_first(X):
    """(N, T, C) → (N, C, T)  — VULGARIS expects (batch, in_channels, T)"""
    return X.transpose(0, 2, 1)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL CONFIG  (small = fast CPU training)
# ─────────────────────────────────────────────────────────────────────────────

def build_config():
    cfg = ModelConfig()
    cfg.input_dim  = N_CHANNELS
    cfg.output_dim = N_CLASSES
    cfg.n_classes  = N_CLASSES

    cfg.ase.n_scales   = 4
    cfg.ase.n_filters  = 8
    cfg.ase.filter_len = 16
    cfg.ase.latent_dim = 64
    cfg.ase.dropout    = 0.1

    cfg.sssr.state_dim = 64
    cfg.sssr.d_inner   = 128
    cfg.sssr.n_heads   = 4

    cfg.htd.n_levels        = 4
    cfg.htd.bottleneck_dim  = 16
    cfg.htd.time_constants  = [0.01, 0.1, 1.0, 10.0]

    cfg.crg.n_nodes = 16
    cfg.crg.max_edges = 64

    cfg.hmb.buffer_size   = 64
    cfg.hmb.archive_size  = 256
    cfg.hmb.embed_dim     = 64
    cfg.hmb.compress_dim  = 16

    cfg.shcal.fisher_samples = 50

    cfg.dah.n_domains    = 4
    cfg.dah.adapter_rank = 8
    cfg.dah.meta_dim     = 32

    cfg.ese.max_rules   = 16
    cfg.ese.min_samples = 10

    cfg.training.lr             = LR
    cfg.training.batch_size     = BATCH
    cfg.training.warmup_steps   = 50
    cfg.training.max_steps      = 2000
    cfg.training.grad_clip      = 1.0
    cfg.training.spectral_clip  = 2.0
    cfg.training.beta_hmb       = 0.05
    cfg.training.gamma_crg      = 0.005
    cfg.training.delta_ewc      = 0.05
    cfg.training.checkpoint_dir = os.path.join(ROOT, "checkpoints", "telecom")

    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# ASCII VISUALIZATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

WIDTH  = 70
HEIGHT = 12

def _sparkline(values, width=60, vmin=None, vmax=None):
    """Single-line ASCII sparkline."""
    chars = " ▁▂▃▄▅▆▇█"
    if len(values) == 0:
        return ""
    vmin = vmin if vmin is not None else min(values)
    vmax = vmax if vmax is not None else max(values)
    rng = vmax - vmin + 1e-9
    out = []
    step = max(1, len(values) // width)
    for i in range(0, len(values), step):
        v = values[i]
        idx = int((v - vmin) / rng * (len(chars) - 1))
        out.append(chars[max(0, min(idx, len(chars) - 1))])
    return "".join(out[:width])


def plot_training_curve(train_losses, val_losses, title="Training Curve"):
    all_vals = train_losses + val_losses
    if not all_vals:
        return
    vmin, vmax = min(all_vals), max(all_vals)
    rng = vmax - vmin + 1e-9

    bar = "─" * WIDTH
    print(f"\n┌{bar}┐")
    print(f"│  {title:<{WIDTH-2}}│")
    print(f"│  Loss range: [{vmin:.4f}, {vmax:.4f}]{' ' * (WIDTH - 28 - len(f'{vmin:.4f}') - len(f'{vmax:.4f}'))}│")
    print(f"├{bar}┤")

    def _row(losses, label, color_char):
        spark = _sparkline(losses, width=WIDTH - 14, vmin=vmin, vmax=vmax)
        last  = f"{losses[-1]:.4f}" if losses else "---"
        pad   = WIDTH - 14 - len(spark)
        print(f"│ {label:<6} {spark}{' ' * pad}  {last:>7} │")

    _row(train_losses, "TRAIN", "░")
    _row(val_losses,   "VAL  ", "▓")
    print(f"└{bar}┘")
    print(f"  Epoch 1{' ' * (WIDTH - 16)}Epoch {len(train_losses)}")


def plot_confusion_matrix(cm, labels):
    n = len(labels)
    max_label = max(len(l) for l in labels)
    col_w = 9

    bar = "─" * (max_label + 2 + n * (col_w + 1) + 1)
    print(f"\n┌{bar}┐")
    print(f"│  CONFUSION MATRIX  (rows=True, cols=Predicted){' ' * (len(bar) - 49)}│")
    print(f"├{bar}┤")

    # Header row
    header = f"│ {'':>{max_label}} │"
    for l in labels:
        header += f" {l[:col_w-1]:^{col_w-1}}│"
    print(header)
    print(f"├{bar}┤")

    for i, row_label in enumerate(labels):
        row_total = cm[i].sum()
        line = f"│ {row_label:>{max_label}} │"
        for j in range(n):
            val = cm[i, j]
            pct = 100 * val / (row_total + 1e-9)
            cell = f"{val:>3}({pct:>3.0f}%)"
            if i == j:
                cell = f"[{cell}]"
            else:
                cell = f" {cell} "
            line += f"{cell:>{col_w}}│"
        print(line)

    print(f"└{bar}┘")


def print_metrics_table(metrics_per_class, labels, overall_acc):
    bar = "─" * 62
    print(f"\n┌{bar}┐")
    print(f"│  PER-CLASS METRICS{' ' * 43}│")
    print(f"├{'─'*14}┬{'─'*9}┬{'─'*9}┬{'─'*9}┬{'─'*9}┤")
    print(f"│ {'Class':<12} │ {'Prec':^7} │ {'Recall':^7} │ {'F1':^7} │ {'Support':^7} │")
    print(f"├{'─'*14}┼{'─'*9}┼{'─'*9}┼{'─'*9}┼{'─'*9}┤")

    for i, label in enumerate(labels):
        p, r, f, s = metrics_per_class[i]
        print(f"│ {label:<12} │ {p:^7.3f} │ {r:^7.3f} │ {f:^7.3f} │ {s:^7} │")

    print(f"├{'─'*14}┴{'─'*9}┴{'─'*9}┴{'─'*9}┴{'─'*9}┤")
    print(f"│  Overall Accuracy: {overall_acc*100:.2f}%{' ' * (62 - 22 - len(f'{overall_acc*100:.2f}'))}│")
    print(f"└{bar}┘")


def print_sample_predictions(X_test, y_test, pipeline, cfg, n_show=8):
    model = pipeline.model
    model.eval()

    indices = np.random.choice(len(y_test), min(n_show, len(y_test)), replace=False)
    X_batch = to_channels_first(X_test[indices]).astype(np.float32)
    y_batch = y_test[indices]

    x_t = Tensor(X_batch, requires_grad=False)
    out, _ = model(x_t, domain_idx=0)
    probs = out.data   # (n_show, N_CLASSES)
    preds = probs.argmax(axis=1)

    bar = "─" * 72
    print(f"\n┌{bar}┐")
    print(f"│  SAMPLE PREDICTIONS — what the operator sees{' ' * 27}│")
    print(f"├{bar}┤")

    for i, idx in enumerate(indices):
        true_cls  = y_batch[i]
        pred_cls  = preds[i]
        conf      = probs[i, pred_cls] * 100
        true_name = EVENT_NAMES[true_cls]
        pred_name = EVENT_NAMES[pred_cls]
        correct   = "✓" if true_cls == pred_cls else "✗"

        # Find top-2 suspects
        top2 = probs[i].argsort()[::-1][:2]

        print(f"│  {correct} Sample {idx:>4}  |  True: {true_name:<13}  |  "
              f"Predicted: {pred_name:<13}  Conf: {conf:5.1f}%  │")

        # Causal hint
        if pred_cls == 1:
            hint = "→ bandwidth saturation → queue buildup → packet drops"
        elif pred_cls == 2:
            hint = "→ temperature spike → RF degradation → error cascade"
        elif pred_cls == 3:
            hint = "→ sudden bandwidth flood → CPU overload → mass drops"
        elif pred_cls == 4:
            hint = "→ oscillating latency → TTL expiry loop detected"
        else:
            hint = "→ all signals within normal operating bounds"

        print(f"│     Causal chain: {hint:<53}│")

        # Prob bar
        prob_bar = ""
        for c in range(N_CLASSES):
            fill = int(probs[i, c] * 10)
            prob_bar += f"{EVENT_NAMES[c][0]}{'█'*fill}{'░'*(10-fill)} "
        print(f"│     {prob_bar:<67}│")
        print(f"│{'─'*72}│")

    print(f"└{bar}┘")


def plot_channel_sample(X_test, y_test, cls=1):
    """Print ASCII sparklines of each channel for a sample of given class."""
    idxs = np.where(y_test == cls)[0]
    if len(idxs) == 0:
        return
    sample = X_test[idxs[0]]  # (T, C)

    bar = "─" * 60
    print(f"\n┌{bar}┐")
    print(f"│  SIGNAL PREVIEW — class: {EVENT_NAMES[cls]:<34}│")
    print(f"├{'─'*15}┬{'─'*44}┤")
    print(f"│ {'Channel':<13} │ {'Sparkline (time →)':<42} │")
    print(f"├{'─'*15}┼{'─'*44}┤")
    for c, name in enumerate(CHANNEL_NAMES):
        spark = _sparkline(sample[:, c], width=42)
        print(f"│ {name:<13} │ {spark:<42} │")
    print(f"└{'─'*15}┴{'─'*44}┘")


def print_causal_graph(model):
    """Print top causal edges discovered by CRG."""
    W = model.crg.W.data   # (n_nodes, n_nodes)
    n = W.shape[0]
    W_abs = np.abs(W)
    np.fill_diagonal(W_abs, 0)

    # Top 10 edges
    flat = W_abs.flatten()
    top_k = min(10, (flat > 0.01).sum())
    top_idx = flat.argsort()[::-1][:top_k]

    bar = "─" * 55
    print(f"\n┌{bar}┐")
    print(f"│  CAUSAL ROUTING GRAPH — top discovered edges{' ' * 9}│")
    print(f"├{'─'*8}┬{'─'*8}┬{'─'*12}┬{'─'*24}┤")
    print(f"│ {'From':^6} │ {'To':^6} │ {'Strength':^10} │ {'Bar':^22} │")
    print(f"├{'─'*8}┼{'─'*8}┼{'─'*12}┼{'─'*24}┤")
    max_w = flat[top_idx[0]] + 1e-9 if top_k > 0 else 1.0
    for k in range(top_k):
        i, j = divmod(top_idx[k], n)
        w = W_abs[i, j]
        bar_len = int(w / max_w * 20)
        bar_s   = "█" * bar_len + "░" * (20 - bar_len)
        print(f"│ N{i:>4}  │ N{j:>4}  │ {w:>10.4f} │ {bar_s:<20}   │")
    print(f"└{'─'*8}┴{'─'*8}┴{'─'*12}┴{'─'*24}┘")


def print_model_summary(model, cfg):
    total = model.n_base_params() + model.n_adapter_params()
    adapt = model.n_adapter_params()
    bar   = "─" * 50
    print(f"\n┌{bar}┐")
    print(f"│  VULGARIS MODEL SUMMARY{' '*26}│")
    print(f"├{bar}┤")
    print(f"│  Input channels  : {cfg.input_dim:<29}│")
    print(f"│  Output classes  : {cfg.n_classes:<29}│")
    print(f"│  Latent dim      : {cfg.ase.latent_dim:<29}│")
    print(f"│  HTD levels      : {cfg.htd.n_levels:<29}│")
    print(f"│  SSSR heads      : {cfg.sssr.n_heads:<29}│")
    print(f"│  CRG nodes       : {cfg.crg.n_nodes:<29}│")
    print(f"│  Base params     : {total - adapt:>10,}  (frozen during adapt)   │")
    print(f"│  Adapter params  : {adapt:>10,}  (LoRA, domain-switched)  │")
    print(f"│  Total params    : {total:>10,}{' '*19}│")
    print(f"└{bar}┘")


def print_section(title):
    pad = (72 - len(title) - 4) // 2
    print(f"\n{'═'*72}")
    print(f"{'═'*pad}  {title}  {'═'*(72-pad-len(title)-4)}")
    print(f"{'═'*72}")


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(model, X, y, batch_size=64, desc=""):
    model.eval()
    all_preds, all_true = [], []
    n = len(y)

    for start in range(0, n, batch_size):
        xb = to_channels_first(X[start:start + batch_size]).astype(np.float32)
        yb = y[start:start + batch_size]
        x_t = Tensor(xb, requires_grad=False)
        out, _ = model(x_t, domain_idx=0)
        preds = out.data.argmax(axis=1)
        all_preds.extend(preds.tolist())
        all_true.extend(yb.tolist())

    all_preds = np.array(all_preds)
    all_true  = np.array(all_true)

    # Confusion matrix
    cm = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
    for t, p in zip(all_true, all_preds):
        cm[t, p] += 1

    accuracy = (all_preds == all_true).mean()

    # Per-class precision, recall, F1
    metrics_per_class = []
    for c in range(N_CLASSES):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        support = cm[c, :].sum()
        prec   = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1     = 2 * prec * recall / (prec + recall + 1e-9)
        metrics_per_class.append((prec, recall, f1, support))

    return accuracy, cm, metrics_per_class


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print_section("VULGARIS — NETWORK TELEMETRY TEST")
    print(f"  Date    : 2026-05-21")
    print(f"  Device  : CPU")
    print(f"  Samples : {N_SAMPLES}  |  Window: {WINDOW}ts  |  Channels: {N_CHANNELS}")
    print(f"  Classes : {[EVENT_NAMES[i] for i in range(N_CLASSES)]}")

    # ── 1. BUILD DATASET ─────────────────────────────────────────────────────
    print_section("1 / 6  —  GENERATING DATASET")

    X, y = build_dataset()
    n     = len(y)
    n_tr  = int(n * TRAIN_FRAC)
    n_val = int(n * VAL_FRAC)
    n_te  = n - n_tr - n_val

    X_train, y_train = X[:n_tr],           y[:n_tr]
    X_val,   y_val   = X[n_tr:n_tr+n_val], y[n_tr:n_tr+n_val]
    X_test,  y_test  = X[n_tr+n_val:],     y[n_tr+n_val:]

    X_train, X_val, X_test, mu, std = normalize_dataset(X_train, X_val, X_test)

    print(f"  Train : {n_tr:>4} samples")
    print(f"  Val   : {n_val:>4} samples")
    print(f"  Test  : {n_te:>4} samples")

    # Class distribution
    print(f"\n  Class distribution (train):")
    for c in range(N_CLASSES):
        cnt = (y_train == c).sum()
        bar = "█" * (cnt // 5)
        print(f"    {EVENT_NAMES[c]:<14}  {cnt:>4}  {bar}")

    # Signal preview
    plot_channel_sample(X_train, y_train, cls=1)   # show congestion pattern
    plot_channel_sample(X_train, y_train, cls=3)   # show DDoS pattern

    # ── 2. BUILD MODEL ───────────────────────────────────────────────────────
    print_section("2 / 6  —  BUILDING MODEL")

    cfg   = build_config()
    model = Vulgaris(cfg)
    print_model_summary(model, cfg)

    loss_fn   = VulgarisLoss(cfg, task="classification")
    optimizer = SpectralAdamW(
        model.parameters(),
        lr=LR,
        weight_decay=0.01,
        grad_clip=1.0,
        spectral_clip=2.0,
    )
    total_steps = EPOCHS * (n_tr // BATCH)
    scheduler = CosineSchedule(
        optimizer,
        warmup_steps=cfg.training.warmup_steps,
        max_steps=total_steps,
        min_lr=cfg.training.min_lr,
    )
    pipeline = TrainingPipeline(model, cfg, loss_fn, optimizer, scheduler)

    # ── 3. TRAIN ─────────────────────────────────────────────────────────────
    print_section("3 / 6  —  TRAINING")

    train_losses, val_losses = [], []

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        # Shuffle train data each epoch
        perm = np.random.permutation(n_tr)
        X_tr_shuf = X_train[perm]
        y_tr_shuf = y_train[perm]

        # Batch iterator
        def batch_iter():
            for start in range(0, n_tr - BATCH + 1, BATCH):
                xb = to_channels_first(X_tr_shuf[start:start + BATCH]).astype(np.float32)
                yb = y_tr_shuf[start:start + BATCH].astype(np.float32)
                yield xb, yb

        epoch_metrics = pipeline.train_epoch(batch_iter(), domain_idx=0)
        t_train = time.time() - t0

        # Validation
        val_metrics = {}
        model.eval()
        val_losses_ep = []
        for start in range(0, n_val - BATCH + 1, BATCH):
            xb = to_channels_first(X_val[start:start + BATCH]).astype(np.float32)
            yb = y_val[start:start + BATCH].astype(np.float32)
            vm = pipeline.eval_step(xb, yb, domain_idx=0)
            val_losses_ep.append(vm.get("total_loss", 0.0))
        val_loss = float(np.mean(val_losses_ep)) if val_losses_ep else 0.0

        tr_loss = epoch_metrics.get("total_loss", 0.0)
        train_losses.append(tr_loss)
        val_losses.append(val_loss)

        # Quick accuracy on a val subset
        val_acc_x = to_channels_first(X_val[:128]).astype(np.float32)
        val_acc_t = Tensor(val_acc_x, requires_grad=False)
        model.eval()
        val_out, _ = model(val_acc_t, domain_idx=0)
        val_preds  = val_out.data.argmax(axis=1)
        val_acc    = (val_preds == y_val[:128]).mean()

        print(f"  Epoch {epoch:>2}/{EPOCHS}  "
              f"train_loss={tr_loss:.4f}  val_loss={val_loss:.4f}  "
              f"val_acc={val_acc*100:.1f}%  "
              f"lr={optimizer.lr:.2e}  "
              f"time={t_train:.1f}s")
        gc.collect()   # free autograd graphs from this epoch

    plot_training_curve(train_losses, val_losses, title="VULGARIS — Network Telemetry Training")

    # ── 4. EVALUATE ──────────────────────────────────────────────────────────
    print_section("4 / 6  —  EVALUATION")

    labels = [EVENT_NAMES[i] for i in range(N_CLASSES)]

    print("\n  [Validation set]")
    val_acc, val_cm, val_metrics = evaluate(model, X_val, y_val, desc="val")
    plot_confusion_matrix(val_cm, labels)
    print_metrics_table(val_metrics, labels, val_acc)

    print("\n  [Test set — held-out]")
    test_acc, test_cm, test_metrics = evaluate(model, X_test, y_test, desc="test")
    plot_confusion_matrix(test_cm, labels)
    print_metrics_table(test_metrics, labels, test_acc)

    # ── 5. SAMPLE PREDICTIONS ────────────────────────────────────────────────
    print_section("5 / 6  —  SAMPLE ALERT PREDICTIONS")
    print_sample_predictions(X_test, y_test, pipeline, cfg, n_show=10)

    # ── 6. CAUSAL GRAPH ──────────────────────────────────────────────────────
    print_section("6 / 6  —  CAUSAL GRAPH & DIAGNOSTICS")
    print_causal_graph(model)

    # Latency benchmark
    print(f"\n  Latency benchmark (100 streaming steps, CPU):")
    state  = model.init_state(batch_size=1)
    model.eval()
    times  = []
    sample = to_channels_first(X_test[:1]).astype(np.float32)   # (1, C, T)
    for step_i in range(100):
        x_step = Tensor(sample[:, :, step_i % WINDOW], requires_grad=False)
        t0 = time.perf_counter()
        out_step, state = model.step(x_step, state, domain_idx=0)
        times.append((time.perf_counter() - t0) * 1000)
    times = sorted(times)
    p50 = times[50]; p95 = times[94]; p99 = times[98]
    print(f"    p50={p50:.2f}ms  p95={p95:.2f}ms  p99={p99:.2f}ms")
    spark_lat = _sparkline(times, width=50, vmin=0)
    print(f"    latency distribution: [{spark_lat}]")

    # EWC / SHCAL status
    shcal_stats = model.shcal.structural_update()
    print(f"\n  SHCAL adaptation counters:")
    print(f"    pruned weights : {shcal_stats.get('n_pruned', 0)}")
    print(f"    grown  weights : {shcal_stats.get('n_grown', 0)}")
    print(f"    sparsity       : {shcal_stats.get('sparsity', 0):.4f}")

    # ── FINAL SUMMARY ─────────────────────────────────────────────────────────
    bar = "═" * 72
    print(f"\n{bar}")
    print(f"  FINAL RESULTS")
    print(f"{bar}")
    print(f"  Validation accuracy : {val_acc*100:.2f}%")
    print(f"  Test accuracy       : {test_acc*100:.2f}%")
    macro_f1 = np.mean([m[2] for m in test_metrics])
    print(f"  Test macro-F1       : {macro_f1:.4f}")
    print(f"  Streaming latency   : p50={p50:.2f}ms  p95={p95:.2f}ms")
    print(f"  Total params        : {model.n_base_params() + model.n_adapter_params():,}")
    print(f"{bar}")
    print(f"\n  Model is running completely end-to-end on CPU.")
    print(f"  All 6 stages passed: data → train → val → test → alerts → causal graph.\n")

    # Save checkpoint
    pipeline.save_checkpoint(tag="telecom_final")
    print(f"  Checkpoint saved → checkpoints/telecom/checkpoint_telecom_final.npz\n")


if __name__ == "__main__":
    main()
