#!/usr/bin/env python3
"""
VULGARIS — Company capability showcase (CPU-only, ~3–8 minutes).

Runs end-to-end: train on synthetic network telemetry, measure streaming latency,
register domains, print causal graph, save checkpoint + JSON/Markdown report.

Usage (from repo root):
    python demo/company_showcase.py
    python demo/company_showcase.py --quick    # fewer samples / epochs
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from config import ModelConfig
from engine.tensor import Tensor
from model.vulgaris import Vulgaris
from training.loss import VulgarisLoss
from training.optimizer import SpectralAdamW, CosineSchedule
from training.pipeline import TrainingPipeline

# Reuse proven synthetic network generators from the telecom test
from tests.network_telecom_test import (  # noqa: E402
    CHANNEL_NAMES,
    EVENT_NAMES,
    N_CHANNELS,
    build_dataset,
    build_config,
    normalize_dataset,
    print_causal_graph,
    print_model_summary,
    to_channels_first,
)

OUTPUT_DIR = os.path.join(ROOT, "demo", "output")

DIFFERENTIATORS = [
    ("Streaming O(1) state", "model.step()", "Cloud TS FMs need full-window batch / GPU"),
    ("Edge CPU deploy", "Pure NumPy, ~1M params default", "Billions-param models, CUDA APIs"),
    ("Multi-site one model", "DAH domain adapters", "Separate model per asset"),
    ("Causal root cause", "CRG DAG + Granger updates", "Correlation-only anomaly scores"),
    ("Rare event memory", "HMB surprise buffer + VAE archive", "No episodic memory"),
    ("Explainability", "ESE CART rules + attribution", "Black-box scores"),
    ("Expert rules in training", "Rule engine -> DAH (optional)", "Post-hoc filters only"),
    ("Safety projection", "CBF safety head (optional)", "Unconstrained outputs"),
    ("On-prem / air-gap", "No PyTorch/cloud dependency", "Data leaves the facility"),
    ("Continual learning", "SHCAL EWC + online_adapt", "Full retrain pipelines"),
]


def _banner(title: str) -> None:
    w = 72
    print("\n" + "=" * w)
    print(f"  {title}")
    print("=" * w)


def print_differentiators() -> None:
    _banner("WHAT WE BUILT - CAPABILITIES OTHERS TYPICALLY DO NOT SHIP TOGETHER")
    print(f"  {'Capability':<28} {'In VULGARIS':<22} {'Typical alternative'}")
    print("  " + "-" * 68)
    for cap, ours, alt in DIFFERENTIATORS:
        print(f"  {cap:<28} {ours:<22} {alt}")


def train_and_evaluate(model: Vulgaris, cfg: ModelConfig, n_samples: int, epochs: int):
    _banner("LIVE TRAIN - SYNTHETIC NETWORK TELEMETRY (router/NOC style KPIs)")
    print(f"  Channels: {', '.join(CHANNEL_NAMES)}")
    print(f"  Events:   {list(EVENT_NAMES.values())}")

    X, y = build_dataset(n_samples=n_samples, window=50)
    n = len(y)
    n_tr, n_val = int(n * 0.7), int(n * 0.15)
    X_tr, y_tr = X[:n_tr], y[:n_tr]
    X_val, y_val = X[n_tr : n_tr + n_val], y[n_tr : n_tr + n_val]
    X_te, y_te = X[n_tr + n_val :], y[n_tr + n_val :]

    X_tr, X_val, X_te, _, _ = normalize_dataset(X_tr, X_val, X_te)

    loss_fn = VulgarisLoss(cfg)
    opt = SpectralAdamW(model.parameters(), lr=cfg.training.lr)
    sched = CosineSchedule(
        opt,
        warmup_steps=cfg.training.warmup_steps,
        max_steps=max(cfg.training.max_steps, epochs * (n_tr // cfg.training.batch_size + 1)),
        min_lr=1e-6,
    )
    pipeline = TrainingPipeline(model, cfg, loss_fn, opt, sched)

    t0 = time.perf_counter()
    for ep in range(epochs):
        model.train()
        perm = np.random.permutation(n_tr)
        losses = []
        for start in range(0, n_tr, cfg.training.batch_size):
            idx = perm[start : start + cfg.training.batch_size]
            xb = to_channels_first(X_tr[idx]).astype(np.float32)
            yb = y_tr[idx]
            m = pipeline.train_step(xb, yb, domain_idx=0)
            losses.append(float(m.get("loss", 0)))
        print(f"  epoch {ep + 1}/{epochs}  loss={np.mean(losses):.4f}")

    train_sec = time.perf_counter() - t0

    model.eval()
    correct, total = 0, 0
    for start in range(0, len(y_te), cfg.training.batch_size):
        xb = to_channels_first(X_te[start : start + cfg.training.batch_size]).astype(np.float32)
        yb = y_te[start : start + cfg.training.batch_size]
        out, _ = model(Tensor(xb, requires_grad=False), domain_idx=0)
        pred = out.data.argmax(axis=1)
        correct += int((pred == yb).sum())
        total += len(yb)
    acc = correct / max(total, 1)

    return {
        "train_seconds": round(train_sec, 2),
        "test_accuracy": round(float(acc), 4),
        "n_train": n_tr,
        "n_test": len(y_te),
        "epochs": epochs,
    }


def measure_streaming_latency(model: Vulgaris, n_steps: int = 200) -> dict:
    _banner("STREAMING INFERENCE - ONE TIMESTEP PER CALL (edge / PLC style)")
    model.eval()
    B = 1
    state = model.init_state(B)
    x_sample = np.random.randn(B, N_CHANNELS).astype(np.float32) * 0.5

    # Warmup
    for _ in range(20):
        xt = Tensor(x_sample, requires_grad=False)
        _, state = model.step(xt, state)

    state = model.init_state(B)
    times_ms = []
    for _ in range(n_steps):
        xt = Tensor(x_sample, requires_grad=False)
        t0 = time.perf_counter()
        _, state = model.step(xt, state)
        times_ms.append((time.perf_counter() - t0) * 1000)

    arr = np.array(times_ms)
    return {
        "steps": n_steps,
        "latency_ms_mean": round(float(arr.mean()), 3),
        "latency_ms_p95": round(float(np.percentile(arr, 95)), 3),
        "latency_ms_max": round(float(arr.max()), 3),
    }


def demo_domains(model: Vulgaris) -> dict:
    _banner("MULTI-DOMAIN - ONE MODEL, MANY SITES (no full retrain)")
    sites = [
        ("pop_london", {"region": "eu-west", "role": "core_router"}),
        ("pop_nyc", {"region": "us-east", "role": "core_router"}),
        ("fab_line_3", {"region": "on-prem", "role": "process"}),
    ]
    registered = []
    for name, meta in sites:
        idx = model.dah.register_domain(name, meta)
        registered.append({"name": name, "domain_idx": int(idx), "metadata": meta})
        print(f"  registered {name!r} -> domain_idx={idx}")
    return {"domains": registered}


def demo_causal_and_explain(model: Vulgaris, cfg: ModelConfig) -> dict:
    _banner("CAUSAL GRAPH (CRG) - ROOT-CAUSE STRUCTURE OVER LATENT NODES")
    print_causal_graph(model)

    out = {"causal_graph_printed": True}
    x_demo = Tensor(
        np.random.randn(1, N_CHANNELS, 50).astype(np.float32),
        requires_grad=False,
    )
    y_demo = Tensor(np.array([[1.0, 0, 0, 0, 0]], dtype=np.float64), requires_grad=False)
    try:
        expl = model.ese.explain(x_demo, y_demo)
        rules = expl.get("rules", [])[:5]
        top = expl.get("top_features", [])[:5]
        out["sample_rules"] = rules
        out["top_features"] = [(n, round(v, 4)) for n, v in top]
        if rules:
            print("\n  Sample extracted rules (ESE):")
            for r in rules[:3]:
                print(f"    - {r}")
    except Exception as e:
        out["explain_note"] = str(e)
    return out


def save_artifacts(model: Vulgaris, report: dict) -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ckpt = os.path.join(OUTPUT_DIR, "showcase_checkpoint")
    model.save(ckpt)
    report["checkpoint_dir"] = ckpt
    report["generated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    json_path = os.path.join(OUTPUT_DIR, "showcase_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    md_path = os.path.join(OUTPUT_DIR, "showcase_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(_markdown_report(report))

    print(f"\n  Saved: {json_path}")
    print(f"  Saved: {md_path}")
    print(f"  Saved: {ckpt}/")
    return md_path


def _markdown_report(r: dict) -> str:
    lines = [
        "# VULGARIS — Capability Showcase Report",
        "",
        f"Generated: **{r.get('generated_at', 'n/a')}**",
        "",
        "## Executive summary",
        "",
        "VULGARIS is a **streaming industrial AI model** (~1M parameters, CPU/edge) that combines "
        "state-space sequence modeling, **causal routing**, **episodic memory**, **multi-domain adapters**, "
        "**explainability**, and optional **safety certification** in one on-prem deployable stack.",
        "",
        "## Live run metrics",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Test accuracy (synthetic NOC) | {r.get('training', {}).get('test_accuracy', 'n/a')} |",
        f"| Training time (s) | {r.get('training', {}).get('train_seconds', 'n/a')} |",
        f"| Streaming latency mean (ms) | {r.get('streaming', {}).get('latency_ms_mean', 'n/a')} |",
        f"| Streaming latency p95 (ms) | {r.get('streaming', {}).get('latency_ms_p95', 'n/a')} |",
        f"| Total parameters | {r.get('model', {}).get('total_params', 'n/a'):,} |",
        f"| Base parameters | {r.get('model', {}).get('base_params', 'n/a'):,} |",
        f"| Adapter parameters (DAH) | {r.get('model', {}).get('adapter_params', 'n/a'):,} |",
        "",
        "## Differentiators vs typical cloud / single-purpose ML",
        "",
    ]
    for cap, ours, alt in DIFFERENTIATORS:
        lines.append(f"- **{cap}** — VULGARIS: {ours}; typical: {alt}")
    lines.extend([
        "",
        "## Registered demo domains",
        "",
    ])
    for d in r.get("domains", {}).get("domains", []):
        lines.append(f"- `{d['name']}` → index `{d['domain_idx']}`")
    lines.extend([
        "",
        "## How to reproduce",
        "",
        "```bash",
        "python demo/company_showcase.py",
        "docker compose up   # optional REST API on :8000",
        "```",
        "",
        "---",
        "*Synthetic data demo — replace with customer historian/SNMP/OTel pipeline for pilots.*",
    ])
    return "\n".join(lines)


def _configure_stdout() -> None:
    """Avoid UnicodeEncodeError on Windows cp1252 consoles."""
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def main():
    _configure_stdout()
    parser = argparse.ArgumentParser(description="VULGARIS company capability showcase")
    parser.add_argument("--quick", action="store_true", help="Faster run (fewer samples/epochs)")
    args = parser.parse_args()

    n_samples = 600 if args.quick else 1200
    epochs = 3 if args.quick else 6

    print_differentiators()

    _banner("MODEL FOOTPRINT - EDGE-SCALE, NOT BILLION-PARAMETER CLOUD")
    cfg = build_config()
    model = Vulgaris(cfg)
    print_model_summary(model, cfg)

    report = {
        "product": "VULGARIS",
        "tagline": "Streaming causal industrial intelligence — on-prem, CPU, explainable",
        "model": {
            "base_params": model.n_base_params(),
            "adapter_params": model.n_adapter_params(),
            "total_params": model.n_base_params() + model.n_adapter_params(),
            "d_model": model.d_model,
            "input_channels": N_CHANNELS,
            "channel_names": CHANNEL_NAMES,
            "event_classes": EVENT_NAMES,
        },
        "differentiators": [
            {"capability": a, "vulgaris": b, "typical_alternative": c}
            for a, b, c in DIFFERENTIATORS
        ],
    }

    report["training"] = train_and_evaluate(model, cfg, n_samples, epochs)
    report["streaming"] = measure_streaming_latency(model)
    report["domains"] = demo_domains(model)
    report["explainability"] = demo_causal_and_explain(model, cfg)

    _banner("DONE - LEAVE-BEHIND ARTIFACTS FOR THE COMPANY")
    md = save_artifacts(model, report)
    print(f"\n  Open for the meeting: {md}")
    print("  Optional API demo:  docker compose up  ->  http://localhost:8000/health")
    print("  Full network test:  python tests/network_telecom_test.py")


if __name__ == "__main__":
    main()
