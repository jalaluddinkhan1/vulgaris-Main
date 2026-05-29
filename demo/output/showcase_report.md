# VULGARIS — Capability Showcase Report

Generated: **2026-05-29T08:14:44Z**

## Executive summary

VULGARIS is a **streaming industrial AI model** (~1M parameters, CPU/edge) that combines state-space sequence modeling, **causal routing**, **episodic memory**, **multi-domain adapters**, **explainability**, and optional **safety certification** in one on-prem deployable stack.

## Live run metrics

| Metric | Value |
|--------|-------|
| Test accuracy (synthetic NOC) | 0.9556 |
| Training time (s) | 82.41 |
| Streaming latency mean (ms) | 13.207 |
| Streaming latency p95 (ms) | 18.693 |
| Total parameters | 278,004 |
| Base parameters | 139,536 |
| Adapter parameters (DAH) | 138,468 |

## Differentiators vs typical cloud / single-purpose ML

- **Streaming O(1) state** — VULGARIS: model.step(); typical: Cloud TS FMs need full-window batch / GPU
- **Edge CPU deploy** — VULGARIS: Pure NumPy, ~1M params default; typical: Billions-param models, CUDA APIs
- **Multi-site one model** — VULGARIS: DAH domain adapters; typical: Separate model per asset
- **Causal root cause** — VULGARIS: CRG DAG + Granger updates; typical: Correlation-only anomaly scores
- **Rare event memory** — VULGARIS: HMB surprise buffer + VAE archive; typical: No episodic memory
- **Explainability** — VULGARIS: ESE CART rules + attribution; typical: Black-box scores
- **Expert rules in training** — VULGARIS: Rule engine -> DAH (optional); typical: Post-hoc filters only
- **Safety projection** — VULGARIS: CBF safety head (optional); typical: Unconstrained outputs
- **On-prem / air-gap** — VULGARIS: No PyTorch/cloud dependency; typical: Data leaves the facility
- **Continual learning** — VULGARIS: SHCAL EWC + online_adapt; typical: Full retrain pipelines

## Registered demo domains

- `pop_london` → index `3`
- `pop_nyc` → index `0`
- `fab_line_3` → index `1`

## How to reproduce

```bash
python demo/company_showcase.py
docker compose up   # optional REST API on :8000
```

---
*Synthetic data demo — replace with customer historian/SNMP/OTel pipeline for pilots.*