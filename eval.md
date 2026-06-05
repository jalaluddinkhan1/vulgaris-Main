# VULGARIS — Evaluation & Benchmarking Master Plan

VULGARIS is an industrial time-series model. It must be evaluated as one.
This document defines the complete evaluation stack — datasets, metrics,
baselines, visualizations, and Kaggle execution plan.

---

## What We Are Proving

| Claim | How We Prove It | Demo # |
|---|---|---|
| O(1) streaming memory | RAM vs sequence length chart | Demo 1 |
| Competitive forecasting accuracy | ETTh1 MAE/MSE vs DLinear, PatchTST, LSTM | Demo 1 |
| Anomaly detection | F1 / AUROC on MSL, SMAP, SWaT | Demo 2 |
| Causal structure discovery | Learned CRG vs known SWaT physical graph | Demo 3 |
| Domain adaptation | CMAPSS FD001→FD003 few-shot RUL | Demo 4 |
| TTT under distribution shift | Error curve with/without TTT | Demo 5 |
| Failure propagation | Fault injection → CRG BFS attribution | Demo 3 |

---

## Datasets

### Tier 1 — Standard Benchmarks (every reviewer knows these)

| Dataset | Task | Sensors | Length | Where |
|---|---|---|---|---|
| ETTh1 / ETTh2 | Multi-horizon forecasting | 7 | 17,420 rows | GitHub (ETDataset) |
| ETTm1 / ETTm2 | Multi-horizon forecasting | 7 | 69,680 rows | GitHub (ETDataset) |
| Weather | Forecasting | 21 | 52,696 rows | Monash repo |
| Electricity | Forecasting | 321 | 26,304 rows | UCI |

### Tier 2 — Industrial Anomaly Detection

| Dataset | Task | Sensors | Labels | Where |
|---|---|---|---|---|
| MSL (Mars Science Lab) | Anomaly detection | 55 | Point anomaly | NASA/Kaggle |
| SMAP | Anomaly detection | 25 | Point anomaly | NASA/Kaggle |
| SWaT | Anomaly + causal | 51 | Attack windows | iTrust Singapore |
| SKAB | Streaming anomaly | 8 | Point + contextual | Kaggle |

### Tier 3 — VULGARIS Differentiators

| Dataset | Task | Why | Where |
|---|---|---|---|
| NASA CMAPSS | RUL prediction (4 domains) | Tests DAH domain adaptation | Kaggle |
| KPI-TSAD (Alibaba) | Telecom KPI anomaly | Tests telecom domain | Kaggle |
| METR-LA | Traffic forecasting | Multi-sensor causal | Kaggle |

---

## Baselines

### Forecasting
| Model | Type | Fair comparison? |
|---|---|---|
| DLinear | Linear, no GPU needed | Yes — same compute class |
| NLinear | Linear | Yes |
| LSTM | RNN, PyTorch | Yes — same problem class |
| PatchTST | Transformer, needs GPU | Use published numbers |
| iTransformer | Transformer, needs GPU | Use published numbers |
| TimesNet | CNN-based | Use published numbers |

### Anomaly Detection
| Model | Type |
|---|---|
| LSTM-VAE | Generative baseline |
| USAD | GAN-based anomaly |
| Anomaly Transformer | SOTA published |
| OmniAnomaly | Probabilistic |

---

## Metrics

### Forecasting
```
MAE  = mean(|ŷ - y|)
MSE  = mean((ŷ - y)²)
RMSE = sqrt(MSE)

Horizons: 96, 192, 336, 720 steps
```

### Anomaly Detection
```
F1   = 2 * P * R / (P + R)   — point-adjust F1 (standard in literature)
AUROC                          — threshold-free
Precision / Recall @ optimal threshold
```

### RUL (CMAPSS)
```
RMSE
Score = sum(exp(-ŷ/13) - 1  if ŷ < 0 else exp(ŷ/10) - 1)
```

### VULGARIS-Specific
```
Memory efficiency:  peak RAM (MB) at T = {1k, 10k, 100k, 1M} steps
Inference latency:  p50 / p95 ms per step (streaming)
Regime purity:      silhouette score of RMC assignments vs known regimes
Causal precision:   fraction of learned CRG edges matching known physical graph
```

---

## Visualization Stack (Kaggle)

| Library | Used for |
|---|---|
| `plotly` | Interactive forecast comparison, memory curves, latency |
| `matplotlib` + `seaborn` | Publication-quality static figures |
| `networkx` | CRG causal graph rendering |
| `sklearn` | Metrics, UMAP of latent space |
| `pandas` | Results tables |
| `rich` | Beautiful terminal output / Kaggle cell output |
| `tqdm` | Training progress |

---

## Demo Notebooks (Kaggle)

### Demo 1 — ETTh1 Multi-Horizon Forecasting + O(1) Memory
**File:** `eval/demo_01_forecasting.py`
**Runtime:** ~25 min on Kaggle CPU
**Outputs:**
- Forecast vs ground truth (plotly, 4 horizons)
- MAE/MSE table vs DLinear, LSTM
- Memory vs sequence length (the O(1) proof chart)
- Regime heatmap over the test period
- Inference latency p50/p95

### Demo 2 — MSL/SMAP Anomaly Detection
**File:** `eval/demo_02_anomaly.py`
**Runtime:** ~20 min on Kaggle CPU
**Outputs:**
- Anomaly score time series (plotly)
- ROC curve vs LSTM-VAE, USAD
- F1 / AUROC table
- Precision-Recall curve

### Demo 3 — SWaT Causal Graph Discovery
**File:** `eval/demo_03_causal.py`
**Runtime:** ~30 min on Kaggle CPU
**Outputs:**
- Learned CRG graph (networkx + plotly)
- Overlay on known SWaT physical pipeline
- Edge precision/recall vs ground truth
- Failure propagation demo: inject fault → show downstream attribution

### Demo 4 — CMAPSS Domain Adaptation
**File:** `eval/demo_04_domain_adaptation.py`
**Runtime:** ~20 min on Kaggle CPU
**Outputs:**
- RUL prediction curves per sub-dataset
- RMSE table: scratch vs DAH few-shot
- UMAP of latent space coloured by domain

### Demo 5 — TTT Under Distribution Shift
**File:** `eval/demo_05_ttt.py`
**Runtime:** ~15 min on Kaggle CPU
**Outputs:**
- Error over time: baseline vs TTT-enabled
- Adaptation curve (aux loss decreasing)
- Final error comparison table

---

## Model Config for Kaggle (fast but real)

```python
ModelConfig(
    input_dim   = 7,        # ETTh1 channels
    output_dim  = 1,
    d_model     = 64,       # small but real
    ase         = ASEConfig(n_filters=8, n_scales=4, filter_len=32, latent_dim=64),
    sssr        = SSSRConfig(state_dim=64, n_heads=4, d_inner=128),
    crg         = CRGConfig(n_nodes=7, n_lags=3),
    rmc         = RMCConfig(n_experts=4),
    training    = TrainingConfig(lr=3e-4, batch_size=32, seq_len=96),
    forecast_horizons = [96, 192, 336, 720],
)
```

Trains in ~15 min on Kaggle CPU. Full config (d_model=256) needs GPU or overnight.

---

## Execution Order

```
Week 1:  Demo 1  — ETTh1 forecasting  (establishes credibility)
Week 2:  Demo 2  — MSL/SMAP anomaly   (industrial relevance)
Week 3:  Demo 3  — SWaT causal graph  (unique differentiator)
Week 4:  Demo 4  — CMAPSS adaptation  (domain transfer proof)
Week 5:  Demo 5  — TTT shift         (online adaptation proof)
Week 6:  Write paper Sections 5–10 using real numbers from all demos
```

---

## What "World Class" Means for This Model

Not: beating PatchTST-large on ETTh1 (it runs on 8×A100, we run on Kaggle CPU).

Yes:
1. **Honest comparison** at the same compute tier (VULGARIS-small vs DLinear vs LSTM)
2. **Unique capabilities shown** that no baseline can demonstrate (causal graph, O(1) memory)
3. **Beautiful, reproducible notebooks** anyone can run on Kaggle free tier
4. **Statistical rigour** — 3 seeds, confidence intervals, not cherry-picked runs
5. **Published results** — Kaggle notebooks public, linked from GitHub and paper

One Kaggle notebook with a fair comparison and a compelling visualization is worth
more than any number of new features.
