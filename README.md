![VULGARIS](https://raw.githubusercontent.com/jalaluddinkhan1/vulgaris-Main/main/VULGARIS-Logo.png)

# VULGARIS

**Streaming causal state-space foundation model for industrial and edge intelligence.**

[![PyPI version](https://img.shields.io/pypi/v/vulgaris.svg)](https://pypi.org/project/vulgaris/)
[![Python](https://img.shields.io/pypi/pyversions/vulgaris.svg)](https://pypi.org/project/vulgaris/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![CI](https://github.com/jalaluddinkhan1/vulgaris-Main/actions/workflows/ci.yml/badge.svg)](https://github.com/jalaluddinkhan1/vulgaris-Main/actions)
[![Tests](https://img.shields.io/badge/tests-220%20passed-brightgreen.svg)](tests/)

---

VULGARIS is an open-source foundational model for real-time telemetry, sensor fusion, and time-series intelligence across industrial environments. It is written entirely in **NumPy** with a custom reverse-mode autograd engine — no PyTorch, no TensorFlow, no CUDA dependency. Trains on CPU. Deploys on edge hardware.

## Why VULGARIS

| Requirement | Transformers | Mamba | Chronos | **VULGARIS** |
|---|---|---|---|---|
| O(1) streaming state | ✗ | ✓ | ✗ | ✓ |
| Multi-rate sensor fusion | ✗ | ✗ | ✗ | ✓ |
| Causal structure discovery | ✗ | ✗ | ✗ | ✓ |
| Certified safety output | ✗ | ✗ | ✗ | ✓ |
| Online continual learning | ✗ | ✗ | ✗ | ✓ |
| Edge deployable (no GPU) | ✗ | ✗ | ✗ | ✓ |
| Multi-horizon forecasting | ~ | ~ | ✓ | ✓ |

## Architecture

![VULGARIS Architecture](https://raw.githubusercontent.com/jalaluddinkhan1/vulgaris-Main/main/Flow.png)

| Module | Role |
|---|---|
| **ASE** — Adaptive Signal Embedding | Morlet wavelet filterbank; per-channel frequency extraction before cross-channel mixing |
| **HTD** — Hierarchical Timescale Decomposition | Four-level parallel SSM; captures dynamics from milliseconds to hours |
| **SSSR** — Selective State-Space Recurrence | ZOH-discretised SSM with HiPPO-LegS init, adaptive timestep, Hebbian online adaptation |
| **RMC** — Regime Mixture Core | Soft Mixture-of-Experts over operating regimes; sparse top-k routing with load-balancing |
| **CRG** — Causal Routing Graph | DAGMA-constrained DAG; Neural Granger mask; regime-conditioned adjacency; failure propagation |
| **HMB** — Hierarchical Memory Bank | VAE-compressed episodic slots; surprise-triggered writes; uncertainty-weighted retrieval |
| **DAH** — Domain-Adaptive Hypernetwork | LoRA-style adapter generation; zero-shot domain switching at inference |
| **ICL** — In-Context Learning | Attention-pooled context encoder; persistent episodic memory across sessions |
| **ESE** — Explainability Engine | CART rule extraction from latent activations; gradient attribution; counterfactuals |
| **SHCAL** | EWC + Hebbian continual learning; prevents catastrophic forgetting |
| **CMLA** | InfoNCE cross-modal latent alignment; multi-sensor fusion |
| **Safety** | Control Barrier Function head; Lipschitz-certified safe action projection |

## Installation

```bash
pip install vulgaris
```

```bash
pip install "vulgaris[serve]"   # + FastAPI REST server
pip install "vulgaris[train]"   # + tqdm + rich progress bars
pip install "vulgaris[all]"     # everything
```

**Requirements:** Python ≥ 3.10 · NumPy ≥ 1.26 · SciPy ≥ 1.11 · PyYAML ≥ 6.0

---

## Quick Start

```python
import numpy as np
from vulgaris import Vulgaris, ModelConfig, Tensor

model = Vulgaris(ModelConfig(input_dim=9, output_dim=1))

# Batch inference: (batch, channels, timesteps)
x = Tensor(np.random.randn(8, 9, 64).astype(np.float32))
prediction, aux = model(x)
print(prediction.data.shape)   # (8, 1)
```

## Streaming (O(1) Memory)

```python
state = model.init_state(batch_size=1)

for sensor_reading in live_stream:
    x_t = Tensor(sensor_reading.astype(np.float32))   # (1, 9)
    output_t, state = model.step(x_t, state)
    print(output_t.data)
```

## Training

```python
from vulgaris import (
    Vulgaris, ModelConfig,
    TrainingPipeline, VulgarisLoss,
    SpectralAdamW, CosineSchedule,
)

config    = ModelConfig(input_dim=9, output_dim=1)
model     = Vulgaris(config)
optimizer = SpectralAdamW(model.parameters(), lr=3e-4)
scheduler = CosineSchedule(optimizer, warmup_steps=1000, max_steps=50_000)
pipeline  = TrainingPipeline(model, config, VulgarisLoss(config), optimizer, scheduler)

metrics = pipeline.train_step(x_np, y_np)
# → {"loss": 0.043, "rmc_balance_loss": 0.001, "dag_penalty": 0.002, ...}
```

## Multi-Horizon Forecasting

```python
# Forecast at horizons 1, 5, and 20 steps ahead — all in one pass
config = ModelConfig(input_dim=9, output_dim=1, forecast_horizons=[1, 5, 20])
model  = Vulgaris(config)

prediction, aux = model(x)
mh = aux["multi_horizon"]   # (batch, 3, 1) — one forecast per horizon
print(mh.shape)              # (8, 3, 1)
```

## Causal Graph & Failure Propagation

```python
# Which sensors caused this fault?
model.crg.update_structure(latent_states)

# Forward-propagate a fault from sensor 3
affected = model.crg.propagate_failure(
    triggered_nodes=[3],
    max_hops=4,
    decay=0.85,
)
# → {5: 0.72, 7: 0.61, 12: 0.48}  — node: failure probability
```

## Regime-Aware Processing

```python
# Which operating regime is active at each timestep?
regimes = model.rmc.regime_assignments(z)   # (batch, T) — 0…K-1

# Visualise routing: which expert each timestep uses
import numpy as np
counts = np.bincount(regimes.flatten(), minlength=4)
print("Expert utilisation:", counts / counts.sum())
```

## Persistent Episodic Memory

```python
from vulgaris import EpisodicMemory, InContextLearning

# Episodes persist across inference calls
icl = InContextLearning(
    d_model=256,
    episodic_memory=EpisodicMemory(capacity=512, d_model=256),
)

# First call stores the context; subsequent calls retrieve similar past episodes
output = icl.encode_context(z_ref, retrieve_k=4)
```

## Causal Memory Queries

```python
# Discover which sensors causally precede sensor 7
causes = model.causal_memory.query_causes(effect=7, min_confidence=0.3)
# → [(3, 0.81), (5, 0.67), (9, 0.44)]  — (cause_node, confidence)

# Top strongest causal relationships in the model's learned graph
edges = model.causal_memory.strongest_edges(top_k=10)
```

## Domain Adaptation (Zero-Shot)

```python
# Switch domain at inference — no retraining, no gradient step
output_factory, _ = model(x, domain_idx=0)
output_telecom, _ = model(x, domain_idx=3)
```

## Anomaly Detection with RevIN

```python
prediction, aux = model(x)

# RevIN stats preserved for anomaly detection
mean  = aux["revin_mean"]          # per-channel mean of this window
std   = aux["revin_std"]           # per-channel std
energy = aux["anomaly_energy"]     # L2 norm of denorm residual — spikes on anomalies
```

## Drift Detection

```python
from vulgaris import DriftDetector

detector = DriftDetector(window_size=200)
detector.set_reference(reference_data)
result = detector.update(new_batch)

if result["drift_detected"]:
    print(f"KS={result['ks_stat']:.3f}  MMD={result['mmd_stat']:.3f}")
```

## Save / Load

```python
model.save("checkpoints/my_run")   # writes weights.npz + config.yaml + metadata.json
model = Vulgaris.load("checkpoints/my_run")
```

No pickle. No binary blobs. Checkpoints are inspectable with `np.load`.

## Configuration

```python
from vulgaris import ModelConfig
from vulgaris.config import SSSRConfig, RMCConfig, CRGConfig

config = ModelConfig(
    input_dim=32,
    output_dim=1,
    d_model=256,
    forecast_horizons=[1, 5, 20],
    sssr=SSSRConfig(state_dim=512, n_heads=16),
    rmc=RMCConfig(n_experts=4, top_k=2),
    crg=CRGConfig(n_nodes=32, lambda_dag=0.01),
)
config.to_yaml("config.yaml")

# Load from environment variables (containerised deployments)
# VULGARIS_INPUT_DIM=32 VULGARIS_D_MODEL=256
config = ModelConfig.from_env()
```

---

## Docker

### Run the inference server

```bash
docker compose up
```

Starts the **VULGARIS inference server** on port `8000` and a **Prometheus** sidecar on port `9090`.

### Configure at runtime

```bash
VULGARIS_INPUT_DIM=32 \
VULGARIS_D_MODEL=256 \
VULGARIS_API_KEYS=secret123 \
docker compose up
```

All `VULGARIS_*` variables can be set in a `.env` file:

```env
VULGARIS_INPUT_DIM=32
VULGARIS_OUTPUT_DIM=1
VULGARIS_N_CLASSES=0
VULGARIS_D_MODEL=256
VULGARIS_CHECKPOINT=checkpoints/production
VULGARIS_API_KEYS=key1,key2
```

### Load a checkpoint

```bash
VULGARIS_CHECKPOINT=checkpoints/my_run docker compose up
```

Checkpoints are mounted from `./checkpoints` into the container — no rebuild needed.

### Build only (no Prometheus)

```bash
docker compose up vulgaris
```

### Resource limits

| Service | Memory | CPU |
|---|---|---|
| vulgaris | 2 GB | 2 cores |
| prometheus | 512 MB | — |

### REST API

| Method | Route | Description |
|---|---|---|
| `POST` | `/predict` | Batch prediction |
| `POST` | `/stream/start` | Start stateful streaming session |
| `POST` | `/stream/{id}/step` | Single-step streaming |
| `GET` | `/stream/{id}/stats` | Session latency stats |
| `DELETE` | `/stream/{id}` | Close session |
| `GET` | `/health` | Model info + degradation level |
| `GET` | `/metrics` | Prometheus exposition |
| `POST` | `/explain` | CART rule extraction |
| `POST` | `/counterfactual` | Gradient-based counterfactual |

---

## Inference Server (without Docker)

```bash
export VULGARIS_INPUT_DIM=9
export VULGARIS_N_CLASSES=5
vulgaris-serve
```

---

## Repository Structure

```
vulgaris/
├── vulgaris/           public API, config, utils
├── engine/             autograd engine — Tensor, Module, layers, parallel_scan, fft_conv
├── modules/            ASE · HTD · SSSR · RMC · CRG · HMB · DAH · ICL · ESE
│                       Safety · SHCAL · CMLA · OntologyEmbedding · RuleEngine
├── model/              full Vulgaris model assembly + MultiHorizonHead
├── memory/             EpisodicMemory · CausalMemory
├── training/           loss · optimizer · pipeline · conformal · distillation
│                       active_learning · distributed · self_supervised
├── inference/          StreamingInference · SpeculativeRollout · EventBuffer
├── serve/              FastAPI server · auth · metrics · degradation · versioning
├── monitoring/         DriftDetector
├── preprocessing/      IndustrialTokenizer · LogEncoder
├── federated/          DP-SGD federated learning
├── benchmarks/         baselines · synthetic datasets
├── scripts/            distributed training launcher
├── deploy/             prometheus.yml
└── tests/              220 unit tests — gradients, numerical stability, modules
```

---

## Self-Supervised Pretraining

```python
from training.self_supervised import SelfSupervisedTrainer

trainer = SelfSupervisedTrainer(model, in_channels=9, d_model=256)
metrics = trainer.pretrain_step(x_np)
# masked reconstruction + temporal InfoNCE — no labels required
```

## Federated Learning

```python
from federated.protocol import FederatedCoordinator

coord = FederatedCoordinator(model, n_clients=10, dp_epsilon=1.0)
coord.aggregate([client_grad_1, client_grad_2, ...])
# Byzantine detection + trimmed mean + DP-SGD noise injection
```

---

## Reproducibility

```python
from vulgaris import set_seed
set_seed(42)
```

---

## Citation

```bibtex
@software{vulgaris2026,
  title   = {VULGARIS: Streaming Causal State-Space Foundation Model for Industrial Intelligence},
  author  = {Khan, Jalaluddin},
  year    = {2026},
  version = {0.7.0},
  url     = {https://github.com/jalaluddinkhan1/vulgaris-Main},
  license = {Apache-2.0},
}
```

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
