![VULGARIS](https://raw.githubusercontent.com/jalaluddinkhan1/vulgaris-Main/main/VULGARIS-Logo.png)

# VULGARIS

**A foundational model for streaming industrial intelligence.**

[![PyPI version](https://img.shields.io/pypi/v/vulgaris.svg)](https://pypi.org/project/vulgaris/)
[![Python](https://img.shields.io/pypi/pyversions/vulgaris.svg)](https://pypi.org/project/vulgaris/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![CI](https://github.com/jalaluddinkhan1/vulgaris-Main/actions/workflows/ci.yml/badge.svg)](https://github.com/jalaluddinkhan1/vulgaris-Main/actions)

---

VULGARIS is an open-source foundational model designed for real-time telemetry, sensor fusion, and time-series intelligence across industrial environments. It is written entirely in NumPy with a custom reverse-mode automatic differentiation engine — no PyTorch, no TensorFlow, no CUDA dependency.

The architecture combines state-space recurrence, causal graph learning, hierarchical memory, and domain-adaptive hypernetworks into a single unified model that trains on a CPU and deploys on edge hardware.

## Architecture

![VULGARIS Architecture](https://raw.githubusercontent.com/jalaluddinkhan1/vulgaris-Main/main/Flow.png)

| Module | Role |
|---|---|
| **ASE** — Adaptive Signal Embedding | Morlet wavelet filterbank; converts raw channels to latent sequences |
| **HTD** — Hierarchical Timescale Decomposition | Four-level parallel SSM; captures dynamics from milliseconds to minutes |
| **SSSR** — Selective State-Space Recurrence | Diagonal ZOH state-space core; selective gating per timestep |
| **DAH** — Domain-Adaptive Hypernetwork | LoRA-style adapter generation; zero-shot domain switching at inference |
| **CRG** — Causal Routing Graph | NOTEARS-constrained DAG; learns causal structure from data online |
| **HMB** — Hierarchical Memory Bank | VAE-compressed episodic slots; retrieval-augmented state transitions |
| **ESE** — Explainability Engine | CART rule extraction from latent activations; human-readable decisions |
| **Safety** | Control Barrier Function head; certified safe action projection |
| **SHCAL** | EWC + Hebbian continual learning; prevents catastrophic forgetting |
| **CMLA** | InfoNCE cross-modal latent alignment; multi-sensor fusion |

## Installation

```bash
pip install vulgaris
```

Optional extras:

```bash
pip install "vulgaris[serve]"   # REST inference server (FastAPI + uvicorn)
pip install "vulgaris[train]"   # Training utilities (tqdm, rich)
pip install "vulgaris[all]"     # Everything
```

**Requirements:** Python ≥ 3.10, NumPy ≥ 1.26, SciPy ≥ 1.11, PyYAML ≥ 6.0

## Quick Start

```python
import numpy as np
from vulgaris import Vulgaris, ModelConfig, Tensor

config = ModelConfig(
    input_dim=9,    # input channels
    n_classes=5,    # classification targets (set 0 for regression)
    output_dim=1,
)

model = Vulgaris(config)

# Batch inference: (batch, channels, timesteps)
x = Tensor(np.random.randn(8, 9, 64).astype(np.float32))
output, aux = model(x)
print(output.data.shape)  # (8, 5)
```

## Streaming Inference

VULGARIS processes one timestep at a time with stateful recurrence — suitable for hard real-time systems:

```python
state = model.init_state(batch_size=1)

for t in range(sequence_length):
    x_t = Tensor(sensor_reading[t].astype(np.float32))  # (1, 9)
    output_t, state = model.step(x_t, state)
```

## Training

```python
from vulgaris import (
    Vulgaris, ModelConfig,
    TrainingPipeline, VulgarisLoss,
    SpectralAdamW, CosineSchedule,
)

config    = ModelConfig(input_dim=9, n_classes=5)
model     = Vulgaris(config)
loss_fn   = VulgarisLoss(config)
optimizer = SpectralAdamW(model.parameters(), lr=3e-4)
scheduler = CosineSchedule(optimizer, warmup_steps=1000,
                           max_steps=50_000, min_lr=1e-6)
pipeline  = TrainingPipeline(model, config, loss_fn, optimizer, scheduler)

metrics = pipeline.train_step(x_np, y_np)
```

## Save and Load

```python
model.save("checkpoints/my_run")
model = Vulgaris.load("checkpoints/my_run")
```

Checkpoints store `weights.npz`, `config.yaml`, and `metadata.json` — no pickle, no binary blobs.

## Configuration

All hyperparameters are dataclasses — composable and serialisable to YAML:

```python
from vulgaris import ModelConfig
from vulgaris.config import SSSRConfig, DAHConfig

config = ModelConfig(
    input_dim=16,
    n_classes=10,
    sssr=SSSRConfig(state_dim=512, n_heads=16),
    dah=DAHConfig(n_domains=64, adapter_rank=32),
)
config.to_yaml("config.yaml")
```

Load from environment variables for containerised deployments:

```python
# VULGARIS_INPUT_DIM=16 VULGARIS_N_CLASSES=10 VULGARIS_D_MODEL=256
config = ModelConfig.from_env()
```

## Inference Server

```bash
export VULGARIS_INPUT_DIM=9
export VULGARIS_N_CLASSES=5
vulgaris-serve
```

Endpoints: `POST /predict` · `GET /health` · `GET /metrics` · `GET /versions`

Supports API key authentication (`VULGARIS_API_KEYS`), Prometheus metrics exposition, and three-level graceful degradation.

## Docker

```bash
docker compose up
```

Brings up the inference server and a Prometheus sidecar. Memory limit: 2 GB.

## Reproducibility

```python
from vulgaris import set_seed
set_seed(42)
```

## Self-Supervised Pretraining

```python
from vulgaris.training import SelfSupervisedTrainer

trainer = SelfSupervisedTrainer(model, in_channels=9, d_model=64)
metrics = trainer.pretrain_step(x_np)  # masked reconstruction + temporal InfoNCE
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

## Multi-Domain Adaptation

```python
# Switch domain at inference time — no retraining
output, aux = model(x, domain_idx=3)
```

## Repository Structure

```
vulgaris/
├── vulgaris/       public API and config
├── engine/         autograd engine: Tensor, Module, layers
├── modules/        ASE · HTD · SSSR · DAH · CRG · HMB · ESE · Safety · CMLA · SHCAL
├── model/          full Vulgaris model assembly
├── training/       loss · optimizer · pipeline · conformal · self-supervised
├── inference/      FastAPI server · streaming
├── serve/          auth · metrics · degradation · versioning
├── monitoring/     drift detection
├── federated/      federated continual learning with differential privacy
├── benchmarks/     baselines · ETT · NAB · EdgeTelemetry dataset loaders
└── tests/          unit tests with numerical gradient verification
```

## Documentation

Full technical documentation: [doc.md](https://github.com/jalaluddinkhan1/vulgaris/blob/main/doc.md)

## Citation

```bibtex
@software{vulgaris2026,
  title   = {VULGARIS: Foundational Model for Streaming Industrial Intelligence},
  author  = {Khan, Jalaluddin},
  year    = {2026},
  url     = {https://github.com/jalaluddinkhan1/vulgaris},
  license = {Apache-2.0},
}
```

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
