# Changelog

All notable changes to VULGARIS are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

### Planned
- Pretrained weights for telecom-small and edge-small variants
- ONNX-compatible export for fixed-topology subgraphs
- INT8 post-training quantization for edge targets
- Real dataset loaders (SWaT, CICIDS, NASA turbofan)

---

## [0.1.0] — 2026-05-21

### Added
- Core autograd engine (`vulgaris.engine`): Tensor, Module, Linear, LayerNorm, RMSNorm, SwiGLU
- Adaptive Signal Embedding (ASE) with Morlet wavelets
- Selective State-Space Recurrence (SSSR) with diagonal ZOH
- Causal Routing Graph (CRG) with NOTEARS DAG constraint
- Hierarchical Memory Bank (HMB) with VAE slot compression
- Domain-Adaptive Hypernetwork (DAH) with LoRA adapters
- Explainability Engine (ESE) with CART rule extraction
- Cross-Modal Latent Alignment (CMLA) with InfoNCE
- Hierarchical Timescale Decomposition (HTD, 4 levels)
- Safety Policy Head with Control Barrier Function
- SHCAL: Synaptic Homeostatic Continual Adaptive Learning (EWC + Hebbian)
- SpectralAdamW optimizer with spectral norm clipping
- CosineSchedule with linear warmup
- Non-stationary conformal prediction with coverage guarantees
- Full TrainingPipeline with checkpointing, NaN guard, DAH cache management
- Self-supervised pretraining: masked reconstruction + temporal InfoNCE
- MultiTaskHead: forecast, anomaly, classification, uncertainty
- DriftDetector: KS, MMD, Wasserstein-1D (pure numpy)
- FastAPI inference server with auth, metrics, degradation control, versioning
- Prometheus-style metrics exposition
- Federated continual learning with differential privacy
- Benchmark suite: LastValue, MovingAverage, ARIMA_lite, LSTMLite
- Dataset loaders: ETT, NAB, EdgeTelemetry (synthetic fallback)
- Versioned checkpoint format (weights.npz + config.yaml + metadata.json)
- `model.save()` and `Vulgaris.load()` API
- Reproducibility via `vulgaris.set_seed()`
- Full unit test suite with numerical gradient verification
- GitHub Actions CI: test matrix (Python 3.10–3.12, Ubuntu + Windows)
- Docker + docker-compose deployment
- Apache 2.0 license

### Architecture
- Pure NumPy — no PyTorch, no TensorFlow, no CUDA required
- Runs on CPU, edge hardware, and ARM devices
- Input format: `(batch, channels, timesteps)` float32
- Streaming single-step `.step()` API for real-time inference
