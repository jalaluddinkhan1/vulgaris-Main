# Changelog

All notable changes to VULGARIS are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [0.8.0] — 2026-06-02

### Added
- **TestTimeTrainer + TTTConfig** (`modules/ttt.py`): test-time training via masked channel reconstruction; wired into `Vulgaris.enable_ttt()` / `disable_ttt()` so every `model(x)` call automatically runs K inner-loop adaptation steps — zero call-site changes required.
- **41 new public symbols** exported from `vulgaris.__init__`: sub-classes `SSSRHead`, `HTDLevel`, `MemoryVAE`, `AdapterLayer`, `CARTExtractor`, `DecisionNode`, `ModalityEncoder`, `CBFLayer`, `SpectralNormLinear`, `MultiTaskHead`, `InContextAdapter`, `ContextEncoder`, `RuleConditionLoss`, `RuleDistiller`, `RuleLifecycleManager`, `RevIN` — plus engine utilities `zeros`, `ones`, `randn`, `rand`, `cat`, `stack`.
- **modules/revin.py, modules/causal_memory.py, modules/episodic_memory.py**: thin re-export stubs so `RevIN`, `CausalMemory`, `EpisodicMemory` are discoverable under `modules.*`.
- **ONNX export** (`serve/onnx_export.py`): `model.export_onnx(path)` bridges numpy weights to a PyTorch trace and writes a self-contained `.onnx` file; `benchmark_onnx()` runs latency profiling via ONNX Runtime. Needs `pip install "vulgaris[export]"`.
- **CRGConfig** gains `ci_threshold: float = 0.05` and `n_regimes: int = 4` — previously accessed via `getattr` fallbacks; now proper dataclass fields.
- **CMLAConfig wired into CrossModalLatentAlignment**: `__init__` now accepts `config: CMLAConfig | None` and applies `contrastive_temp` / `contrastive_weight` from it.

### Fixed
- **`Vulgaris.load()` crash**: `ModelConfig.from_yaml()` was deserializing nested configs (`ase`, `sssr`, etc.) as plain dicts instead of dataclass instances, causing `AttributeError` on load. Now reconstructs each nested field as the correct dataclass type.
- **`revin_mean` / `anomaly_energy` always None**: vulgaris.py checked `hasattr(revin, "last_mean")` but RevIN stores `self._mean`. Fixed to use `self.revin._mean` / `self.revin._std` directly.
- **RMC regime_weights never flowed to CRG**: `rmc.forward()` had the `return z_out, balance_loss` line missing `regime_weights` despite computing it. Changed to `return z_out, balance_loss, regime_weights`; vulgaris.py updated to 3-value unpack.
- **ASE filter cache**: `_build_filters()` now caches kernel computation and only recomputes when wavelet parameters change.
- **ASE missing-value interpolation**: masked sensor positions are now linearly interpolated instead of zero-padded, eliminating the FFT frequency bias.
- **`vulgaris_version` in save() metadata**: was `"0.1.0"`, now `"0.8.0"`.
- **Type annotations**: `crg.attach_causal_memory`, `icl.__init__` (episodic_memory), `dah.attach_ontology_embedding`, `dah.attach_rule_encoder` all annotated with `TYPE_CHECKING` guards.

### Changed
- `vulgaris/__init__.py` public API: 68 → **110 symbols**
- `modules/__init__.py` re-exports all 17 module files with full sub-class surface

## [Unreleased]

### Planned
- Pretrained weights for telecom-small and edge-small variants
- ONNX-compatible export for fixed-topology subgraphs
- INT8 post-training quantization for edge targets
- Real dataset loaders (SWaT, CICIDS, NASA turbofan)

---

## [0.7.0] — 2026-05-31

### Added
- **Multi-horizon output head** (`model/vulgaris.py`): `MultiHorizonHead` produces forecasts at multiple horizons in one forward pass (default `[1, 5, 20]` steps). Mean-pools the full latent sequence, then applies a separate `Linear` per horizon. Output stored in `aux_losses["multi_horizon"]` as `(B, n_horizons, output_dim)`.
- **RevIN anomaly preservation** (`model/vulgaris.py`): Reverse-instance normalisation statistics (`revin_mean`, `revin_std`) and `anomaly_energy` (L2 norm of the de-normalised residual) are stored in `aux_losses` every forward pass, so downstream consumers and loss functions can detect level-shift anomalies that would otherwise be erased by normalisation.
- **Pre-norm architecture** (`model/vulgaris.py`): Eight `RMSNorm` layers (`norm_htd`, `norm_sssr`, `norm_attn`, `norm_icl`, `norm_dah`, `norm_rmc`, `norm_crg`, `norm_hmb`) applied before each major module in the forward stack, following GPT-3/LLaMA pre-LN convention for training stability.
- **Channel-independent ASE** (`modules/ase.py`): Wavelet bank is now applied to each input channel independently before channel mixing (`channel_mix`). This matches the iTransformer finding that per-channel frequency extraction before cross-channel aggregation avoids polluting CRG's causal discovery with premature feature conflation.
- **DAGMA DAG penalty** (`modules/crg.py`): Replaces the NOTEARS matrix-exponential `tr(exp(W²))-n` with the strictly-convex DAGMA penalty `-log det(sI - W⊙W) - n·log(s)`. Better-conditioned around DAG solutions, avoids local minima from the Cayley-Hamilton expansion, and uses a Cholesky solve instead of a matrix exponential.

### Changed
- **RMC returns regime weights** (`modules/rmc.py`): `forward()` now returns a 3-tuple `(z_out, balance_loss, regime_weights)`. `regime_weights` is `(K,)` float32 — the mean soft routing probabilities across the batch, computed once so callers (CRG, step) never need to re-run `gate_proj`.
- **RMC → CRG ordering** (`model/vulgaris.py`): RMC now runs before CRG in the forward stack so the regime weights returned by RMC can condition CRG's effective adjacency matrix (`W_regime_bias` blending).
- **Timestamps → dt → SSSR** (`model/vulgaris.py`): Raw timestamps are differenced (`np.diff`) and clipped to `[1e-4, 10.0]` before being passed as `dt` to SSSR, enabling correct continuous-time discretisation for irregular-rate sensor streams.
- **Float32 throughout** (`modules/ase.py`, `modules/sssr.py`, `modules/crg.py`, `modules/rmc.py`): All `np.float64` dtypes and `.astype(np.float64)` calls replaced with `np.float32`. Halves memory for intermediate arrays and avoids silent double-precision promotion.
- `pyproject.toml` and `vulgaris/__init__.py` bumped to version **0.7.0**
- `vulgaris/__init__.py` exports `MultiHorizonHead` (total public API: 94 symbols)

---

## [0.6.0] — 2026-05-31

### Added
- **SSSR: log-space recurrence** (`engine/parallel_scan.py`): Hillis-Steele scan now tracks cumulative a-products in log space (`_numpy_scan_logspace`), preventing underflow to zero on sequences longer than ~1000 steps. The backward pass also uses the clamped `a` values for consistency. Log-space scan is now the default numpy backend.
- **CRG: Neural Granger mask** (`modules/crg.py`): Learned parameter `M` (n_nodes × n_nodes logits) gates each adjacency edge via `sigmoid(M)`. L1 penalty on `sigmoid(M)` drives unused edges toward zero during training. `update_structure()` also nudges M toward ±3 based on the EMA Granger signal, so the mask and W reinforce each other. Adds genuine learned causal discovery on top of the existing correlation-based heuristic.
- **CRG: Regime-conditioned causal graphs** (`modules/crg.py`): `W_regime_bias` parameter (K × n_nodes × n_nodes) stores one additive bias per regime. `forward()` accepts optional `regime_weights: np.ndarray` (K mean routing weights from RMC) and blends the regime biases: `W_eff = (W + Σ_k w_k * W_bias_k) * sigmoid(M)`. Different operating regimes now have distinct causal graph structures.
- **CRG: Failure propagation** (`modules/crg.py`): `propagate_failure(triggered_nodes, ...)` forward-propagates anomaly signals through the causal graph, assigning a failure probability to each downstream node (decays multiplicatively with edge weight and hop count). Directly useful for industrial fault diagnosis — given sensor anomaly nodes, returns ranked list of affected downstream nodes.
- **CRG: CausalMemory integration** (`modules/crg.py`): `attach_causal_memory(causal_memory)` lets a `CausalMemory` instance be wired to CRG; discovered edges with confidence > 0.1 are written into it automatically during `update_structure()`.
- **RMC: Sparse top-k routing** (`modules/rmc.py`): `RegimeMixtureCore.__init__` gains `top_k` parameter. When `top_k < n_experts`, only the top-k experts by gating weight are activated per token; weights are re-normalised over the sparse set. Straight-through gradient: the softmax backward uses the full pre-mask weights so all experts still receive gradient signal even when inactive.
- **Unified memory** (`memory/episodic.py`, `memory/causal.py`, `memory/__init__.py`): Two new standalone modules:
  - `EpisodicMemory` — fixed-capacity ring buffer of (key, value) pairs with cosine-similarity retrieval; used by ICL to persist context episodes across inference sessions.
  - `CausalMemory` — append-only store of (cause, effect, confidence) triples with O(1) inverted-index lookup by cause or effect node; written by CRG, readable by any module.
- **ICL: Episodic retrieval memory** (`modules/icl.py`): `InContextLearning` gains optional `episodic_memory: EpisodicMemory` parameter. When attached, `encode_context()` (a) stores each newly encoded context vector in memory and (b) retrieves the top-k most similar past episodes (via `retrieve_k` argument) and prepends them to the context set — giving ICL persistent cross-session memory without any gradient updates.

### Changed
- `pyproject.toml` and `vulgaris/__init__` bumped to version 0.6.0
- `memory*` added to `pyproject.toml` package discovery
- `vulgaris/__init__.py` exports `EpisodicMemory`, `CausalMemory` (total public API: 93 symbols)

---

## [0.5.0] — 2026-05-31

### Added
- **SSSR: HiPPO-LegS timescale initialization** (`modules/sssr.py`): `SSSRHead.log_A` now initialised with log-uniformly spaced decay rates (linspace −4.0→−0.2) instead of a flat log(0.5), giving each state dimension a distinct temporal scale from long-range (state 0) to short-range (state N−1); directly improves multi-scale sequence modelling without any hyperparameter change
- **SSSR: adaptive timestep** (`modules/sssr.py`): `SSSRHead.forward()`, `SSSRHead.step()`, `SelectiveSSR.forward()`, and `SelectiveSSR.step()` accept an optional `dt` tensor — when provided, it overrides the learned dt projection; enables correct handling of irregular-rate sensor streams where inter-sample intervals are known
- **SSSR: streaming causal-conv cache** (`modules/sssr.py`): `SelectiveSSR` now maintains a `_conv_buf` ring buffer across `step()` calls so the causal depthwise conv uses real history instead of zero-padding; `reset_conv_cache()` clears it between sequences; eliminates a systematic boundary artefact in online inference
- **ICL: attention pooling** (`modules/icl.py`): `ContextEncoder` replaces mean-pool-over-time with a learned attention pool — a single-head dot-product attention with a learned query (`pool_q: Linear(d_model, 1)`) computes softmax weights over reference timesteps and takes a weighted sum; full backward pass through both z_ref and pool_q is implemented analytically

### Changed
- `pyproject.toml` and `vulgaris/__init__` bumped to version 0.5.0

---

## [0.4.0] — 2026-05-25

### Added
- **RegimeMixtureCore (RMC)** (`modules/rmc.py`): Switch-Transformer-style soft Mixture-of-Experts with K=4 Linear experts (d_model→d_model), softmax gating with temperature τ, load-balancing loss L = K·Σ_k f_k·P_k (hard argmax for f_k, mean routing probability for P_k), and `regime_assignments()` returning (B,T) hard assignment indices
- **ActiveLearner** (`training/active_learning.py`): pool-based uncertainty sampling with four acquisition functions — output variance (`uncertainty`), Shannon entropy (`entropy`), top-2 margin (`margin`), and `random`; MC dropout support via n_mc>1; labeled-set exclusion tracking; `query()` returns top-k indices
- **DistillationLoss + DistillationTrainer** (`training/distillation.py`): Hinton 2015 knowledge distillation with temperature-scaled KL soft targets (L_soft = T²·KL), hint-layer MSE, and combined loss L = α·L_hard + (1−α)·(T²·L_soft + β·L_hint); DistillationTrainer freezes all teacher parameters and auto-creates hint projector when teacher/student dimensions differ
- **SpeculativeRollout** (`inference/speculative.py`): WorldModelHead drafts γ steps in latent space, full model verifies via output_head; accepts if ‖y_draft − y_verify‖_∞ < threshold, rejects and resets draft otherwise; exposes `acceptance_rate` and `effective_speedup` stats, plus `rollout_single()` for single-step use
- **config.py** additions: `RMCConfig` (n_experts, tau, balance_weight), `CMLAConfig` (n_modalities, contrastive_temp, contrastive_weight), `ICLConfig` (max_context, gate_init); `d_model` field on `ModelConfig`; `from_env()` now maps `VULGARIS_D_MODEL` → `cfg.d_model`
- All package `__init__.py` files updated to export new symbols (total public API: 89 symbols)

### Changed
- `pyproject.toml` bumped to version 0.4.0
- Dockerfile and docker-compose updated with new environment variables for RMC, CMLA, and ICL configuration

---

## [0.3.0] — 2026-05-24

### Added
- **EventBuffer** (`inference/event_buffer.py`): lock-free shared-memory ring buffer using `multiprocessing.shared_memory`; power-of-2 capacity with bitwise mask indexing; back-pressure `put()`; batch `drain()`
- **Distributed training** (`training/distributed.py`): `init_process_group`, `broadcast_parameters`, `allreduce_gradients`, `allreduce_scalar`, `barrier` — uses torch.distributed as transport, operates on numpy parameter arrays
- **DistributedSampler** (`training/distributed_sampler.py`): Knuth-hash epoch shuffling with padding to ensure equal shard sizes across all ranks
- **`scripts/train_distributed.py`**: torchrun entrypoint for multi-node/multi-GPU distributed training
- **`serve/migration.py`**: `migrate_checkpoint` (v1→v2 SHA-256 backfill), `HotSwapAdapter` (thread-safe live model swap with a reentrant lock), `get_checkpoint_version`

### Changed
- **StreamingInference** (`inference/streaming.py`): added delta-threshold gating (skip update if ‖x_t − x_{t-1}‖ < threshold), poisoning detection (Z-score > 6σ flagging after 30-sample warmup), and out-of-order timestamp detection
- **DegradationController** (`serve/degradation.py`): full FULL→REDUCED→ALERT_ONLY finite-state machine with configurable p95-latency and error-rate thresholds; added **CanaryController** with PRIMARY/CANARY/SHADOW routing modes

---

## [0.2.0] — 2026-05-22

### Added
- **InContextLearning** (`modules/icl.py`): `ContextEncoder` + `InContextAdapter`; cross-attention mechanism with near-zero gate initialisation enabling zero-shot adaptation from reference examples without weight updates
- **RuleEngine** (`modules/rule_engine.py`): `Rule` dataclass, `RuleRegistry`, `RuleEncoder` (masked mean-pool → linear projection → meta_dim), `RuleConditionLoss` (sigmoid gate soft constraint), `RuleDistiller` (CART↔registry comparison), `RuleLifecycleManager` (decay / prune / merge)
- **OntologyEmbedding** (`modules/ontology_embedding.py`): 80 industrial terms across 12 semantic clusters; mean-pool term embeddings; `OntologyRegistry` for per-domain term storage
- **PreprocessingIndustrialTokenizer** (`preprocessing/industrial_tokenizer.py`): `ChannelSpec`, `TokenType` enum (CONTINUOUS / DISCRETE / BINARY / EVENT), per-channel tokenisation with configurable d_model
- **LogEncoder** (`preprocessing/log_encoder.py`): Drain3-style fixed-depth prefix trie with O(1) amortised template matching; UUID / IP / hex / number normalisation; outputs `(template_id, severity)` tuples
- **DriftDetector** (`monitoring/drift.py`): KS statistic, RBF-kernel MMD with median-heuristic bandwidth, and Wasserstein-1D per feature (pure numpy)
- **AuditLogger** (`serve/audit.py`): structured per-request audit trail for compliance and traceability
- **SelfSupervisedTrainer** (`training/self_supervised.py`): MAE masked reconstruction loss combined with TF-C temporal InfoNCE contrastive objective

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

added 
Assessment: What's Worth Building
Short answer: About 40% of the list is high-value. The rest is premature optimization or complexity that doesn't match the numpy-only, no-GPU, industrial IoT use case.

Tier 1 — Build Now (High Impact, Core Value)
SSSR (state space — your most important module):

HiPPO/S4 initialization — proven to fix long-range vanishing gradients, directly solves a real problem
Log-space recurrence — numerical stability for long sequences, critical for industrial time-series
Adaptive timestep hierarchy — essential for irregular-rate sensor data (the actual use case)
Streaming conv cache — needed for real-time inference
CRG (causal reasoning — "VERY IMPORTANT" per your note):

Neural Granger causality — core differentiator, replaces correlation with actual causal discovery
Dynamic causal graphs — causal structure should change across regimes (e.g., fault vs. normal)
Failure propagation modeling — directly valuable for industrial fault diagnosis
ICL:

Attention pooling (replace mean) — the mean-pooling is a known weak point, 5 lines to fix
Episodic retrieval memory — critical if you want ICL to actually work across sessions
Unified Global Memory (partial):

Episodic + causal memory — replace the fragmented copies across SSSR/HTD/ICL
Skip semantic/working memory — too abstract for industrial time-series
Tier 2 — Phase 6 (Good Ideas, Not Urgent)
RMC: sparse top-k routing, temporal routing consistency
DAH: continual domain replay, uncertainty-aware adaptation
MULTITASK HEAD: quantile prediction, calibrated confidence intervals
ASE: learnable frequency gating (skip complex wavelets — overkill)
SAFETY HEAD: differentiable QP, uncertainty-aware margins (only if targeting safety-critical deployments)
Tier 3 — Skip or Defer Indefinitely
Complex state dynamics / complex wavelets — complex arithmetic in numpy is slow; no GPU means the cost outweighs the benefit
Latent counterfactual rollouts, graph temporal diffusion — requires differentiable graph layers you'd need to build from scratch
Physics-informed latent constraints, mechanistic latent system modeling — these need domain-specific priors you don't have
Neural digital twin simulation, predictive system rollout engine — "global system" features that are a separate product, not a module
Dynamic ontology learning, evolving ontology discovery — the ontology is a structured prior, not a learned module
Symbolic planning, rule contradiction detection — rules system should stay simple; this is a logic engine, not your scope
Structured regrowth, meta-learned plasticity — NAS-style complexity in numpy is not feasible
Implementation Plan
Phase 5 — Numerical Core (v0.5.0)
Goal: Fix the real weaknesses in SSSR and CRG before adding features.

SSSR: HiPPO-LegS initialization — replace random init of A matrix with HiPPO-LegS structured matrix. Fixes long-range memory for free.
SSSR: log-space recurrence — compute recurrence in log-space, exponentiate at output. Prevents underflow on long sequences (T > 1000).
SSSR: streaming cache — add hidden_state buffer for autoregressive inference without recomputing the full sequence.
SSSR: adaptive timestep — accept dt tensor per timestep; fold into discretization of A.
ICL: replace mean-pool with attention — 10-line change, significant quality improvement.
Phase 6 — Causal Intelligence (v0.6.0)
Goal: Make CRG actually do causal discovery, not just correlation.

CRG: Neural Granger — add a learned mask M[i,j] that gates whether signal j Granger-causes signal i. Train with L1 sparsity.
CRG: Regime-conditioned causal graphs — condition causal graph structure on current regime from RMC. Different failure modes have different causal structure.
CRG: Failure propagation — given a causal graph and anomaly detection, trace which upstream signals caused a downstream fault.
RMC: Sparse top-k routing — replace soft routing with hard top-k using straight-through gradient. Cleaner expert specialization.
Phase 7 — Unified Memory (v0.7.0)
Goal: Replace the 4 fragmented memory stores with one coherent system.

memory/episodic.py — ring buffer of (context, outcome) pairs with similarity-based retrieval. Replaces ad-hoc buffers in SSSR, HTD, ICL.
memory/causal.py — append-only store of (cause, effect, confidence) triples. CRG writes here; ICL and RMC read here.
Wire into SSSR, HTD, ICL, SHCAL — replace internal state with shared memory reads/writes.
vulgaris/__init__.py — export EpisodicMemory, CausalMemory.
Phase 8 — Forecasting + Safety (v0.8.0)
Goal: Add quantile forecasting and differentiable safety constraints.

MULTITASK HEAD: quantile regression — add pinball loss, output (low, mid, high) bounds. Directly useful for alarm thresholds.
MULTITASK HEAD: multi-horizon — predict t+1, t+5, t+20 simultaneously with shared representation.
SAFETY HEAD: differentiable QP — implement a small quadratic program solver (OSQP-style, ~100 lines) to project actions onto safe set.
SAFETY HEAD: reachability — forward-simulate SSSR for K steps under current policy; flag if trajectory exits safe region.