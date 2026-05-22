# VULGARIS: A Foundational Model for Industrial Intelligence

*A complete technical reference for the architecture, mathematics, training, and deployment of the VULGARIS system.*

---

## Table of Contents

1. [What This Is and Why It Exists](#1-what-this-is-and-why-it-exists)
2. [The Problem with Existing Approaches](#2-the-problem-with-existing-approaches)
3. [Architecture Overview](#3-architecture-overview)
4. [Module Deep Dives](#4-module-deep-dives)
   - 4.1 [Adaptive Signal Embedding (ASE)](#41-adaptive-signal-embedding-ase)
   - 4.2 [Selective State-Space Recurrence (SSSR)](#42-selective-state-space-recurrence-sssr)
   - 4.3 [Causal Routing Graph (CRG)](#43-causal-routing-graph-crg)
   - 4.4 [Hierarchical Memory Bank (HMB)](#44-hierarchical-memory-bank-hmb)
   - 4.5 [Self-Healing and Continuous Adaptation Layer (SHCAL)](#45-self-healing-and-continuous-adaptation-layer-shcal)
   - 4.6 [Domain-Adaptive Hypernetwork (DAH)](#46-domain-adaptive-hypernetwork-dah)
   - 4.7 [Explainability and Symbolic Extraction Engine (ESE)](#47-explainability-and-symbolic-extraction-engine-ese)
   - 4.8 [Cross-Modal Latent Alignment (CMLA)](#48-cross-modal-latent-alignment-cmla)
   - 4.9 [Hierarchical Timescale Decomposition (HTD)](#49-hierarchical-timescale-decomposition-htd)
   - 4.10 [Safety-Critical Policy Head](#410-safety-critical-policy-head)
5. [The Unified Loss Function](#5-the-unified-loss-function)
6. [Training from Scratch](#6-training-from-scratch)
7. [Domain Adaptation](#7-domain-adaptation)
8. [Streaming Inference](#8-streaming-inference)
9. [Federated Deployment](#9-federated-deployment)
10. [Configuration Reference](#10-configuration-reference)
11. [Benchmarks and Expected Performance](#11-benchmarks-and-expected-performance)
12. [The Custom Engine](#12-the-custom-engine)
13. [CUDA and Rust Acceleration](#13-cuda-and-rust-acceleration)
14. [Theoretical Guarantees](#14-theoretical-guarantees)
15. [Known Limitations and Future Work](#15-known-limitations-and-future-work)

---

## 1. What This Is and Why It Exists

VULGARIS is a foundational model built specifically for industrial intelligence — the kind of AI that has to work reliably on a factory floor at 3am when a bearing starts to fail, or inside a 5G base station deciding in milliseconds whether a cell is degrading, or running on a $30 edge chip inside a smart meter watching for grid anomalies.

The word "foundational" is used deliberately. This is not a specialized fault-detection model, not a time-series classifier, not a process optimization script. It is a general-purpose model that can be pretrained on unlabeled multivariate telemetry and then adapted — cheaply, quickly, with almost no data — to any industrial domain. One model, many deployments.

The design philosophy has four core beliefs.

**Signals are not tokens.** Treating industrial telemetry like text — chopping it into discrete tokens, feeding it to a transformer — throws away everything that makes physical signals special: their continuity, their multi-rate nature, their causal physical relationships. A vibration sensor sampled at 25 kHz and a temperature sensor sampled at 1 Hz carry fundamentally different information, and forcing them through the same vocabulary is architecturally wrong.

**Causality is structure, not a pattern to learn.** In physical systems, the causal relationships between signals are often knowable or discoverable. The pump pressure affects the pipe flow. The motor current spikes before the bearing temperature rises. A model that learns these relationships explicitly — as a graph — is more interpretable, more sample-efficient, and more trustworthy than one that buries them in attention weights.

**The model must adapt without retraining.** Industrial environments change. New equipment, seasonal load patterns, firmware updates, sensor drift. A model that requires a week of retraining to handle these is useless in production. VULGARIS adapts online: Hebbian weight updates, conformal recalibration, and domain adapter switching happen continuously, in minutes, without touching the base model.

**If you cannot explain it, you cannot deploy it.** In critical infrastructure, a black-box anomaly score is insufficient. An operator needs to know which sensor, what causal path, what counterfactual would resolve it. Every prediction in VULGARIS is traceable to a finite path through the causal graph.

---

## 2. The Problem with Existing Approaches

It is worth being specific about why existing approaches fall short, because the architectural choices in VULGARIS are direct responses to concrete failures.

**Transformers at scale.** Large language models adapted for time series inherit attention's O(N²) memory complexity. A single 5G cell producing 100 KPIs at 1-second intervals generates 8.6 million timesteps per day. Fitting this in a KV cache is not feasible on edge hardware. Additionally, tokenizing continuous signals introduces quantization artifacts that corrupt the high-frequency information most relevant to fault detection.

**Classical SSMs (Mamba, S4, HiPPO).** These solve the attention scaling problem but are designed for text. They assume fixed sampling rates, single-modality inputs, and static weights. Industrial telemetry is multi-rate, multi-modal, and non-stationary. More importantly, these architectures provide no explainability — their recurrent state is a dense latent vector with no semantic interpretation. You cannot ask Mamba "why did it predict a fault on sensor 7?"

**Specialized industrial models (LSTM variants, Prophet, ARIMA).** These are domain-specific in the worst way: they generalize poorly, they do not learn causal structure, they require significant feature engineering, and each one is a standalone artifact. A factory with 200 machine types needs 200 models, each maintained separately. The operational burden alone makes this unviable.

**Foundation models for time series (Moirai, Chronos, MOMENT).** A significant step forward, but these are forecasting models. They predict the next value, not the causal structure. They do not adapt online, do not provide certified safety guarantees, and are not designed for the latency and memory constraints of edge deployment.

VULGARIS was designed to fill the gap: a model that handles the full scope of industrial intelligence — from raw sensor ingestion to certified-safe control outputs — in a single architecture that runs on a Cortex-A55 and can be explained to a maintenance engineer.

---

## 3. Architecture Overview

The VULGARIS forward pass is a sequential pipeline of modules, each adding a different kind of processing. Data flows through them in order, with each module adding a residual contribution to the shared latent representation.

```
Raw telemetry x  (batch, channels, time)
        |
        v
+-----------------------------------+
|  Adaptive Signal Embedding (ASE)  |  Morlet wavelets -> latent space
|                                   |  Output: (batch, time, d_model)
+-----------------------------------+
        | residual add
        v
+-----------------------------------+
|  Hierarchical Timescale (HTD)     |  4 nested SSMs at different tau
|                                   |  Output: (batch, time, d_model)
+-----------------------------------+
        | residual add
        v
+-----------------------------------+
|  Selective State-Space (SSSR)     |  O(N) recurrence + Hebbian adapt
|                                   |  Output: (batch, time, d_model)
+-----------------------------------+
        | residual add
        v
+-----------------------------------+
|  Causal Routing Graph (CRG)       |  Sparse causal message passing
|                                   |  Output: (batch, time, d_model)
|                                   |  + dag_penalty (scalar loss term)
+-----------------------------------+
        | residual add
        v
+-----------------------------------+
|  Hierarchical Memory Bank (HMB)   |  Surprise-driven event retrieval
|                                   |  Output: (batch, time, d_model)
|                                   |  + memory_loss (VAE term)
+-----------------------------------+
        |
        v
+-----------------------------------+
|  Domain Adapter (DAH)             |  Modifies SSSR weights per domain
|                                   |  No latency -- cached adapters
+-----------------------------------+
        |
        v
+-----------------------------------+
|  Output Head                      |  Regression or classification
+-----------------------------------+
        |
        v
+-----------------------------------+
|  Safety Filter (optional)         |  CBF projection, Lipschitz bounds
+-----------------------------------+
        |
        v
    Prediction + uncertainty interval
```

For multi-modal inputs (vibration + thermal + current + logs), the Cross-Modal Latent Alignment (CMLA) module runs before ASE to fuse the modalities into a single tensor. The Explainability Engine (ESE) runs passively alongside the main pipeline, recording latent states for offline rule extraction and responding to on-demand attribution queries.

The key properties of this pipeline:

- **No attention anywhere.** Every module runs in O(N) time and O(1) memory with respect to sequence length.
- **Fully causal.** At every point, the model only sees past information. Suitable for both offline analysis and real-time streaming.
- **Modular.** Each module can be disabled for simpler deployments. A minimal VULGARIS for a constrained edge device might run only ASE + SSSR + OutputHead.
- **Residual throughout.** Every module adds to the shared representation rather than replacing it. This makes training stable and allows individual modules to be frozen or replaced independently.

---

## 4. Module Deep Dives

### 4.1 Adaptive Signal Embedding (ASE)

**File:** `modules/ase.py`

The fundamental problem with applying standard deep learning to industrial telemetry is that physical signals live in continuous time and carry information across many frequency scales simultaneously. A bearing fault shows up at a specific characteristic frequency — the BPFO (ball pass frequency, outer race) — that depends on bearing geometry and shaft speed. A gradual thermal drift has a time constant of hours. A relay trip is microseconds. No fixed temporal resolution captures all of these faithfully.

ASE solves this by replacing the embedding lookup with a learnable continuous-time filter bank. Instead of mapping discrete tokens to vectors, it maps raw signal windows to a latent space using a bank of K×S filters where K is the number of filters per scale and S is the number of temporal scales (dilation levels).

Each filter is a parameterized Morlet wavelet:

```
psi_k(t) = A_k * exp(-0.5 * (t * exp(-log_sigma_k))^2) * cos(omega_k * t + phi_k)
```

The four parameters per filter — amplitude A_k, log-width log_sigma_k, center frequency omega_k, and phase phi_k — are all learned end-to-end via backpropagation. The filter bank adapts to the specific frequency content of the target domain during pretraining.

The multi-scale aspect is handled by dilated convolution. At scale s, the wavelet filter is applied with dilation 2^s, effectively sampling a wider temporal window without increasing the filter length. This is computationally efficient and allows the same filter to capture the same oscillation pattern at different temporal resolutions.

After the multi-scale filterbank, all scale outputs are concatenated along the channel dimension and projected to the model's latent dimension D via a linear layer followed by RMSNorm. Every position in the output encodes a rich multi-frequency representation of the local signal neighborhood.

**Handling irregular sampling.** When timestamps are provided, ASE normalizes the inter-sample intervals and appends them as an extra input channel. This allows the model to learn representations that are time-aware without requiring imputation or resampling, which would corrupt the signal statistics.

**Why not a standard CNN?** A standard 1D CNN uses fixed rectangular filters initialized randomly. The wavelet parameterization is a strong inductive bias: it forces the filters to be oscillatory and localized, which is exactly the structure of physical signals. This means ASE reaches useful representations with far less data than a generic CNN would need.

### 4.2 Selective State-Space Recurrence (SSSR)

**File:** `modules/sssr.py`

The sequence model at the heart of VULGARIS. It replaces self-attention entirely and provides O(N) time and O(1) memory inference. The fundamental recurrence is:

```
h_t = A_t * h_{t-1} + B_t * x_t     (elementwise, A is diagonal)
y_t = C_t · h_t + D * x_t
```

**Stability guarantee.** The transition coefficients A_t are computed from learnable log parameters:

```
A_t = exp(-exp(log_A) * dt_t)
```

Since exp(-exp(anything)) always returns a value in (0, 1), every element of A_t is strictly between zero and one. This guarantees that the hidden state is bounded for all inputs, regardless of sequence length. Exploding gradients through time are impossible by construction.

**Input-dependent gating.** Unlike classical SSMs where A is fixed, in SSSR the timescale dt_t and the input/output projections B_t and C_t all depend on the current input:

```
dt_t = softplus(W_dt * x_t + b)  clamped to [dt_min, dt_max]
B_t  = W_B * x_t
C_t  = W_C * x_t
```

This selectivity is what makes SSSR powerful. For a motor monitoring system, the model can automatically increase memory retention (smaller dt → larger A → slower forgetting) when it detects unusual behavior, and speed up forgetting (larger dt → smaller A) during normal steady-state operation.

**Hebbian online adaptation.** After each forward pass during training, SSSR applies an in-place Hebbian update to the log_A parameters:

```
delta_log_A = eta * mean over (batch, time) of (h_t * h_{t-1} - h_t^2)
```

This update strengthens state dimensions that co-activate across timesteps and weakens those that activate independently. It runs directly on the numpy arrays without going through autograd, so it adds negligible overhead. The parameters are clamped to [-5, 0] after each update.

**Architecture.** The full SSSR block follows an expanded, gated design:

1. Input is projected to d_inner = 2 * d_model (expansion)
2. A causal depthwise convolution (kernel size 4, left-padded for causality) adds local context
3. The expanded signal feeds multiple independent SSM heads in parallel
4. Outputs are multiplied elementwise by a SiLU gate computed from a separate branch on the input
5. A linear skip connection from input to output is added

Multi-head design allows each head to specialize in different frequency bands or signal aspects. The outputs are concatenated before the final projection back to d_model.

**Training vs. inference.** During training, SSSR runs a sequential scan storing all intermediate hidden states — needed for the Hebbian update and for SHCAL's Fisher computation. During streaming inference, it runs a single recurrence step per timestep, requiring O(state_dim) memory regardless of history length.

### 4.3 Causal Routing Graph (CRG)

**File:** `modules/crg.py`

The most novel module in VULGARIS. The core idea is to replace dense learned attention patterns with a sparse, differentiably-discovered causal graph that reflects the actual physical dependency structure of the system being monitored.

**The graph.** CRG maintains a learnable adjacency matrix W ∈ R^(n×n) where W_ij represents the causal influence from node i to node j. The model first maps the latent representation to n graph nodes:

```
node_states = linear_embed(x)    shape: (batch, time, n_nodes)
```

Then performs sparse message passing for each node j:

```
updated_j = node_states_j + sum over i of (W_ij * node_states_i)
             only for edges where |W_ij| > threshold
```

Finally maps back to the latent space:

```
output = linear_out(updated_states)
```

**The DAG constraint.** An unconstrained adjacency matrix can produce cycles — "A causes B which causes A" — which is physically nonsensical and numerically problematic. CRG enforces acyclicity using the NOTEARS constraint:

```
h(W) = trace(expm(W * W)) - n = 0
```

where expm is the matrix exponential. This is the standard continuous relaxation of the combinatorial acyclicity condition. In VULGARIS it is approximated via a truncated power series (6 terms) for computational efficiency and differentiated using the chain rule on the trace.

This constraint enters the total loss as a penalty. During pretraining, the model discovers whatever causal graph minimizes the task loss subject to acyclicity and L1 sparsity on W.

**Structure discovery.** During pretraining, CRG also updates W using Granger causality scores computed from the current node activations — the lagged correlation between node_i and node_j over several lag lengths. This provides a data-driven initialization signal that pushes W toward physically meaningful relationships before the gradient takes over.

**Traceability.** The `explain(query_node)` method traverses the discovered graph via BFS from a target node, following the strongest incoming edges. The output is a ranked list of (node_index, cumulative_influence) pairs — a machine-generated causal chain from inputs to prediction. This is the audit trail that regulators, operators, and certification bodies need.

**Why this wins over attention.** Attention computes all N² pairwise relationships at every timestep. CRG computes only the E active edges where E is typically 2–5% of all possible edges. For 64 monitored signals, attention computes 4096 relationships; CRG computes roughly 80. More importantly, those 80 relationships are interpretable — they correspond to real causal dependencies in the physical system.

### 4.4 Hierarchical Memory Bank (HMB)

**File:** `modules/hmb.py`

Transformer models address memory by storing all past hidden states in a KV cache. This cache grows linearly with sequence length and has no mechanism for forgetting irrelevant information. For a system running 24/7 for years, this is untenable.

HMB takes a different approach: it remembers events, not timesteps. An event is a moment when something surprising happened — when the model's prediction was significantly wrong, or when the signal statistics changed regime. Between events, the system is behaving as expected and nothing needs to be stored.

**The surprise metric.** After every forward pass, HMB computes a surprise score:

```
surprise_t = ||h_actual_t - h_predicted_t||^2 / (2 * sigma^2)
```

where h_predicted is what the SSM predicted before seeing the current input, and sigma^2 is a running estimate of the variance of this error. When surprise_t exceeds the threshold, the current hidden state is immediately archived.

**Three-tier memory structure.**

1. **Working buffer** — a ring buffer (deque) of the most recent hidden states, timestamps, and uncertainty estimates. Always in memory, accessed at full speed.

2. **Event archive** — a fixed-size dictionary of compressed events indexed by causal tags. Events are stored as (mean, logvar) pairs from the VAE encoder. A complex event takes the same storage as a simple one.

3. **Consolidation** — periodically (every consolidation_interval steps), the working buffer is batch-compressed through the VAE and written to the event archive. Old and rarely-accessed entries are evicted using LRU policy.

**The memory VAE.** The compression module is a variational autoencoder with a two-layer MLP encoder and decoder, trained jointly with the rest of the model:

```
L_memory = E[||h - Dec(Enc(h))||^2] + beta * KL(N(mu, sigma^2) || N(0, I))
```

The beta parameter controls the compression-fidelity tradeoff. Higher beta means smaller compressed representations but less faithful reconstruction.

**Retrieval.** When the model needs past context, it queries the archive using cosine similarity between the current hidden state and all archived entries. The retrieved context is a weighted sum of the top-k matches, weighted by uncertainty:

```
retrieved = sum_k softmax(sim_k / tau) * (1 / uncertainty_k) * Dec(z_k)
```

This is added to the current hidden state as a residual. The model can condition on past events — a bearing failure from six months ago, a regime change last Tuesday — without attending to every past timestep.

### 4.5 Self-Healing and Continuous Adaptation Layer (SHCAL)

**File:** `modules/shcal.py`

The hardest unsolved problem in deploying AI on industrial systems is distribution shift. The environment changes continuously: equipment ages, processes drift, operating conditions evolve. A static model trained once becomes stale. Continuous full retraining is too expensive and risks destroying previous knowledge. SHCAL is the answer.

**Elastic Weight Consolidation.** When the model finishes learning a task or domain, SHCAL computes the Fisher information diagonal for each parameter:

```
F_i ≈ (1/N) * sum_n (d log p(y_n|x_n) / d theta_i)^2
```

This estimates how important each parameter is for the current knowledge. The EWC penalty then prevents important parameters from drifting:

```
L_ewc = (lambda_ewc / 2) * sum_i F_i * (theta_i - theta_star_i)^2
```

where theta_star are the parameters at the end of the last task. This is parameter-specific L2 regularization: loose for parameters that did not matter to previous tasks, tight for parameters that did. The model can update freely in parts of parameter space that do not affect previous tasks.

**Hebbian plasticity.** For fast local adaptation without backpropagation, SHCAL applies Hebbian updates to the linear layers:

```
delta W_ij = eta * (post_i * pre_j - decay * W_ij)
```

This is Oja's rule: strengthen connections between co-activating neurons, include decay to prevent unbounded growth. It runs entirely in numpy, adding almost no overhead.

Before applying any Hebbian update, SHCAL validates it in shadow mode: it applies the update to a temporary copy of the weights and checks whether the loss on the current batch would increase by more than 10%. If so, the update is rejected.

**Structural plasticity.** Over longer timescales, SHCAL can add or remove connections. Connections whose weight magnitude stays below prune_threshold for several consecutive steps are zeroed out. Connections that were previously pruned but whose gradient consistently exceeds grow_threshold are re-enabled with a small random weight. The model reorganizes its capacity as task demands change.

**Conformal recalibration.** SHCAL maintains a rolling history of normalized prediction errors: s_t = |y_t - y_hat_t| / sigma_t. If the coverage rate drops below 1 - alpha, trigger_adaptation is set, which propagates to the training pipeline and triggers additional adaptation.

### 4.6 Domain-Adaptive Hypernetwork (DAH)

**File:** `modules/dah.py`

In a large industrial deployment you might have 50 different pump models, 30 motor types, 10 grid configurations. Training a separate model for each wastes resources and ignores the fact that most of the underlying physics is shared. DAH enables a single pretrained model to serve all of them via lightweight domain-specific adapters.

**Architecture.** Given a domain identifier, the hypernetwork produces a low-rank weight update for each target layer:

```
z_domain = MetaMLP(DomainEmbed(domain_idx))          dimension: meta_dim

A_l = reshape(hyper_A_l(z_domain))    shape: (d_out, rank)
B_l = reshape(hyper_B_l(z_domain))    shape: (rank, d_in)
delta_W_l = A_l @ B_l                 shape: (d_out, d_in)
```

The adapted forward pass for each layer is:

```
output = W_base @ x + exp(log_scale_l) * A_l @ (B_l @ x)
```

The base weights are completely frozen after pretraining. All domain-specific knowledge lives in the adapters. Total adapter parameters per domain: typically less than 1% of the base model size.

**Switching domains.** When `model.set_domain(domain_idx)` is called, the hypernetwork runs once to generate all adapter matrices and caches them. Subsequent forward passes use cached adapters with zero additional compute. Domain switch latency is under 1ms on any modern hardware.

**New domains at runtime.** `model.dah.register_domain(name, metadata)` assigns a new domain index based on a stable hash of the metadata dictionary. The hypernetwork immediately generates adapters for this domain using its learned mapping from metadata to adapter weights — no retraining needed. For truly novel domains, the adapters provide a reasonable starting point that improves with a few dozen labeled examples.

**Cost breakdown.** A production VULGARIS with d_model=256 and rank=16 has approximately 5M frozen base parameters and roughly 120K adapter parameters per domain. Running 32 simultaneous domains adds only 4M parameters total — less than the base model.

### 4.7 Explainability and Symbolic Extraction Engine (ESE)

**File:** `modules/ese.py`

ESE is not in the forward path. It does not process the signal or modify predictions. It is a passive observer that records latent states during normal operation and answers explanation queries on demand.

**Rule induction.** During operation, ESE accumulates (hidden_state, prediction) pairs. Periodically, it fits a CART decision tree to these pairs. CART is a custom pure-numpy implementation — no sklearn — using binary recursive splitting on Gini impurity for classification and MSE for regression.

The result is a set of human-readable rules:

```
IF latent[2] > 0.83 AND latent[7] < -0.31
THEN prediction = FAULT (confidence = 0.91, n_samples = 147)
```

These rules are approximations of the model's behavior, not exact descriptions. But they capture the dominant patterns and provide an interpretable proxy that operators can reason about, audit, and challenge.

**Causal attribution.** When asked to attribute a specific prediction, ESE computes the gradient of the output with respect to the input, then weights each dimension by the strength of its CRG edges:

```
attr_i = sum_j (W_ij * |d y / d x_j|)
```

This combines gradient sensitivity (how much would the output change?) with causal graph connectivity (how strongly does this node influence others?). The result is more physically meaningful than pure gradient sensitivity, which can be dominated by saturation effects near decision boundaries.

**Counterfactual generation.** Given a current state and a desired target prediction, ESE finds the minimal perturbation via gradient descent:

```
minimize over delta:  ||delta||^2 + alpha * ||delta||_1
subject to:          ||model(h + delta) - y_target|| < epsilon
```

The L2 term keeps the perturbation small. The L1 term encourages sparsity — change as few dimensions as possible. Together they produce counterfactuals that are both minimal and interpretable.

**Report format.** ESE.format_report() produces a structured text report:

```
PREDICTION: FAULT (confidence 0.94)

ATTRIBUTION (top-5):
  sensor_04 [bearing_temp]    0.34
  sensor_11 [vibration_z]     0.21
  sensor_02 [motor_current]   0.18

RULE: IF latent[2] > 0.83 AND latent[7] < -0.31 THEN FAULT (conf=0.91)

COUNTERFACTUAL: If sensor_04 decreased by 12.3 C (-18%), prediction changes to NORMAL.
```

This is designed to be directly usable by maintenance engineers without any ML knowledge.

### 4.8 Cross-Modal Latent Alignment (CMLA)

**File:** `modules/cmla.py`

Industrial systems produce data from many sensor types capturing different physical phenomena: vibration accelerometers, thermal cameras, current transformers, pressure transducers, event logs. These cannot simply be concatenated — they have different units, sampling rates, noise profiles, and information content.

CMLA aligns all modalities in a shared latent space using contrastive learning. Each modality has its own encoder (2-layer MLP with RMSNorm and L2 normalization) that maps raw values to the shared D-dimensional space. Encoders are trained jointly using InfoNCE:

```
L_align = -E[ log( exp(sim(z_m, z_m') / tau) / sum_k exp(sim(z_m, z_k) / tau) ) ]
```

Here z_m and z_m' are embeddings of the same physical event from different modalities (positive pairs). This loss rewards producing similar embeddings when two sensors observe the same event and dissimilar embeddings otherwise.

After alignment, modalities are fused using inverse-variance weighting:

```
z_fused = (sum_m z_m / u_m) / (sum_m 1 / u_m)
```

where u_m is the uncertainty of modality m, estimated as its distance from the mean of all modality embeddings for that timestep. A noisy or poorly-calibrated sensor automatically receives less weight — no manual tuning required.

### 4.9 Hierarchical Timescale Decomposition (HTD)

**File:** `modules/htd.py`

Physical systems operate simultaneously at many timescales. A pump has mechanical dynamics at ~100ms (shaft rotation), thermal dynamics at ~10 minutes, and degradation dynamics at ~months. A single SSM with a fixed timescale will either be too fast (missing slow trends) or too slow (missing fast transients).

HTD runs four parallel SSMs with timescales tau = [0.01s, 0.1s, 1.0s, 10.0s], each constrained to operate at its designated timescale:

```
dt_i = tau_i * 2 * sigmoid(linear_dt_i(x))    constrained near tau_i
A_bar_i = exp(-exp(log_A_i) * dt_i)
h_t^i = A_bar_i * h_{t-1}^i + (1 - A_bar_i) * B_i(x_t)
```

The levels communicate via bottleneck projections:

- **Fast to slow:** The slow level's input is augmented with a compressed version of the fast level's hidden state. A spike in vibration tells the slow model that something changed, even before temperature follows.
- **Slow to fast feedback:** The fast level receives a bias from the slow level's state. Historical context (the system has been running hot for three hours) modulates how the fast model interprets current readings.

Slower levels process subsampled inputs — level i processes one input for every 2^i tokens at level 0. All level outputs are upsampled to the original resolution and concatenated before projection.

Every output position encodes information from the immediate past (fast level), the recent past (intermediate levels), and the long-term trend (slow level). This is genuinely multi-timescale, not just multi-resolution.

### 4.10 Safety-Critical Policy Head

**File:** `modules/safety.py`

For control applications — adjusting setpoints, triggering alarms, commanding actuators — VULGARIS includes a certified-safe output layer. This is not an optional guardrail; it is a hard mathematical constraint on the output space.

**Control Barrier Functions.** A CBF h: S → R defines a safe set S = {s : h(s) >= 0}. For discrete-time systems, the CBF condition states:

```
h(s_{t+1}) >= (1 - gamma) * h(s_t)    for all admissible inputs u
```

If h(s_0) >= 0 then h(s_t) >= 0 for all t. The safety filter projects a nominal (uncertified) action onto the closest safe action:

```
minimize over u:  ||u - u_nom||^2
subject to:       grad_h(s) * f(s,u) + gamma * h(s) >= 0
```

The closed-form projection for this linear constraint is:

```
u_safe = u_nom + max(0, -grad_h * u_nom - gamma * h) / (||grad_h||^2 + eps) * grad_h
```

This is differentiable with respect to u_nom, so gradients flow back through the safety filter during training. The policy learns to produce actions requiring minimal correction.

**Lipschitz certification.** All layers in the policy network use spectral normalization (3-step power iteration) to enforce:

```
||f(x) - f(y)||_2 <= L * ||x - y||_2
```

This provides a certified bound on output sensitivity to input perturbations — critical for proving robustness to sensor noise.

---

## 5. The Unified Loss Function

All training objectives are unified into a single loss derived from rate-distortion-complexity principles:

```
L = L_task                    primary objective (MSE or cross-entropy)
  + beta  * L_memory          HMB: compression, I(memory; past)
  + gamma * L_dag             CRG: acyclicity + sparsity penalty
  + delta * L_ewc             SHCAL: elastic weight consolidation
  + eps   * L_conformal       coverage gap: max(0, alpha - actual_miscoverage)
  + zeta  * L_cbf             safety: CBF constraint violations
  + eta   * L_temporal        HTD coherence: mean ||h_t - h_{t-1}||^2
  + theta * L_contrastive     CMLA: InfoNCE cross-modal alignment
```

From a variational inference perspective: L_task is the negative log-likelihood, L_memory is the rate term penalizing memory bloat, L_dag and L_temporal are complexity regularizers preferring simpler structures, L_ewc is a prior over parameters, and L_conformal and L_cbf enforce hard requirements.

**Default hyperparameter weights:**

| Term | Parameter | Default | When to increase |
|------|-----------|---------|-----------------|
| L_task | 1.0 | always | — |
| L_memory | beta | 0.1 | more compression needed |
| L_dag | gamma | 0.01 | sparser causal graph needed |
| L_ewc | delta | 0.1 | catastrophic forgetting observed |
| L_conformal | eps | 0.05 | coverage undershoot |
| L_cbf | zeta | 1.0 | safety violations in production |
| L_temporal | eta | 0.01 | hidden states too jumpy |
| L_contrastive | theta | 0.1 | multi-modal alignment weak |

---

## 6. Training from Scratch

### Installation

```bash
pip install numpy scipy scikit-learn pyyaml tqdm rich
# for inference server only:
pip install fastapi uvicorn pydantic
```

### Minimal training script

```python
import numpy as np
from config import ModelConfig
from model.vulgaris import Vulgaris
from training.loss import VulgarisLoss
from training.optimizer import SpectralAdamW, CosineSchedule
from training.pipeline import TrainingPipeline

config = ModelConfig(input_dim=32, output_dim=1)
config.ase.latent_dim = 256
config.training.max_steps = 50000

model = Vulgaris(config)
print(f"Parameters: {model.n_params():,}")

loss_fn   = VulgarisLoss(config, task='regression')
optimizer = SpectralAdamW(list(model.parameters()), lr=3e-4, weight_decay=0.01)
scheduler = CosineSchedule(optimizer, warmup_steps=1000, max_steps=50000, min_lr=1e-6)
pipeline  = TrainingPipeline(model, config, loss_fn, optimizer, scheduler)

for step in range(50000):
    x = np.random.randn(32, 32, 1024).astype('float32')   # batch, channels, time
    y = np.random.randn(32, 1).astype('float32')
    metrics = pipeline.train_step(x, y)
    if step % 500 == 0:
        print(f"step {step}  loss={metrics['total_loss']:.4f}")
    if step % 5000 == 0:
        pipeline.save_checkpoint(f'step_{step}')
```

### Pretraining strategy

There are three recommended phases.

**Phase 1 — Self-supervised signal mastery.** Mask 15–30% of input channels at random and train to reconstruct them from the remaining channels. This drives CRG structure discovery (which signals are correlated?), ASE filter learning (what frequencies matter?), and SSSR memory development (which history is useful for reconstruction?).

**Phase 2 — Temporal contrastive pretraining.** Train the model to predict whether two signal windows are from the same timestep (positive) or different timesteps (negative). This sharpens the representations and builds temporal invariance to sensor noise.

**Phase 3 — Causal structure hardening.** Run several epochs with high gamma (DAG penalty). This forces W to converge to a sparse, acyclic graph that captures the dominant causal relationships. Inspect this graph afterward — it should match your engineering knowledge.

### Compute estimates

| Config | Params | GPU Memory | Training time (A100) |
|--------|--------|------------|----------------------|
| Edge (latent=64) | ~150K | 2 GB | ~6 hours |
| Standard (latent=256) | ~2.5M | 8 GB | ~2 days |
| Server (latent=512) | ~10M | 24 GB | ~1 week |
| Research (latent=1024) | ~40M | 80 GB | ~3 weeks |

---

## 7. Domain Adaptation

After pretraining, adapting to a new domain takes approximately 6 hours on a single GPU and requires only 1,000–10,000 labeled examples.

```python
# Step 1: register domain
domain_idx = model.dah.register_domain(
    domain_name='wind_turbine_offshore',
    metadata={'equipment': 'wind_turbine', 'env': 'offshore', 'rated_kw': 5000}
)

# Step 2: freeze base, train only adapters
model.freeze_base()
print(f"Trainable: {model.n_adapter_params():,} params")

adapter_params = [p for name, p in model.named_parameters()
                  if 'hyper_' in name or 'adapter' in name]
optimizer = SpectralAdamW(adapter_params, lr=1e-3)

for step in range(5000):
    x, y = next(domain_data_gen)
    metrics = pipeline.train_step(x, y, domain_idx=domain_idx)

# Step 3: deploy
model.set_domain(domain_idx)
from inference.streaming import StreamingInference
engine = StreamingInference(model, batch_size=1)
result = engine.step(sensor_reading())
```

Adapter parameters can be distributed to edge devices independently of the base model weights, significantly reducing deployment overhead.

---

## 8. Streaming Inference

VULGARIS is a streaming model first. The `StreamingInference` class manages all statefulness, normalization, and latency tracking.

```python
from inference.streaming import StreamingInference

engine = StreamingInference(model=model, batch_size=1, mode='predictive')

# Single-step inference
result = engine.step(np.array([[...]], dtype='float32'))  # (batch, channels)
# result keys: 'prediction', 'uncertainty', 'latency_ms', 'step'

# Latency statistics after warmup
stats = engine.get_latency_stats()
# keys: 'mean_ms', 'p50_ms', 'p95_ms', 'p99_ms', 'max_ms'
```

**Mode switching:**

```python
engine.switch_mode('diagnostic')    # enables ESE recording (+5-10ms overhead)
engine.switch_mode('prescriptive')  # enables safety filter (control applications)
engine.switch_mode('predictive')    # standard, minimal overhead
```

**INT8 quantization for constrained hardware:**

```python
engine.quantize_model(bits=8)
# Symmetric per-tensor quantization: scale = max(|w|) / 127
# Typical accuracy loss: <0.5% on well-trained models
```

**REST API:**

```python
from inference.server import InferenceServer
server = InferenceServer('checkpoints/latest.npz', config=config)
server.run()   # 0.0.0.0:8000
```

Available endpoints:

| Method | Path | Purpose |
|--------|------|---------|
| POST | /predict | Single window |
| POST | /stream/start | Start stateful session |
| POST | /stream/{id}/step | Push one timestep |
| GET | /stream/{id}/stats | Latency + coverage |
| DELETE | /stream/{id} | End session |
| POST | /explain | Attribution + rules |
| POST | /counterfactual | Generate counterfactual |
| GET | /domains | List domains |
| POST | /domains/register | Register new domain |

---

## 9. Federated Deployment

Industrial data is almost always siloed — by legal requirement, competitive sensitivity, or physical isolation. VULGARIS implements federated continual learning with differential privacy so that a global model can be improved by a fleet of devices without any device sharing its raw data.

```python
from federated.protocol import FederatedContinualLearning

# Server
fed = FederatedContinualLearning(
    global_model=model,
    config=config,
    dp_noise=1.0,      # Gaussian noise sigma
    compression=0.01,  # top-1% gradient compression
)

# Client: train locally, then submit gradients
client_id = 'plant_43_machine_7'
domain_idx = fed.register_client(client_id, metadata={'plant': 43, 'machine': 7})

local_grads = {name: param.grad for name, param in model.named_parameters()
               if param.grad is not None}
result = fed.client_update(client_id, local_grads, n_samples=1024)

# Server: aggregate and update global model
delta_norms = fed.aggregate(round_id=current_round)

# Privacy accounting
privacy = fed.privacy_report()
print(f"Privacy budget: epsilon={privacy['epsilon']:.2f}, delta={privacy['delta']}")
```

**FedProx** — add to the client's local loss to prevent drift on non-IID data:

```python
penalty = fed.fedprox_penalty({name: p.data for name, p in model.named_parameters()})
total_loss = task_loss + penalty
```

The (ε, δ)-DP guarantee uses Rényi differential privacy accounting with an alpha sweep over [2, 64], which is tighter than the classical moments accountant for practical parameters.

Byzantine fault tolerance is implemented as a trimmed mean aggregation: the top and bottom 10% of client updates (by L2 norm) are discarded before averaging. This makes the aggregation robust to up to 10% of clients being compromised.

---

## 10. Configuration Reference

Full configuration is in `config.py` via `ModelConfig`. All sub-configs have reasonable defaults.

### Core dimensions

| Parameter | Default | Impact |
|-----------|---------|--------|
| input_dim | 64 | Number of input channels |
| output_dim | 64 | Number of output dimensions |
| n_classes | 0 | 0 = regression; >0 = classification |
| ase.latent_dim | 256 | Model width d_model. Most impactful parameter |
| ase.n_scales | 8 | Temporal dilation scales for wavelet bank |
| ase.n_filters | 16 | Wavelets per scale |

### SSM tuning

| Parameter | Default | Notes |
|-----------|---------|-------|
| sssr.state_dim | 256 | Hidden state size per head |
| sssr.d_inner | 512 | Expansion (should be 2 * latent_dim) |
| sssr.n_heads | 8 | Parallel SSM heads |
| sssr.dt_min | 0.001 | Fastest timescale capturable |
| sssr.dt_max | 0.1 | Slowest per-step timescale |
| sssr.hebbian_lr | 1e-4 | Online adaptation rate |

### Memory configuration

| Parameter | Default | Notes |
|-----------|---------|-------|
| hmb.buffer_size | 512 | Working buffer (recent steps) |
| hmb.archive_size | 4096 | Event archive capacity |
| hmb.surprise_threshold | 2.0 | Sigma above which to archive |
| hmb.consolidation_interval | 1000 | Steps between compressions |

### Training

| Parameter | Default | Notes |
|-----------|---------|-------|
| training.lr | 3e-4 | Peak learning rate |
| training.warmup_steps | 1000 | Linear warmup steps |
| training.grad_clip | 1.0 | Global gradient norm clip |
| training.spectral_clip | 2.0 | Max spectral norm for 2D matrices |
| training.weight_decay | 0.01 | AdamW decoupled L2 |

---

## 11. Benchmarks and Expected Performance

The benchmark suite (`benchmarks/suite.py`) generates four high-fidelity synthetic datasets.

**Power Grid Monitoring** — 32 sensors at 60 Hz (voltage, current, frequency harmonics). Fault types include voltage sag, overcurrent, and harmonic distortion. Task: 5-class fault classification.

**SCADA Process Control** — 24 sensors: temperature, pressure, flow rate, valve positions. Three operating regimes with transitions, gradual drift, and sudden component failures. Task: anomaly detection with root cause attribution.

**5G RAN Performance** — 16 cells, 4 KPIs each (PRB utilization, SINR, throughput, latency). Traffic bursts, handover events, inter-cell interference. Task: multi-cell KPI prediction and congestion forecasting.

**Predictive Maintenance** — 20 sensors: 3-axis vibration, bearing temperature, motor current. Bearing degradation following physical wear equations with BPFO fault frequency signatures. Task: Remaining Useful Life regression.

### Expected metrics at standard config (latent_dim=256)

| Dataset | Task | MAPE / F1 | Conformal Coverage | CPU p99 |
|---------|------|-----------|--------------------|---------|
| Power Grid | Classification | F1 > 0.90 | 89–91% | <20ms |
| SCADA | Anomaly detection | F1 > 0.85 | 88–92% | <15ms |
| 5G RAN | Regression | MAPE 5–8% | 88–92% | <18ms |
| Predictive Maint. | RUL regression | MAPE 12–18% | 87–93% | <15ms |

Coverage measured as the fraction of true values within the 90% conformal prediction interval. Target is 90% ± 2%.

---

## 12. The Custom Engine

VULGARIS does not use PyTorch, TensorFlow, or JAX. The entire computation stack is built from scratch using numpy as the numerical backend. This is deliberate.

**Deployment flexibility.** PyTorch adds ~500 MB of dependency weight and requires CUDA toolkits. On a $30 edge SBC with 512 MB of RAM and no GPU, this is untenable. The VULGARIS engine has no mandatory dependency beyond numpy and scipy.

**Full control over backward passes.** Custom operations like the SSM parallel scan and the NOTEARS matrix exponential have non-standard gradients. Implementing these directly in the engine is cleaner and easier to audit than wrapping them in framework-specific custom autograd functions.

**Transparency.** Every operation is readable Python. No JIT compilation, no kernel fusion, no graph optimization that obscures what is actually happening. This matters for regulated industries where the model's computation must be auditable by a third party.

**The tradeoff.** Without PyTorch's fused CUDA kernels, VULGARIS is slower at training time — roughly 3–5x slower than an equivalent PyTorch model on the same hardware. The CUDA kernels in `core/kernels/` partially address this for the two hottest paths.

### How the autograd works

`engine/tensor.py` implements tape-based reverse-mode differentiation. Every operation creates a closure that computes the backward pass and registers `_prev` pointers (the inputs that contributed to this output). When `backward()` is called on a scalar loss:

1. Topological sort of the computation graph via DFS from the loss node
2. Traversal in reverse order, calling each operation's backward closure
3. Each closure accumulates into `.grad` using `(existing_grad or 0) + contribution`

Broadcasting is handled by `_unbroadcast(grad, target_shape)`, which sums out any dimensions that were broadcast during the forward pass. This is the most common source of bugs in custom autograd implementations.

---

## 13. CUDA and Rust Acceleration

### CUDA kernels

**SSM parallel scan** (`core/kernels/ssm_scan.cu`) implements the Blelloch work-efficient parallel prefix scan for h_t = a_t * h_{t-1} + b_t. The scan uses shared memory within each CUDA block and a two-pass strategy for sequences longer than the block size. Expected speedup over numpy: 40–60x for sequences of length 1024+.

**Wavelet convolution** (`core/kernels/wavelet.cu`) implements batched 1D convolution with dilation for the ASE filter bank. Texture memory is used for the read-only filter bank. Expected speedup: 20–30x.

```bash
cd core
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4
```

The Python wrappers in `core/__init__.py` detect compiled `.so` files and use CUDA automatically, falling back to numpy otherwise.

### Rust event streaming

The Rust runtime (`runtime/`) provides a high-performance event ingestion layer for asynchronous multi-sensor streams. It handles event buffering, sliding-window assembly with linear interpolation for missing values, backpressure tracking, and direct numpy array handoff via PyO3.

```bash
cd runtime
pip install maturin
maturin develop --release
```

Usage:

```python
from vulgaris_runtime import PyEventProcessor

proc = PyEventProcessor(n_sensors=32, window_size=256, window_step=64)
proc.push(timestamp=time.time(), sensor_id=7, value=23.4, quality=100)

batch = proc.try_get_batch()   # np.ndarray (32, 256) float32, or None
if batch is not None:
    result = engine.step(batch[np.newaxis])
```

---

## 14. Theoretical Guarantees

**Stability of SSSR.** The hidden state h_t is bounded for all inputs. Since A_t = exp(-exp(log_A) * dt_t) and all terms are positive, every element of A_t lies strictly in (0, 1). The recurrence h_t = A_t * h_{t-1} + B_t * x_t is therefore contractive for bounded inputs.

**Acyclicity of CRG.** The NOTEARS constraint h(W) = trace(expm(W * W)) - n = 0 is satisfied when and only when W has no entry-wise nonzero cycle. When enforced as a hard constraint via augmented Lagrangian, the learned graph is guaranteed to be a DAG (Zheng et al., NEURIPS 2018).

**Conformal coverage.** For exchangeable test data, the conformal predictor guarantees P(y_T in C_T) >= 1 - alpha. For non-stationary data, the exponential forgetting mechanism provides adaptive coverage. Coverage is guaranteed to be at least 1 - alpha on average over any sliding window of length W when the forgetting factor lambda >= 2 / (W * alpha).

**Privacy of federated updates.** With noise multiplier sigma, clipping norm C, and sampling ratio q = batch / n_local, each aggregation round satisfies (epsilon, delta)-differential privacy where epsilon is computed via Rényi DP moments accountant. Budget accumulates sublinearly in the number of rounds.

**Lipschitz bound.** With spectral normalization applied to all layers with maximum spectral norm L_max, the policy network satisfies ||f(x) - f(y)||_2 <= L_max^(n_layers) * ||x - y||_2. For a 2-layer network with L_max=10, the global Lipschitz constant is at most 100.

---

## 15. Known Limitations and Future Work

**Training is slower than PyTorch.** The custom numpy autograd engine is correct but not optimized. The parallel SSM scan backward pass runs a Python loop where a CUDA kernel would be orders of magnitude faster. Until the full kernel library is complete, training large configurations requires either many CPU cores or patience.

**CRG structure discovery has latent confounder blindness.** The Granger causality scores used to initialize W are lagged correlations, which do not correctly identify causal structure when unobserved confounders exist. The NOTEARS constraint ensures acyclicity but not causal correctness. For well-instrumented systems where confounders are observable, the discovered graph is accurate. For systems with hidden shared causes, a variational inference approach is needed.

**Hebbian updates are local heuristics.** The SHCAL Hebbian rule updates parameters using only local activations. There is no global convergence guarantee. The shadow mode validation catches obvious failures, but subtle degradation can accumulate over thousands of steps. Running periodic gradient-based fine-tuning on a held-out validation set is recommended for production deployments.

**Hypernetwork generalization is bounded.** Domain adapters for domains significantly outside the pretraining distribution will not generalize well. The hypernetwork learned a mapping from metadata to adapter weights, but that mapping is reliable only near the training distribution. For radically novel domains, 10–100 labeled examples for adapter fine-tuning are required.

**No multi-GPU training.** The engine does not implement data or model parallelism across devices. For configurations larger than what fits on one GPU's memory, this is a real limitation. Adding gradient synchronization is straightforward conceptually but requires significant implementation work.

**ESE rules require sufficient data volume.** The CART tree is meaningful only when at least a few thousand (hidden_state, prediction) pairs have been collected. Rules improve continuously with more data. In early deployment, treat them as provisional and supplement with domain expertise.

---

*VULGARIS is a research prototype implementing novel ideas at the intersection of industrial AI, continuous learning, and certified safety. It is designed to be studied, extended, and improved. The codebase is intentionally readable — there are no obfuscated abstractions, no performance-over-clarity tradeoffs. Every design choice described in this document is directly reflected in the source code. Read the source alongside this document, not instead of it.*
