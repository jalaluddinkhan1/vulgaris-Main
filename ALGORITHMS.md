# VULGARIS — Algorithms & Mathematics Reference

Complete mathematical reference for every algorithm used in VULGARIS.

---

## Table of Contents

1. [Autograd Engine — Tensor & Core Ops](#1-autograd-engine--tensor--core-ops)
2. [ASE — Adaptive Signal Embedding](#2-ase--adaptive-signal-embedding)
3. [SSSR — Selective State-Space Recurrence](#3-sssr--selective-state-space-recurrence)
4. [HTD — Hierarchical Timescale Decomposition](#4-htd--hierarchical-timescale-decomposition)
5. [DAH — Domain-Adaptive Hypernetwork](#5-dah--domain-adaptive-hypernetwork)
6. [CRG — Causal Routing Graph](#6-crg--causal-routing-graph)
7. [HMB — Hierarchical Memory Bank](#7-hmb--hierarchical-memory-bank)
8. [ESE — Explainability Engine](#8-ese--explainability-engine)
9. [Safety — Control Barrier Functions](#9-safety--control-barrier-functions)
10. [SHCAL — Self-Healing Continual Adaptation Layer](#10-shcal--self-healing-continual-adaptation-layer)
11. [CMLA — Cross-Modal Latent Alignment](#11-cmla--cross-modal-latent-alignment)
12. [Loss — Unified Variational Loss](#12-loss--unified-variational-loss)
13. [Optimizer — SpectralAdamW, PCGrad & CosineSchedule](#13-optimizer--spectraladamw-pcgrad--cosine-schedule)
14. [Distributed Training — Data-Parallel Allreduce](#14-distributed-training--data-parallel-allreduce)
15. [Log Encoder — O(1) Template Extraction](#15-log-encoder--o1-template-extraction)
16. [Event Buffer — Shared-Memory Ring Buffer](#16-event-buffer--shared-memory-ring-buffer)
17. [Security Hardening](#17-security-hardening)
18. [Rule Engine — Neuro-Symbolic Integration](#18-rule-engine--neuro-symbolic-integration)
19. [Model Architecture — Vulgaris Forward Pass](#19-model-architecture--vulgaris-forward-pass)
20. [Training Pipeline — Augmentation & Curriculum](#20-training-pipeline--augmentation--curriculum)
21. [Conformal Prediction — Non-Stationary Coverage](#21-conformal-prediction--non-stationary-coverage)
22. [Federated Learning — DP-SGD & Byzantine Robustness](#22-federated-learning--dp-sgd--byzantine-robustness)
23. [Drift Detection — KS, MMD & Wasserstein](#23-drift-detection--ks-mmd--wasserstein)
24. [Degradation & Serving Infrastructure](#24-degradation--serving-infrastructure)
25. [Benchmarking — Datasets, Metrics & Baselines](#25-benchmarking--datasets-metrics--baselines)
26. [RMC — Regime Mixture Core](#26-rmc--regime-mixture-core)
27. [Knowledge Distillation](#27-knowledge-distillation)
28. [Active Learning — Pool-Based Uncertainty Sampling](#28-active-learning--pool-based-uncertainty-sampling)
29. [Speculative Rollout — Draft-Verify Autoregressive Decoding](#29-speculative-rollout--draft-verify-autoregressive-decoding)
30. [ICL — In-Context Learning](#30-icl--in-context-learning)
31. [Ontology Embedding — Industrial Semantic Context](#31-ontology-embedding--industrial-semantic-context)

---

## 1. Autograd Engine — Tensor & Core Ops

**Files:** `engine/tensor.py`, `engine/ops.py`, `engine/module.py`

Reverse-mode automatic differentiation. A dynamic computation graph is built on the forward pass; `backward()` traverses it in reverse topological order and accumulates gradients. All neural operations are implemented as pure numpy with explicit backward closures.

### Core Operations

**Matrix multiplication**

```
Forward:  out = A @ B
Backward: dA = grad_out @ B.T
          dB = A.T @ grad_out
```

**Element-wise multiply**

```
Forward:  out = a * b
Backward: da = grad_out * b
          db = grad_out * a
```

**Sum / Mean (reduction)**

```
Forward:  out = sum(x, axis)
Backward: dx = broadcast(grad_out, shape(x))
```

**Softmax**

```
Forward:  shifted = x - max(x)          # numerical stability
          s = exp(shifted) / sum(exp(shifted))
Backward: dx = s * (grad_out - sum(grad_out * s))
```

**Layer Normalisation**

```
Forward:  μ = mean(x),  σ² = var(x)
          x̂ = (x - μ) / √(σ² + ε)
          out = γ * x̂ + β
Backward: standard LN gradient through normalisation
```

**SiLU**

```
Forward:  out = x * σ(x)             where σ(x) = 1 / (1 + exp(-x))
Backward: dx  = σ(x) * (1 + x * (1 - σ(x))) * grad_out
```

**GELU**

```
Forward:  out = x * 0.5 * (1 + erf(x / √2))
Backward: dx  = (Φ(x) + x * φ(x)) * grad_out
          where Φ = CDF, φ = PDF of N(0,1)
```

**Power**

```
Forward:  out = x ^ n
Backward: dx  = n * x^(n-1) * grad_out
```

### Conv1D (Grouped, Dilated)

Input `x ∈ ℝ^{B × C_in × L}`, weight `W ∈ ℝ^{C_out × (C_in/groups) × K}`:

```
L_out = (L + 2·padding − K) // stride + 1

Algorithm:
  1. Pad: x_pad = pad(x, [(0,0), (0,0), (padding, padding)])
  2. Im2col: windows ∈ ℝ^{B × C_in × K × L_out}
  3. Flatten: col ∈ ℝ^{B × (C_in·K) × L_out}
  4. Per group g:  out_g = einsum("oi,bil→bol", W_g_flat, col_g)
  5. Concatenate groups; add bias
```

### SSM Parallel Scan (Associative)

Mamba-style scan over `A ∈ ℝ^{B×L×N}`, `B ∈ ℝ^{B×L×N}`:

```
Monoid:  (a₁, b₁) ⊕ (a₂, b₂) = (a₂ · a₁,  a₂ · b₁ + b₂)

Forward recurrence:
  h[0] = B[0]
  h[t] = A[t] · h[t-1] + B[t]

y[t] = dot(C[t], h[t])

Backward (reverse scan via chain rule accumulation)
```

### Selective Scan Step (Single-Timestep Streaming)

```
dt_sp     = log(1 + exp(dt))                     (softplus)
dt        = clip(dt_sp, dt_min=0.001, dt_max=0.1)
A_bar     = exp(−exp(A_log) · dt)                (ZOH decay)
B_bar     = (1 − A_bar) · B
h_new     = A_bar · h_prev + B_bar · x_t
y         = C · h_new + D · x_t                   (skip connection D)
```

### Matrix Exponential Trace (NOTEARS Gradient)

```
tr(expm(W)) computed via scipy.linalg.expm
Gradient:  d/dW tr(expm(W)) = expm(W)^T
```

Used inside the NOTEARS penalty: `h(W) = tr(expm(W ⊙ W)) − n`.

### Top-K Sparse (Straight-Through)

```
threshold = k-th largest value (via argpartition)
hard_mask = (scores ≥ threshold).astype(float)
Backward:  pass gradient through hard_mask unchanged (straight-through estimator)
```

### Differentiable Top-K

```
Hard mask: argpartition → top-k indices
Soft mask (within top-k only):
  s_soft[j] = k · exp(score[j]/T) / Σ_{i∈topk} exp(score[i]/T)

Jacobian (for backward, within top-k):
  d(s_j·k)/d(score_m) = (k/T) · s_j · (δ[j==m] − s_m)
```

---

## 2. ASE — Adaptive Signal Embedding

**File:** `modules/ase.py`

Converts raw sensor channels into latent sequences using a learnable Morlet wavelet filterbank with dilated convolution.

### Morlet Wavelet Kernel

Each filter k is a Morlet wavelet parameterised by learnable scalars:

```
ψ_k(t) = A_k · exp(−0.5 · (t / σ_k)²) · cos(ω_k · t + φ_k)

where:
  A_k   = exp(log_A_k)          (amplitude, always positive)
  σ_k   = exp(log_σ_k)          (width, always positive)
  ω_k   = learnable frequency
  φ_k   = learnable phase
```

### Filter Normalisation

```
energy = √( Σ_t ψ_k(t)² + ε )
ψ_k    = ψ_k / energy           (unit-energy normalisation)
```

### Dilated Convolution

Scale s uses dilation `d = 2^s`. The effective kernel span is:

```
K_eff = (K − 1) · d + 1
```

### Channel Mixing

```
x_mixed = W_mix · x              (W_mix ∈ ℝ^{n_filters × C_in})
```

Implemented as einsum: `x_mixed[b, f, t] = Σ_i W_mix[f, i] · x[b, i, t]`

---

## 3. SSSR — Selective State-Space Recurrence

**File:** `modules/sssr.py`

Multi-head Mamba-style SSM core with ZOH discretisation, selective gating, and online Hebbian adaptation.

### ZOH Discretisation

Continuous-time system `ḣ = −λh + Bu` is discretised with Zero-Order Hold:

```
A_t = exp(−λ · Δt)              where λ = exp(log_A), Δt = softplus(Δt_raw)
h_t = A_t ⊙ h_{t-1} + B_t ⊙ x_t
```

`Δt` is clamped to `[Δt_min, Δt_max]` via:

```
softplus(x) = log(1 + exp(clip(x, −20, 20)))
```

### Output Projection

```
y_t = Σ_j C_t[j] · h_t[j]  +  D ⊙ x_t        (skip connection)
```

### Gated Output (SiLU gate)

```
out = y_SSM ⊙ SiLU(z)
```

### Hebbian Online Adaptation

After each forward pass the decay constants are updated in-place:

```
δ = mean_{b,t}( h_curr ⊙ h_prev − h_curr² )    ∈ ℝ^N
log_A ← clip( log_A + η_hebb · δ,  −5, 0 )
```

This biologically-inspired rule strengthens stable states and weakens oscillatory ones.

---

## 4. HTD — Hierarchical Timescale Decomposition

**File:** `modules/htd.py`

Four parallel SSM levels operating at progressively coarser timescales, coupled by learned bottleneck projections.

### Per-Level State Update

```
Δt   = sigmoid(Δt_raw) · 2τ           (τ is the level-specific timescale)
Ā    = exp(−exp(log_A) · Δt)
h_t  = Ā ⊙ h_{t-1} + (1 − Ā) ⊙ (B · x_t)
```

### Subsampling

Level i processes one frame every `2^i` timesteps:

```
indices = {0, 2^i, 2·2^i, …, T}
```

### Upsampling (nearest-neighbour)

```
y_i_up = repeat(y_i, repeats=T / T_i, axis=time)
```

### Cross-Level Coupling

**Fast → Slow (bottleneck down):**

```
aug  = W_down_{i-1} · h_last_{i-1}       ∈ ℝ^D
x_i  = x_i + aug                          (broadcast over time)
```

**Slow → Fast (bottleneck up):**

```
bias = W_up_i · h_last_{i+1}              ∈ ℝ^{state_dim}
h_t  = h_t + bias
```

---

## 5. DAH — Domain-Adaptive Hypernetwork

**File:** `modules/dah.py`

LoRA-style adapters whose weights are generated by a small hypernetwork conditioned on a learned domain embedding.

### LoRA Forward Pass

```
out = x W_base^T  +  (x B^T) A^T · exp(log_scale)

where:
  W_base ∈ ℝ^{d_out × d_in}   (frozen base weight)
  A ∈ ℝ^{d_out × r}           (adapter up-projection)
  B ∈ ℝ^{r × d_in}            (adapter down-projection)
  r ≪ d_out, d_in              (rank)
```

### Hypernetwork Adapter Generation

```
z       = domain_embedding[idx]         ∈ ℝ^{d_embed}
A_flat  = f_A(z)                        ∈ ℝ^{d_out · r}
B_flat  = f_B(z)                        ∈ ℝ^{r · d_in}
A       = reshape(A_flat, (d_out, r))
B       = reshape(B_flat, (r, d_in))
```

### Hash-Based Domain Indexing

```
h     = SHA-256(JSON(metadata))  →  integer
idx   = h mod n_domains
```

Collisions resolved by linear probing.

---

## 6. CRG — Causal Routing Graph

**File:** `modules/crg.py`

Learns a sparse directed acyclic graph over sensor nodes using the NOTEARS penalty and online Granger-style structure updates.

### NOTEARS DAG Penalty

A weighted adjacency matrix W represents no cycles iff:

```
h(W) = tr(expm(W ⊙ W)) − n = 0

expm(A) ≈ I + A + A²/2! + A³/3! + A⁴/4! + A⁵/5! + A⁶/6!
where A = W ⊙ W   (element-wise square)
```

### Message Passing

```
updated = node_states + node_states @ W_sparse     (W_sparse is top-k entries of W)
```

### Granger Causality (Online)

For each lag `l ∈ {1, …, max_lag}`:

```
corr_ij(l) = (x_i[:-l] − μ_i)^T (x_j[l:] − μ_j)
             ─────────────────────────────────────
             (T − l) · σ_i · σ_j

G_ij = mean_l |corr_ij(l)|
```

### EMA Structure Update

```
acc ← 0.9 · acc + 0.1 · G                  (exponential moving average)
W   ← W  + lr_struct · (acc − |W|)          lr_struct = 1e-3
```

### Regularisation

```
L_dag  = λ_dag  · h(W)
L_spar = λ_spar · Σ_{i,j} |W_ij|
L_reg  = L_dag + L_spar
```

### Conditional Independence Pruning

After every `ci_update_interval` steps, each active edge (i→j) is tested for spurious correlation.

**Confounder selection** — score each candidate node k by the harmonic mean of its Pearson correlations with both i and j:

```
score(k) = 2 · |r_ik| · |r_jk| / (|r_ik| + |r_jk| + ε)
```

Top-2 highest-scoring nodes form the conditioning set S.

**Partial correlation** — residualise i and j on S via OLS, then correlate residuals:

```
x_i_res = x_i − X_S · (X_S^T X_S + εI)^{-1} X_S^T x_i
x_j_res = x_j − X_S · (X_S^T X_S + εI)^{-1} X_S^T x_j

pcorr(i,j | S) = corr(x_i_res, x_j_res)
```

**Edge decision:**

```
if |pcorr(i,j | S)| < ci_threshold:
    W[i,j] ← 0               (prune spurious edge)
    edge_label[i,j] ← "associative"
else:
    edge_label[i,j] ← "causal"
```

### NaN Collapse Guard

Before every structure update, W is checked for non-finite values:

```
if any non-finite in W:
    W ← N(0, 0.01²)  with diagonal zeroed
    granger_accumulator ← 0
```

This prevents NaN propagation from corrupting the DAG penalty gradient.

---

## 7. HMB — Hierarchical Memory Bank

**File:** `modules/hmb.py`

Event-driven episodic memory with VAE compression, surprise-triggered writes, cosine-similarity retrieval, and uncertainty-weighted aggregation.

### VAE Compression (Encoder)

```
μ, log σ² = encoder(h)
z = μ + ε · exp(0.5 · log σ²)        ε ~ N(0, I)   (reparameterisation)
```

### VAE Loss

```
L_recon = ‖ decoder(z) − h ‖²
L_KL    = −0.5 · Σ (1 + log σ² − μ² − σ²)
L_VAE   = L_recon + L_KL
```

### Surprise Signal

```
surprise = ‖ h_actual − h_predicted ‖² / (2 · running_var + ε)
running_var ← 0.99 · running_var + 0.01 · batch_var
```

A slot is written to memory when `surprise > threshold`.

### Cosine Similarity Retrieval

```
q̂ = q / (‖q‖ + ε)
ĉ_m = c_m / (‖c_m‖ + ε)
sim_{i,m} = q̂_i · ĉ_m
```

### Uncertainty-Weighted Attention

```
w_m  = 1 / (u_m + 1e-4)                 (inverse uncertainty)
logits_{i,m} = sim_{i,m} / τ + log w_m  τ = 0.1
α    = softmax(logits)
ctx  = α · C                             (B × D)
```

### Memory Pressure Eviction Policy

When RSS exceeds the eviction threshold, entries are removed in priority order:

```
eviction_score(k) = (uncertainty_k, −timestamp_k − access_count_k)
```

The entry minimising this score is evicted first — low-uncertainty memories are cheapest to re-learn and are therefore expendable before high-uncertainty ones. Within equal uncertainty, LRU (oldest, least accessed) entries are removed first.

Three escalating eviction stages:

```
RSS > RSS_WARN_MB  (3000 MB): evict adapter cache entries
RSS > RSS_EVICT_MB (4500 MB): evict HMB working buffer entries
                             → evict HMB archive entries
```

---

## 8. ESE — Explainability Engine

**File:** `modules/ese.py`

Produces human-readable rules via CART tree extraction, feature attributions via gradient averaging, and adversarial counterfactuals via projected gradient descent.

### CART Split Criterion

**Classification — Gini Impurity:**

```
Gini(y) = 1 − Σ_c p_c²            p_c = |{y=c}| / |y|

gain = Gini(y_parent) − (n_L/n) · Gini(y_L) − (n_R/n) · Gini(y_R)
```

**Regression — Variance:**

```
MSE(y) = Var(y) = E[y²] − (E[y])²
```

### Gradient Attribution

```
attr_j = mean_{b,t} |∂L/∂h_j|           ∈ ℝ^{d_model}
attr   = attr / Σ_j attr_j               (normalised to sum to 1)
```

### Counterfactual Generation

Find minimal perturbation δ such that the model output matches a target:

```
L_cf = ‖ f(h + δ) − y_target ‖²  +  ‖δ‖²  +  0.01 · ‖δ‖₁
δ ← δ − lr_cf · ∂L_cf/∂δ                (gradient descent on δ)
```

---

## 9. Safety — Control Barrier Functions

**File:** `modules/safety.py`

Certified safe action projection using a learned Control Barrier Function (CBF) and Lipschitz-constrained spectral normalisation.

### Spectral Normalisation (Power Iteration)

```
for k = 1 … n_iter:
    v ← W^T u / ‖W^T u‖
    u ← W v   / ‖W v‖
σ_max ≈ u^T W v

if σ_max > L_max:
    W ← W · (L_max / σ_max)
```

### CBF Safety Filter

Given nominal action u_nom and barrier value h(s):

```
Lie_deriv ≈ Σ_j (∂h/∂s_j) · u_nom_j        (first-order approximation)
margin    = Lie_deriv + γ · h(s)            (CBF condition: margin ≥ 0)

violation = ReLU(−h(s))
```

**Correction when margin < 0:**

```
norm_sq   = ‖∂h/∂s‖²  + ε
corr_mag  = max(0, −margin) / norm_sq
u_safe    = u_nom + corr_mag · ∂h/∂s        (minimal projection)
```

**CBF Training Loss:**

```
L_cbf = Σ_t ReLU( −( h(s') − (1 − γ) · h(s) ) )
```

### Gradient of h via Finite Differences

```
∂h/∂s_j ≈ (h(s + ε e_j) − h(s − ε e_j)) / 2ε      ε = 1e-4
```

---

## 10. SHCAL — Self-Healing Continual Adaptation Layer

**File:** `modules/shcal.py`

Prevents catastrophic forgetting via Elastic Weight Consolidation (EWC), strengthens important connections via Hebbian learning, and prunes/grows weights via structural plasticity.

### EWC Penalty

```
L_ewc = (λ/2) · Σ_i F_i · (θ_i − θ*_i)²

where:
  θ*_i = parameters at end of previous task
  F_i  = diagonal Fisher: F_i = E[(∂L/∂θ_i)²]
```

### Fisher Diagonal Estimation

```
F_i ← (1/N) · Σ_{x} (∂L/∂θ_i)²
```

### Hebbian Weight Update

```
ΔW_ij = η · (post_i · pre_j − κ · W_ij)

η = plasticity_rate
κ = 1e-3  (weight decay)
```

As a batch outer product:

```
ΔW = η · (post_avg ⊗ pre_avg − κ · W)
```

### Structural Plasticity

**Pruning:**

```
if |W_ij| < θ_prune  for prune_window steps:
    mask_ij ← 0    (permanently disconnect)
```

**Growing:**

```
if ‖∂L/∂W_ij‖² > θ_grow  for 3 steps  AND  mask_ij = 0:
    mask_ij ← 1
    W_ij    ← N(0, 0.01²)
```

### Conformal Recalibration Trigger

```
score      = ‖ŷ − y‖ / σ̂                   (normalised residual)
threshold  = quantile(scores_calib, 1 − α)   α = 0.1
coverage   = P(score < threshold)

if coverage < 1 − α:
    trigger full recalibration
```

---

## 11. CMLA — Cross-Modal Latent Alignment

**File:** `modules/cmla.py`

Fuses representations from multiple sensor modalities using inverse-variance weighting and aligns them with an InfoNCE contrastive loss.

### L2 Normalisation

```
ẑ_m = z_m / (‖z_m‖₂ + ε)
```

### Per-Modality Uncertainty

```
u_m = ‖ẑ_m − ẑ_mean‖² / (D · σ²_base)       D = d_model, σ²_base = 1.0
u_m = clip(u_m, 1e-6, ∞)
```

### Inverse-Variance Fusion

```
w_m = 1 / u_m
w̃_m = w_m / Σ_m w_m              (normalised weights)
z_fused = Σ_m w̃_m · ẑ_m
```

### EMA Alignment Matrix

```
new_align[i,j] = (μ_i / ‖μ_i‖) · (μ_j / ‖μ_j‖)    (cosine similarity of means)
Align ← 0.99 · Align + 0.01 · new_align
```

### InfoNCE Loss

With temperature τ = 0.07:

```
sim[i,j] = ẑ_i · ẑ_j / τ

positive mask: pos[i,j] = 1  iff same (batch,time) index but different modality

L_InfoNCE = − (1/N) · Σ_i log( Σ_{j∈pos(i)} exp(sim[i,j]) / Σ_{j≠i} exp(sim[i,j]) )
```

---

## 12. Loss — Unified Variational Loss

**File:** `training/loss.py`

All sub-losses are summed with tunable coefficients:

```
L = L_task
  + β      · L_memory       (HMB VAE loss)
  + γ      · L_dag          (CRG NOTEARS + L1)
  + δ      · L_ewc          (SHCAL EWC penalty)
  + ε      · L_conformal    (conformal calibration)
  + 0.1    · L_cbf          (CBF safety)
  + 0.01   · L_temporal     (temporal coherence)
  + 0.01   · L_contrastive  (CMLA InfoNCE)

β = beta_hmb,  γ = gamma_crg,  δ = delta_ewc,  ε = epsilon_conformal
```

### Task Loss

**Classification (cross-entropy):**

```
L_task = −(1/B) · Σ_b log p_{b, y_b}
```

**Regression (MSE):**

```
L_task = (1/BT) · ‖ŷ − y‖²_F
```

### Temporal Coherence

Penalises sudden jumps in hidden state:

```
L_temporal = mean ‖h_{t} − h_{t-1}‖²
```

---

## 13. Optimizer — SpectralAdamW, PCGrad & CosineSchedule

**File:** `training/optimizer.py`

AdamW with spectral-norm clipping, PCGrad multi-task gradient surgery, and a staged loss activation schedule.

### AdamW Update

```
m_t = β₁ m_{t-1} + (1 − β₁) g_t
v_t = β₂ v_{t-1} + (1 − β₂) g_t²

m̂_t = m_t / (1 − β₁^t)           (bias correction)
v̂_t = v_t / (1 − β₂^t)

θ ← θ − lr · m̂_t / (√v̂_t + ε)   (Adam step)
θ ← θ − lr · λ · θ                (decoupled weight decay)
```

Defaults: β₁ = 0.9, β₂ = 0.999, ε = 1e-8

### Global Gradient Clipping

```
‖g‖ = √( Σ_p ‖g_p‖² )
if ‖g‖ > clip_max:
    g_p ← g_p · clip_max / ‖g‖
```

### Spectral Clipping (Power Iteration, 3 steps)

Applied to 2-D weight matrices:

```
for _ in range(3):
    v ← W^T u / ‖W^T u‖
    u ← W v   / ‖W v‖
σ_max = u^T W v

if σ_max > spectral_clip:
    W ← W · spectral_clip / σ_max
```

### Cosine Schedule with Linear Warmup

```
                lr_base · (step + 1) / warmup_steps          if step < warmup_steps
lr_t =  {
                lr_min + (lr_base − lr_min)
                  · 0.5 · (1 + cos(π · progress))           otherwise

progress = (step − warmup_steps) / (max_steps − warmup_steps)
```

### PCGrad — Gradient Surgery for Multi-Task Learning

When two loss gradients conflict (negative cosine similarity), the conflicting component of one is projected out before accumulation.

For each pair of per-loss gradient vectors g_i and g_j:

```
if  g_i · g_j < 0:
    g_i ← g_i − (g_i · g_j / ‖g_j‖²) · g_j     (remove component along g_j)
```

Final parameter update uses the sum of all corrected gradients:

```
g_corrected = Σ_i g_i_corrected
optimizer.step(g_corrected)
```

This prevents dominant losses from interfering with the gradient signal of weaker losses.

### StagedLossSchedule — Curriculum Activation

Loss terms are activated in three waves to stabilise early training:

| Step range    | Active losses                                      |
|---------------|----------------------------------------------------|
| 0 →           | task, temporal                                     |
| 500 →         | + DAG penalty (CRG), InfoNCE (CMLA)               |
| 1000 →        | + conformal, memory (HMB VAE), EWC, CBF safety    |

```
weight_i(step) = 1.0  if step ≥ activation_step_i
               = 0.0  otherwise
```

PCGrad operates only on the active subset returned by `filter_losses(step, losses)`.

---

## 14. Distributed Training — Data-Parallel Allreduce

**Files:** `training/distributed.py`, `training/distributed_sampler.py`, `scripts/train_distributed.py`

Single-node and multi-node data-parallel training via `torch.distributed` (gloo backend for CPU, nccl for GPU) with custom numpy parameter broadcast.

### Gradient Allreduce

After each backward pass, gradients are summed across all ranks and divided by world size:

```
g_global = AllReduce_SUM(g_local)     (ring allreduce via torch.distributed)
g_avg    = g_global / world_size
```

Implementation: numpy array → `torch.from_numpy()` → `dist.all_reduce(op=SUM)` → numpy view back.

### Parameter Broadcast (Initialisation)

Rank 0 sends its weights to all other ranks once at startup (and after checkpoint load):

```
for each parameter θ:
    dist.broadcast(θ, src=0)          (all ranks receive rank-0 values)
```

This guarantees identical initialisation across replicas.

### DistributedSampler — Epoch-Shuffled Sharding

Each rank receives a disjoint, non-overlapping shard of the dataset per epoch.

```
epoch_seed = seed XOR (epoch × 0x9E3779B9)    (Knuth multiplicative hash)
shuffled   = RNG(epoch_seed).permutation(N)

# Padding to ensure equal shard size
padded_len = ceil(N / world_size) × world_size
shuffled   = tile(shuffled, ceil(padded_len / N))[:padded_len]

rank_indices = shuffled[rank :: world_size]
```

Epoch barrier: all ranks call `dist.barrier()` at the end of each epoch before proceeding.

---

## 15. Log Encoder — O(1) Template Extraction

**File:** `preprocessing/log_encoder.py`

Converts free-text log lines into dense 2-D feature vectors using a Drain3-style fixed-depth prefix trie for O(1) amortised template matching.

### Token Normalisation

Before trie lookup, tokens are normalised to collapse high-cardinality values:

```
UUID   →  <UUID>          regex: [0-9a-f]{8}-[0-9a-f]{4}-…
IPv4   →  <IP>            regex: \d{1,3}(\.\d{1,3}){3}
Hex    →  <HEX>           regex: 0x[0-9a-fA-F]+
Number →  <NUM>           regex: \b\d+(\.\d+)?\b
```

### Fixed-Depth Prefix Trie (depth = 4)

```
path = [token_0, token_1, …, token_{depth-1}]     (first D normalised tokens)
node = root
for token in path:
    node = node.children[token]           (create if absent)

template = node.template                  (list of tokens, wildcards = <*>)
```

### Template Merge Rule

When a new log line reaches a leaf but one token differs from the stored template:

```
if template[i] ≠ new_token[i]:
    template[i] ← "<*>"                  (position becomes wildcard)
```

Templates are never split; divergence only adds wildcards.

### Output Encoding

```
encode(line) → [template_id, severity]    ∈ ℝ²

template_id ∈ {0, …, max_templates}       (0 = unknown / cap exceeded)
severity    ∈ {DEBUG=0, INFO=1, WARN=2, ERROR=3, CRITICAL=4}
```

---

## 16. Event Buffer — Shared-Memory Ring Buffer

**File:** `inference/event_buffer.py`

Zero-copy producer/consumer queue using `multiprocessing.shared_memory` for cross-process event streaming without serialisation overhead.

### Memory Layout

```
Header region  (64 bytes):
  [0:8]   write_cursor  (int64)
  [8:16]  read_cursor   (int64)

Data region    (capacity × 2 × 8 bytes):
  [i]  = event vector of shape (2,) float64
```

Capacity must be a power of 2; slot index uses bitwise mask:

```
slot = cursor & (capacity − 1)            (replaces modulo)
```

### Back-Pressure (Put)

```
while (write_cursor − read_cursor) >= capacity:
    sleep(1 ms)                           (spin-wait with yield)
    if elapsed > timeout:
        return False                      (buffer full — drop or retry)

data[write_cursor & mask] ← event
write_cursor ← write_cursor + 1
return True
```

### Batch Drain (Get)

```
n_avail = write_cursor − read_cursor
if n_avail == 0:
    return None                           (empty)

batch = data[read_cursor & mask : (read_cursor + n_avail) & mask]
                                          (may wrap; handle with two-slice copy)
read_cursor ← read_cursor + n_avail
return batch                              shape (n_avail, 2)
```

---

## 17. Security Hardening

**Files:** `model/vulgaris.py`, `inference/streaming.py`, `serve/migration.py`

### Checkpoint Integrity — SHA-256

On save:

```
hash = SHA-256(weights.npz byte content)
metadata["weights_sha256"] = hex(hash)
```

On load:

```
recomputed = SHA-256(weights.npz byte content)
if recomputed ≠ metadata["weights_sha256"]:
    raise ValueError(f"Integrity check failed: …")
```

Old checkpoints without the field skip the check (backwards compatible).

### Telemetry Poisoning Detection — EMA Z-Score

Per input channel, exponential moving average mean and standard deviation are maintained:

```
μ_t = (1 − α) · μ_{t-1} + α · x_t          α = 0.05
σ²_t = (1 − α) · σ²_{t-1} + α · (x_t − μ_t)²

z_t = |x_t − μ_t| / (√σ²_t + ε)
```

After a 30-sample warmup:

```
if z_t > z_thresh (default 6.0):
    x_t ← clip(x_t, μ_t − z_thresh · σ_t, μ_t + z_thresh · σ_t)
    poisoning_flag ← True
```

### Malformed Packet Guard

At the start of each streaming step:

```
if shape(x) ≠ expected_shape:
    raise ValueError("malformed packet")

x ← where(isfinite(x), x, 0.0)           (replace NaN/Inf with 0)
```

### Out-of-Order Timestamp Detection

```
if timestamp < last_timestamp:
    ooo_event ← True
last_timestamp ← max(last_timestamp, timestamp)
```

---

## 18. Rule Engine — Neuro-Symbolic Integration

**File:** `modules/rule_engine.py`

Encodes symbolic IF-THEN rules as dense vectors, injects them into the DAH hypernetwork, enforces them during training via soft penalty, and manages their lifecycle automatically.

### Rule Encoding

Each rule is encoded as a fixed-length vector of dimension `rule_dim = 10`:

```
v = [feat_idx/n_features, op_code/n_ops, threshold (normalised),
     consequence_code/n_consequences, value (clipped to [−1,1]),
     severity/4.0, confidence, support_count/1000 (clipped),
     0.0, 0.0]                                   (reserved)
```

### Domain Rule Injection into DAH

The `RuleEncoder` aggregates all active rules for a domain via masked mean pooling, then projects to `meta_dim`:

```
R         ∈ ℝ^{max_rules × rule_dim}     (stacked rule encodings)
mask      ∈ {0,1}^{max_rules}            (1 = valid rule)

z_rule = out_proj(mean_{valid rules}( rule_proj(R) ))   ∈ ℝ^{1 × meta_dim}

z_final = z_domain + z_rule              (additive injection after meta_mlp)
```

This makes adapter weights rule-aware without changing the base hypernetwork architecture.

### RuleConditionLoss — Soft Constraint Penalty

For each active rule and each sample in the batch, a sigmoid gate measures how strongly the condition holds:

```
gate = σ( (x[feature] − threshold) · direction )   ∈ (0, 1)

direction = +1  if op ∈ {">", "≥"}
          = −1  if op ∈ {"<", "≤"}
          =  0  (no gate) if op = "=="
```

Consequence-type penalty:

```
clamp_output:   penalty = gate · ReLU(ŷ − value)²
suppress:       penalty = gate · ŷ²
boost:          penalty = gate · ReLU(−ŷ + value)²
alert / noop:   penalty = 0
```

Total rule penalty:

```
L_rules = penalty_scale · mean_{rules, batch}(penalty)
```

### Confidence Update (Feedback Loop)

After each inference step with ground-truth feedback:

```
correct_count ← correct_count + 1_{rule fired correctly}
support_count ← support_count + 1_{rule fired}
confidence    = correct_count / (support_count + ε)
```

### Inactivity Decay

Rules that have not fired recently decay in confidence:

```
factor = 0.5 ^ (steps_since_last_fired / decay_half_life)     decay_half_life = 2000
confidence ← confidence × factor                               (manual rules exempt)
```

### Pruning Criterion

A rule is removed when it has enough evidence but consistently underperforms:

```
prune if:  support_count ≥ min_support (20)
       AND confidence < min_confidence (0.30)
       AND source ≠ "manual"
```

### Rule Merging

Two rules targeting the same (feature, op, consequence) with thresholds within `merge_rtol = 10%`:

```
|threshold_i − threshold_j| / max(|threshold_i|, |threshold_j|) < merge_rtol
```

Merged rule:

```
threshold_merged = (support_i · threshold_i + support_j · threshold_j)
                   / (support_i + support_j)
confidence_merged = max(confidence_i, confidence_j)
source            = "merged"
```

### ESE→Registry Feedback (CART Distillation)

After each CART extraction, `RuleDistiller.compare()` classifies each CART rule:

- **agreed** — rule exists and thresholds match within tolerance
- **refined** — rule exists but threshold drifted; update if source ≠ "manual"
- **missing** — not in registry; add if seen in ≥ 2 extractions with `confidence = cart_init_conf (0.60)`
- **emergent** — in registry but absent from CART (model has learned beyond the tree)

---

## 19. Model Architecture — Vulgaris Forward Pass

**File:** `model/vulgaris.py`

Vulgaris is a modular foundational model whose forward pass chains all seven core modules in a fixed residual topology.

### Reversible Instance Normalisation (RevIN)

Applied to raw input `x ∈ ℝ^{B×C×T}` before ASE:

```
μ_c = mean_T(x_{b,c,:})
σ_c = std_T(x_{b,c,:}) + ε
x̂_{b,c,t} = (x_{b,c,t} − μ_c) / σ_c      (normalise)

After output head:
ŷ ← ŷ · σ_c + μ_c                         (denormalise)
```

Statistics are stored per batch and restored on the output before returning predictions.

### Multi-Modal Fusion Path

When `x` is a list of tensors (one per modality):

```
Each x_m: (B, C_m, T) → transpose → (B, T, C_m)
CMLA fuses all modalities → z_fused ∈ ℝ^{B × T × d_model}
ASE is bypassed; z_fused enters the core stack directly.
```

### Core Processing Stack

Single-modality path `x ∈ ℝ^{B×C_in×T}`:

```
z = ASE(RevIN(x))              → (B, T, d_model)
z = z + HTD(z)                 → multi-timescale residual
z = z + SSSR(z)                → SSM residual
z = z + CausalAttention(z)     → attention residual
z = ICL(z, context) if context provided
z = DAH.forward(z, layer_name="skip_proj", domain_idx)
z = z + CRG(z)                 → causal routing + dag_penalty
z = z + HMB(z)                 → memory retrieval + memory_loss
ŷ = OutputHead(z[:, -1, :])    → last-timestep projection
ŷ = SafetyHead(ŷ) if use_safety
ESE.record(z) during training
```

### Causal Self-Attention

Multi-head dot-product attention with causal (lower-triangular) mask:

```
n_heads = d_model // 64
Q, K, V = W_Q z, W_K z, W_V z     each ∈ ℝ^{B × T × d_model}

Split heads: reshape to (B, n_heads, T, head_dim)

scores = Q @ K^T / √head_dim
mask: scores[i,j] = −∞  for j > i     (causal)
α = softmax(scores)
out = α @ V
out = reshape(B, T, d_model) @ W_O
```

### In-Context Learning (ICL)

When a context (reference examples) is provided at inference time:

```
context: (B, n_examples, C, T) → encode each example via ASE+HTD → (B, n_examples, d_model)
cross-attention: query=z, key/value=context_encodings
z_icl = z + CrossAttention(z, context_encodings)
```

Zero-shot adaptation: no gradient update, purely from reference examples in the input.

### Single-Step Streaming

```
step(x_t, state):
  x_t: (B, C_in) → unsqueeze → (B, C_in, 1)
  Full forward pass with T=1; state carries SSSR and HTD hidden states
  Returns: (B, output_dim), updated VulgarisState
```

### Autoregressive Rollout

```
rollout(x_context, horizon):
  Warm-up: forward on context (B, C, T)
  For step t = 1 … horizon:
    ŷ_t = step(last_output, state)
    Feed ŷ_t back as next input
  Returns: (B, horizon, output_dim)
```

---

## 20. Training Pipeline — Augmentation & Curriculum

**File:** `training/pipeline.py`

### Time-Series Augmentation

Applied independently to `x ∈ ℝ^{B×C×T}` during training (each fires with probability `aug_prob = 0.5`):

```
Gaussian noise:      x += N(0, noise_std²)           noise_std = 0.01
Channel dropout:     x[:, mask, :] = 0               mask ~ Bernoulli(p=0.1) per channel
Magnitude scaling:   x[:, c, :] *= U(0.8, 1.2)       per channel
Time warp:           x = roll(x, shift, axis=2)       shift ~ U(−4, 4) steps (circular)
```

### Curriculum Sequence Length

Training uses progressively longer sequences during warmup:

```
warmup_steps = max(1, total_steps × warmup_frac)     warmup_frac = 0.3

if step < warmup_steps:
    length = t_min + (t_max − t_min) · (step / warmup_steps)
else:
    length = t_max

x = x[:, :, :length]     (crop to current curriculum length)
```

### Online Adaptation (No Gradient Step)

Used at inference time to adapt to a new sample without retraining:

```
1. Forward pass in eval mode → ŷ
2. SHCAL Hebbian update on SSSR linear layers using current activations
3. Conformal calibration update: NonStationaryConformal.update(ŷ, y)
4. No optimizer.step() — only Hebbian in-place update
```

### Conformal Calibration Schedule

During training, conformal scores are updated every 50 steps:

```
if step % 50 == 0:
    conformal.update(ŷ_batch, y_batch, sigma=uncertainty_estimate)
```

---

## 21. Conformal Prediction — Non-Stationary Coverage

**File:** `training/conformal.py`

EnbPI-style conformal predictor with exponential forgetting for non-stationary time-series.

### Conformity Score

```
s_t = |y_true − y_pred| / (σ_t + 1e-8)      (normalised absolute residual)
```

### Weighted Empirical Quantile

Older scores receive exponentially decaying weights:

```
w_t = exp(−λ · (T − t))                      λ = forgetting_factor = 0.01

q̂ = inf { q :  Σ_t w_t · 1[s_t ≤ q] / Σ_t w_t  ≥  1 − α }

α = 0.05  →  target 95% coverage
```

Implementation: sort (score, weight) pairs, cumulative weight sum, searchsorted.

### Prediction Interval

```
lower = ŷ − q̂ · σ
upper = ŷ + q̂ · σ
```

### Coverage Loss

```
target_coverage = 1 − α
actual_coverage = fraction(last 500 predictions inside interval)
gap  = max(0, target_coverage − actual_coverage)
L_conformal = gap²
```

### Calibration Criterion

Detector is considered calibrated when:

```
n_scores ≥ 50  AND  |actual_coverage − (1 − α)| ≤ 0.02
```

---

## 22. Federated Learning — DP-SGD & Byzantine Robustness

**File:** `federated/protocol.py`

### Differential Privacy — Gaussian Mechanism

Per-parameter gradient clipping and noise injection:

```
Clip:  g ← g · min(1, C / (‖g‖ + ε))        C = max_grad_norm = 1.0
Noise: g_noisy = g + N(0, (σ · C)²)          σ = noise_multiplier = 1.0
```

### Rényi DP Accounting (Abadi et al. 2016)

```
q = batch_size / n_samples                    (sampling ratio)

For each order α ∈ {2, 3, …, 64}:
  RDA(α) = (q² · α) / (2 · σ²)              (per-step Rényi divergence)
  After n_steps:
  ε(α) = n_steps · RDA(α) + log(1/δ) / (α − 1)

ε_reported = min_α ε(α)                       (tightest bound)
```

### Gradient Compression (Top-K with Error Feedback)

```
k = max(1, int(d · compression_ratio))        compression_ratio = 0.01

Error accumulation:
  accumulated = residual + g
  top_k_idx   = argpartition(|accumulated|, −k)[−k:]
  transmitted = (top_k_idx, accumulated[top_k_idx], shape)
  residual    = accumulated;  residual[top_k_idx] = 0

Receiver:
  g_dense = zeros(shape)
  g_dense[top_k_idx] = top_k_vals
```

### Byzantine Detection

Before aggregation, each client gradient is compared to the round median:

```
median = median(stacked_client_grads)
std    = std(stacked_client_grads)

distance = ‖g_client − median‖
threshold = 2.0 · std · √(grad.size)

if distance > threshold:
    client flagged Byzantine → excluded from aggregation
```

### Robust Aggregation — Trimmed Mean

```
Sort clients by ‖g_client‖ (gradient norm)
Remove top 10% and bottom 10% by norm
g_agg = mean(remaining clients)
global_params ← global_params − g_agg
```

### FedProx Penalty

Regularises local updates toward global parameters:

```
L_fedprox = (μ/2) · ‖θ_local − θ_global‖²     μ = 0.01
```

Added to local loss before each client backward pass.

---

## 23. Drift Detection — KS, MMD & Wasserstein

**File:** `monitoring/drift.py`

Online distributional shift detection comparing a reference window to an incoming current window.

### Kolmogorov-Smirnov Statistic

Computed per feature, reported as max:

```
KS_feat = max_x | ECDF_ref(x) − ECDF_cur(x) |
KS      = max_{features} KS_feat

Drift detected if KS > ks_threshold = 0.1
```

Implementation: sort both samples, merge, compute ECDFs at combined grid, take max absolute difference.

### Maximum Mean Discrepancy (RBF kernel)

```
σ = median( pairwise ‖a − b‖ for a,b in combined sample )     (median heuristic)

k(a, b) = exp( −‖a − b‖² / (2σ²) )

MMD² = E_{r,r'}[k(r,r')] + E_{c,c'}[k(c,c')] − 2·E_{r,c}[k(r,c)]

MMD = √max(0, MMD²)

Drift detected if MMD > mmd_threshold = 0.05
```

Estimated with subsampling: n_samples = 100 per split.

### Wasserstein Distance (1-D per Feature)

```
For each feature d:
  Sort ref → r_sorted, cur → c_sorted
  Interpolate both to common grid of length max(n_ref, n_cur)
  W_d = mean_t | r_interp(t) − c_interp(t) |

W = mean_{features} W_d

Drift detected if W > wasserstein_threshold = 0.1
```

### Detection Logic

A step is flagged as drift if any one statistic exceeds its threshold:

```
drift_detected = (KS > 0.1) OR (MMD > 0.05) OR (W > 0.1)
```

Reference reset: call `reset_reference()` to promote current window to new reference after confirmed shift.

---

## 24. Degradation & Serving Infrastructure

**Files:** `serve/degradation.py`, `serve/versioning.py`, `serve/metrics.py`, `serve/auth.py`, `inference/server.py`

### Latency-Driven Degradation Controller

Maintains a sliding window of 500 observed latencies and 60-second error window:

```
p95 = percentile(last_200_latencies, 95)
error_rate = errors_in_60s / total_in_60s

State machine:
  FULL       → REDUCED     if p95 > 2.0 s
  REDUCED    → ALERT_ONLY  if p95 > 5.0 s  OR  error_rate > 0.1
  Any        → upgrade     if 60 s elapsed below threshold
```

**Behaviour by level:**

| Level        | ESE | DAH | Output                        |
|--------------|-----|-----|-------------------------------|
| FULL         | ✓   | ✓   | Full predictions + uncertainty|
| REDUCED      | ✗   | ✗   | Cached domain 0 adapter       |
| ALERT_ONLY   | ✗   | ✗   | Anomaly score only            |

### In-Process Metrics

Thread-safe counters/gauges/histograms with no external dependency:

```
Counter:    atomic float add; Prometheus text exposition
Gauge:      set / inc / dec
Histogram:  buckets [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
            sliding window max 1000 obs; p95/p99 via np.percentile

Tracked metrics:
  requests_total    (counter)
  requests_errors   (counter)
  request_latency   (histogram, seconds)
  active_requests   (gauge)
  model_version     (gauge)
```

Exposed at `GET /metrics` in Prometheus text format.

### Model Version Registry

```
register(path, tag, metadata):
  hash = SHA-256(file, 65KB chunks)
  entry = {tag, path, checksum, size_bytes, registered_at, metadata}
  index["versions"].append(entry)

promote(tag):
  index["previous"] = index["active"]
  index["active"]   = tag

rollback():
  swap active ↔ previous

verify(tag) → bool:
  recompute SHA-256; compare to stored checksum
```

### REST API Surface (inference/server.py)

| Method | Route | Purpose |
|--------|-------|---------|
| POST | `/predict` | Batch prediction with degradation awareness |
| POST | `/stream/start` | Create stateful streaming session |
| POST | `/stream/{id}/step` | Single-step streaming inference |
| GET  | `/stream/{id}/stats` | Latency and step statistics |
| DELETE | `/stream/{id}` | Close session |
| GET  | `/health` | Model params, active sessions, degradation |
| GET  | `/metrics` | Prometheus exposition |
| GET  | `/degradation` | Current level + p95 + error rate |
| POST | `/domains/register` | Hash-assign domain index |
| POST | `/explain` | ESE CART rule extraction |
| POST | `/counterfactual` | 50-step gradient-based counterfactual |
| POST | `/versions/register` | Register checkpoint with SHA-256 |
| POST | `/versions/{tag}/promote` | Set active version |
| POST | `/versions/rollback` | Revert to previous version |

### Counterfactual Search (Server-Side)

```
For step t = 1 … 50:
  ŷ = model.step(h + δ)
  L = ‖ŷ − y_target‖² + ‖δ‖²
  ∂L/∂δ_j ≈ (L(δ + ε·e_j) − L(δ − ε·e_j)) / 2ε    ε = 1e-3
  δ ← δ − lr · ∂L/∂δ                               lr = req.lr (default 0.01)
```

---

## 25. Benchmarking — Datasets, Metrics & Baselines

**Files:** `benchmarks/suite.py`, `benchmarks/baselines.py`

### Synthetic Dataset Formulas

**Power Grid** (32 sensors, window=128, 50 Hz base):

```
V(t) = A_load(t) · sin(2π·50·t) + 0.1·sin(2π·150·t) + 0.05·sin(2π·250·t) + N(0, 0.02)
A_load(t) = 0.5 + 0.4·sin(t)
Fault (5% prob): V ← V · U(0.5, 0.85)  for 10–30 steps
y = [voltage_norm, freq_deviation, fault_probability, load·(1+freq_dev)]
```

**Industrial Process** (24 sensors, window=64, 3 regimes):

```
Regime switches every 2000 samples
Within regime: drift = 0.001 · (i mod 2000)
Anomaly (3% prob): sensor spike ← 3× amplitude
y = [base_temp + drift, base_pressure, anomaly_flag]
```

**Telecom RAN** (n_cells×4 sensors, window=60):

```
Traffic load(hour) = 0.3 + 0.6·exp(−0.5·((hour−8)²)/4) + 0.4·exp(−0.5·((hour−18)²)/4)
Anomaly (2% prob): throughput ← 0.1×, SINR ← SINR − 10 dB  for 10 steps
y = [avg_throughput / n_cells, std(X)]
```

**Predictive Maintenance** (20 sensors, window=128):

```
health(i) = clip(1 − (i/lifespan)², 0, 1)           (degradation trajectory)
RUL(i) = (lifespan − i) / lifespan · 100

Vibration(t) = (1 + 3·deg)·sin(2π·f·t) + deg·2·sin(2π·4.7f·t) + N(0, 0.1+0.5·deg)
Temperature(t) = 60 + deg·40 + 5·sin(2π·0.1·t) + N(0, 1+2·deg)
Current(t) = 5 + deg·3 + 0.5·sin(2π·50·t) + N(0, 0.2+deg)

y = [RUL, failure_within_100_steps]
```

### Regression Metrics

```
MAE  = mean(|ŷ − y|)
RMSE = √mean((ŷ − y)²)
MAPE = mean(|ŷ − y| / (|y| + ε)) × 100
R²   = 1 − SS_res / SS_tot
```

### Classification Metrics (Per-Class)

```
Precision = TP / (TP + FP + ε)
Recall    = TP / (TP + FN + ε)
F1        = 2 · Precision · Recall / (Precision + Recall + ε)
```

### Split-Conformal Coverage (Benchmark Evaluation)

```
Split at midpoint:
q_level = ceil((1−α) · (mid+1)) / mid        α = 0.05
q̂ = quantile(|residuals_cal|, q_level)
coverage = fraction(|residuals_test| ≤ q̂)    (target 90% and 95%)
```

### Baseline Models

| Model | Algorithm |
|-------|-----------|
| LastValue | `ŷ = x[last_timestep, channel_0]` |
| MovingAverage(k=5) | `ŷ = mean(x[:, 0, −k:])` |
| ExponentialSmoothing(α=0.3) | `s_t = α·x_t + (1−α)·s_{t-1}` |
| ARIMA-lite(p=5) | AR(5) via ridge OLS: `(X^T X + λI)^{-1} X^T y`, λ=1e-3 |
| LSTMLite (random init) | Full LSTM gates (numpy): `c_t = f⊙c_{t-1} + i⊙g`, `h_t = o⊙tanh(c_t)`, readout = `mean(h)` |

### Latency Benchmarking

```
Warmup: 10 inference steps (discard)
Measure: 100 steps via perf_counter
Report: {p50, p95, p99, mean, max} in milliseconds

Memory estimate: n_params × 8 bytes (float64) / 1024²  → MB
FLOPs estimate:  2 × n_params  (weight multiply-accumulate count)
```

---

*Generated from source at `d:\Microsoft\Vulgaris`. All equations derived from actual implementation in the corresponding Python files.*

---

## 26. RMC — Regime Mixture Core

**File:** `modules/rmc.py`

Switch-Transformer-style soft Mixture-of-Experts that routes each token independently to a weighted combination of K linear experts, with an auxiliary load-balancing loss to prevent expert collapse.

### Gating

Given input $\mathbf{x} \in \mathbb{R}^{B \times T \times d}$, a learned gate linear layer produces unnormalised logits which are scaled by temperature $\tau$ and normalised via softmax:

```
logits = x @ W_gate.T                        shape (B, T, K)
g = softmax(logits / τ)                      routing weights
```

### Expert Computation

K independent linear experts, each mapping $\mathbb{R}^d \to \mathbb{R}^d$:

```
e_k(x) = x @ W_k.T + b_k                    k = 1 … K

out = Σ_k  g[:,:,k:k+1] · e_k(x)            weighted sum of expert outputs
```

### Load-Balancing Loss

Hard expert assignment (no gradient):

```
a = argmax(logits, axis=-1)                  shape (B, T)
f_k = mean over (B,T) of [a == k]            fraction of tokens hard-routed to k
P_k = mean over (B,T) of g[:,:,k]            mean routing probability (differentiable)

L_balance = balance_weight · K · Σ_k  f_k · P_k
```

This loss (Fedus et al., 2021) penalises configurations where $f_k$ and $P_k$ are simultaneously large for the same expert, pushing the router towards uniform utilisation without using a non-differentiable operation in the backward pass.

### Regime Assignments

```
regime_assignments() → argmax(logits, axis=-1)    (B, T)  int64
```

Returns the single most-likely expert index per token, used downstream for interpretability and per-regime metric tracking.

**Design decisions:**
- Temperature $\tau > 1$ softens the routing distribution, smoothing gradients early in training; $\tau = 1$ recovers hard top-1 routing in the limit.
- Experts are plain Linear layers (no bias on $W_k$) keeping the module parameter count at $K \times d^2$ — O(1) in sequence length.
- The load-balancing coefficient `balance_weight` defaults to 0.01 and is exposed in `RMCConfig`.

---

## 27. Knowledge Distillation

**File:** `training/distillation.py`

Hinton (2015) knowledge distillation: a smaller student model is trained to match the soft probability distribution produced by a larger, frozen teacher model, in addition to fitting the hard ground-truth labels.

### Temperature-Scaled Soft Targets

For a batch of inputs, both teacher and student produce logit vectors. Soft probabilities at temperature $T$:

```
p_teacher = softmax(z_teacher / T)
p_student = softmax(z_student / T)
```

The soft-target loss is the KL divergence scaled by $T^2$ (the $T^2$ factor compensates for the $1/T$ gradient shrinkage):

```
L_soft = T² · KL(p_teacher ‖ p_student)
       = T² · Σ_c  p_teacher_c · log(p_teacher_c / p_student_c)
```

### Hint Loss

Intermediate layer representations can be aligned via MSE. When teacher and student have different hidden dimensions, `DistillationTrainer` auto-creates a linear projector $W_h \in \mathbb{R}^{d_s \times d_t}$:

```
h_student_proj = h_student @ W_h.T          (d_t-dimensional)
L_hint = (1/BT) · ‖h_student_proj − h_teacher‖²_F
```

### Combined Loss

```
L = α · L_hard + (1 − α) · (T² · L_soft + β · L_hint)
```

where $\alpha$ balances hard and soft objectives and $\beta$ weights the hint term independently.

### DistillationTrainer

```
1. Freeze all teacher parameters (requires_grad = False for all W, b).
2. For each batch:
   a. Forward teacher (no gradient tape).
   b. Forward student.
   c. Compute L_hard, L_soft, L_hint.
   d. Backward on student only.
   e. Optimizer step on student parameters.
```

**Design decisions:**
- Teacher freeze is enforced in `__init__`, not per-step, so gradient computation for teacher parameters never allocates memory.
- The hint projector is created lazily on the first forward pass, after tensor shapes are known.
- Default $T = 4$, $\alpha = 0.5$, $\beta = 0.1$ — all configurable.

---

## 28. Active Learning — Pool-Based Uncertainty Sampling

**File:** `training/active_learning.py`

Pool-based active learning selects the most informative unlabelled examples from a fixed candidate pool to query for annotation, maximising model improvement per labelling cost.

### Acquisition Functions

Given model output $\mathbf{p} \in \mathbb{R}^{B \times C}$ (softmax probabilities over C classes):

**Uncertainty (output variance):**

```
score = var(p, axis=-1)         scalar variance across class probs
```

**Entropy (Shannon):**

```
score = −Σ_c  p_c · log(p_c + ε)
```

**Margin (top-2 difference):**

```
p_sorted = sort(p, descending=True)
score = −(p_sorted[:,0] − p_sorted[:,1])    (negative so higher = more uncertain)
```

**Random:**

```
score = uniform(0, 1, size=B)
```

### MC Dropout

When `n_mc > 1`, the model is queried $n_{\text{mc}}$ times with dropout active; the per-sample mean output is used for entropy and margin, and the per-sample variance across runs is used for the uncertainty acquisition:

```
p_runs = stack([model(x_pool, training=True) for _ in range(n_mc)])   (n_mc, B, C)
p_mean = mean(p_runs, axis=0)
score_uncertainty = var(p_runs, axis=0).mean(axis=-1)
```

### Query

```
query(pool, k):
  scores = acquisition(pool)
  scores[labeled_set] = −∞               exclude already-labelled indices
  indices = argsort(scores)[-k:]          top-k highest uncertainty
  labeled_set ∪= indices
  return indices
```

**Design decisions:**
- The labeled set is maintained as a Python `set` of integer indices. Exclusion is O(1) per index.
- Acquisition function is selected by string name at construction time, not per-call, to avoid repeated dispatch overhead in large loops.

---

## 29. Speculative Rollout — Draft-Verify Autoregressive Decoding

**File:** `inference/speculative.py`

Speculative decoding (Leviathan et al., 2023) accelerates autoregressive generation by using a cheap draft model to propose $\gamma$ future steps, then verifying them in a single forward pass of the full model. VULGARIS adapts this to latent-space regression via an infinity-norm acceptance criterion.

### Draft Phase

`WorldModelHead` maps the current latent state $\mathbf{z}_t$ to a sequence of $\gamma$ draft predictions autoregressively in latent space:

```
z_draft_0 = z_t
for i in 1 … γ:
    z_draft_i = WorldModelHead(z_draft_{i-1})    (lightweight MLP)
y_draft = output_head(z_draft_γ)
```

### Verify Phase

The full VULGARIS model is advanced $\gamma$ steps from $\mathbf{z}_t$:

```
z_verify_γ, _ = model.step_n(x_{t+1:t+γ}, state_t)
y_verify = output_head(z_verify_γ)
```

### Acceptance Criterion

```
δ = ‖y_draft − y_verify‖_∞

if δ < threshold:
    accept draft; advance state to z_draft_γ
    accepted_steps += γ
else:
    reject; state stays at z_verify_γ (full model advance is not wasted)
    accepted_steps += 0
```

### Statistics

```
acceptance_rate   = accepted_steps / total_proposed_steps
effective_speedup = (accepted_steps · γ + rejected_steps) / total_full_model_steps
```

**Design decisions:**
- The infinity-norm criterion is conservative (it bounds worst-case per-output-dimension error) and avoids requiring calibrated per-output thresholds.
- On rejection the full model's verified state is retained, so no computation is discarded.
- `rollout_single()` exposes a simplified path: draft exactly one step, verify, return accepted output.

---

## 30. ICL — In-Context Learning

**File:** `modules/icl.py`

Zero-shot adaptation at inference time without weight updates: a small set of reference (context) examples is encoded into a context vector which is injected into the main stream via cross-attention with a gated residual.

### ContextEncoder

Given $N$ reference examples $\{(\mathbf{x}^{(i)}, \mathbf{y}^{(i)})\}_{i=1}^N$, each pair is independently encoded and mean-pooled:

```
h_i = MLP([x_i ‖ y_i])                      per-example encoding
c   = (1/N) · Σ_i h_i                        context vector  ∈ ℝ^{d_ctx}
```

### InContextAdapter (Cross-Attention)

The main stream query $\mathbf{q} \in \mathbb{R}^{B \times T \times d}$ attends to the context:

```
Q = q @ W_Q
K = c @ W_K                                  context keys (broadcast over B, T)
V = c @ W_V                                  context values

A = softmax(Q @ K.T / √d_head)
ctx_out = A @ V                               cross-attended context  ∈ ℝ^{B×T×d}
```

### Gated Residual

The context output is mixed into the main stream with a scalar gate $g$ initialised near zero, so early training leaves the base model unmodified:

```
g = sigmoid(g_raw)                            g_raw initialised to small negative value
h_out = h + g · ctx_out
```

Near-zero gate initialisation is critical: it ensures the adapter is a near-identity at the start of fine-tuning and does not disrupt pretrained representations.

**Design decisions:**
- Mean-pooling over context examples is permutation-invariant and makes the adapter robust to context ordering.
- Gate is a single scalar per adapter (not per-head), minimising the number of parameters that require initialisation care.
- `ICLConfig.max_context` caps N at inference time to bound the cross-attention compute.

---

## 31. Ontology Embedding — Industrial Semantic Context

**File:** `modules/ontology_embedding.py`

A lightweight structured vocabulary of industrial terminology that injects domain-semantic priors into the model's embedding space without requiring a full language model.

### Term Vocabulary

80 industrial terms are grouped into 12 semantic clusters (e.g., *thermal*, *vibration*, *electrical*, *control*, *network*, *safety*). Each term is a short string; the vocabulary is hardcoded in the module and versioned with the model.

### Cluster Embedding

Each of the 12 clusters is assigned a learned embedding $\mathbf{e}_k \in \mathbb{R}^{d_{\text{ont}}}$. The embedding for a term in cluster $k$ is the cluster embedding (terms within a cluster share an embedding):

```
e_term = E[cluster_id[term]]                  E ∈ ℝ^{12 × d_ont}
```

### Domain Embedding

Given a list of terms active in a domain, the domain embedding is the mean-pool of their term embeddings:

```
e_domain = (1/|terms|) · Σ_{t ∈ terms}  E[cluster_id[t]]
```

This vector is appended to the DAH domain conditioning signal, injecting semantic information about the domain into the hypernetwork.

### OntologyRegistry

```
register(domain_name, term_list):
    e_domain = mean_pool([E[cluster_id[t]] for t in term_list])
    registry[domain_name] = e_domain

lookup(domain_name) → e_domain ∈ ℝ^{d_ont}
```

**Design decisions:**
- Sharing embeddings at cluster granularity (not per-term) reduces the parameter count from 80·d to 12·d while preserving the semantic grouping structure.
- Mean-pooling is differentiable and order-invariant, so domain embeddings can be updated by gradient-based domain adaptation without requiring a fixed term ordering.
- The 80-term vocabulary covers the major IEC 61131 / IEC 61508 / 3GPP industrial sensor categories; extensions are added by registering new terms to existing clusters.
