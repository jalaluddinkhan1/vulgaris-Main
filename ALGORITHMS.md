# VULGARIS — Algorithms & Mathematics Reference

Complete mathematical reference for every algorithm used in VULGARIS.

---

## Table of Contents

1. [Autograd Engine (Tensor)](#1-autograd-engine-tensor)
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
13. [Optimizer — SpectralAdamW & CosineSchedule](#13-optimizer--spectraladamw--cosineschedule)

---

## 1. Autograd Engine (Tensor)

**File:** `engine/tensor.py`

Reverse-mode automatic differentiation. A dynamic computation graph is built on the forward pass; `backward()` traverses it in reverse topological order and accumulates gradients.

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

## 13. Optimizer — SpectralAdamW & CosineSchedule

**File:** `training/optimizer.py`

AdamW with an extra spectral-norm clipping step to control Lipschitz constants during training.

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

---

*Generated from source at `d:\Microsoft\Vulgaris`. All equations derived from actual implementation in the corresponding Python files.*
