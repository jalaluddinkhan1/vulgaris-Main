# VULGARIS — Complete System Overview

> A foundational model for streaming industrial intelligence.
> Built from the ground up for real-time sensor data, edge deployment, and regulated industries.

---

## Table of Contents

1. [What Is VULGARIS](#1-what-is-vulgaris)
2. [The Problem It Solves](#2-the-problem-it-solves)
3. [How Someone Uses It](#3-how-someone-uses-it)
4. [The Full Architecture](#4-the-full-architecture)
5. [Every Module Explained](#5-every-module-explained)
6. [Industry-Specific Use Cases](#6-industry-specific-use-cases)
7. [What Makes It Uniquely Different](#7-what-makes-it-uniquely-different)
8. [How It Compares to Alternatives](#8-how-it-compares-to-alternatives)
9. [The Complete Technical Stack](#9-the-complete-technical-stack)
10. [Deployment Lifecycle](#10-deployment-lifecycle)
11. [Who Built It and Why](#11-who-built-it-and-why)

---

## 1. What Is VULGARIS

VULGARIS is a foundational AI model built specifically for **industrial time-series data** — the kind of data that comes from sensors, machines, power grids, network equipment, semiconductor fabs, and any physical system that produces continuous streams of measurements.

It is not a chatbot. It is not a language model. It is not a general-purpose classifier.

It is a **single unified model** that can simultaneously:
- Detect anomalies in real-time sensor streams
- Predict failures before they happen (Remaining Useful Life)
- Classify machine states and operating regimes
- Learn causal relationships between sensors
- Adapt to entirely new machines or factories without full retraining
- Explain every decision in plain human-readable rules
- Guarantee safe output within certified operational boundaries

The entire system is written in **pure NumPy** with a custom automatic differentiation engine. It has zero dependency on PyTorch, TensorFlow, or any GPU framework. It trains on a CPU and deploys on edge hardware.

---

## 2. The Problem It Solves

### The Industrial AI Problem Today

A power plant has 10,000 sensors. A semiconductor fab has 50,000 process variables. A telecom network has millions of performance counters updated every second.

Today, industrial companies handle this with:

- **Rule-based systems** written by domain experts — brittle, cannot adapt, require constant manual updates
- **Separate models per use case** — one model for anomaly detection, another for classification, another for forecasting — maintained separately, no shared knowledge
- **Generic cloud AI APIs** — data must leave the factory, violating data sovereignty and privacy regulations
- **Academic models** (scikit-learn, simple LSTM) — trained from scratch per deployment, no memory, no causal understanding, no safety guarantees

The result: companies spend years building and maintaining fragile, disconnected AI pipelines. When a new machine is installed, or a process changes, everything breaks and must be rebuilt.

### What VULGARIS Replaces

VULGARIS replaces the entire fragmented stack with one model that:

- Runs **on-premise** — data never leaves the facility
- Learns **causal structure** from the sensors themselves — no expert rule writing
- Adapts to **new domains** (new machines, new factories) in 500 training steps instead of months
- Operates **continuously** — updating itself as machines age, conditions drift, and processes evolve
- Provides **certified uncertainty** — every prediction comes with a mathematically guaranteed confidence interval
- Explains **why** it made every decision — in terms operators and regulators understand

---

## 3. How Someone Uses It

VULGARIS follows a simple three-phase lifecycle for any deployment:

### Phase 1 — Connect and Configure (Day 1)

```python
from vulgaris import Vulgaris, ModelConfig

config = ModelConfig(
    input_dim=32,       # number of sensors
    n_classes=4,        # operating states: normal, degraded, fault, critical
    output_dim=1,       # regression output (e.g. Remaining Useful Life)
)
model = Vulgaris(config)
```

Tell the model about the domain — what kind of industrial environment this is:

```python
domain_idx = model.dah.register_domain("pump_station_A", {
    "sector": "water_treatment",
    "signal_type": "vibration_pressure_flow",
    "n_sensors": 32,
    "sample_rate_hz": 100,
})
```

Optionally inject known engineering rules before any data is seen:

```python
registry.add_rule(domain_idx, Rule(
    feature=7,               # vibration_rms sensor index
    op=">",
    threshold=3.5,           # 3.5g RMS is abnormal for this pump
    consequence="alert",
    name="vibration_alarm",
    source="manual",
))
```

### Phase 2 — Train on Your Own Data (Days 1–7)

No pre-trained weights needed. The model learns entirely from the customer's own sensor history:

```python
from vulgaris import TrainingPipeline, VulgarisLoss, SpectralAdamW, CosineSchedule

optimizer = SpectralAdamW(model.parameters(), lr=3e-4)
scheduler = CosineSchedule(optimizer, warmup_steps=500, max_steps=10_000)
loss_fn   = VulgarisLoss(config)
pipeline  = TrainingPipeline(model, config, loss_fn, optimizer, scheduler)

for epoch in range(20):
    for x_batch, y_batch in dataloader:
        metrics = pipeline.train_step(x_batch, y_batch, domain_idx=domain_idx)
```

Typical training: **500–2,000 steps** to reach useful performance on a single CPU. No GPU required.

### Phase 3 — Deploy and Stream Forever

```python
# Real-time streaming: one sensor reading at a time
state = model.init_state(batch_size=1)

while plant_is_running:
    sensor_reading = read_sensors()          # shape: (1, 32) — one timestep
    output, state  = model.step(sensor_reading, state)

    print(f"State: {output.prediction}")
    print(f"Confidence: {output.uncertainty}")
    print(f"Reason: {output.explanation}")   # human-readable rule
```

The model **never stops learning**. After deployment it calls `online_adapt()` on every new reading:

```python
pipeline.online_adapt(x_new, y_new, domain_idx=domain_idx)
# Updates Hebbian weights + recalibrates uncertainty — no retraining, no downtime
```

When a new machine is added to the facility — **no retraining of the full model**:

```python
# Register new domain, train only adapter weights (500 steps)
new_domain = model.dah.register_domain("pump_station_B", {...})
model.freeze_base()    # lock all shared weights
pipeline.train_step(x_new_machine, y_new_machine, domain_idx=new_domain)
# Only 50,000 adapter parameters update — not the full model
```

---

## 4. The Full Architecture

VULGARIS processes sensor data through a **fixed residual stack** of 10 specialist modules, each doing one thing exceptionally well:

```
Raw Sensors (B × C_sensors × T_timesteps)
        │
        ▼
┌─────────────────┐
│ RevIN           │  Instance normalisation per channel — removes mean/variance shift
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ ASE             │  Learnable Morlet wavelet filterbank — converts raw signal to
│                 │  time-frequency latent representation
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ HTD             │  4 parallel SSM levels at timescales 0.01s / 0.1s / 1s / 10s
│                 │  Captures fast transients AND slow drifts simultaneously
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ SSSR            │  Mamba-style selective state-space recurrence
│                 │  Linear O(n) complexity — handles 1M+ timestep streams
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Causal          │  Multi-head causal self-attention
│ Attention       │  Captures long-range dependencies within the sequence
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ ICL             │  In-Context Learning — zero-shot adaptation from reference examples
│                 │  No gradient update needed; adapts from examples in context window
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ DAH             │  Domain-Adaptive Hypernetwork — generates domain-specific LoRA
│                 │  adapter weights on the fly. Switch domains at inference time.
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ CRG             │  Causal Routing Graph — learns a DAG (directed acyclic graph)
│                 │  over sensors. Routes information along causal paths, not correlations.
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ HMB             │  Hierarchical Memory Bank — VAE-compressed episodic memory
│                 │  Stores rare events; retrieves similar past situations
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Output Head     │  Final classification or regression projection
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Safety Head     │  Control Barrier Function — certifies output is within safe bounds
│                 │  Projects any unsafe prediction back to the safe set
└─────────────────┘
        │
        ▼
 Prediction + Uncertainty + Explanation + Safety Certificate
```

**Supporting modules** run in parallel during training:

- **ESE** — Explainability Engine: extracts human-readable IF-THEN rules from latent activations
- **SHCAL** — Self-Healing Continual Adaptation: prevents catastrophic forgetting via EWC + structural plasticity
- **CMLA** — Cross-Modal Latent Alignment: fuses data from multiple sensor types with uncertainty weighting
- **Rule Engine** — Neuro-symbolic layer: encodes domain rules as vectors, enforces them during training

---

## 5. Every Module Explained

### ASE — Adaptive Signal Embedding
**What it does:** Converts raw sensor channels into a rich latent representation.

**How:** Instead of a fixed FFT or hand-designed filter, ASE learns a bank of **Morlet wavelets** — filters that simultaneously capture both time and frequency structure. Each wavelet's frequency, width, phase, and amplitude are learned from data. This means the model automatically discovers the relevant frequencies in your sensor (e.g., bearing resonance frequency, grid frequency harmonics) without being told.

**Why it matters:** Different sensors have different physics. A vibration sensor needs different frequency bands than a temperature sensor or a network packet counter. ASE learns the right time-frequency decomposition for each sensor type from data.

---

### HTD — Hierarchical Timescale Decomposition
**What it does:** Processes sensor data simultaneously at four timescales: 10ms, 100ms, 1s, and 10s.

**How:** Four parallel SSM (state-space model) channels, each subsampled at a different rate and coupled by learned bottleneck projections. Slow channels inform fast channels via bias injection; fast channels inform slow channels via learned downsampling.

**Why it matters:** Industrial events happen at multiple scales simultaneously. A bearing failure produces high-frequency vibration bursts (milliseconds), a temperature rise (seconds), and a torque drift (minutes). HTD captures all three with one pass through the model.

---

### SSSR — Selective State-Space Recurrence
**What it does:** The core sequence processing engine. Handles arbitrarily long sensor streams in linear time.

**How:** A multi-head Mamba-style SSM with Zero-Order Hold discretisation. Unlike transformers (O(n²) in sequence length), SSSR is O(n) — it processes a 1-million-timestep sensor stream with the same cost per step as a 100-timestep stream. Gating is input-selective: the model learns which inputs are important enough to update the hidden state.

**Unique feature:** Online Hebbian adaptation — after each forward pass, the decay constants are updated in-place based on activity patterns. States that are stable get reinforced; oscillatory states get damped. The model literally rewires itself as it processes data.

---

### DAH — Domain-Adaptive Hypernetwork
**What it does:** Allows one model to serve unlimited industrial domains simultaneously, switching instantly at inference time.

**How:** A small hypernetwork (domain embedding → meta MLP → weight generators) produces domain-specific LoRA adapter matrices on the fly. When you say `domain_idx=7`, the hypernetwork generates new A and B adapter matrices for every target layer and computes: `output = x @ W_base + (x @ B) @ A × scale`. The base weights are frozen; only the adapters change per domain.

**Why this is revolutionary:** A traditional approach requires one separate model per domain — one for Pump A, one for Pump B, one for the new factory. DAH requires one model. New domains are registered in under a second. The adapter generation costs microseconds.

**OOD Guard:** DAH monitors whether a new domain falls outside the training distribution using Mahalanobis distance on domain embeddings. If a domain is genuinely novel, it warns rather than silently producing unreliable outputs.

---

### CRG — Causal Routing Graph
**What it does:** Learns which sensors actually cause changes in other sensors — not just which ones correlate.

**How:** CRG maintains a weighted adjacency matrix W over all sensor nodes, constrained to represent a Directed Acyclic Graph (DAG) using the NOTEARS penalty. It updates this graph online using Granger causality scores across multiple lags. A conditional independence test prunes edges that are merely correlational rather than causal, labelling remaining edges as "causal" or "associative".

**Why it matters:** Correlation is not causation. In a power plant, when a pump fails, dozens of sensors change together — but only 2–3 are actually in the causal chain. CRG routes information along causal paths only, preventing the model from being confused by spurious correlations. This is what gives VULGARIS its diagnostic accuracy — it knows the *cause* of an anomaly, not just that one occurred.

---

### HMB — Hierarchical Memory Bank
**What it does:** Remembers rare but important events so they can inform future decisions.

**How:** A two-tier episodic memory with a fast working buffer (512 slots) and a slow compressed archive (4096 slots). Events are written when surprise exceeds a threshold (`‖h_actual − h_predicted‖² / variance`). Events are compressed via a VAE to 64 dimensions. Retrieval uses cosine similarity with uncertainty-weighted attention — high-uncertainty memories contribute more to the retrieved context.

**Why it matters:** Industrial rare events (equipment failure modes, abnormal process conditions) may occur only once every few months. Standard neural networks forget them. HMB retains them and uses them to recognise similar patterns when they reappear — even months later. A failure mode seen once is never forgotten.

---

### ESE — Explainability Engine
**What it does:** Extracts human-readable IF-THEN rules from the model's internal decisions.

**How:** Three methods simultaneously:
1. **CART rule extraction** — fits a decision tree to the model's latent activations, producing rules like `IF vibration_rms > 3.5 AND bearing_temp > 85°C THEN state = fault`
2. **Gradient attribution** — ranks which sensors matter most for each prediction, normalised to sum to 100%
3. **Counterfactual generation** — computes the minimal sensor change that would flip the prediction: "If bearing temperature had been 12°C lower, this would not have been flagged"

**Why it matters:** Operators do not trust black boxes. Regulators require explanations. ESE makes VULGARIS auditable and certifiable in regulated industries.

---

### Safety Head — Control Barrier Functions
**What it does:** Provides a mathematical guarantee that model outputs never recommend unsafe actions.

**How:** A learned barrier function h(s) defines the safe set — the region of sensor-space and action-space that is operationally safe. Before any prediction is returned, the output is checked: if the implied action violates `dh/dt + γ·h(s) ≥ 0`, the output is projected back to the safe set via a minimal correction. Safety is enforced by construction, not just by training.

**Why it matters:** In power grids, nuclear plants, and aviation, a wrong recommendation can cause a catastrophic failure. CBF safety provides a correctness guarantee that cannot be violated regardless of model uncertainty. This is the difference between "probably safe" and "certified safe".

---

### SHCAL — Self-Healing Continual Adaptation Layer
**What it does:** Prevents the model from forgetting old knowledge when it learns new things.

**How:** Three mechanisms combined:
1. **EWC (Elastic Weight Consolidation)** — penalises changes to weights that were important for past tasks, weighted by their Fisher information
2. **Hebbian learning** — strengthens connections between co-active neurons: `ΔW = η · (post ⊗ pre − κW)`
3. **Structural plasticity** — permanently prunes inactive connections; grows new connections where gradient signal appears

**Why it matters:** In long-running industrial deployments, conditions change. A model trained on summer operations must remain accurate in winter. SHCAL ensures new learning never destroys old competence.

---

### CMLA — Cross-Modal Latent Alignment
**What it does:** Fuses data from sensors of completely different types into a coherent representation.

**How:** Each sensor modality is normalised and weighted by its inverse uncertainty. Modalities that agree with each other (low variance) are trusted more; modalities that are outlying (high variance) are down-weighted automatically. An InfoNCE contrastive loss during training forces representations of the same event from different sensor types to align in latent space.

**Why it matters:** A pump failure shows up as vibration (accelerometer), heat (temperature), draw (current), and noise (acoustic). CMLA fuses all four modalities into one coherent diagnosis. If one sensor is broken or missing, the others compensate.

---

### Rule Engine — Neuro-Symbolic Integration
**What it does:** Bridges the gap between learned AI and domain expert knowledge.

**How:** Domain experts encode rules as structured objects (`IF sensor_7 > 3.5 THEN alert`). These rules are encoded as 10-dimensional vectors and injected directly into the DAH hypernetwork as an additive signal — making the adapter weights rule-aware from the very first training step. The rule engine has a full lifecycle: rules accumulate confidence from feedback, decay when inactive, are pruned when proven wrong, and merge when redundant. New rules are automatically proposed from CART extractions.

**Why it matters:** In industrial AI, domain experts know things that cannot be learned from data. A metallurgist knows that a particular alloy crystallises at 423°C — there may be no historical fault data to learn this from. The rule engine lets this knowledge be injected directly and maintained automatically.

---

## 6. Industry-Specific Use Cases

### Power Grid & Energy

**Sensors:** Voltage (per phase), current, frequency, power factor, transformer temperature — typically 32–128 channels at 60 Hz.

**What VULGARIS does:**

| Capability | Detail |
|-----------|--------|
| Fault detection | Detects voltage sags, swells, harmonics, and phase imbalance within 1–2 cycles (16–33ms at 60Hz) |
| Frequency deviation | Monitors grid frequency deviation with sub-Hz sensitivity; triggers alerts before protection relays trip |
| Cascade prediction | CRG identifies which substations are causally upstream of others; predicts cascade failure propagation paths |
| Load forecasting | HTD captures both short-term load cycles (hourly) and long-term seasonal drift simultaneously |
| Safe dispatch | CBF safety head certifies that no recommended dispatch action violates N-1 contingency constraints |

**Seed rules example:**
```
IF voltage_pu < 0.85  THEN fault_probable    (ANSI/IEEE undervoltage threshold)
IF freq_dev > 0.5 Hz  THEN alert_system_op   (NERC reliability standard)
IF THD > 0.05         THEN power_quality_event
```

---

### Industrial Process Manufacturing

**Sensors:** Temperature (multiple zones), pressure (upstream/downstream), flow rate, valve position, pH, conductivity — typically 24–64 channels at 0.1–10 Hz.

**What VULGARIS does:**

| Capability | Detail |
|-----------|--------|
| Regime detection | Identifies production regimes (startup, steady-state, CIP, shutdown) automatically |
| Anomaly detection | Detects sensor drift, valve sticking, heat exchanger fouling, and pump cavitation |
| Root cause analysis | CRG causal graph identifies which upstream sensor caused the downstream anomaly |
| Process optimisation | Predicts yield or quality from in-process measurements; suggests setpoint adjustments |
| Changeover prediction | HTD tracks long-term degradation trends; predicts optimal cleaning/maintenance timing |

---

### Telecom & RAN (Radio Access Networks)

**Sensors:** PRB utilisation, SINR, throughput, latency, handover rate, interference — per cell, per frequency band, typically 64–256 channels per site at 1–60s granularity.

**What VULGARIS does:**

| Capability | Detail |
|-----------|--------|
| Congestion prediction | Predicts PRB utilisation 5–15 minutes ahead; enables proactive traffic steering |
| Anomaly detection | Detects hardware faults, interference events, and misconfiguration automatically |
| RAN anomaly root cause | CRG identifies whether a cell's degradation is caused by interference from neighbours or internal hardware |
| Traffic pattern learning | HTD learns daily/weekly traffic patterns; separates routine load from genuine network events |
| Domain per-site | DAH treats each cell as a domain — one model, hundreds of sites, each with its own adapter |

---

### Network Infrastructure (Data Centre / Enterprise)

**Sensors:** Packet loss rate, RTT, jitter, bandwidth utilisation, CPU/memory on network devices, BGP route counts — typically 32–512 metrics at 1s–1min granularity.

**What VULGARIS does:**

| Capability | Detail |
|-----------|--------|
| Traffic classification | Distinguishes normal traffic from DDoS attacks, scanning, and exfiltration patterns |
| Failure prediction | Predicts link failures and device crashes from subtle telemetry drift before outage |
| Capacity planning | Forecasts bandwidth growth trajectories per link per time-of-day pattern |
| SLA monitoring | Conformal prediction provides guaranteed coverage intervals around latency/loss predictions |
| Root cause isolation | CRG identifies which network hop is the causal origin of end-to-end degradation |

---

### Semiconductor Manufacturing

**Sensors:** Chamber pressure, RF power, gas flow rates, etch rate, wafer temperature, endpoint detection signals — typically 50–200 channels per process step at 1–100 Hz.

**What VULGARIS does:**

| Capability | Detail |
|-----------|--------|
| Process drift detection | Detects chamber condition drift (deposition on chamber walls, electrode erosion) before yield impact |
| Fault detection & classification (FDC) | Classifies process excursions by fault type with CART-extracted rules for operator review |
| Virtual metrology | Predicts wafer-level electrical parameters (Vt, Idsat) from in-situ process signals — no physical measurement needed until needed |
| Run-to-run control support | Provides real-time process state estimates for advanced process control (APC) systems |
| Recipe adaptation | DAH allows one model to cover multiple process recipes and chamber configurations |

**Why VULGARIS is especially suited to semiconductor:**
- Data sovereignty — wafer process data cannot leave the fab; VULGARIS runs entirely on-premise
- High-dimensional multivariate — CMLA fuses plasma, thermal, and mechanical signals simultaneously
- Rare fault modes — HMB retains rare process excursion signatures that happen only once per quarter

---

### Predictive Maintenance

**Sensors:** Vibration (triaxial accelerometer), bearing temperature, motor current, oil pressure, acoustic emission — typically 8–20 channels at 10–1000 Hz.

**What VULGARIS does:**

| Capability | Detail |
|-----------|--------|
| RUL estimation | Predicts Remaining Useful Life in operating hours/cycles with conformal uncertainty bounds |
| Fault classification | Classifies fault type: bearing inner race, outer race, ball, gear tooth, imbalance, misalignment |
| Degradation tracking | HTD tracks slow degradation trends (weeks) while SSSR detects sudden events (milliseconds) |
| Maintenance scheduling | Combines RUL prediction with conformal coverage: "95% confidence: failure within 127–189 hours" |
| Fleet learning | DAH enables one model across an entire fleet — each machine is a domain; shared fault knowledge transfers between similar machines |

**Counterfactual diagnostics (powered by ESE):**
> "This bearing is predicted to fail within 50 hours. If vibration RMS were below 2.8g, expected life would extend to 200+ hours. Root cause: outer race defect developing since 2026-04-12."

---

## 7. What Makes It Uniquely Different

### 1. One Model — Infinite Domains

Every competitor requires a separate model per machine, per factory, per use case. VULGARIS uses the **Domain-Adaptive Hypernetwork** to serve unlimited domains from a single model. Adding a new machine takes 500 training steps on a CPU, not a full retraining project.

No other open-source industrial AI framework does this.

### 2. Causal Understanding, Not Just Correlation

Every competing approach (from simple LSTM to TimesFM) predicts outputs from inputs. VULGARIS additionally **learns the causal graph** of the sensor network. It knows that Sensor A causes Sensor B — not just that they move together. This enables root-cause analysis, not just anomaly detection.

### 3. Certified Safe Output

The CBF Safety Head provides a mathematical guarantee on outputs. Competing models can predict anything — including recommendations that would damage equipment or violate safety constraints. VULGARIS certifies that no output violates the defined safety envelope, by construction.

### 4. Neuro-Symbolic Rule Integration

Engineers know things about their machines that cannot be learned from data (regulatory limits, physical constraints, safety interlocks). VULGARIS is the only model that allows this knowledge to be **injected as symbolic rules** that actively shape the learned representations — not as post-hoc filters, but as first-class training signals.

### 5. Never Stops Learning — Never Forgets

SHCAL continual learning means the model updates itself at every inference step via Hebbian plasticity and conformal recalibration, without ever forgetting what it learned before. It handles machine ageing, seasonal drift, and process changes automatically.

### 6. Full Explainability Built In

Every prediction comes with:
- Which sensors contributed most (gradient attribution, %)
- Which rule fired (CART-extracted, human-readable)
- What would need to change for a different prediction (counterfactual)
- How confident the model is (conformal prediction interval, certified coverage)

This is not bolted on — it is part of the core architecture.

### 7. Runs Anywhere — No Cloud Required

Pure NumPy. No CUDA. No Docker GPU support needed. Runs on:
- Raspberry Pi 4
- Industrial edge computers (Advantech, Beckhoff)
- Standard factory servers
- Air-gapped secure environments

Data never leaves the facility. This is the only compliant option for nuclear, defence, and critical infrastructure customers.

### 8. Federated Learning Across Sites

Multiple facilities can train collaboratively without sharing raw data. Differential privacy (DP-SGD) is built in. Byzantine-robust aggregation prevents a compromised site from poisoning the shared model. Each site retains its own domain adapter; only shared knowledge is federated.

---

## 8. How It Compares to Alternatives

### VULGARIS vs Generic Time-Series Foundation Models (TimesFM, MOIRAI, Chronos)

| Factor | TimesFM / MOIRAI / Chronos | VULGARIS |
|--------|---------------------------|---------|
| Training data | Massive public datasets (100B+ points) | Your own sensor data |
| Data privacy | Data goes to Google/AWS/API | Fully on-premise, air-gap capable |
| Domain adaptation | Fine-tune full model (expensive) | DAH adapter swap (500 steps, CPU) |
| Causal understanding | None | CRG learns causal DAG over sensors |
| Safety guarantees | None | CBF certified safe output |
| Explainability | None | CART rules + attribution + counterfactuals |
| Rule injection | None | Full neuro-symbolic rule engine |
| Regulatory compliance | Poor (cloud dependency) | Designed for regulated industries |
| Multivariate sensor fusion | Limited | CMLA inverse-variance fusion |
| Rare event memory | None | HMB episodic memory (4096 slots) |
| Edge deployment | No (cloud API) | Yes — CPU only, <2GB RAM |
| Multi-domain (one model) | No | Yes — DAH hypernetwork |

### VULGARIS vs Traditional Industrial AI (OSIsoft PI, GE Predix, Siemens MindSphere)

| Factor | Traditional Industrial Platforms | VULGARIS |
|--------|----------------------------------|---------|
| Model type | Rule-based + simple regression | Deep neural architecture |
| Causal discovery | Manual by engineers | Automatic from data |
| Adaptation | Manual reconfiguration | Automatic continual learning |
| Multi-domain | Separate deployment per asset | Single model, unlimited domains |
| Cost | $100K–$1M per deployment | Open-source, self-hosted |
| Explainability | Basic rule display | Full causal + attribution + CF |
| Safety certification | Process interlock (external) | Built into model output |

### VULGARIS vs Custom PyTorch Models Built Per Customer

| Factor | Custom PyTorch Per Customer | VULGARIS |
|--------|-----------------------------|---------|
| Time to first model | 3–6 months | Days |
| Maintenance | Per-customer engineering team | One model, configuration only |
| Multi-domain | Rebuild from scratch | Register domain, 500 steps |
| Continual learning | Manual retraining pipeline | Built in |
| Explainability | Usually bolted on or absent | Core architecture |
| Safety | Absent | CBF certified |
| Federated learning | Custom build each time | Built in |

---

## 9. The Complete Technical Stack

### Core Engine
- **Language:** Python 3.10+, pure NumPy
- **Autograd:** Custom reverse-mode automatic differentiation — Tensor, Parameter, Module classes
- **Ops:** Conv1D (grouped, dilated), SSM parallel scan (associative), selective scan step, matrix exponential trace, differentiable top-k

### Model Architecture

| Component | Parameters (default config) | Purpose |
|-----------|----------------------------|---------|
| ASE (8 scales × 16 filters) | ~50K | Time-frequency embedding |
| HTD (4 levels, d=256) | ~200K | Multi-timescale processing |
| SSSR (8 heads, state=256) | ~300K | Linear-complexity recurrence |
| Causal Attention (d=256) | ~250K | Long-range dependencies |
| DAH (32 domains, rank=16) | ~100K | Domain adaptation |
| CRG (64 nodes) | ~30K | Causal graph |
| HMB (512 buffer, 64 embed) | ~150K | Episodic memory |
| ESE + Safety | ~50K | Explainability + safety |
| SHCAL | Masks only | Plasticity control |
| Output Head | ~10K | Final prediction |
| **Total** | **~1.1M parameters** | |

### Training Infrastructure
- **Optimizer:** SpectralAdamW — AdamW with spectral norm clipping; PCGrad gradient surgery for multi-task
- **Schedule:** Cosine schedule with linear warmup; StagedLossSchedule activates loss terms in curriculum waves
- **Augmentation:** Gaussian noise, channel dropout, magnitude scaling, time warp (all configurable)
- **Distributed:** torch.distributed gloo/nccl backend; DDP allreduce; DistributedSampler with Knuth-hash epoch shuffling
- **Conformal:** Non-stationary conformal prediction with exponential forgetting (EnbPI-style)

### Loss Function (8 terms, staged activation)

```
L = L_task                           # supervised signal
  + 0.10 · L_memory                  # HMB VAE reconstruction + KL
  + 0.01 · L_dag                     # CRG NOTEARS DAG penalty + L1 sparsity
  + 0.10 · L_ewc                     # SHCAL catastrophic forgetting penalty
  + 0.05 · L_conformal               # coverage calibration
  + 0.10 · L_cbf                     # safety barrier violation penalty
  + 0.01 · L_temporal                # temporal coherence (smooth hidden states)
  + 0.01 · L_contrastive             # CMLA InfoNCE cross-modal alignment
```

### Serving Infrastructure
- **API:** FastAPI REST server — 14 endpoints covering prediction, streaming, domains, versions, explainability
- **Degradation:** Three-level automatic degradation (FULL → REDUCED → ALERT_ONLY) based on p95 latency and error rate
- **Metrics:** In-process Prometheus-compatible metrics (requests, latency histogram, active sessions)
- **Versioning:** SHA-256 checkpoint integrity, promote/rollback, migration path between versions
- **Auth:** API key authentication via environment variable
- **Logging:** Structured JSON logging with context fields

### Security
- SHA-256 checkpoint integrity verification at load
- EMA Z-score poisoning detection on incoming telemetry (z > 6σ → clamp + flag)
- Malformed packet guard (shape validation + NaN/Inf replacement)
- Out-of-order timestamp detection
- API key gating on all prediction endpoints

### Monitoring
- **Drift detection:** Kolmogorov-Smirnov, MMD (RBF kernel, median heuristic), Wasserstein-1D
- **OOD detection:** Mahalanobis distance on DAH domain embeddings
- **Health stats:** poisoning rate, OOO event rate, per-session latency

### Federated Learning
- Differential privacy: Gaussian mechanism with Rényi DP accounting (Abadi et al.)
- Gradient compression: top-k sparse with error feedback residuals (1% compression ratio default)
- Byzantine detection: median-distance filtering (2σ threshold)
- Robust aggregation: trimmed mean (remove top/bottom 10% by gradient norm)
- FedProx penalty for local update regularisation

---

## 10. Deployment Lifecycle

```
┌─────────────────────────────────────────────────────────┐
│                   CUSTOMER FACILITY                     │
│                                                         │
│  Step 1: Install                                        │
│    pip install vulgaris                                 │
│    docker compose up   (optional — serves REST API)     │
│                                                         │
│  Step 2: Configure                                      │
│    Define sensors, domain, seed rules                   │
│    Register domain in DAH                               │
│                                                         │
│  Step 3: Train on historical data (CPU, 1–7 days)       │
│    500–10,000 steps depending on dataset size           │
│    Checkpoint every epoch; rollback if needed           │
│                                                         │
│  Step 4: Deploy streaming inference                     │
│    model.step(sensor_t, state)   — one call per tick    │
│    Returns: prediction + uncertainty + rule + safety    │
│                                                         │
│  Step 5: Continuous self-update (forever)               │
│    pipeline.online_adapt()       — every inference step │
│    Hebbian update + conformal recalibration             │
│    No downtime, no retraining, no human intervention    │
│                                                         │
│  Step 6: New machine / new factory                      │
│    register_domain() + 500 fine-tune steps              │
│    Base model unchanged; only adapter weights update    │
│                                                         │
│  Step 7: Multiple facilities (federated)                │
│    Sites share gradient updates with DP privacy         │
│    Each site retains its domain adapters                │
│    Byzantine-robust aggregation protects all sites      │
└─────────────────────────────────────────────────────────┘
```

---

## 11. Who Built It and Why

VULGARIS was built to answer a specific question: **why does industrial AI keep failing in production?**

The answer is not model accuracy. Academic benchmarks show excellent numbers. The failures come from:

1. Models that cannot adapt when machines age or processes change
2. Models that cannot explain their decisions to engineers and regulators
3. Models that require expensive cloud infrastructure and send sensitive data offsite
4. Models that require months of rebuilding every time a new machine is added
5. Models that have no safety guarantees and cannot be certified

VULGARIS was designed so that each of these is architecturally impossible to fail:

- **Cannot fail to adapt** — SHCAL + online_adapt() + conformal recalibration run at every step
- **Cannot fail to explain** — ESE is part of the forward pass, not optional
- **Cannot send data offsite** — pure numpy, no cloud dependencies, air-gap capable
- **Cannot require months per new domain** — DAH registers a domain in milliseconds
- **Cannot output unsafe values** — CBF safety head is the final gate before every output

The name VULGARIS is Latin for "common" — the idea being that this level of industrial AI capability should be ordinary and accessible, not the exclusive domain of companies with 100-person ML teams and unlimited cloud budgets.

---

*Full mathematical reference: [ALGORITHMS.md](ALGORITHMS.md)*
*Changelog: [CHANGELOG.md](CHANGELOG.md)*
*Contributing: [CONTRIBUTING.md](CONTRIBUTING.md)*
*License: Apache 2.0*
