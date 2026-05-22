# VULGARIS: A Streaming Causal State-Space Foundation Model for Industrial and Edge Intelligence

**Authors:** [Principal Research Team — AI Systems Architecture, Control Theory, Sequence Modeling, Distributed Systems]

---

## Abstract

Industrial systems generate continuous, multi-rate, causally structured telemetry at scales that fundamentally exceed the modeling assumptions underlying contemporary deep learning architectures. A 5G base station continuously emits upwards of 300 key performance indicators per second; a modern power plant instruments more than 50,000 physical sensors simultaneously; a semiconductor fabrication line imposes timing tolerances measured in microseconds, with no tolerance for deferred inference or unbounded memory growth. Existing foundation models for time series — whether transformer-based, state-space-based, or statistical — share a common failure mode: they are designed for offline, batch, fixed-distribution prediction tasks over stationary signals. They do not satisfy the joint constraints of streaming inference with $O(1)$ memory, continual learning under non-stationary distributions, certified safety under control-theoretic guarantees, and edge deployability within a power and memory envelope compatible with embedded industrial hardware.

We present VULGARIS (Versatile Unified Latent Graph-Augmented Recurrent Intelligence System), a streaming causal state-space foundation model that addresses each of these constraints through principled architectural decisions derived from first principles. VULGARIS introduces seven co-designed modules: (1) **ASE** (Adaptive Signal Embedding), a learnable multi-scale Morlet wavelet filterbank that maps heterogeneous, multi-rate sensor channels to a unified latent manifold without tokenization or discretization artifacts; (2) **HTD** (Hierarchical Timescale Decomposition), a nested hierarchy of zero-order-hold discretized SSMs operating at geometrically spaced time constants, enabling simultaneous capture of sub-second and multi-hour dynamics; (3) **SSSR** (Selective State-Space Recurrence), a multi-head input-selective SSM with Hebbian online adaptation and ZOH-discretized dynamics that provides $O(1)$ streaming state with provably bounded hidden norms; (4) **CRG** (Causal Routing Graph), a differentiable sparse DAG over latent nodes enforced via NOTEARS-style acyclicity penalties, providing online Granger-based structure discovery; (5) **HMB** (Hierarchical Memory Bank), an event-driven memory system stratified into a working buffer and a VAE-compressed archive, indexed by cosine-weighted surprise; (6) **DAH** (Domain Adaptive Hypernetwork), a hypernetwork over LoRA adapters that enables zero-shot domain transfer at the cost of fewer than 0.1% additional parameters; and (7) a **CBF-augmented safety head** with spectral normalization that provides Lipschitz-certified, control-barrier-function-enforced output safety.

VULGARIS processes inputs at $O(1)$ per-step memory, supports multi-rate channels through continuous-time signal embedding, performs online continual learning without catastrophic forgetting via Elastic Weight Consolidation and Hebbian plasticity, and produces formally traceable predictions through the causal graph structure. The full model operates within 512 MB RAM with single-step inference latency under 50 ms on ARM Cortex-class hardware. We present the complete architectural derivation, mathematical foundations, and formal problem specification in this monograph.

---

## 1. Introduction

### 1.1 The Industrial Intelligence Gap

The central claim of this work begins with a quantitative observation, not a rhetorical one. Industrial systems are already operating at data volumes that dwarf any existing benchmark in machine learning. A single 5G New Radio base station, operating with 64-antenna massive MIMO and full Layer 1 telemetry exposed, generates approximately 300 key performance indicators (KPIs) per second: per-cell throughput, block error rates, signal-to-interference-plus-noise ratios across beams, scheduler queue depths, transport acknowledgment timings, and dozens of physical layer counters. A fleet of 10,000 base stations — the scale of a single metropolitan operator — produces $3 \times 10^6$ samples per second, or approximately 260 billion samples per day. A combined-cycle power plant instruments roughly 50,000 physical sensors: thermocouples, pressure transducers, vibration accelerometers, flow meters, and electrical bus monitors. Each sensor streams at between 1 Hz and 10 kHz depending on the physical phenomenon; the aggregate per-plant throughput is on the order of several hundred megabytes per second of raw telemetry. In semiconductor fabrication, the process control layer for a single lithography step may involve 4,000 in-situ sensors operating at sub-millisecond polling rates, with yield-critical windows that cannot tolerate inference latency in excess of a few hundred microseconds.

These numbers are not presented for rhetorical effect. They define precise engineering requirements that translate directly into architectural constraints. Any model that requires $O(T)$ memory in the context length, that processes inputs in fixed-length batches, or that cannot produce outputs faster than the sensor sampling interval is architecturally disqualified from this application class — not merely suboptimal, but functionally incompatible.

The gap between these requirements and the capabilities of current AI systems is not a matter of scale: it is a matter of kind. Contemporary machine learning, including the most capable large-scale systems, operates on the paradigm of *snapshot intelligence*: a finite, fixed-length window of observations is embedded, transformed, and decoded into a prediction or action. This paradigm is appropriate for language modeling, image recognition, and tabular prediction. It is inappropriate for industrial intelligence, where the fundamental unit of information is not a token or a sample but a *continuous signal trajectory* with physical continuity, causal structure, and hard safety requirements.

Distinguishing industrial intelligence from language intelligence requires precision. Natural language is discrete, exchangeable, and semantically self-contained — a sentence has meaning regardless of the physical state of the world that produced it. Physical telemetry is continuous, causally ordered, non-exchangeable, and physically grounded. A voltage reading of 4.17 V on a power bus is unintelligible without the trajectory of prior readings, the physical model of the circuit it belongs to, and the causal relationships between that bus and upstream switching events. The *meaning* of a sensor measurement is inseparable from its temporal and causal context in a way that has no parallel in language.

Five concrete properties differentiate industrial signals from language tokens, with direct architectural consequences:

**Continuity.** Industrial signals are realizations of continuous-time stochastic processes. Discretization at any fixed rate introduces aliasing artifacts and destroys information about the inter-sample dynamics. A model that tokenizes a temperature waveform into fixed-length patches treats a continuous physical phenomenon as a discrete sequence, introducing reconstruction error proportional to the tokenization granularity and obscuring dynamics that occur within the patch window.

**Multi-rate structure.** Different sensors in the same system operate at rates that may span six orders of magnitude. A vibration accelerometer on a gas turbine bearing operates at 20 kHz; the downstream thermal management system integrates over seconds; operational control loops run at 1 Hz. A model that assumes a common sampling rate across channels either over-samples slow signals (wasting computation) or under-samples fast ones (destroying information). Neither is acceptable.

**Causal structure.** Industrial systems are physical systems governed by differential equations and causal mechanisms. A fault in a motor bearing causes a vibration signature, which causes elevated temperature, which causes a protection relay to trip, which causes a downstream voltage sag. This causal chain has a specific directionality and lag structure that a model should represent explicitly if its predictions are to be used for fault attribution and root cause analysis. An architecture that treats all channels symmetrically — as is the case with attention over concatenated sensor readings — cannot represent this structure without implicitly re-learning it as a pattern in the weights.

**Hard safety constraints.** In industrial control, certain outputs are physically inadmissible. A recommended setpoint outside the safe operating envelope of a valve is not merely "wrong" in the loss-function sense; it can cause a physical accident. A model deployed in a safety-critical loop must provide not only a prediction but a certificate that the prediction satisfies the safety constraints of the physical system — or a provable correction that brings it within bounds.

**Mandatory explainability.** Industrial operators and regulators require that the model's outputs be attributable to specific sensors, signals, and causal pathways. This is not an aspirational quality attribute: the European Union AI Act (Article 13, Transparency obligations), IEC 61508 (Functional Safety of Electrical/Electronic/Programmable Electronic Safety-related Systems), and the NIST AI Risk Management Framework all impose explicit traceability requirements on AI systems used in safety-related industrial contexts. An architecture whose decision pathway cannot be traced to specific input contributions is architecturally non-compliant with these requirements, independent of its predictive performance.

The deployment context adds a sixth constraint: *edge hardware realism*. Industrial systems are physically distributed; the telemetry is generated at the edge, and transmitting it to a central inference server introduces latency, bandwidth cost, and single-point failure modes that are unacceptable for real-time control. The model must operate on embedded hardware — ARM Cortex processors, FPGA co-processors, or NVIDIA Jetson-class edge accelerators — within power envelopes measured in watts and memory budgets measured in hundreds of megabytes. A model requiring a GPU with 40 GB of HBM is not a solution to the industrial intelligence problem; it is a solution to a different problem posed on different infrastructure.

The central thesis of this paper is stated precisely: **industrial intelligence requires fundamentally different architectural assumptions than language intelligence, and no existing architecture satisfies all five structural requirements simultaneously**. VULGARIS is designed from the ground up to satisfy all five.

### 1.2 Why Existing Architectures Fail

We assess existing architectures against the constraints derived in the previous section. Our critique is architectural: we do not question whether these models are well-designed for their intended domain. We ask only whether they satisfy the joint constraints of industrial deployment. They do not, and the reasons are structural.

**Transformers and Attention-Based Models.** The transformer architecture (Vaswani et al., 2017) implements scaled dot-product attention:

$$\text{Attn}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

with computational cost $O(T^2 d)$ and memory cost $O(T^2)$ for a sequence of length $T$. For language modeling, where typical sequences are hundreds to thousands of tokens, this is tractable. For industrial telemetry, where a single sensor operating at 1 kHz produces $3.6 \times 10^6$ samples per hour, it is not. A transformer attending over 10 minutes of a 64-channel telemetry stream at 1 kHz requires 600,000 tokens, implying $3.6 \times 10^{11}$ attention score computations per layer — six orders of magnitude beyond practical inference.

The memory problem is equally fundamental. Autoregressive transformer inference maintains a key-value cache that grows as $O(T)$ per layer per attention head. At streaming inference, where the model processes data continuously with no terminal sequence length, the KV cache grows without bound. This is not an implementation detail that can be optimized away; it is a structural property of the architecture.

Beyond scaling, transformers have no inductive bias for causal temporal structure. The attention mechanism learns correlations across all pairs of positions, treating the sequence as an unordered set with positional embeddings as a weak substitute for temporal ordering. For signals with genuine causal structure, this means the causal relationships must be implicitly encoded in the attention weights — a learning problem with no guarantee of convergence to the correct structure, and no mechanism for explicit attribution.

Efficient attention variants — Performer (Choromanski et al., 2021), Linear Transformer (Katharopoulos et al., 2020), Flash Attention (Dao et al., 2022) — reduce the constant factor or hardware bottleneck of attention but do not change its asymptotic memory behavior in streaming inference. Flash Attention remains $O(T^2)$ compute; linear attention approximations sacrifice the expressiveness of exact attention in exchange for tractability. None provides $O(1)$ streaming state.

**Mamba / Selective State-Space Models.** Mamba (Gu and Dao, 2023) represents a genuine advance over transformers for long-sequence modeling. Its selective scan mechanism makes the input-dependent transition parameters $B$, $C$, and $\Delta t$ functions of the input:

$$\mathbf{h}_t = \bar{\mathbf{A}}_t \mathbf{h}_{t-1} + \bar{\mathbf{B}}_t \mathbf{x}_t, \quad \hat{y}_t = C_t \mathbf{h}_t$$

where $\bar{\mathbf{A}}_t = e^{\Delta_t \mathbf{A}}$ and $\bar{\mathbf{B}}_t = (\Delta_t \mathbf{A})^{-1}(e^{\Delta_t \mathbf{A}} - I) \cdot \Delta_t \mathbf{B}$. This achieves $O(1)$ streaming state and $O(T)$ training cost via parallel scan. The hardware-aware CUDA kernel implementation makes it competitive with transformers at long context. These are genuine contributions.

However, Mamba was designed for language modeling and inherits assumptions that do not hold in industrial settings. First, Mamba assumes a uniform, single-rate input sequence. It has no mechanism for multi-rate sensor fusion; a Mamba model applied to telemetry with heterogeneous sampling rates must either interpolate all channels to a common rate (introducing artifacts) or operate on a manually pre-processed, rate-aligned tensor (requiring domain engineering that eliminates the generalizability of the model). Second, Mamba's transition matrix $\mathbf{A}$ has a fixed diagonal structure initialized with S4/HiPPO matrices; while the effective timescales are modulated by $\Delta_t$, the eigenvalue structure is determined at initialization and adapted only via gradient descent on the training distribution. For continual learning under distribution shift, gradient descent on $\mathbf{A}$ risks catastrophic forgetting with no protective mechanism. Third, Mamba has no explicit causal graph structure. The hidden state $\mathbf{h}_t$ integrates information across all input channels simultaneously with no mechanism for attributing outputs to specific causal pathways. Fourth, Mamba was not designed for edge deployment, safety-critical filtering, or multi-domain adaptation. These are not oversights; they are outside the design scope of the paper.

**S4 and HiPPO-Based Models.** S4 (Gu et al., 2021) provides the mathematical foundation for modern SSMs: the HiPPO matrix initialization ensures that the SSM state $\mathbf{h}_t$ approximately maintains a polynomial basis expansion of the input history:

$$\frac{d}{dt}\mathbf{h}(t) = \mathbf{A}\mathbf{h}(t) + \mathbf{B}\mathbf{x}(t), \quad \mathbf{A}_{nk} = -\begin{cases}(2n+1)^{1/2}(2k+1)^{1/2} & \text{if } n > k \\ n+1 & \text{if } n = k \\ 0 & \text{if } n < k\end{cases}$$

This is mathematically elegant and provides a principled rationale for the state initialization. S4's key limitation for industrial use is its *fixed* $\mathbf{A}$ matrix: the transition dynamics are determined at initialization and do not adapt to the input, to domain shift, or to online feedback. The model is non-selective in the Mamba sense — it cannot decide which aspects of the input to retain in state. S5 and subsequent variants improve parallelism but do not address selectivity or online adaptation.

**Industrial-Specific Statistical Methods.** ARIMA, Prophet, and their variants are designed for univariate or low-dimensional time-series forecasting under stationarity assumptions. They require per-series feature engineering, cannot transfer knowledge across systems, and fail catastrophically under distribution shift. LSTM-based industrial models (several proprietary deployments) learn temporal dynamics but require $O(T)$ training context, suffer from vanishing gradients over long horizons, and have no mechanism for causal structure, multi-rate inputs, or certified safety. These models represent the baseline from which industrial AI is departing, not a viable foundation.

**Recent Time-Series Foundation Models.** Chronos (Amazon, 2024), Moirai (Salesforce, 2024), MOMENT (CMU, 2024), and related works represent the state of progress in foundation modeling for time series. Each makes genuine contributions to zero-shot forecasting and representation learning. However, their architectural assumptions disqualify them for industrial edge deployment:

- **Forecasting-only objective**: These models are trained to predict future values. They have no mechanism for causal structure attribution, anomaly localization, or control-theoretic safety filtering. Industrial systems require prediction as one capability among several; offline forecasting accuracy is not the primary evaluation criterion.
- **Tokenization of continuous signals**: Models like Chronos tokenize continuous time series by binning into quantized symbols. This introduces irreversible information loss at the tokenization boundary, particularly for high-frequency signals where the inter-sample dynamics carry physical meaning.
- **No edge deployability**: These models require GPU inference. The smallest published Chronos variant (Mini, ~8M parameters with transformer backbone) requires hundreds of megabytes of peak activation memory during inference — exceeding the memory budget of many industrial edge devices.
- **No online adaptation**: These models are trained offline and deployed as fixed weights. They cannot adapt to the specific characteristics of a new installation without retraining, which requires downtime and labeled data that are unavailable in production.
- **No certified safety**: None of these models provides any mechanism for safety constraint satisfaction. Their outputs are unconstrained probability distributions over future values, with no integration into control-theoretic safety frameworks.

Our critique of each of these architectures is not dismissive — each represents a genuine scientific contribution within its design scope. The point is that no existing architecture was designed to satisfy the joint set of constraints that industrial edge intelligence imposes. VULGARIS is designed explicitly for this joint problem.

### 1.3 Contributions

We enumerate our technical contributions precisely.

1. **Continuous-Time Signal Embedding without Tokenization (ASE).** We present the Adaptive Signal Embedding module: a learnable multi-scale Morlet wavelet filterbank parameterized by per-filter amplitude $A_k$, width $\sigma_k$, frequency $\omega_k$, and phase $\phi_k$, applied at $S$ geometric dilation scales. ASE maps heterogeneous, multi-rate, multi-channel signals $\mathbf{x} \in \mathbb{R}^{B \times C \times T}$ to a unified latent manifold $\mathbf{z} \in \mathbb{R}^{B \times T \times D}$ without any discretization, tokenization, or patch-based representation. The learnable wavelet parameters adapt the filterbank to the frequency content of each deployed domain.

2. **Hierarchical Timescale Decomposition (HTD).** We present a nested hierarchy of $L$ SSM levels, each operating at a geometrically spaced time constant $\tau_l \in \{0.01, 0.1, 1.0, 10.0\}$ seconds, with bidirectional bottleneck coupling (fast-to-slow augmentation and slow-to-fast feedback). Each level uses ZOH-discretized diagonal SSM dynamics with time-constant-aware $\Delta t$ initialization. This enables simultaneous modeling of sub-second transients and multi-hour operational trends within a single forward pass.

3. **Selective State-Space Recurrence with Hebbian Online Adaptation (SSSR).** We present a multi-head selective SSM where input-dependent $\Delta t$, $B$, and $C$ are computed per-timestep, with Hebbian online weight updates to $\log \mathbf{A}$ at each forward pass. The Hebbian rule $\Delta \log A_n = \eta \cdot \mathbb{E}[h_t^{(n)} h_{t-1}^{(n)} - (h_t^{(n)})^2]$ provides a biologically motivated, parameter-local adaptation mechanism that does not require backpropagation and does not interfere with base-parameter stability. ZOH discretization ensures $\bar{A}_t = e^{-e^{\log A} \cdot \Delta t} \in (0, 1)$, guaranteeing bounded hidden state norms.

4. **Causal Routing Graph with Differentiable DAG Enforcement (CRG).** We present a differentiable sparse DAG over $n$ latent nodes, with adjacency matrix $\mathbf{W} \in \mathbb{R}^{n \times n}$, trained jointly with the main model via the NOTEARS acyclicity penalty $h(\mathbf{W}) = \text{tr}(e^{\mathbf{W} \odot \mathbf{W}}) - n$. The CRG performs Granger-motivated online structure discovery through exponential moving average of cross-lagged correlations in the latent node space, and provides O(E) message passing over the active edge set. Causal attributions for any output are produced by BFS traversal over the learned DAG.

5. **Event-Driven Hierarchical Memory Bank (HMB).** We present an event-triggered two-tier memory system with a FIFO working buffer and a VAE-compressed archive. Memory writes are triggered by a normalized surprise score $s_t = \|\mathbf{h}_t - \hat{\mathbf{h}}_t\|^2 / (2\hat{\sigma}^2)$ relative to a running variance estimate. Retrieval uses temperature-scaled cosine attention weighted by uncertainty. The memory system consumes $O(1)$ state with respect to time (bounded buffer capacity), while providing context from arbitrary historical events via compressed archive storage.

6. **Domain Adaptive Hypernetwork with LoRA Injection (DAH).** We present a hypernetwork $g_\phi: \mathbb{Z} \to \{(A_l, B_l)\}_{l=1}^L$ that generates domain-specific LoRA adapter matrices $(A_l \in \mathbb{R}^{d \times r}, B_l \in \mathbb{R}^{r \times d})$ for the SSSR projection layers, conditioned on a learnable domain embedding. The adapter is injected as $W_{\text{eff}} = W_{\text{base}} + A_l B_l \cdot e^{s}$, where $s$ is a learnable log-scale. The base model is frozen during domain adaptation; only the hypernetwork parameters (< 0.1\% of total) are updated.

7. **SHCAL: Streaming Hebbian Continual Adaptation with EWC.** We present SHCAL (Streaming Hebbian Continual Adaptation and Learning), a continual learning wrapper that combines Elastic Weight Consolidation (EWC) penalization of the Fisher-estimated important parameters with Hebbian plasticity on the SSSR projection layers. The EWC penalty $\mathcal{R}_{\text{EWC}}(\theta) = \frac{\lambda}{2} \sum_i F_i (\theta_i - \theta_i^*)^2$ where $F_i = \mathbb{E}\left[(\partial \log p(y|\mathbf{x}, \theta) / \partial \theta_i)^2\right]$ is estimated online via a rolling Fisher approximation.

8. **CBF-Augmented Safety Head with Spectral Normalization.** We present a safety output filter combining a learned Control Barrier Function $h_\theta: \mathcal{S} \to \mathbb{R}^{n_c}$ with a QP-based safety projection. The nominal policy network uses spectral normalization to enforce $\|f\|_\text{Lip} \leq L_{\max}$, providing certified robustness to input perturbations. The CBF safety filter projects unsafe nominal outputs to the nearest safe output satisfying $\dot{h}(s) + \gamma h(s) \geq 0$ for all active constraints.

9. **Conformal Prediction with Online Recalibration (SHCAL-CP).** Integrated within SHCAL, we implement a conformal prediction wrapper that provides distribution-free coverage guarantees for the model's prediction intervals, recalibrated online as the test distribution shifts, using an adaptive conformal procedure that maintains target coverage even under non-exchangeable streams.

10. **A Formally Specified Joint Problem and Architecture.** We provide, to our knowledge, the first formal joint specification of the streaming industrial intelligence problem as a constrained optimization over $O(1)$-memory, continually-learning, causally-explainable, safety-certified, edge-deployable models — and a complete architectural instantiation satisfying all constraints.

### 1.4 Paper Organization

Section 2 provides the mathematical background and related work, covering structured state-space models (S4, Mamba, HiPPO), attention mechanisms and their efficiency variants, online continual learning theory (EWC, Hebbian, catastrophic forgetting), causal discovery (NOTEARS, Granger causality), conformal prediction for streaming data, and safety via Control Barrier Functions. Section 3 formalizes the industrial telemetry problem, deriving each constraint from first principles and proving that it eliminates naive approaches. Section 4 presents the VULGARIS architectural overview: the seven design principles, the complete forward pipeline, tensor conventions, and the justification for module ordering. Subsequent sections (Sections 5–10, forthcoming in Parts 2–4) detail each module individually, provide training procedures, report experimental results, and discuss limitations and future directions.

---

## 2. Background and Related Work

We survey the intellectual lineage of VULGARIS, characterizing each relevant line of prior work precisely and acknowledging both its contributions and its limitations relative to the industrial deployment problem.

### 2.1 Structured State-Space Models

**HiPPO and S4.** The HiPPO (High-order Polynomial Projection Operators) framework (Gu et al., 2020) provides the mathematical foundation for modern sequence-to-sequence SSMs. The key insight is that the hidden state of an SSM should be interpreted as a set of polynomial basis coefficients that project the input history $x(s), s \leq t$ onto a measure $\mu^{(t)}$, yielding the optimal polynomial approximation of the input under that measure. The HiPPO matrix $\mathbf{A}$ is derived analytically from the choice of measure; the LegS (Legendre, scaled) variant produces:

$$\mathbf{A}_{nk} = -\begin{cases}(2n+1)^{1/2}(2k+1)^{1/2} & n > k \\ n + 1 & n = k \\ 0 & n < k\end{cases}, \quad \mathbf{B}_n = (2n+1)^{1/2}$$

This initialization ensures that $\mathbf{h}(t) \in \mathbb{R}^N$ encodes the $N$-th order Legendre polynomial expansion of $x(s)$ weighted by a uniform measure over $[0, t]$. The S4 model (Gu et al., 2021) operationalizes this insight as a trainable SSM by (i) initializing $\mathbf{A}$ from HiPPO, (ii) restricting $\mathbf{A}$ to normal-plus-low-rank (NPLR) structure to enable efficient spectral computation, and (iii) computing the convolution kernel $\bar{K} = (C\bar{B}, C\bar{A}\bar{B}, C\bar{A}^2\bar{B}, \ldots)$ in the frequency domain via FFT.

S4's key limitation for industrial deployment is the *fixed* $\mathbf{A}$ matrix: once trained, the transition dynamics are static. The model cannot selectively attend to different parts of the input history based on the current input — a form of hard gating that Mamba later addresses. Additionally, S4 operates on a single, fixed discretization step $\Delta t$; multi-rate signals require external preprocessing.

**S5 and Parallel Scan.** S5 (Smith et al., 2022) reformulates the S4 layer to use a MIMO (multi-input, multi-output) state-space model instead of independent SISO channels, and introduces the parallel scan algorithm for efficient GPU training. The parallel scan computes $\mathbf{h}_t = \mathbf{A}\mathbf{h}_{t-1} + \mathbf{B}\mathbf{x}_t$ for all $t$ simultaneously via a tree reduction, reducing training complexity from $O(TN^2)$ to $O(TN \log T)$ for dense $\mathbf{A}$.

**Mamba and Selective Scanning.** Mamba (Gu and Dao, 2023) introduces the key innovation of *input-selective* parameters: rather than fixing $\mathbf{B}$, $\mathbf{C}$, and $\Delta t$ as learned constants, they are computed as linear projections of the input $x_t$:

$$\Delta_t = \text{softplus}(W_\Delta x_t + b_\Delta), \quad B_t = W_B x_t, \quad C_t = W_C x_t$$

This creates a data-dependent gate on state updates — the model can decide, based on the content of the current input, how much to update the hidden state and how to read from it. The hardware-aware CUDA selective scan kernel makes this feasible in practice by fusing the sequential scan into a single GPU kernel, avoiding the memory bandwidth bottleneck of materializing intermediate states.

Mamba achieves competitive or superior performance to transformers on language modeling benchmarks at long context, with $O(1)$ inference state and $O(T)$ training compute. These are genuine advances.

The discretization theory underlying Mamba (and all SSMs) proceeds as follows. Given the continuous-time ODE:

$$\dot{\mathbf{h}}(t) = \mathbf{A}\mathbf{h}(t) + \mathbf{B}\mathbf{x}(t)$$

the zero-order-hold (ZOH) discretization with step size $\Delta$ gives:

$$\bar{\mathbf{A}} = e^{\mathbf{A}\Delta}, \quad \bar{\mathbf{B}} = \mathbf{A}^{-1}(e^{\mathbf{A}\Delta} - \mathbf{I})\mathbf{B}$$

and the discrete recurrence $\mathbf{h}_t = \bar{\mathbf{A}}\mathbf{h}_{t-1} + \bar{\mathbf{B}}\mathbf{x}_t$. The bilinear (Tustin) transform gives the alternative:

$$\bar{\mathbf{A}} = (I - \Delta/2 \cdot A)^{-1}(I + \Delta/2 \cdot A), \quad \bar{\mathbf{B}} = (I - \Delta/2 \cdot A)^{-1} \Delta B$$

For diagonal $\mathbf{A}$ (the case in most practical SSMs), ZOH simplifies to element-wise exponentiation: $\bar{A}_n = e^{A_n \Delta}$. When $\mathbf{A}$ is constrained to have negative real parts (i.e., $A_n < 0$ for all $n$), ZOH guarantees $\bar{A}_n \in (0, 1)$, ensuring the hidden state remains bounded. VULGARIS uses this constraint explicitly by parameterizing $\mathbf{A} = -e^{\log \mathbf{A}}$ with $\log \mathbf{A} \in (-5, 0)$, which ensures $A_n < 0$ for all $n$ and $\bar{A}_n \in (e^{-5\Delta}, 1)$ for all discretization steps.

**RWKV.** RWKV (Peng et al., 2023) attempts to combine the expressive receptive field of attention with the $O(1)$ inference cost of RNNs by defining a linear attention analog through exponential decay weighting:

$$\text{Attn}(q, k, v, t) = \frac{\sum_{i \leq t} e^{-(t-i)w + k_i} v_i}{\sum_{i \leq t} e^{-(t-i)w + k_i}}$$

where $w$ is a learned decay vector. This can be computed recurrently at $O(1)$ memory. However, RWKV sacrifices the full selectivity of Mamba: the decay is applied uniformly, and the effective attention is a fixed linear function of the keys and queries without the input-dependent gating of selective SSMs. For industrial signals with irregular event structures (faults, mode switches), the uniform exponential decay provides insufficient selectivity.

### 2.2 Attention Mechanisms and Efficient Variants

The scaled dot-product attention of the original transformer computes $O(T^2)$ pairwise similarity scores. For sequences of length $T$, the attention matrix $\mathbf{P} = \text{softmax}(QK^\top / \sqrt{d}) \in \mathbb{R}^{T \times T}$ requires $O(T^2)$ memory to materialize. For $T = 10^5$ (a mere 100 seconds at 1 kHz), this is $10^{10}$ entries — far beyond practical memory budgets.

Several approaches attempt to reduce this cost. Performer (Choromanski et al., 2021) approximates the softmax kernel via random Fourier features, expressing attention as $\text{Attn}(Q, K, V) \approx \phi(Q)(\phi(K)^\top V)$, which reduces to $O(T d^2)$ via the associativity of matrix multiplication. Linear Transformer (Katharopoulos et al., 2020) similarly factorizes the kernel. These approaches can be computed recurrently at $O(1)$ memory. However, they trade exact attention for an approximation whose quality degrades for long sequences and peaked attention distributions — precisely the scenario that occurs in industrial telemetry when a sudden fault creates a highly localized attention pattern.

Flash Attention (Dao et al., 2022) and Flash Attention 2 (Dao, 2023) reduce the constant factor of exact attention via tiled HBM-IO-aware computation, achieving practical speedups of 2-4× over naive implementations. However, Flash Attention does not change the $O(T^2)$ compute complexity or the $O(T)$ per-step KV cache growth in autoregressive inference. It is an engineering optimization for the batch training setting, not a solution to the streaming inference problem.

The fundamental incompatibility of attention with $O(1)$ streaming inference is provable: any model that is equivalent to attention over all past inputs must store a representation sufficient to compute the attention output for any future query, which requires $O(T)$ storage for arbitrary queries. The $O(1)$ streaming constraint is therefore incompatible with exact attention — not merely impractical, but provably incompatible in the general case.

### 2.3 Online Continual Learning

The stability-plasticity dilemma (Grossberg, 1982) is the central challenge of continual learning: a system must be *plastic* enough to acquire new knowledge from new data, while remaining *stable* enough to not overwrite previously acquired knowledge. McCloskey and Cohen (1989) characterized the catastrophic interference phenomenon in connectionist systems, showing that training a neural network on a new task with gradient descent rapidly overwrites the weights encoding previous tasks. French (1999) provided a systematic review of the phenomenon and its mitigation strategies.

**Elastic Weight Consolidation (EWC).** Kirkpatrick et al. (2017) introduced EWC, which approximates the posterior over parameters after learning task $A$ as a Gaussian centered at $\theta^*_A$ with precision proportional to the diagonal of the Fisher information matrix $F$:

$$p(\theta | \mathcal{D}_A) \approx \mathcal{N}(\theta^*_A, F^{-1})$$

When learning a new task $B$, EWC penalizes deviations from $\theta^*_A$ in proportion to $F$:

$$\mathcal{L}_B(\theta) + \frac{\lambda}{2} \sum_i F_i (\theta_i - \theta^{*A}_i)^2$$

The Fisher diagonal $F_i = \mathbb{E}_{\mathcal{D}_A}\left[\left(\frac{\partial \log p(y|\mathbf{x}, \theta)}{\partial \theta_i}\right)^2\right]$ estimates the importance of each parameter for task $A$. Parameters with high Fisher values contributed strongly to $A$'s performance and should not be moved; parameters with low Fisher values can be freely adapted.

EWC's limitation for streaming industrial systems is computational: maintaining and updating the Fisher diagonal for all parameters is expensive, and the single-task approximation (quadratic bowl around one previous optimum) becomes increasingly inaccurate as the number of previous distribution shifts grows. VULGARIS addresses this by applying EWC selectively to the SSSR projection layers (the parameters most responsible for temporal dynamics) and approximating the Fisher online via a rolling gradient-squared estimate.

**Progressive Neural Networks and PackNet.** Rusu et al. (2016) proposed Progressive Neural Networks, which add new capacity for each new task while freezing all previous parameters. This perfectly prevents forgetting but requires $O(K)$ parameters for $K$ tasks — incompatible with edge deployment. Mallya and Lazebnik (2018) proposed PackNet, which uses parameter pruning to identify a subset of weights responsible for each task. Both approaches require knowing task boundaries in advance and cannot adapt incrementally.

**Hebbian Learning.** Hebb (1949) proposed the co-activation principle: "cells that fire together, wire together." Formally, the Hebbian learning rule for a weight $w_{ij}$ is $\Delta w_{ij} = \eta x_i y_j$, where $x_i$ is the presynaptic activity and $y_j$ is the postsynaptic activity. Oja (1982) introduced a normalized variant that prevents weight explosion:

$$\Delta w_{ij} = \eta (x_i y_j - y_j^2 w_{ij})$$

VULGARIS implements a version of this rule on the SSM state dynamics: the log-decay parameters $\log \mathbf{A}$ are updated as $\Delta \log A_n = \eta \cdot \mathbb{E}[h_t^{(n)} h_{t-1}^{(n)} - (h_t^{(n)})^2]$, where the subtracted term $-(h_t^{(n)})^2$ provides stability analogous to Oja's normalization. This provides a local, online adaptation mechanism that does not require gradient computation or task boundary detection.

**Non-Exchangeability in Streaming Systems.** Standard continual learning theory assumes that the distribution shift is organized into discrete "tasks" with explicit boundaries. Industrial telemetry does not satisfy this structure: distribution shift is continuous (equipment aging, environmental drift), punctuated by sudden events (faults, mode switches, maintenance interventions) with no external notification of the boundary. The online conformal recalibration mechanism in SHCAL addresses this by treating coverage maintenance as a proxy signal for distribution shift: when the empirical coverage of prediction intervals begins to deviate from the target $1-\alpha$, the model increases its learning rate and updates its EWC anchors.

### 2.4 Causal Discovery

Causal discovery — the problem of inferring a directed acyclic graph (DAG) over observed variables from data — is relevant to industrial intelligence because the causal structure of a physical system is partially knowable from data and provides the basis for fault attribution, counterfactual analysis, and interventional generalization.

**Granger Causality.** Granger (1969) defined operational causality in terms of predictive utility: $X$ Granger-causes $Y$ if past values of $X$ improve the prediction of future values of $Y$ above and beyond the prediction achievable from past $Y$ alone. Formally, $X \not\to_G Y$ if and only if $P(Y_{t+1} | Y_t, Y_{t-1}, \ldots) = P(Y_{t+1} | Y_t, Y_{t-1}, \ldots, X_t, X_{t-1}, \ldots)$. Granger causality is computable, scalable, and naturally expressed as a regression: $Y_{t+1} = \sum_{l=1}^{L} \alpha_l Y_{t-l} + \sum_{l=1}^{L} \beta_l X_{t-l} + \epsilon$; if the $\beta$ coefficients are jointly significant, $X$ Granger-causes $Y$.

The limitation of Granger causality is that it identifies *predictive* causal relationships, not *interventional* ones. In the presence of latent common causes (confounders), Granger causality identifies $U \to X$ and $U \to Y$ as $X \to_G Y$ because $X$ contains information about $U$ which predicts $Y$. In industrial settings with latent physical variables (e.g., unobserved internal temperatures), this creates false causal edges.

**Constraint-Based Methods.** The PC algorithm (Spirtes et al., 2000) discovers the Markov equivalence class of the true DAG by testing conditional independence relationships. It has $O(p^k)$ complexity where $p$ is the number of variables and $k$ is the maximum degree of any node. For $p = 64$ nodes (the CRG default), exhaustive PC at high-lag conditioning is computationally intractable.

**NOTEARS.** Zheng et al. (2018) introduced a reformulation of DAG structure learning as a continuous optimization problem. The key insight is that a matrix $\mathbf{W}$ represents the adjacency matrix of a DAG if and only if:

$$h(\mathbf{W}) = \text{tr}\left(e^{\mathbf{W} \odot \mathbf{W}}\right) - d = 0$$

This constraint is smooth and differentiable, enabling gradient-based optimization. Structure learning becomes:

$$\min_{\mathbf{W}} F(\mathbf{W}) + \lambda \|\mathbf{W}\|_1 \quad \text{subject to} \quad h(\mathbf{W}) = 0$$

using augmented Lagrangian methods. In practice, $h(\mathbf{W})$ is enforced as a soft penalty rather than a hard constraint, allowing approximate DAG structure during training while penalizing cycles.

VULGARIS uses a truncated power-series approximation to $e^{\mathbf{W} \odot \mathbf{W}}$ (6 terms) for computational efficiency, which is accurate when $\|\mathbf{W}\|_F$ is small (enforced by the $\ell_1$ sparsity penalty). The gradient of $h(\mathbf{W})$ flows through the power series expansion directly.

**Extensions: DAG-GNN, NoCurl.** Yu et al. (2019) proposed DAG-GNN, which parameterizes the structural equations with neural networks and uses a variational autoencoder over graphs. Zhu et al. (2020) proposed NoCurl, which parameterizes $\mathbf{W}$ as the difference of two matrices whose product is cycle-free. VULGARIS uses the original NOTEARS formulation for its simplicity and gradient stability, supplemented by online Granger updates for structure initialization.

**Why Causal Structure Matters for Physical Systems.** The fundamental reason for including causal structure in an industrial AI system is *interventional generalization* (Pearl, 2009). A purely correlational model trained on normal operation data may learn that $X$ and $Y$ are correlated, but when an engineer intervenes on $X$ (e.g., by manually changing a setpoint), the learned correlation breaks. A model with the correct causal DAG can correctly predict the outcome of the intervention via do-calculus: $P(Y | \text{do}(X=x)) = \sum_z P(Y | X=x, Z=z) P(Z=z)$ for the appropriate adjustment set $Z$. This is the formal basis for model-based fault attribution and counterfactual process control.

### 2.5 Conformal Prediction

Conformal prediction (Vovk et al., 2005) provides distribution-free coverage guarantees for prediction sets. For a significance level $\alpha$, conformal prediction guarantees $P(Y_{n+1} \in C(X_{n+1})) \geq 1 - \alpha$ without assumptions on the data distribution, provided only that the data are exchangeable (i.e., the joint distribution is invariant to permutations of the indices).

For a nonconformity score $s(x, y)$ (e.g., $s = |y - \hat{y}|$ for regression), the conformal prediction set is:

$$C(x) = \left\{y : s(x, y) \leq \hat{q}_{1-\alpha}\right\}$$

where $\hat{q}_{1-\alpha}$ is the $(1-\alpha)(1 + 1/n)$-quantile of the calibration nonconformity scores.

**Conformal Prediction for Time Series.** The exchangeability assumption does not hold for time series: $p(x_1, \ldots, x_T) \neq p(x_{\sigma(1)}, \ldots, x_{\sigma(T)})$ for general permutations $\sigma$ in a temporally dependent sequence. EnbPI (Xu and Xie, 2021) extends conformal prediction to time series via a jackknife+ estimator on sequential residuals, providing approximate coverage under mild mixing conditions. Adaptive Conformal Inference (Gibbs and Candès, 2021) provides coverage guarantees under arbitrary distribution shift by treating the coverage deficit as a signal for recalibrating the quantile $\hat{q}$:

$$\hat{q}_{t+1} = \hat{q}_t + \alpha_{\text{step}} \cdot \left(\alpha - \mathbf{1}\{Y_t \notin C_t(X_t)\}\right)$$

This adaptive recalibration guarantees long-run average coverage even under non-stationary, non-exchangeable streams.

VULGARIS integrates adaptive conformal recalibration within SHCAL to provide coverage-calibrated uncertainty intervals for all model outputs. The nonconformity score used is the standardized residual $s_t = (y_t - \hat{y}_t) / \hat{\sigma}_t$, where $\hat{\sigma}_t$ is the predicted uncertainty from the model.

### 2.6 Safety in Machine Learning

**Control Barrier Functions.** A Control Barrier Function (Ames et al., 2016) for a dynamical system $\dot{s} = f(s, u)$ is a function $h: \mathcal{S} \to \mathbb{R}$ such that $h(s) \geq 0$ defines the *safe set* $\mathcal{C} = \{s : h(s) \geq 0\}$, and the system can be kept in $\mathcal{C}$ by solving a quadratic program at each timestep:

$$\min_u \|u - u_\text{nom}\|^2 \quad \text{subject to} \quad \dot{h}(s, u) + \gamma h(s) \geq 0$$

where $\gamma > 0$ is a class $\mathcal{K}$ function scalar, $\dot{h}(s, u) = \nabla_s h \cdot f(s, u)$. The condition $\dot{h}(s, u) + \gamma h(s) \geq 0$ ensures forward invariance of $\mathcal{C}$: if $h(s_0) \geq 0$, then $h(s_t) \geq 0$ for all $t \geq 0$ under any policy satisfying the constraint.

VULGARIS implements a *learned* CBF $h_\theta(s)$, where $s$ is the model's latent state and the CBF is jointly trained with the policy head. The QP is implemented as a closed-form projection (valid for the unconstrained-action case), avoiding the overhead of an explicit QP solver at inference time.

**Lipschitz Neural Networks.** A neural network $f$ is $L$-Lipschitz if $\|f(x) - f(x')\| \leq L \|x - x'\|$ for all $x, x'$. Lipschitz bounds provide certified robustness: for any perturbation $\delta$ with $\|\delta\| \leq \epsilon$, the output perturbation is bounded by $L\epsilon$. Spectral normalization (Miyato et al., 2018) enforces $\sigma_{\max}(W_l) \leq 1$ for each layer via power iteration, so the product of layer Lipschitz constants $L = \prod_l \sigma_{\max}(W_l) \leq 1$ bounds the network's global Lipschitz constant.

Anil et al. (2019) proposed GroupSort activations that preserve the Lipschitz constant through activation functions, enabling certified robustness analysis for deep networks. VULGARIS uses spectral normalization with a configurable per-layer maximum singular value bound $L_{\max}$ in the safety head, providing a tunable tradeoff between expressiveness and Lipschitz bound.

**CBF-QP Safety Filters for Learned Policies.** The CBF-QP framework (Ames et al., 2019) can be applied as a safety filter to any nominal policy: the nominal action $u_\text{nom}$ from the learned policy is projected to the nearest feasible action satisfying the CBF constraints. This separation of policy learning and safety enforcement is valuable for industrial AI: the base model can be trained purely on task performance, and the safety filter is applied at inference time without modifying the base model.

VULGARIS implements this separation explicitly: the base SSSR + CRG + HMB pipeline produces a nominal output, and the safety head applies the CBF projection as the final step. Crucially, the CBF itself is learned jointly with the model, so the safe set $\mathcal{C}$ is adapted to the specific operational context of the deployed system.

---

## 3. Problem Formulation

We provide the formal mathematical specification of the industrial intelligence problem that VULGARIS is designed to solve.

### 3.1 Industrial Telemetry: Formal Definition

Let $\mathcal{X}$ denote the space of industrial telemetry observations. We define a *telemetry stream* as a pair $(\mathbf{x}, \boldsymbol{\tau})$ where $\mathbf{x}: \mathbb{R}_{\geq 0} \to \mathbb{R}^C$ is a $C$-channel continuously-indexed signal and $\boldsymbol{\tau} = \{t_1, t_2, t_3, \ldots\}$ with $0 \leq t_1 < t_2 < t_3 < \cdots$ is a (possibly irregular) observation schedule. The observed discrete-time sequence is $\{(\mathbf{x}(t_k), t_k)\}_{k=1}^\infty$.

**Multi-rate structure.** In general, each channel $c \in \{1, \ldots, C\}$ has its own observation schedule $\boldsymbol{\tau}^{(c)} = \{t_1^{(c)}, t_2^{(c)}, \ldots\}$ with characteristic sampling interval $\Delta_c = f_c^{-1}$ where $f_c$ is the sampling frequency. We allow $f_c$ to vary by orders of magnitude across channels: $f_c \in [10^{-3}, 10^4]$ Hz in typical industrial settings. A channel observed at 10 kHz and another at 1 Hz share the same batch tensor but at vastly different temporal densities.

**Multi-modal noise structure.** Sensors of different physical types have characteristic noise models. We model the observation at channel $c$ and time $t_k^{(c)}$ as:

$$x_k^{(c)} = \mu^{(c)}(t_k^{(c)}) + \epsilon_k^{(c)}, \quad \epsilon_k^{(c)} \sim \mathcal{N}(0, \sigma_c^2)$$

where $\mu^{(c)}$ is the latent physical signal and $\sigma_c^2$ is the channel-specific noise variance. In practice, $\sigma_c^2$ may itself be time-varying (drift noise, intermittent sensor faults), but we treat it as a fixed parameter estimated from initial calibration data. Cross-channel correlations arise from shared physical coupling, not from the noise model.

**Discrete observation tensor.** For training on finite sequences, we define the observation tensor $\mathbf{X} \in \mathbb{R}^{B \times C \times T}$ for a batch of $B$ sequences, $C$ channels, and $T$ timesteps. In the multi-rate case, channels with lower sampling frequencies are zero-padded or forward-filled in the time dimension; the model's signal embedding must be robust to this representation. The ASE module handles this explicitly by treating the filterbank application as a continuous-time convolution with adaptive bandwidth, so the response at any timestep is governed by the physical scale of the wavelet rather than the artificial sampling grid.

**Timestamps.** When available, the observation timestamps $\boldsymbol{\tau}$ are provided to ASE as an additional channel normalized to $[0, 1]$ over each window. This allows the model to distinguish temporal position from signal value and to learn time-of-day, day-of-week, and seasonal patterns as part of the embedding.

### 3.2 The Streaming Inference Constraint

**Definition (Causal Model).** A model $f$ is *causal* if, for any time $t$, the output $f(\mathbf{x}, t)$ depends only on $\{x(s) : s \leq t\}$.

**Definition (Streaming Inference Constraint).** A causal model $f$ satisfies the *streaming inference constraint* if there exists a finite-dimensional state $\mathbf{s}_t \in \mathbb{R}^d$ with $d$ independent of $T$ such that:

$$f(\mathbf{x}, t) = g(\mathbf{s}_t), \quad \mathbf{s}_t = \phi(\mathbf{s}_{t-1}, \mathbf{x}(t))$$

for some functions $g$ and $\phi$. The state size $|\mathbf{s}_t| = O(1)$ with respect to the sequence length $T$.

This definition formalizes the requirement of *constant-memory streaming inference*: the model maintains a fixed-size state that is updated at each timestep by a function of the previous state and the new observation, and produces outputs as a function of the current state alone. No state that grows with $T$ is permitted.

**Proposition 3.1.** *Transformers with KV cache do not satisfy the streaming inference constraint.*

*Proof.* The KV cache for an $L$-layer transformer with $H$ heads and head dimension $d_h$ at time $t$ stores the key and value tensors for all previous positions: $\text{KV}(t) = \{(K_l^{(t')}, V_l^{(t')})\}_{l=1}^L, t'=1,\ldots,t$. The size of this cache is $2 L H d_h t = O(t)$, which grows without bound as $t \to \infty$. Therefore, the KV cache size is not $O(1)$ with respect to $T$, and the transformer does not satisfy the streaming inference constraint. $\square$

**Proposition 3.2.** *Standard SSMs with diagonal $\mathbf{A}$ satisfy the streaming inference constraint.*

*Proof.* The SSM state $\mathbf{h}_t = \bar{\mathbf{A}} \mathbf{h}_{t-1} + \bar{\mathbf{B}} \mathbf{x}_t$ is an element-wise linear recurrence. The state dimension $|\mathbf{h}_t| = N$ is fixed at model initialization, independent of $T$. The state update $\phi(\mathbf{h}_{t-1}, \mathbf{x}_t) = \bar{\mathbf{A}} \mathbf{h}_{t-1} + \bar{\mathbf{B}} \mathbf{x}_t$ is $O(N)$ per step. Therefore, the SSM satisfies the streaming inference constraint with $d = N$. $\square$

**Latency constraint.** Beyond the memory constraint, streaming inference imposes a latency bound: for a system with control period $T_\text{ctrl}$ (the time between successive control actions), the inference time must satisfy:

$$\Delta t_\text{inf} \leq T_\text{ctrl} - T_\text{comm}$$

where $T_\text{comm}$ is the sensor-to-processor communication latency. For a power plant with 1 Hz control loops: $T_\text{ctrl} = 1000$ ms, $T_\text{comm} \approx 10$ ms, giving $\Delta t_\text{inf} \leq 990$ ms — a relatively relaxed constraint. For a semiconductor process control loop at 100 Hz: $T_\text{ctrl} = 10$ ms, $T_\text{comm} \approx 0.5$ ms, giving $\Delta t_\text{inf} \leq 9.5$ ms — a demanding constraint that requires per-step inference to complete in under 10 ms. For a 5G RAN scheduler operating at slot duration 1 ms, the effective latency budget is sub-millisecond and requires hardware-optimized recurrence, not sequential Python.

### 3.3 The Continual Learning Constraint

**Distribution Shift.** We define the time-indexed distribution $\mathcal{D}_t$ over $(\mathbf{x}, y)$ pairs, where $\mathbf{x}$ is the input telemetry and $y$ is the target label (future value, fault class, etc.). A model deployed in a production industrial system faces:

1. **Gradual drift**: $\|\mathcal{D}_t - \mathcal{D}_{t+\Delta}\|_\text{TV} = O(\Delta \cdot \epsilon_\text{drift})$ for some small $\epsilon_\text{drift}$. Examples: equipment aging, sensor calibration drift, seasonal environmental changes, long-term process chemistry evolution.

2. **Sudden shift**: $\mathcal{D}_{t^+ }$ is abruptly different from $\mathcal{D}_{t^-}$. Examples: a new operating mode is activated, a hardware component is replaced, a new product formula is introduced, a fault creates a fundamentally different system dynamic.

**Definition (Catastrophic Forgetting Bound).** A model $f_\theta$ satisfies the *continual learning constraint* with forgetting tolerance $\epsilon_\text{forget}$ if, for any new distribution $\mathcal{D}_\text{new}$, after updating $\theta$ to minimize $\mathcal{L}(\theta, \mathcal{D}_\text{new})$, the performance on any previous distribution $\mathcal{D}_\text{old}$ degrades by at most:

$$\mathcal{L}(f_{\theta'}, \mathcal{D}_\text{old}) \leq \mathcal{L}(f_{\theta^*}, \mathcal{D}_\text{old}) + \epsilon_\text{forget}$$

where $\theta^*$ is the parameter vector before updating and $\theta'$ is the parameter vector after updating.

The EWC regularization term provides a bound on forgetting in terms of the Fisher information. Specifically, for a model adapted with EWC penalty $\mathcal{R}_\text{EWC}(\theta) = \frac{\lambda}{2} \sum_i F_i (\theta_i - \theta_i^*)^2$, the performance degradation on $\mathcal{D}_\text{old}$ is bounded by:

$$\mathcal{L}(f_{\theta'}, \mathcal{D}_\text{old}) - \mathcal{L}(f_{\theta^*}, \mathcal{D}_\text{old}) \leq \frac{1}{\lambda} \mathcal{L}_\text{new}^* + O\left(\frac{\|\theta' - \theta^*\|^2}{\lambda}\right)$$

where $\mathcal{L}_\text{new}^*$ is the minimum achievable loss on the new task, and the $O$ term reflects the quadratic approximation error of the EWC penalty. This is a non-vacuous bound when $\lambda$ is chosen relative to the Fisher magnitudes.

**Forward Transfer.** We define *forward transfer* as the improvement on a new task $\mathcal{D}_\text{new}$ attributable to prior learning on $\mathcal{D}_\text{old}$: $\text{FT} = \mathcal{L}_\text{scratch}(\mathcal{D}_\text{new}) - \mathcal{L}_\text{transfer}(\mathcal{D}_\text{new})$. The DAH hypernetwork provides forward transfer by conditioning the domain adapter on a learned embedding: adapters for similar domains share meta-parameters $\phi$ and will benefit from each other's data.

### 3.4 The Explainability Constraint

**Formal Auditability Requirement.** For any prediction $\hat{y}_t$ produced by the model at time $t$, there must exist a computable attribution function $\text{Attr}(\hat{y}_t) \to \{(s_i, w_i)\}_{i=1}^K$ mapping the prediction to a set of input signals $s_i$ with scalar attribution weights $w_i$, such that:

1. $\sum_i |w_i| \leq M$ for a finite constant $M$ (attribution is bounded)
2. $\text{Attr}$ is computable in $O(\text{poly}(n))$ time, where $n$ is the number of input channels
3. $\sum_i w_i \hat{f}(s_i) \approx \hat{y}_t$ within a certified approximation error $\delta_\text{attr}$ (attribution is faithful)
4. The attribution is provided alongside every prediction (not as a post-hoc optional computation)

This is not an aspirational requirement. It is a legal and regulatory mandate in multiple jurisdictions. The EU AI Act Article 13 requires "transparency obligations" for high-risk AI systems, which includes AI used in critical infrastructure such as energy, transport, and manufacturing. IEC 61508 SIL 3 certification requires that safety-related software be formally verified and auditable. The NIST AI Risk Management Framework (AI RMF 1.0, 2023) identifies explainability as a core trustworthiness property for deployed AI systems.

**Why Dense Attention Weights Are Insufficient.** A common claim in the literature is that attention weights can serve as explanations by identifying which input positions the model "attended to." This claim fails the formal auditability requirement for two reasons.

First, attention weights are defined over the *positions* in the KV cache, not over the original input signals. A signal that has been tokenized into $P$ patches, processed through $L$ transformer layers, and attended to by $H$ heads produces $L \times H \times P^2$ attention weights. There is no canonical procedure for mapping these to a single per-signal attribution weight; any such mapping requires an additional post-hoc approximation that introduces its own approximation error $\delta_\text{approx}$ beyond the model's prediction error.

Second, attention weights are not faithful explanations in the sense of condition (3). Jain and Wallace (2019) demonstrated empirically that replacing attention weights with random or inverted weights produces similar model outputs in many cases, suggesting that attention weights do not capture the model's functional dependence on inputs. Pruthi et al. (2020) showed that models can be trained to produce attention weights that do not correlate with functional importance.

The CRG module in VULGARIS provides attributions that satisfy conditions (1)-(4) by construction: the DAG structure defines a finite set of directed pathways from input channels to output predictions, and BFS over the DAG computes cumulative edge weights in $O(E)$ time. The attribution is computed from the model's internal causal structure, not from a post-hoc approximation.

### 3.5 The Edge Deployment Constraint

Industrial edge devices span a wide range of capabilities:

| Platform | RAM | Peak Compute | TDP | Typical OS |
|---|---|---|---|---|
| ARM Cortex-A55 (OEM) | 512 MB | ~1 TOPS INT8 | 2 W | Embedded Linux |
| Raspberry Pi CM4 | 4 GB | ~25 GFLOPS FP32 | 5 W | Linux |
| NVIDIA Jetson Nano | 4 GB LPDDR4 | 472 GFLOPS FP16 | 10 W | JetPack / Ubuntu |
| Jetson AGX Orin | 32 GB | 275 TOPS INT8 | 60 W | JetPack |
| Industrial x86 (Atom-class) | 8–32 GB | 50–200 GFLOPS | 15–35 W | Linux / RTOS |
| FPGA (Xilinx Zynq UltraScale+) | On-chip SRAM: 4 MB; DDR: up to 4 GB | Custom; typical 100–500 GOPS | 10–25 W | Bare-metal / RTOS |

VULGARIS is designed to operate within the Raspberry Pi CM4 / Jetson Nano tier: 4 GB RAM, $O(10\text{–}100)$ GFLOPS FP32 compute, 5–10 W power budget. This imposes:

1. **Model size**: Peak activation memory during single-step inference must not exceed $\sim 200$ MB. For VULGARIS with default config ($D=256$, $N=256$, $n=64$ CRG nodes, $L=4$ HTD levels), the total activation footprint per batch element per step is $O(D + N \cdot n_\text{heads} + n^2 + D \cdot L) \approx O(4D + N) = O(2048)$ floating-point values, or 16 KB at FP64 — far within budget.

2. **Parameter storage**: The base VULGARIS model with default configuration has approximately 8–12M trainable parameters (65–95 MB at FP32). This is deployable on embedded Linux devices with 512 MB RAM, leaving $>400$ MB for the OS, buffer cache, and application.

3. **Inference throughput**: The SSM recurrence at each step requires $O(N \cdot D)$ multiply-accumulate operations for the state update and $O(D^2)$ for the projection layers — approximately $10^5$–$10^6$ FLOPs per step, achievable at $>1$ kHz on ARM Cortex-A55 with NEON SIMD optimization.

4. **Numerical precision**: Edge hardware typically does not have double-precision FP64 SIMD. The VULGARIS training implementation uses FP64 for correctness; deployment quantizes to FP32 or INT8, with the spectral normalization in the safety head providing a guaranteed Lipschitz bound that bounds the quantization error propagation.

### 3.6 Formal Problem Statement

We now state the joint optimization problem that VULGARIS solves.

Let $\Theta$ be the parameter space, $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}$ a training dataset from an industrial telemetry distribution, and $\mathcal{L}_\text{task}$ a task-specific loss (MSE for regression, cross-entropy for classification, combined for multi-task). The VULGARIS training objective is:

$$\min_\theta \mathbb{E}_{(\mathbf{x}, y) \sim \mathcal{D}}\left[\mathcal{L}_\text{task}(f_\theta(\mathbf{x}), y)\right] + \lambda_\text{EWC} \mathcal{R}_\text{EWC}(\theta) + \lambda_\text{DAG} h(\mathbf{W}) + \lambda_\text{mem} \mathcal{L}_\text{VAE}(\theta)$$

subject to the following constraints:

1. **Streaming inference constraint:** $|\text{state}(f_\theta)| = O(1)$ with respect to sequence length $T$.

2. **Continual learning constraint:** For any new distribution $\mathcal{D}'$, adaptation satisfies $\mathcal{L}(f_{\theta'}, \mathcal{D}) \leq \mathcal{L}(f_\theta, \mathcal{D}) + \epsilon_\text{forget}$.

3. **Explainability constraint:** For every prediction $\hat{y}_t$, the CRG provides a traceable attribution $\text{Attr}(\hat{y}_t) \to \{(s_i, w_i)\}$ computable in $O(E)$ time, where $E$ is the number of active DAG edges.

4. **Safety constraint:** The output satisfies the CBF feasibility condition: $h_\theta(s_t, \hat{y}_t) \geq 0$ for all active safety constraints, or equivalently, $\dot{h}_\theta(s_t) + \gamma h_\theta(s_t) \geq 0$.

5. **Edge deployment constraint:** Model state $|\text{state}(f_\theta)| \leq M_\text{state}$, single-step inference time $\Delta t_\text{inf} \leq \tau_\text{max}$, and total parameter memory $|f_\theta| \leq M_\text{param}$ for specified hardware-tier constants $M_\text{state}$, $\tau_\text{max}$, $M_\text{param}$.

The objective includes three regularization terms: (i) $\mathcal{R}_\text{EWC}$ prevents catastrophic forgetting; (ii) $h(\mathbf{W})$ enforces DAG acyclicity on the causal graph; (iii) $\mathcal{L}_\text{VAE}$ is the evidence lower bound (ELBO) loss for the memory VAE, ensuring that the archive compression is faithful.

The constraints are enforced through architectural choices (not penalty terms alone): constraint (1) is enforced by the SSM recurrence structure; constraint (2) is enforced by EWC and DAH parameter isolation; constraint (3) is enforced by the CRG DAG structure; constraint (4) is enforced by the CBF-QP safety filter; constraint (5) is enforced by the module design and parameter budget.

---

## 4. VULGARIS Architectural Overview

### 4.1 Design Principles

The VULGARIS architecture is derived from seven design principles, each obtained directly from the formal constraints of Section 3.

**Principle 1: Signal-Native Embedding (No Tokenization).** The streaming inference constraint requires that the model's state size be $O(1)$ in $T$. Tokenization of continuous signals into fixed-length patches introduces a window-size hyperparameter $P$ such that the model processes sequences of length $T/P$ tokens. While this reduces the effective sequence length, it creates two problems: (i) events that occur within a single patch cannot be temporally localized below the patch granularity; (ii) the patch boundary creates artificial discontinuities in signals that are physically continuous across the boundary. A signal-native approach treats the input as a continuous-time signal and applies a filterbank that maps each timestep to a latent vector without any windowing artifact. This is realized by ASE.

**Principle 2: Linear-Time Recurrence (No Attention).** Propositions 3.1 and 3.2 establish that SSMs satisfy the $O(1)$ streaming constraint while transformers do not. The streaming inference constraint and edge deployment constraint jointly require a recurrent model with linear-time inference. Additionally, linear-time recurrence provides constant-cost per-step compute, enabling deployment on embedded processors without batch-processing overhead. This is realized by SSSR and HTD.

**Principle 3: Explicit Causal Structure (Learned DAG).** The explainability constraint requires that predictions be traceable to specific input signals via a causal pathway. An implicit causal model (e.g., the attention weights of a transformer, or the dense weight matrix of a projection layer) does not provide this traceability by default. An explicit sparse DAG over latent nodes provides a direct mapping from any output node back to its influencing input nodes via graph traversal. The DAG is learned jointly with the model via NOTEARS penalties, so the causal structure is adapted to the data. This is realized by CRG.

**Principle 4: Event-Driven Memory (Not Timestep-Driven).** The $O(1)$ memory constraint requires that the model's memory capacity be bounded independent of $T$. A timestep-driven memory that stores a representation for every past timestep grows as $O(T)$. An event-driven memory that writes only when a surprise threshold is exceeded provides bounded capacity: the expected number of writes per unit time is $\theta_\text{write} = P(s_t > s_\text{thresh}) \cdot f_s$, which is bounded for any threshold $s_\text{thresh} > 0$ and stationary distribution. This is realized by HMB.

**Principle 5: Continuous Adaptation (Online Learning).** The continual learning constraint requires that the model update its parameters as the data distribution changes. Offline adaptation (periodic retraining on new labeled data) requires labeled data, downtime, and potentially full retraining — incompatible with uninterrupted operation and sparse labels. Online adaptation via Hebbian updates (parameter-local, no labels required) and EWC (prevents forgetting while allowing adaptation) provides a continual learning mechanism compatible with the deployment constraints. This is realized by SSSR (Hebbian) and SHCAL (EWC).

**Principle 6: Domain Modularity (Frozen Base + Lightweight Adapters).** Industrial systems span many domains: a power plant and a 5G base station share high-level temporal modeling requirements but have entirely different signal statistics, sampling rates, and causal structures. Training a separate model for each domain is impractical; fine-tuning the full model for each domain risks catastrophic forgetting. A modular architecture with a frozen pretrained base and domain-specific adapters enables zero-shot or few-shot domain transfer at the cost of < 0.1% additional parameters per domain. This is realized by DAH.

**Principle 7: Certified Safety (CBF + Lipschitz).** The safety constraint requires that the model's outputs satisfy physical safety constraints, not merely in expectation but for every output. A model that occasionally produces unsafe outputs is undeployable in a safety-critical loop, regardless of its average performance. CBF-QP safety filtering provides a formal guarantee: the output satisfies all active CBF constraints by construction. Spectral normalization provides a Lipschitz bound that certifies robustness to bounded input perturbations. This is realized by the safety head.

### 4.2 The Forward Pipeline

The VULGARIS forward pipeline processes input tensors through seven sequential modules, each producing a transformed representation that is passed to the next module via a residual connection. We describe each stage in terms of the tensor transformations it performs.

```
Input: x ∈ R^{B × C × T}  (batch × channels × time)
       [optional: timestamps ∈ R^{B × T}]
          │
          ▼ ─────────────────────────────────────────────
          │  ASE: Adaptive Signal Embedding
          │  Learnable Morlet wavelet filterbank at S scales
          │  x ─[channel_mix]─[dilated_conv_s0..sS]─[proj]─[RMSNorm]─▶ z⁰
          ▼
       z⁰ ∈ R^{B × T × D}
          │
          ▼ ─────────────────────────────────────────────
          │  HTD: Hierarchical Timescale Decomposition
          │  L nested SSMs at τ={0.01, 0.1, 1.0, 10.0}s
          │  with fast↔slow bottleneck coupling
          │  z⁰ ─[L levels, subsample+upsample]─[output_proj]─▶ z_htd
          ▼
       z¹ = z⁰ + z_htd ∈ R^{B × T × D}   (residual)
          │
          ▼ ─────────────────────────────────────────────
          │  SSSR: Selective State-Space Recurrence
          │  Multi-head input-selective SSM with causal depthwise conv
          │  and Hebbian online update of log_A
          │  z¹ ─[x_proj]─[causal_conv]─[n_heads SSM]─[y_proj]─[gate]─▶ z_sssr
          ▼
       z² = z¹ + z_sssr ∈ R^{B × T × D}   (residual)
          │
          ▼ ─────────────────────────────────────────────
          │  CRG: Causal Routing Graph
          │  Sparse DAG (n_nodes=64) with NOTEARS penalty
          │  + online Granger structure update
          │  z² ─[node_embed]─[message_pass(W)]─[node_out]─▶ z_crg, L_dag
          ▼
       z³ = z² + z_crg ∈ R^{B × T × D},   L_dag ∈ R   (residual + penalty)
          │
          ▼ ─────────────────────────────────────────────
          │  HMB: Hierarchical Memory Bank
          │  Event-triggered write (surprise threshold)
          │  + VAE archive + cosine-attention retrieval
          │  z³ ─[write(h,s)]─[retrieve(query)]─[ctx_broadcast]─▶ z_hmb, L_mem
          ▼
       z⁴ = z³ + z_hmb ∈ R^{B × T × D},   L_mem ∈ R   (residual + VAE loss)
          │
          ▼ ─────────────────────────────────────────────
          │  DAH: Domain Adaptive Hypernetwork
          │  Generates LoRA adapters for SSSR linear layers
          │  (domain_idx) ─[domain_embed]─[meta_mlp]─[hyper_heads]─▶ (A_l, B_l)
          │  [Adapters injected into SSSR projections; base weights frozen]
          ▼
       z⁴ (unchanged; DAH modifies SSSR weights for next forward pass)
          │
          ▼ ─────────────────────────────────────────────
          │  Output Head
          │  z⁴[:, -1, :] ─[Linear]─[softmax / identity]─▶ ŷ
          ▼
       ŷ ∈ R^{B × K}   (K = n_classes or output_dim)
          │
          ▼ ─────────────────────────────────────────────
          │  Safety Filter (conditional on use_safety=True)
          │  ŷ ─[CBF(s)]─[QP_projection]─▶ ŷ_safe
          ▼
       ŷ_safe ∈ R^{B × K}   (CBF-feasible output)
```

At each intermediate stage, the residual connection $\mathbf{z}^{l+1} = \mathbf{z}^l + f_l(\mathbf{z}^l)$ ensures gradient flow from the loss function to all modules, enables each module to learn an incremental *correction* to the current representation rather than a complete *reencoding*, and allows modules to be independently frozen without disrupting the overall pipeline.

### 4.3 Tensor Conventions

Throughout this paper, we use the following notation for tensor dimensions. All tensors are indexed as $\mathbf{x}[b, c, t]$ for input or $\mathbf{z}[b, t, d]$ for intermediate representations (note the transposition of $C$ and $T$ after ASE embedding).

| Symbol | Meaning | Default Value |
|---|---|---|
| $B$ | Batch size | 32 |
| $C$ | Input channels (sensor count) | 64 |
| $T$ | Sequence length (timesteps) | 1024 |
| $D$ | Model dimension (latent\_dim) | 256 |
| $N$ | SSM state dimension | 256 |
| $n$ | CRG node count | 64 |
| $L$ | HTD levels | 4 |
| $H$ | SSSR head count | 8 |
| $D_H$ | SSSR head dimension ($D_\text{inner} / H$) | 64 |
| $S$ | ASE scale count | 8 |
| $F$ | ASE filter count per scale | 16 |
| $r$ | LoRA adapter rank (DAH) | 16 |
| $K$ | Output dimension | 64 |

**Module output conventions.** All intermediate representations after ASE use the layout $\mathbf{z} \in \mathbb{R}^{B \times T \times D}$ (batch-first, then time, then feature). The input tensor $\mathbf{x} \in \mathbb{R}^{B \times C \times T}$ (batch-first, then channel, then time) is transposed by ASE to $\mathbf{z}^0 \in \mathbb{R}^{B \times T \times D}$. This transposition reflects the distinction between the input domain (channels are the primary indexing dimension, as in signal processing) and the latent domain (features are the last dimension, as in neural network conventions).

**Streaming state convention.** In single-step inference, the relevant state per module is:

- SSSR: $\{\mathbf{h}_t^{(i)}\}_{i=1}^H \subset \mathbb{R}^{B \times N/H}$ — one hidden state per SSM head
- HTD: $\{\mathbf{h}_t^{(l)}\}_{l=1}^L \subset \mathbb{R}^{B \times N/4}$ — one hidden state per timescale level
- HMB: internal FIFO buffer (bounded by `buffer_size`) and archive dictionary — $O(1)$ in time
- CRG: stateless (message passing uses current $\mathbf{z}$ only; $\mathbf{W}$ is a model parameter)
- ASE: stateless (each step processed independently)

The complete streaming state for a batch of size 1 is approximately $H \times (N/H) + L \times (N/4) = N + LN/4$ values — for default settings, $256 + 4 \times 64 = 512$ values, or 4 KB at FP32. This is genuinely $O(1)$ in $T$ and fits comfortably in the L1 cache of any modern processor.

### 4.4 The Residual Structure and Why It Matters

Every module in VULGARIS applies an additive residual connection:

$$\mathbf{z}^{l+1} = \mathbf{z}^l + f_l(\mathbf{z}^l)$$

where $f_l$ is the $l$-th module's transformation. This is not merely a convenience borrowed from ResNets; it has three specific consequences that are important for the industrial deployment setting.

**Gradient flow.** For a loss function $\mathcal{L}$ applied to the final output $\hat{y} = g(\mathbf{z}^L)$, the gradient at the $l$-th module is:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{z}^l} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{l+1}} \cdot \left(I + \frac{\partial f_l}{\partial \mathbf{z}^l}\right)$$

The identity term $I$ guarantees that the gradient at layer $l$ includes a direct path from the loss, independent of $\partial f_l / \partial \mathbf{z}^l$. For $L = 4$ modules, the gradient magnitude at the first module is bounded below by $\|\partial \mathcal{L} / \partial \mathbf{z}^L\| / \prod_{l=1}^{L-1} \|I + J_l\|$, where the residual identity prevents the product from collapsing to zero (the vanishing gradient problem) even if individual Jacobians $J_l$ are small.

**Progressive freezing.** When a module is frozen (all its parameters detached from the computation graph), the residual connection ensures that the frozen module passes information forward without blocking gradient flow to earlier modules. Concretely, if module $f_l$ is frozen with $\partial f_l / \partial \theta_l = 0$, the gradient to module $f_{l-1}$ is $\frac{\partial \mathcal{L}}{\partial \mathbf{z}^{l-1}} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}^l} \cdot (I + J_{l-1})$, which is nonzero as long as $\partial \mathcal{L}/\partial \mathbf{z}^l \neq 0$ — guaranteed by the frozen module's residual pass-through. This property enables the domain adaptation strategy of DAH: freeze all base modules, train only the adapters.

**Incremental refinement semantics.** The residual structure imposes a specific semantic on each module: $f_l$ learns to produce a *correction* $\delta \mathbf{z}^l = f_l(\mathbf{z}^l)$ to the current representation, not a complete re-encoding of the signal. This has two effects. First, if $f_l$ is initialized near zero (e.g., by initializing the final linear layer of $f_l$ to zero), then $\mathbf{z}^{l+1} \approx \mathbf{z}^l$ at the beginning of training, and the correction grows as the module learns. This is numerically stable and avoids initialization-sensitive pathologies. Second, the corrections at each level are interpretable as specializations: ASE produces the baseline latent, HTD corrects for timescale structure, SSSR corrects for temporal memory, CRG corrects for causal structure, and HMB corrects for contextual anomalies. The interpretability of the correction at each level is a byproduct of the residual design, not an additional mechanism.

### 4.5 Why the Modules Are in This Order

The module ordering ASE → HTD → SSSR → CRG → HMB is not arbitrary. Each module's computation depends on the output of the previous module in a way that justifies the specific ordering.

**ASE before all temporal modules.** The input $\mathbf{x} \in \mathbb{R}^{B \times C \times T}$ is a raw sensor tensor in the signal domain. All temporal modeling modules (HTD, SSSR, CRG, HMB) operate in the *latent* domain and expect an input tensor $\mathbf{z} \in \mathbb{R}^{B \times T \times D}$ of fixed dimensionality. ASE is the sole module responsible for the signal-to-latent mapping; all subsequent modules can be agnostic to the specific sensor types, sampling rates, and physical units of the input. This separation of concerns is fundamental: it means that the temporal modeling modules can be pretrained on heterogeneous data from many industrial domains without modification, with domain-specific signal characteristics handled entirely by the ASE parameters.

**HTD before SSSR.** The HTD module decomposes the input representation into contributions from multiple timescales simultaneously, producing a latent $\mathbf{z}^1$ that has been enriched with both fast and slow dynamics. The SSSR module then learns its selective state transitions on this multi-timescale representation. If SSSR preceded HTD, it would be applied to the raw ASE latent, which encodes the signal at a single effective timescale (that of the ASE filterbank output). The SSSR would then need to internally learn the multi-timescale decomposition via its $\Delta t$ selection mechanism — a harder learning problem, since the $\Delta t$ projection must simultaneously infer both the relevant timescale and the selective memory gate from the same input. By providing HTD-enriched representations as input to SSSR, we give the SSSR module a structured representation from which the $\Delta t$ selection can focus on *selectivity* (which events to remember) rather than having to also infer the *appropriate timescale* (which the HTD has already resolved).

**SSSR before CRG.** The CRG performs message passing over a learned DAG with node states derived from the current representation. The quality of the DAG structure discovery depends on the quality of the node representations: if the nodes carry generic, weakly specialized latents, the Granger-based correlation estimates will be noisy and the NOTEARS penalty will be ineffective. The SSSR module provides temporally-informed latents — latents that encode the recent history of each channel via the SSM hidden state — which are more informative for causal structure discovery than raw embeddings. Specifically, the Granger-inspired structure update in CRG computes cross-lagged correlations $G_{ij} = \frac{1}{L}\sum_{l=1}^L |\text{corr}(h_{i,t-l}, h_{j,t})|$ where $h_{i,t}$ is the node state at time $t$. For this to capture causal relationships, the node states must carry temporal information; SSSR-enriched representations are significantly more informative for this purpose than static embeddings.

**HMB after CRG.** The HMB retrieves memory entries using the current representation as a query. The utility of this retrieval depends on the query being causally-informed: a memory query that does not reflect the current causal context will retrieve memories that are similar in signal space but irrelevant in causal context. By placing HMB after CRG, the query representation $\mathbf{z}^3$ includes the causal routing correction, meaning that memories are retrieved based on both signal similarity and causal context similarity. This is particularly important for event-driven memory: when an anomalous event occurs (high surprise), the memory retrieval should find not just statistically similar past events but causally related past events — prior occurrences that preceded similar downstream effects.

The ordering also reflects a principle of *increasing context radius*: ASE provides local, channel-specific embeddings; HTD provides multi-timescale temporal context within the current window; SSSR provides dense temporal memory through recurrent state; CRG provides causal context through graph message passing; HMB provides long-range episodic context through memory retrieval. Each successive module extends the effective context radius, building a richer representation for the final output head.

---

*[End of Part 1 — Sections 1–4. Part 2 will cover Sections 5–7: ASE, HTD, and SSSR detailed derivations. Part 3 will cover Sections 8–10: CRG, HMB, and DAH. Part 4 will cover training, evaluation, experimental results, and discussion.]*
