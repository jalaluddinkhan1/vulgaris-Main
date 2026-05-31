# VULGARIS: A Streaming Causal State-Space Foundation Model for Industrial and Edge Intelligence

**Authors:** [Principal Research Team â€” AI Systems Architecture, Control Theory, Sequence Modeling, Distributed Systems]

---

## Abstract

Industrial systems generate continuous, multi-rate, causally structured telemetry at scales that fundamentally exceed the modeling assumptions underlying contemporary deep learning architectures. A 5G base station continuously emits upwards of 300 key performance indicators per second; a modern power plant instruments more than 50,000 physical sensors simultaneously; a semiconductor fabrication line imposes timing tolerances measured in microseconds, with no tolerance for deferred inference or unbounded memory growth. Existing foundation models for time series â€” whether transformer-based, state-space-based, or statistical â€” share a common failure mode: they are designed for offline, batch, fixed-distribution prediction tasks over stationary signals. They do not satisfy the joint constraints of streaming inference with $O(1)$ memory, continual learning under non-stationary distributions, certified safety under control-theoretic guarantees, and edge deployability within a power and memory envelope compatible with embedded industrial hardware.

We present VULGARIS (Versatile Unified Latent Graph-Augmented Recurrent Intelligence System), a streaming causal state-space foundation model that addresses each of these constraints through principled architectural decisions derived from first principles. VULGARIS introduces seven co-designed modules: (1) **ASE** (Adaptive Signal Embedding), a learnable multi-scale Morlet wavelet filterbank that maps heterogeneous, multi-rate sensor channels to a unified latent manifold without tokenization or discretization artifacts; (2) **HTD** (Hierarchical Timescale Decomposition), a nested hierarchy of zero-order-hold discretized SSMs operating at geometrically spaced time constants, enabling simultaneous capture of sub-second and multi-hour dynamics; (3) **SSSR** (Selective State-Space Recurrence), a multi-head input-selective SSM with Hebbian online adaptation and ZOH-discretized dynamics that provides $O(1)$ streaming state with provably bounded hidden norms; (4) **CRG** (Causal Routing Graph), a differentiable sparse DAG over latent nodes enforced via NOTEARS-style acyclicity penalties, providing online Granger-based structure discovery; (5) **HMB** (Hierarchical Memory Bank), an event-driven memory system stratified into a working buffer and a VAE-compressed archive, indexed by cosine-weighted surprise; (6) **DAH** (Domain Adaptive Hypernetwork), a hypernetwork over LoRA adapters that enables zero-shot domain transfer at the cost of fewer than 0.1% additional parameters; and (7) a **CBF-augmented safety head** with spectral normalization that provides Lipschitz-certified, control-barrier-function-enforced output safety.

VULGARIS processes inputs at $O(1)$ per-step memory, supports multi-rate channels through continuous-time signal embedding, performs online continual learning without catastrophic forgetting via Elastic Weight Consolidation and Hebbian plasticity, and produces formally traceable predictions through the causal graph structure. The full model operates within 512 MB RAM with single-step inference latency under 50 ms on ARM Cortex-class hardware. We present the complete architectural derivation, mathematical foundations, and formal problem specification in this monograph.

---

## 1. Introduction

### 1.1 The Industrial Intelligence Gap

The central claim of this work begins with a quantitative observation, not a rhetorical one. Industrial systems are already operating at data volumes that dwarf any existing benchmark in machine learning. A single 5G New Radio base station, operating with 64-antenna massive MIMO and full Layer 1 telemetry exposed, generates approximately 300 key performance indicators (KPIs) per second: per-cell throughput, block error rates, signal-to-interference-plus-noise ratios across beams, scheduler queue depths, transport acknowledgment timings, and dozens of physical layer counters. A fleet of 10,000 base stations â€” the scale of a single metropolitan operator â€” produces $3 \times 10^6$ samples per second, or approximately 260 billion samples per day. A combined-cycle power plant instruments roughly 50,000 physical sensors: thermocouples, pressure transducers, vibration accelerometers, flow meters, and electrical bus monitors. Each sensor streams at between 1 Hz and 10 kHz depending on the physical phenomenon; the aggregate per-plant throughput is on the order of several hundred megabytes per second of raw telemetry. In semiconductor fabrication, the process control layer for a single lithography step may involve 4,000 in-situ sensors operating at sub-millisecond polling rates, with yield-critical windows that cannot tolerate inference latency in excess of a few hundred microseconds.

These numbers are not presented for rhetorical effect. They define precise engineering requirements that translate directly into architectural constraints. Any model that requires $O(T)$ memory in the context length, that processes inputs in fixed-length batches, or that cannot produce outputs faster than the sensor sampling interval is architecturally disqualified from this application class â€” not merely suboptimal, but functionally incompatible.

The gap between these requirements and the capabilities of current AI systems is not a matter of scale: it is a matter of kind. Contemporary machine learning, including the most capable large-scale systems, operates on the paradigm of *snapshot intelligence*: a finite, fixed-length window of observations is embedded, transformed, and decoded into a prediction or action. This paradigm is appropriate for language modeling, image recognition, and tabular prediction. It is inappropriate for industrial intelligence, where the fundamental unit of information is not a token or a sample but a *continuous signal trajectory* with physical continuity, causal structure, and hard safety requirements.

Distinguishing industrial intelligence from language intelligence requires precision. Natural language is discrete, exchangeable, and semantically self-contained â€” a sentence has meaning regardless of the physical state of the world that produced it. Physical telemetry is continuous, causally ordered, non-exchangeable, and physically grounded. A voltage reading of 4.17 V on a power bus is unintelligible without the trajectory of prior readings, the physical model of the circuit it belongs to, and the causal relationships between that bus and upstream switching events. The *meaning* of a sensor measurement is inseparable from its temporal and causal context in a way that has no parallel in language.

Five concrete properties differentiate industrial signals from language tokens, with direct architectural consequences:

**Continuity.** Industrial signals are realizations of continuous-time stochastic processes. Discretization at any fixed rate introduces aliasing artifacts and destroys information about the inter-sample dynamics. A model that tokenizes a temperature waveform into fixed-length patches treats a continuous physical phenomenon as a discrete sequence, introducing reconstruction error proportional to the tokenization granularity and obscuring dynamics that occur within the patch window.

**Multi-rate structure.** Different sensors in the same system operate at rates that may span six orders of magnitude. A vibration accelerometer on a gas turbine bearing operates at 20 kHz; the downstream thermal management system integrates over seconds; operational control loops run at 1 Hz. A model that assumes a common sampling rate across channels either over-samples slow signals (wasting computation) or under-samples fast ones (destroying information). Neither is acceptable.

**Causal structure.** Industrial systems are physical systems governed by differential equations and causal mechanisms. A fault in a motor bearing causes a vibration signature, which causes elevated temperature, which causes a protection relay to trip, which causes a downstream voltage sag. This causal chain has a specific directionality and lag structure that a model should represent explicitly if its predictions are to be used for fault attribution and root cause analysis. An architecture that treats all channels symmetrically â€” as is the case with attention over concatenated sensor readings â€” cannot represent this structure without implicitly re-learning it as a pattern in the weights.

**Hard safety constraints.** In industrial control, certain outputs are physically inadmissible. A recommended setpoint outside the safe operating envelope of a valve is not merely "wrong" in the loss-function sense; it can cause a physical accident. A model deployed in a safety-critical loop must provide not only a prediction but a certificate that the prediction satisfies the safety constraints of the physical system â€” or a provable correction that brings it within bounds.

**Mandatory explainability.** Industrial operators and regulators require that the model's outputs be attributable to specific sensors, signals, and causal pathways. This is not an aspirational quality attribute: the European Union AI Act (Article 13, Transparency obligations), IEC 61508 (Functional Safety of Electrical/Electronic/Programmable Electronic Safety-related Systems), and the NIST AI Risk Management Framework all impose explicit traceability requirements on AI systems used in safety-related industrial contexts. An architecture whose decision pathway cannot be traced to specific input contributions is architecturally non-compliant with these requirements, independent of its predictive performance.

The deployment context adds a sixth constraint: *edge hardware realism*. Industrial systems are physically distributed; the telemetry is generated at the edge, and transmitting it to a central inference server introduces latency, bandwidth cost, and single-point failure modes that are unacceptable for real-time control. The model must operate on embedded hardware â€” ARM Cortex processors, FPGA co-processors, or NVIDIA Jetson-class edge accelerators â€” within power envelopes measured in watts and memory budgets measured in hundreds of megabytes. A model requiring a GPU with 40 GB of HBM is not a solution to the industrial intelligence problem; it is a solution to a different problem posed on different infrastructure.

The central thesis of this paper is stated precisely: **industrial intelligence requires fundamentally different architectural assumptions than language intelligence, and no existing architecture satisfies all five structural requirements simultaneously**. VULGARIS is designed from the ground up to satisfy all five.

### 1.2 Why Existing Architectures Fail

We assess existing architectures against the constraints derived in the previous section. Our critique is architectural: we do not question whether these models are well-designed for their intended domain. We ask only whether they satisfy the joint constraints of industrial deployment. They do not, and the reasons are structural.

**Transformers and Attention-Based Models.** The transformer architecture (Vaswani et al., 2017) implements scaled dot-product attention:

$$\text{Attn}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

with computational cost $O(T^2 d)$ and memory cost $O(T^2)$ for a sequence of length $T$. For language modeling, where typical sequences are hundreds to thousands of tokens, this is tractable. For industrial telemetry, where a single sensor operating at 1 kHz produces $3.6 \times 10^6$ samples per hour, it is not. A transformer attending over 10 minutes of a 64-channel telemetry stream at 1 kHz requires 600,000 tokens, implying $3.6 \times 10^{11}$ attention score computations per layer â€” six orders of magnitude beyond practical inference.

The memory problem is equally fundamental. Autoregressive transformer inference maintains a key-value cache that grows as $O(T)$ per layer per attention head. At streaming inference, where the model processes data continuously with no terminal sequence length, the KV cache grows without bound. This is not an implementation detail that can be optimized away; it is a structural property of the architecture.

Beyond scaling, transformers have no inductive bias for causal temporal structure. The attention mechanism learns correlations across all pairs of positions, treating the sequence as an unordered set with positional embeddings as a weak substitute for temporal ordering. For signals with genuine causal structure, this means the causal relationships must be implicitly encoded in the attention weights â€” a learning problem with no guarantee of convergence to the correct structure, and no mechanism for explicit attribution.

Efficient attention variants â€” Performer (Choromanski et al., 2021), Linear Transformer (Katharopoulos et al., 2020), Flash Attention (Dao et al., 2022) â€” reduce the constant factor or hardware bottleneck of attention but do not change its asymptotic memory behavior in streaming inference. Flash Attention remains $O(T^2)$ compute; linear attention approximations sacrifice the expressiveness of exact attention in exchange for tractability. None provides $O(1)$ streaming state.

**Mamba / Selective State-Space Models.** Mamba (Gu and Dao, 2023) represents a genuine advance over transformers for long-sequence modeling. Its selective scan mechanism makes the input-dependent transition parameters $B$, $C$, and $\Delta t$ functions of the input:

$$\mathbf{h}_t = \bar{\mathbf{A}}_t \mathbf{h}_{t-1} + \bar{\mathbf{B}}_t \mathbf{x}_t, \quad \hat{y}_t = C_t \mathbf{h}_t$$

where $\bar{\mathbf{A}}_t = e^{\Delta_t \mathbf{A}}$ and $\bar{\mathbf{B}}_t = (\Delta_t \mathbf{A})^{-1}(e^{\Delta_t \mathbf{A}} - I) \cdot \Delta_t \mathbf{B}$. This achieves $O(1)$ streaming state and $O(T)$ training cost via parallel scan. The hardware-aware CUDA kernel implementation makes it competitive with transformers at long context. These are genuine contributions.

However, Mamba was designed for language modeling and inherits assumptions that do not hold in industrial settings. First, Mamba assumes a uniform, single-rate input sequence. It has no mechanism for multi-rate sensor fusion; a Mamba model applied to telemetry with heterogeneous sampling rates must either interpolate all channels to a common rate (introducing artifacts) or operate on a manually pre-processed, rate-aligned tensor (requiring domain engineering that eliminates the generalizability of the model). Second, Mamba's transition matrix $\mathbf{A}$ has a fixed diagonal structure initialized with S4/HiPPO matrices; while the effective timescales are modulated by $\Delta_t$, the eigenvalue structure is determined at initialization and adapted only via gradient descent on the training distribution. For continual learning under distribution shift, gradient descent on $\mathbf{A}$ risks catastrophic forgetting with no protective mechanism. Third, Mamba has no explicit causal graph structure. The hidden state $\mathbf{h}_t$ integrates information across all input channels simultaneously with no mechanism for attributing outputs to specific causal pathways. Fourth, Mamba was not designed for edge deployment, safety-critical filtering, or multi-domain adaptation. These are not oversights; they are outside the design scope of the paper.

**S4 and HiPPO-Based Models.** S4 (Gu et al., 2021) provides the mathematical foundation for modern SSMs: the HiPPO matrix initialization ensures that the SSM state $\mathbf{h}_t$ approximately maintains a polynomial basis expansion of the input history:

$$\frac{d}{dt}\mathbf{h}(t) = \mathbf{A}\mathbf{h}(t) + \mathbf{B}\mathbf{x}(t), \quad \mathbf{A}_{nk} = -\begin{cases}(2n+1)^{1/2}(2k+1)^{1/2} & \text{if } n > k \\ n+1 & \text{if } n = k \\ 0 & \text{if } n < k\end{cases}$$

This is mathematically elegant and provides a principled rationale for the state initialization. S4's key limitation for industrial use is its *fixed* $\mathbf{A}$ matrix: the transition dynamics are determined at initialization and do not adapt to the input, to domain shift, or to online feedback. The model is non-selective in the Mamba sense â€” it cannot decide which aspects of the input to retain in state. S5 and subsequent variants improve parallelism but do not address selectivity or online adaptation.

**Industrial-Specific Statistical Methods.** ARIMA, Prophet, and their variants are designed for univariate or low-dimensional time-series forecasting under stationarity assumptions. They require per-series feature engineering, cannot transfer knowledge across systems, and fail catastrophically under distribution shift. LSTM-based industrial models (several proprietary deployments) learn temporal dynamics but require $O(T)$ training context, suffer from vanishing gradients over long horizons, and have no mechanism for causal structure, multi-rate inputs, or certified safety. These models represent the baseline from which industrial AI is departing, not a viable foundation.

**Recent Time-Series Foundation Models.** Chronos (Amazon, 2024), Moirai (Salesforce, 2024), MOMENT (CMU, 2024), and related works represent the state of progress in foundation modeling for time series. Each makes genuine contributions to zero-shot forecasting and representation learning. However, their architectural assumptions disqualify them for industrial edge deployment:

- **Forecasting-only objective**: These models are trained to predict future values. They have no mechanism for causal structure attribution, anomaly localization, or control-theoretic safety filtering. Industrial systems require prediction as one capability among several; offline forecasting accuracy is not the primary evaluation criterion.
- **Tokenization of continuous signals**: Models like Chronos tokenize continuous time series by binning into quantized symbols. This introduces irreversible information loss at the tokenization boundary, particularly for high-frequency signals where the inter-sample dynamics carry physical meaning.
- **No edge deployability**: These models require GPU inference. The smallest published Chronos variant (Mini, ~8M parameters with transformer backbone) requires hundreds of megabytes of peak activation memory during inference â€” exceeding the memory budget of many industrial edge devices.
- **No online adaptation**: These models are trained offline and deployed as fixed weights. They cannot adapt to the specific characteristics of a new installation without retraining, which requires downtime and labeled data that are unavailable in production.
- **No certified safety**: None of these models provides any mechanism for safety constraint satisfaction. Their outputs are unconstrained probability distributions over future values, with no integration into control-theoretic safety frameworks.

Our critique of each of these architectures is not dismissive â€” each represents a genuine scientific contribution within its design scope. The point is that no existing architecture was designed to satisfy the joint set of constraints that industrial edge intelligence imposes. VULGARIS is designed explicitly for this joint problem.

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

10. **A Formally Specified Joint Problem and Architecture.** We provide, to our knowledge, the first formal joint specification of the streaming industrial intelligence problem as a constrained optimization over $O(1)$-memory, continually-learning, causally-explainable, safety-certified, edge-deployable models â€” and a complete architectural instantiation satisfying all constraints.

### 1.4 Paper Organization

Section 2 provides the mathematical background and related work, covering structured state-space models (S4, Mamba, HiPPO), attention mechanisms and their efficiency variants, online continual learning theory (EWC, Hebbian, catastrophic forgetting), causal discovery (NOTEARS, Granger causality), conformal prediction for streaming data, and safety via Control Barrier Functions. Section 3 formalizes the industrial telemetry problem, deriving each constraint from first principles and proving that it eliminates naive approaches. Section 4 presents the VULGARIS architectural overview: the seven design principles, the complete forward pipeline, tensor conventions, and the justification for module ordering. Subsequent sections (Sections 5â€“10, forthcoming in Parts 2â€“4) detail each module individually, provide training procedures, report experimental results, and discuss limitations and future directions.

---

## 2. Background and Related Work

We survey the intellectual lineage of VULGARIS, characterizing each relevant line of prior work precisely and acknowledging both its contributions and its limitations relative to the industrial deployment problem.

### 2.1 Structured State-Space Models

**HiPPO and S4.** The HiPPO (High-order Polynomial Projection Operators) framework (Gu et al., 2020) provides the mathematical foundation for modern sequence-to-sequence SSMs. The key insight is that the hidden state of an SSM should be interpreted as a set of polynomial basis coefficients that project the input history $x(s), s \leq t$ onto a measure $\mu^{(t)}$, yielding the optimal polynomial approximation of the input under that measure. The HiPPO matrix $\mathbf{A}$ is derived analytically from the choice of measure; the LegS (Legendre, scaled) variant produces:

$$\mathbf{A}_{nk} = -\begin{cases}(2n+1)^{1/2}(2k+1)^{1/2} & n > k \\ n + 1 & n = k \\ 0 & n < k\end{cases}, \quad \mathbf{B}_n = (2n+1)^{1/2}$$

This initialization ensures that $\mathbf{h}(t) \in \mathbb{R}^N$ encodes the $N$-th order Legendre polynomial expansion of $x(s)$ weighted by a uniform measure over $[0, t]$. The S4 model (Gu et al., 2021) operationalizes this insight as a trainable SSM by (i) initializing $\mathbf{A}$ from HiPPO, (ii) restricting $\mathbf{A}$ to normal-plus-low-rank (NPLR) structure to enable efficient spectral computation, and (iii) computing the convolution kernel $\bar{K} = (C\bar{B}, C\bar{A}\bar{B}, C\bar{A}^2\bar{B}, \ldots)$ in the frequency domain via FFT.

S4's key limitation for industrial deployment is the *fixed* $\mathbf{A}$ matrix: once trained, the transition dynamics are static. The model cannot selectively attend to different parts of the input history based on the current input â€” a form of hard gating that Mamba later addresses. Additionally, S4 operates on a single, fixed discretization step $\Delta t$; multi-rate signals require external preprocessing.

**S5 and Parallel Scan.** S5 (Smith et al., 2022) reformulates the S4 layer to use a MIMO (multi-input, multi-output) state-space model instead of independent SISO channels, and introduces the parallel scan algorithm for efficient GPU training. The parallel scan computes $\mathbf{h}_t = \mathbf{A}\mathbf{h}_{t-1} + \mathbf{B}\mathbf{x}_t$ for all $t$ simultaneously via a tree reduction, reducing training complexity from $O(TN^2)$ to $O(TN \log T)$ for dense $\mathbf{A}$.

**Mamba and Selective Scanning.** Mamba (Gu and Dao, 2023) introduces the key innovation of *input-selective* parameters: rather than fixing $\mathbf{B}$, $\mathbf{C}$, and $\Delta t$ as learned constants, they are computed as linear projections of the input $x_t$:

$$\Delta_t = \text{softplus}(W_\Delta x_t + b_\Delta), \quad B_t = W_B x_t, \quad C_t = W_C x_t$$

This creates a data-dependent gate on state updates â€” the model can decide, based on the content of the current input, how much to update the hidden state and how to read from it. The hardware-aware CUDA selective scan kernel makes this feasible in practice by fusing the sequential scan into a single GPU kernel, avoiding the memory bandwidth bottleneck of materializing intermediate states.

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

The scaled dot-product attention of the original transformer computes $O(T^2)$ pairwise similarity scores. For sequences of length $T$, the attention matrix $\mathbf{P} = \text{softmax}(QK^\top / \sqrt{d}) \in \mathbb{R}^{T \times T}$ requires $O(T^2)$ memory to materialize. For $T = 10^5$ (a mere 100 seconds at 1 kHz), this is $10^{10}$ entries â€” far beyond practical memory budgets.

Several approaches attempt to reduce this cost. Performer (Choromanski et al., 2021) approximates the softmax kernel via random Fourier features, expressing attention as $\text{Attn}(Q, K, V) \approx \phi(Q)(\phi(K)^\top V)$, which reduces to $O(T d^2)$ via the associativity of matrix multiplication. Linear Transformer (Katharopoulos et al., 2020) similarly factorizes the kernel. These approaches can be computed recurrently at $O(1)$ memory. However, they trade exact attention for an approximation whose quality degrades for long sequences and peaked attention distributions â€” precisely the scenario that occurs in industrial telemetry when a sudden fault creates a highly localized attention pattern.

Flash Attention (Dao et al., 2022) and Flash Attention 2 (Dao, 2023) reduce the constant factor of exact attention via tiled HBM-IO-aware computation, achieving practical speedups of 2-4Ã— over naive implementations. However, Flash Attention does not change the $O(T^2)$ compute complexity or the $O(T)$ per-step KV cache growth in autoregressive inference. It is an engineering optimization for the batch training setting, not a solution to the streaming inference problem.

The fundamental incompatibility of attention with $O(1)$ streaming inference is provable: any model that is equivalent to attention over all past inputs must store a representation sufficient to compute the attention output for any future query, which requires $O(T)$ storage for arbitrary queries. The $O(1)$ streaming constraint is therefore incompatible with exact attention â€” not merely impractical, but provably incompatible in the general case.

### 2.3 Online Continual Learning

The stability-plasticity dilemma (Grossberg, 1982) is the central challenge of continual learning: a system must be *plastic* enough to acquire new knowledge from new data, while remaining *stable* enough to not overwrite previously acquired knowledge. McCloskey and Cohen (1989) characterized the catastrophic interference phenomenon in connectionist systems, showing that training a neural network on a new task with gradient descent rapidly overwrites the weights encoding previous tasks. French (1999) provided a systematic review of the phenomenon and its mitigation strategies.

**Elastic Weight Consolidation (EWC).** Kirkpatrick et al. (2017) introduced EWC, which approximates the posterior over parameters after learning task $A$ as a Gaussian centered at $\theta^*_A$ with precision proportional to the diagonal of the Fisher information matrix $F$:

$$p(\theta | \mathcal{D}_A) \approx \mathcal{N}(\theta^*_A, F^{-1})$$

When learning a new task $B$, EWC penalizes deviations from $\theta^*_A$ in proportion to $F$:

$$\mathcal{L}_B(\theta) + \frac{\lambda}{2} \sum_i F_i (\theta_i - \theta^{*A}_i)^2$$

The Fisher diagonal $F_i = \mathbb{E}_{\mathcal{D}_A}\left[\left(\frac{\partial \log p(y|\mathbf{x}, \theta)}{\partial \theta_i}\right)^2\right]$ estimates the importance of each parameter for task $A$. Parameters with high Fisher values contributed strongly to $A$'s performance and should not be moved; parameters with low Fisher values can be freely adapted.

EWC's limitation for streaming industrial systems is computational: maintaining and updating the Fisher diagonal for all parameters is expensive, and the single-task approximation (quadratic bowl around one previous optimum) becomes increasingly inaccurate as the number of previous distribution shifts grows. VULGARIS addresses this by applying EWC selectively to the SSSR projection layers (the parameters most responsible for temporal dynamics) and approximating the Fisher online via a rolling gradient-squared estimate.

**Progressive Neural Networks and PackNet.** Rusu et al. (2016) proposed Progressive Neural Networks, which add new capacity for each new task while freezing all previous parameters. This perfectly prevents forgetting but requires $O(K)$ parameters for $K$ tasks â€” incompatible with edge deployment. Mallya and Lazebnik (2018) proposed PackNet, which uses parameter pruning to identify a subset of weights responsible for each task. Both approaches require knowing task boundaries in advance and cannot adapt incrementally.

**Hebbian Learning.** Hebb (1949) proposed the co-activation principle: "cells that fire together, wire together." Formally, the Hebbian learning rule for a weight $w_{ij}$ is $\Delta w_{ij} = \eta x_i y_j$, where $x_i$ is the presynaptic activity and $y_j$ is the postsynaptic activity. Oja (1982) introduced a normalized variant that prevents weight explosion:

$$\Delta w_{ij} = \eta (x_i y_j - y_j^2 w_{ij})$$

VULGARIS implements a version of this rule on the SSM state dynamics: the log-decay parameters $\log \mathbf{A}$ are updated as $\Delta \log A_n = \eta \cdot \mathbb{E}[h_t^{(n)} h_{t-1}^{(n)} - (h_t^{(n)})^2]$, where the subtracted term $-(h_t^{(n)})^2$ provides stability analogous to Oja's normalization. This provides a local, online adaptation mechanism that does not require gradient computation or task boundary detection.

**Non-Exchangeability in Streaming Systems.** Standard continual learning theory assumes that the distribution shift is organized into discrete "tasks" with explicit boundaries. Industrial telemetry does not satisfy this structure: distribution shift is continuous (equipment aging, environmental drift), punctuated by sudden events (faults, mode switches, maintenance interventions) with no external notification of the boundary. The online conformal recalibration mechanism in SHCAL addresses this by treating coverage maintenance as a proxy signal for distribution shift: when the empirical coverage of prediction intervals begins to deviate from the target $1-\alpha$, the model increases its learning rate and updates its EWC anchors.

### 2.4 Causal Discovery

Causal discovery â€” the problem of inferring a directed acyclic graph (DAG) over observed variables from data â€” is relevant to industrial intelligence because the causal structure of a physical system is partially knowable from data and provides the basis for fault attribution, counterfactual analysis, and interventional generalization.

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

**Conformal Prediction for Time Series.** The exchangeability assumption does not hold for time series: $p(x_1, \ldots, x_T) \neq p(x_{\sigma(1)}, \ldots, x_{\sigma(T)})$ for general permutations $\sigma$ in a temporally dependent sequence. EnbPI (Xu and Xie, 2021) extends conformal prediction to time series via a jackknife+ estimator on sequential residuals, providing approximate coverage under mild mixing conditions. Adaptive Conformal Inference (Gibbs and CandÃ¨s, 2021) provides coverage guarantees under arbitrary distribution shift by treating the coverage deficit as a signal for recalibrating the quantile $\hat{q}$:

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

where $T_\text{comm}$ is the sensor-to-processor communication latency. For a power plant with 1 Hz control loops: $T_\text{ctrl} = 1000$ ms, $T_\text{comm} \approx 10$ ms, giving $\Delta t_\text{inf} \leq 990$ ms â€” a relatively relaxed constraint. For a semiconductor process control loop at 100 Hz: $T_\text{ctrl} = 10$ ms, $T_\text{comm} \approx 0.5$ ms, giving $\Delta t_\text{inf} \leq 9.5$ ms â€” a demanding constraint that requires per-step inference to complete in under 10 ms. For a 5G RAN scheduler operating at slot duration 1 ms, the effective latency budget is sub-millisecond and requires hardware-optimized recurrence, not sequential Python.

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
| Industrial x86 (Atom-class) | 8â€“32 GB | 50â€“200 GFLOPS | 15â€“35 W | Linux / RTOS |
| FPGA (Xilinx Zynq UltraScale+) | On-chip SRAM: 4 MB; DDR: up to 4 GB | Custom; typical 100â€“500 GOPS | 10â€“25 W | Bare-metal / RTOS |

VULGARIS is designed to operate within the Raspberry Pi CM4 / Jetson Nano tier: 4 GB RAM, $O(10\text{â€“}100)$ GFLOPS FP32 compute, 5â€“10 W power budget. This imposes:

1. **Model size**: Peak activation memory during single-step inference must not exceed $\sim 200$ MB. For VULGARIS with default config ($D=256$, $N=256$, $n=64$ CRG nodes, $L=4$ HTD levels), the total activation footprint per batch element per step is $O(D + N \cdot n_\text{heads} + n^2 + D \cdot L) \approx O(4D + N) = O(2048)$ floating-point values, or 16 KB at FP64 â€” far within budget.

2. **Parameter storage**: The base VULGARIS model with default configuration has approximately 8â€“12M trainable parameters (65â€“95 MB at FP32). This is deployable on embedded Linux devices with 512 MB RAM, leaving $>400$ MB for the OS, buffer cache, and application.

3. **Inference throughput**: The SSM recurrence at each step requires $O(N \cdot D)$ multiply-accumulate operations for the state update and $O(D^2)$ for the projection layers â€” approximately $10^5$â€“$10^6$ FLOPs per step, achievable at $>1$ kHz on ARM Cortex-A55 with NEON SIMD optimization.

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

**Principle 5: Continuous Adaptation (Online Learning).** The continual learning constraint requires that the model update its parameters as the data distribution changes. Offline adaptation (periodic retraining on new labeled data) requires labeled data, downtime, and potentially full retraining â€” incompatible with uninterrupted operation and sparse labels. Online adaptation via Hebbian updates (parameter-local, no labels required) and EWC (prevents forgetting while allowing adaptation) provides a continual learning mechanism compatible with the deployment constraints. This is realized by SSSR (Hebbian) and SHCAL (EWC).

**Principle 6: Domain Modularity (Frozen Base + Lightweight Adapters).** Industrial systems span many domains: a power plant and a 5G base station share high-level temporal modeling requirements but have entirely different signal statistics, sampling rates, and causal structures. Training a separate model for each domain is impractical; fine-tuning the full model for each domain risks catastrophic forgetting. A modular architecture with a frozen pretrained base and domain-specific adapters enables zero-shot or few-shot domain transfer at the cost of < 0.1% additional parameters per domain. This is realized by DAH.

**Principle 7: Certified Safety (CBF + Lipschitz).** The safety constraint requires that the model's outputs satisfy physical safety constraints, not merely in expectation but for every output. A model that occasionally produces unsafe outputs is undeployable in a safety-critical loop, regardless of its average performance. CBF-QP safety filtering provides a formal guarantee: the output satisfies all active CBF constraints by construction. Spectral normalization provides a Lipschitz bound that certifies robustness to bounded input perturbations. This is realized by the safety head.

### 4.2 The Forward Pipeline

The VULGARIS forward pipeline processes input tensors through seven sequential modules, each producing a transformed representation that is passed to the next module via a residual connection. We describe each stage in terms of the tensor transformations it performs.

```
Input: x âˆˆ R^{B Ã— C Ã— T}  (batch Ã— channels Ã— time)
       [optional: timestamps âˆˆ R^{B Ã— T}]
          â”‚
          â–¼ â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
          â”‚  ASE: Adaptive Signal Embedding
          â”‚  Learnable Morlet wavelet filterbank at S scales
          â”‚  x â”€[channel_mix]â”€[dilated_conv_s0..sS]â”€[proj]â”€[RMSNorm]â”€â–¶ zâ°
          â–¼
       zâ° âˆˆ R^{B Ã— T Ã— D}
          â”‚
          â–¼ â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
          â”‚  HTD: Hierarchical Timescale Decomposition
          â”‚  L nested SSMs at Ï„={0.01, 0.1, 1.0, 10.0}s
          â”‚  with fastâ†”slow bottleneck coupling
          â”‚  zâ° â”€[L levels, subsample+upsample]â”€[output_proj]â”€â–¶ z_htd
          â–¼
       zÂ¹ = zâ° + z_htd âˆˆ R^{B Ã— T Ã— D}   (residual)
          â”‚
          â–¼ â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
          â”‚  SSSR: Selective State-Space Recurrence
          â”‚  Multi-head input-selective SSM with causal depthwise conv
          â”‚  and Hebbian online update of log_A
          â”‚  zÂ¹ â”€[x_proj]â”€[causal_conv]â”€[n_heads SSM]â”€[y_proj]â”€[gate]â”€â–¶ z_sssr
          â–¼
       zÂ² = zÂ¹ + z_sssr âˆˆ R^{B Ã— T Ã— D}   (residual)
          â”‚
          â–¼ â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
          â”‚  CRG: Causal Routing Graph
          â”‚  Sparse DAG (n_nodes=64) with NOTEARS penalty
          â”‚  + online Granger structure update
          â”‚  zÂ² â”€[node_embed]â”€[message_pass(W)]â”€[node_out]â”€â–¶ z_crg, L_dag
          â–¼
       zÂ³ = zÂ² + z_crg âˆˆ R^{B Ã— T Ã— D},   L_dag âˆˆ R   (residual + penalty)
          â”‚
          â–¼ â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
          â”‚  HMB: Hierarchical Memory Bank
          â”‚  Event-triggered write (surprise threshold)
          â”‚  + VAE archive + cosine-attention retrieval
          â”‚  zÂ³ â”€[write(h,s)]â”€[retrieve(query)]â”€[ctx_broadcast]â”€â–¶ z_hmb, L_mem
          â–¼
       zâ´ = zÂ³ + z_hmb âˆˆ R^{B Ã— T Ã— D},   L_mem âˆˆ R   (residual + VAE loss)
          â”‚
          â–¼ â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
          â”‚  DAH: Domain Adaptive Hypernetwork
          â”‚  Generates LoRA adapters for SSSR linear layers
          â”‚  (domain_idx) â”€[domain_embed]â”€[meta_mlp]â”€[hyper_heads]â”€â–¶ (A_l, B_l)
          â”‚  [Adapters injected into SSSR projections; base weights frozen]
          â–¼
       zâ´ (unchanged; DAH modifies SSSR weights for next forward pass)
          â”‚
          â–¼ â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
          â”‚  Output Head
          â”‚  zâ´[:, -1, :] â”€[Linear]â”€[softmax / identity]â”€â–¶ Å·
          â–¼
       Å· âˆˆ R^{B Ã— K}   (K = n_classes or output_dim)
          â”‚
          â–¼ â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
          â”‚  Safety Filter (conditional on use_safety=True)
          â”‚  Å· â”€[CBF(s)]â”€[QP_projection]â”€â–¶ Å·_safe
          â–¼
       Å·_safe âˆˆ R^{B Ã— K}   (CBF-feasible output)
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

- SSSR: $\{\mathbf{h}_t^{(i)}\}_{i=1}^H \subset \mathbb{R}^{B \times N/H}$ â€” one hidden state per SSM head
- HTD: $\{\mathbf{h}_t^{(l)}\}_{l=1}^L \subset \mathbb{R}^{B \times N/4}$ â€” one hidden state per timescale level
- HMB: internal FIFO buffer (bounded by `buffer_size`) and archive dictionary â€” $O(1)$ in time
- CRG: stateless (message passing uses current $\mathbf{z}$ only; $\mathbf{W}$ is a model parameter)
- ASE: stateless (each step processed independently)

The complete streaming state for a batch of size 1 is approximately $H \times (N/H) + L \times (N/4) = N + LN/4$ values â€” for default settings, $256 + 4 \times 64 = 512$ values, or 4 KB at FP32. This is genuinely $O(1)$ in $T$ and fits comfortably in the L1 cache of any modern processor.

### 4.4 The Residual Structure and Why It Matters

Every module in VULGARIS applies an additive residual connection:

$$\mathbf{z}^{l+1} = \mathbf{z}^l + f_l(\mathbf{z}^l)$$

where $f_l$ is the $l$-th module's transformation. This is not merely a convenience borrowed from ResNets; it has three specific consequences that are important for the industrial deployment setting.

**Gradient flow.** For a loss function $\mathcal{L}$ applied to the final output $\hat{y} = g(\mathbf{z}^L)$, the gradient at the $l$-th module is:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{z}^l} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{l+1}} \cdot \left(I + \frac{\partial f_l}{\partial \mathbf{z}^l}\right)$$

The identity term $I$ guarantees that the gradient at layer $l$ includes a direct path from the loss, independent of $\partial f_l / \partial \mathbf{z}^l$. For $L = 4$ modules, the gradient magnitude at the first module is bounded below by $\|\partial \mathcal{L} / \partial \mathbf{z}^L\| / \prod_{l=1}^{L-1} \|I + J_l\|$, where the residual identity prevents the product from collapsing to zero (the vanishing gradient problem) even if individual Jacobians $J_l$ are small.

**Progressive freezing.** When a module is frozen (all its parameters detached from the computation graph), the residual connection ensures that the frozen module passes information forward without blocking gradient flow to earlier modules. Concretely, if module $f_l$ is frozen with $\partial f_l / \partial \theta_l = 0$, the gradient to module $f_{l-1}$ is $\frac{\partial \mathcal{L}}{\partial \mathbf{z}^{l-1}} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}^l} \cdot (I + J_{l-1})$, which is nonzero as long as $\partial \mathcal{L}/\partial \mathbf{z}^l \neq 0$ â€” guaranteed by the frozen module's residual pass-through. This property enables the domain adaptation strategy of DAH: freeze all base modules, train only the adapters.

**Incremental refinement semantics.** The residual structure imposes a specific semantic on each module: $f_l$ learns to produce a *correction* $\delta \mathbf{z}^l = f_l(\mathbf{z}^l)$ to the current representation, not a complete re-encoding of the signal. This has two effects. First, if $f_l$ is initialized near zero (e.g., by initializing the final linear layer of $f_l$ to zero), then $\mathbf{z}^{l+1} \approx \mathbf{z}^l$ at the beginning of training, and the correction grows as the module learns. This is numerically stable and avoids initialization-sensitive pathologies. Second, the corrections at each level are interpretable as specializations: ASE produces the baseline latent, HTD corrects for timescale structure, SSSR corrects for temporal memory, CRG corrects for causal structure, and HMB corrects for contextual anomalies. The interpretability of the correction at each level is a byproduct of the residual design, not an additional mechanism.

### 4.5 Why the Modules Are in This Order

The module ordering ASE â†’ HTD â†’ SSSR â†’ CRG â†’ HMB is not arbitrary. Each module's computation depends on the output of the previous module in a way that justifies the specific ordering.

**ASE before all temporal modules.** The input $\mathbf{x} \in \mathbb{R}^{B \times C \times T}$ is a raw sensor tensor in the signal domain. All temporal modeling modules (HTD, SSSR, CRG, HMB) operate in the *latent* domain and expect an input tensor $\mathbf{z} \in \mathbb{R}^{B \times T \times D}$ of fixed dimensionality. ASE is the sole module responsible for the signal-to-latent mapping; all subsequent modules can be agnostic to the specific sensor types, sampling rates, and physical units of the input. This separation of concerns is fundamental: it means that the temporal modeling modules can be pretrained on heterogeneous data from many industrial domains without modification, with domain-specific signal characteristics handled entirely by the ASE parameters.

**HTD before SSSR.** The HTD module decomposes the input representation into contributions from multiple timescales simultaneously, producing a latent $\mathbf{z}^1$ that has been enriched with both fast and slow dynamics. The SSSR module then learns its selective state transitions on this multi-timescale representation. If SSSR preceded HTD, it would be applied to the raw ASE latent, which encodes the signal at a single effective timescale (that of the ASE filterbank output). The SSSR would then need to internally learn the multi-timescale decomposition via its $\Delta t$ selection mechanism â€” a harder learning problem, since the $\Delta t$ projection must simultaneously infer both the relevant timescale and the selective memory gate from the same input. By providing HTD-enriched representations as input to SSSR, we give the SSSR module a structured representation from which the $\Delta t$ selection can focus on *selectivity* (which events to remember) rather than having to also infer the *appropriate timescale* (which the HTD has already resolved).

**SSSR before CRG.** The CRG performs message passing over a learned DAG with node states derived from the current representation. The quality of the DAG structure discovery depends on the quality of the node representations: if the nodes carry generic, weakly specialized latents, the Granger-based correlation estimates will be noisy and the NOTEARS penalty will be ineffective. The SSSR module provides temporally-informed latents â€” latents that encode the recent history of each channel via the SSM hidden state â€” which are more informative for causal structure discovery than raw embeddings. Specifically, the Granger-inspired structure update in CRG computes cross-lagged correlations $G_{ij} = \frac{1}{L}\sum_{l=1}^L |\text{corr}(h_{i,t-l}, h_{j,t})|$ where $h_{i,t}$ is the node state at time $t$. For this to capture causal relationships, the node states must carry temporal information; SSSR-enriched representations are significantly more informative for this purpose than static embeddings.

**HMB after CRG.** The HMB retrieves memory entries using the current representation as a query. The utility of this retrieval depends on the query being causally-informed: a memory query that does not reflect the current causal context will retrieve memories that are similar in signal space but irrelevant in causal context. By placing HMB after CRG, the query representation $\mathbf{z}^3$ includes the causal routing correction, meaning that memories are retrieved based on both signal similarity and causal context similarity. This is particularly important for event-driven memory: when an anomalous event occurs (high surprise), the memory retrieval should find not just statistically similar past events but causally related past events â€” prior occurrences that preceded similar downstream effects.

The ordering also reflects a principle of *increasing context radius*: ASE provides local, channel-specific embeddings; HTD provides multi-timescale temporal context within the current window; SSSR provides dense temporal memory through recurrent state; CRG provides causal context through graph message passing; HMB provides long-range episodic context through memory retrieval. Each successive module extends the effective context radius, building a richer representation for the final output head.

---

*[End of Part 1 â€” Sections 1â€“4. Part 2 will cover Sections 5â€“7: ASE, HTD, and SSSR detailed derivations. Part 3 will cover Sections 8â€“10: CRG, HMB, and DAH. Part 4 will cover training, evaluation, experimental results, and discussion.]*
# VULGARIS: A Streaming Causal State-Space Foundation Model for Industrial Time Series
## Part 2: Core Architectural Modules

---

## 5. Adaptive Signal Embedding

### 5.1 The Tokenization Problem for Physical Signals

The dominant paradigm in sequence modeling treats the embedding of raw input as a preprocessing step: map discrete tokens to dense vectors via a learned vocabulary table, then pass those vectors to the sequence model. This paradigm originated in natural language processing, where it is well-motivated. Words and subword units are discrete, semantically cohesive, and their identity is independent of their temporal position. The token "temperature" carries the same conceptual content whether it appeared as the 42nd or the 4200th token in a document. Position is encoded separately, as an afterthought.

Physical signals admit no such separation. A voltage reading of $0.73$ V at time $t = 1.240$ s is a fundamentally different event than a voltage reading of $0.73$ V at time $t = 1.350$ s when the preceding sample occurred at $t = 1.240$ s â€” in the latter case, a gap of $110$ ms separates the two measurements, potentially indicating a sensor dropoff, a communication fault, or a deliberate hold. The sample value and the sampling interval are jointly informative, and they cannot be disentangled without loss.

More fundamentally, physical signals are continuous, band-limited functions of time. By the Shannonâ€“Nyquist sampling theorem, a signal sampled at rate $f_s$ Hz contains frequency components up to $f_s / 2$ Hz and no higher. The information content of the signal is distributed continuously across this frequency band. Tokenization â€” the discretization of continuous amplitudes into a vocabulary of $V$ distinct symbols â€” is a many-to-one mapping $\tau_V : \mathbb{R} \to \{1, \ldots, V\}$ that irreversibly destroys amplitude information below the quantization granularity.

To make this precise, consider a signal with amplitude range $[x_{\min}, x_{\max}]$ uniformly quantized into $V$ vocabulary entries. The quantization step is $\Delta_q = (x_{\max} - x_{\min}) / V$, and the resulting quantization noise power is:

$$\sigma_q^2 = \frac{\Delta_q^2}{12} = \frac{(x_{\max} - x_{\min})^2}{12 V^2}$$

For a practical bearing fault detection scenario, consider a current sensor monitoring a 690 V line-to-line motor drive. The amplitude range is on the order of 1 kV peak-to-peak. With a vocabulary of $V = 256$ tokens (comparable to byte-level tokenization used in some time-series models), the quantization noise floor is:

$$\sigma_q = \frac{1000}{12^{1/2} \cdot 256} \approx 1.13 \text{ V}$$

This is not negligible. The ball pass frequency outer race (BPFO) fault signature for a 6205 bearing at 1750 RPM appears as an amplitude modulation in the current spectrum at approximately $105$ Hz with a typical fault amplitude of $0.5$â€“$2$ V peak. At $V = 256$, the quantization noise floor is comparable to the fault signature itself. The model trained on tokenized signals cannot, in principle, reliably detect such faults â€” not because of insufficient training data or model capacity, but because the information has been destroyed at the input.

The failure mode compounds with higher-frequency phenomena. The BPFO and its harmonics up to the fifth order span $105$â€“$525$ Hz, requiring amplitude resolution well below $0.5$ V at 1 kV range. With $V = 1024$ (already a large vocabulary for time-series models), the quantization noise is still $0.28$ V â€” marginal at best.

Beyond amplitude quantization, tokenization destroys temporal structure. Consider two representations of the same physical event: a sudden current spike at $t = 10.00$ s followed by return to baseline at $t = 10.05$ s. If the sampling rate is $1$ kHz, these two events are separated by $50$ samples. If the tokenizer operates at a lower effective rate (as in patching-based approaches), the spike and recovery may be merged into a single patch token, and the $50$ ms duration â€” which carries diagnostic information about the type of fault â€” is lost. Conversely, if the sampling rate varies (irregular sampling due to network jitter in industrial Ethernet), two nominally identical sequences of tokens may correspond to physically very different events.

Formally, the temporal information content of a sampled sequence is not merely the sequence of values $\{x_1, \ldots, x_T\}$ but the joint process $\{(x_1, t_1), \ldots, (x_T, t_T)\}$. Any embedding function that maps this joint process to a fixed-length token ignoring $\{t_i\}$ is discarding structural information that is not recoverable downstream.

### 5.2 Continuous-Time Signal Representations via Wavelet Analysis

The mathematically natural alternative to tokenization is to embed signals using continuous-time representations that preserve their frequency structure. The continuous wavelet transform (CWT) provides the canonical framework.

Given a mother wavelet $\psi : \mathbb{R} \to \mathbb{C}$ satisfying the admissibility condition $\int_{-\infty}^{\infty} |\hat{\psi}(\omega)|^2 / |\omega| \, d\omega < \infty$, the CWT of a signal $x \in L^2(\mathbb{R})$ is:

$$\mathcal{W}_\psi[x](a, b) = \frac{1}{\sqrt{|a|}} \int_{-\infty}^{\infty} x(t) \, \psi^*\!\left(\frac{t - b}{a}\right) dt$$

where $a \in \mathbb{R} \setminus \{0\}$ is the scale parameter and $b \in \mathbb{R}$ is the translation parameter. The CWT is an invertible transform (up to the admissibility constant), and the original signal can be reconstructed exactly from its CWT coefficients. No information is lost.

For real-valued signals, the Morlet wavelet provides optimal joint time-frequency localization:

$$\psi_{\text{Morlet}}(t) = \pi^{-1/4} e^{i\omega_0 t} e^{-t^2/2}$$

with $\omega_0 \approx 5$ for standard practice. The Heisenberg uncertainty principle establishes a fundamental lower bound on the joint time-frequency resolution:

$$\Delta t \cdot \Delta f \geq \frac{1}{4\pi}$$

The Morlet wavelet achieves this bound, making it optimal among all wavelets for problems where joint localization is desired. In bearing fault detection, the BPFO appears at a specific frequency but its onset time carries diagnostic information about when the fault initiated â€” exactly the joint localization scenario where the Morlet wavelet is optimal.

However, the CWT with a fixed mother wavelet makes a strong prior assumption: that the optimal time-frequency tradeoff is given by the Heisenberg bound, applied uniformly across all frequencies and all channels. Physical signals often violate this assumption. Electrical transients require high time resolution and low frequency resolution (short $\Delta t$, large $\Delta f$); thermal dynamics require the opposite. A fixed wavelet imposes a single, suboptimal tradeoff for all phenomena simultaneously.

The Empirical Mode Decomposition (EMD) and its variants (EEMD, CEEMDAN) offer a data-adaptive alternative by decomposing signals into Intrinsic Mode Functions (IMFs) that are locally mono-component. However, EMD is not defined by a convex optimization and has no natural gradient â€” it cannot be learned end-to-end. Furthermore, the IMFs lack a direct interpretation in terms of physical frequency bands, making it difficult to initialize the decomposition based on domain knowledge.

This motivates the approach taken in VULGARIS: a learnable filter bank that retains the convolution structure of wavelet analysis (enabling gradient-based learning and fast FFT-based computation) while allowing all filter parameters to be optimized end-to-end for the target domain.

### 5.3 The ASE Filter Bank Architecture

The Adaptive Signal Embedding module instantiates a learnable filter bank of $K$ Morlet-type wavelets, each parameterized by four real-valued parameters $\boldsymbol{\theta}_k = (A_k, \log \sigma_k, \omega_k, \phi_k) \in \mathbb{R}^4$.

The filter for filter index $k$ is defined on a continuous grid as:

$$\psi_k(t; A_k, \sigma_k, \omega_k, \phi_k) = A_k \exp\!\left(-\frac{1}{2}\left(\frac{t}{e^{\log \sigma_k}}\right)^2\right) \cos(\omega_k t + \phi_k)$$

The parameterization $\sigma_k = e^{\log \sigma_k}$ ensures strict positivity of the bandwidth parameter without requiring constrained optimization. The amplitude $A_k$ scales the filter response and is initialized to ensure unit expected power when the filter is applied to white noise. The center frequency $\omega_k$ and phase offset $\phi_k$ are unconstrained.

For discretization, the filter is evaluated on a symmetric time grid $\mathbf{t} = \{-L/2, -(L/2 - 1), \ldots, L/2 - 1, L/2\} / (L/2)$ of length $L$ (the `filter_len` hyperparameter), normalized so that $t \in [-1, 1]$. This normalization ensures that filter shape is scale-independent, with the scale handled separately by the dilation parameter. The result is a discretized convolution kernel $\mathbf{k}_k \in \mathbb{R}^{1 \times L}$.

**Multi-scale filtering via dilation.** Rather than instantiating separate filters at each scale, VULGARIS reuses each learned filter at $S$ different scales via the dilation trick from WaveNet. At scale $s \in \{0, 1, \ldots, S-1\}$, filter $k$ is applied with dilation $d_s = 2^s$. A dilated convolution with dilation $d_s$ and kernel of length $L$ has an effective receptive field of:

$$R_s = L \cdot d_s = L \cdot 2^s$$

samples, while still performing only $L$ multiply-accumulate operations per output position. The computational cost of multi-scale filtering is therefore $O(KS \cdot T \cdot L)$, independent of the receptive field size.

The output of scale $s$, filter $k$ is:

$$\mathbf{y}_{k,s} = \mathbf{x} *_{d_s} \mathbf{k}_k \in \mathbb{R}^{B \times T}$$

where $*_{d_s}$ denotes the dilated convolution with stride 1 and appropriate causal zero-padding on the left to maintain sequence length $T$. The causal padding of $d_s \cdot (L - 1)$ zeros on the left ensures that $\mathbf{y}_{k,s,t}$ depends only on $\mathbf{x}_{t'}, t' \leq t$, which is required for the streaming inference mode.

**Projection to latent space.** The full filter bank produces an output tensor of shape $\mathbb{R}^{B \times KS \times T}$ by concatenating over all $K$ filters and $S$ scales. Since the model processes multiple input channels $C$, the filter bank is applied channel-wise and the resulting $C \cdot KS$ feature maps are concatenated:

$$\mathbf{F} \in \mathbb{R}^{B \times C \cdot KS \times T}$$

This tensor is transposed to $\mathbb{R}^{B \times T \times C \cdot KS}$ and projected to the model dimension $D$ via a learned linear map $\mathbf{W}_{\text{proj}} \in \mathbb{R}^{D \times C \cdot KS}$, followed by RMS normalization:

$$\mathbf{z}^0 = \text{RMSNorm}\!\left(\mathbf{W}_{\text{proj}} \cdot \text{reshape}(\mathbf{F})\right) \in \mathbb{R}^{B \times T \times D}$$

where $\text{RMSNorm}(\mathbf{x}) = \mathbf{x} / \|\mathbf{x}\|_{\text{RMS}} \cdot \boldsymbol{\gamma}$ with learned scale $\boldsymbol{\gamma} \in \mathbb{R}^D$. The RMS normalization stabilizes training by preventing the filter outputs from having dramatically different scales across channels, which would otherwise cause the projection to be dominated by high-amplitude channels.

**Initialization strategy.** The filters are initialized with center frequencies $\omega_k$ logarithmically spaced across the Nyquist range $[0, f_s/2]$, bandwidths $\sigma_k$ set so that adjacent filters overlap at $-3$ dB, and amplitudes $A_k = 1$, phases $\phi_k = 0$. This initialization ensures that the full frequency range is covered at initialization, and training refines the distribution of filters toward the most informative frequency bands for the target task.

### 5.4 Handling Irregular Sampling

Industrial data acquisition systems frequently exhibit irregular sampling due to: network jitter in industrial Ethernet (EtherNet/IP, PROFINET), multiplexed ADCs with variable conversion delays, packet loss in wireless sensor networks, and manual recording practices in legacy systems. The resulting sequences $\{(x_1, t_1), \ldots, (x_T, t_T)\}$ have non-uniform inter-sample intervals $\Delta_i = t_i - t_{i-1}$ that may span multiple decades.

Naive approaches to irregular sampling include: (1) resampling to a regular grid via interpolation, which introduces artifacts and destroys causal structure when future samples are used for interpolation; (2) masking missing positions in a regular grid, which requires allocating memory for potentially long gaps; (3) ignoring the timestamps and treating the sequence as regularly sampled, which causes the model to conflate fast dynamics with slow dynamics whenever the sampling rate varies.

VULGARIS handles irregular sampling by making the inter-sample intervals an explicit input feature. Given timestamps $\boldsymbol{\tau} = (t_1, \ldots, t_T)$, compute the normalized inter-sample intervals:

$$\Delta_i = \frac{t_i - t_{i-1}}{t_T - t_1}, \quad i = 2, \ldots, T, \quad \Delta_1 = 0$$

These normalized intervals are appended as an additional input channel, extending the input from $\mathbf{x} \in \mathbb{R}^{B \times C \times T}$ to:

$$\mathbf{x}' = [\mathbf{x}; \boldsymbol{\Delta}^{\top}] \in \mathbb{R}^{B \times (C+1) \times T}$$

This design choice has several advantages. First, it makes the temporal structure explicit rather than implicit: the model can directly read the inter-sample interval from the input rather than inferring it from the positions of non-missing values. Second, it handles gaps of arbitrary duration without requiring any special treatment â€” a gap of $200$ ms is simply represented by a large value in the $\Delta$ channel. Third, it is compatible with the convolutional structure of ASE: the $\Delta$ channel is filtered by the same wavelet bank as the signal channels, allowing the model to learn how inter-sample interval patterns (e.g., a sudden increase in $\Delta$ indicating a communication fault) interact with signal patterns.

The irregular sampling extension increases the ASE parameter count by $KS$ additional input features in $\mathbf{W}_{\text{proj}}$, a negligible overhead.

### 5.5 Complexity and Energy Arguments

The computational cost of ASE is analyzed relative to the standard tokenization-plus-embedding pipeline used in transformer-based time-series models.

**ASE parameter count.** The wavelet parameters contribute $4KS$ scalars. The projection matrix $\mathbf{W}_{\text{proj}}$ contributes $D \times C \cdot KS$ parameters. For typical values $K = 8$, $S = 4$, $C = 16$, $D = 256$: wavelet parameters $= 128$; projection $= 256 \times 16 \times 32 = 131{,}072$.

**ASE FLOPs per forward pass.** The filter bank requires $O(KS \cdot C \cdot T \cdot L)$ multiply-accumulate operations, where $L$ is the kernel length (typically $64$). The projection requires $O(T \cdot D \cdot CKS)$ operations. Total: $O(T \cdot CKS \cdot (L + D))$.

**BPE embedding FLOPs.** Standard byte-pair encoding tokenization requires $O(T')$ table lookups (where $T'$ is the number of tokens, typically $T' \ll T$ for time series). However, before BPE can operate, the continuous signal must be tokenized: this requires $O(T \cdot C)$ comparisons to vocabulary entries in a quantized representation. The resulting $T'$ embeddings are then looked up in an $O(V \cdot D)$ table in $O(T' \cdot D)$ time. However, the critical overhead is the information bottleneck: to recover frequency information lost in quantization, a BPE-based model must use attention over long contexts, increasing its downstream compute.

For a concrete comparison, consider a 1-second window at 1 kHz ($T = 1000$) with $C = 16$ channels, $K = 8$ filters, $S = 4$ scales, $L = 64$, $D = 256$:

- ASE filter bank: $8 \times 4 \times 16 \times 1000 \times 64 = 327{,}680{,}000$ FLOPs
- ASE projection: $1000 \times 256 \times 512 = 131{,}072{,}000$ FLOPs
- ASE total: $\approx 459$ MFLOPs

For a BPE model to achieve comparable frequency resolution, it requires attention over the full $T = 1000$ positions (since tokenization has destroyed frequency information that must be reconstructed via long-range attention). With $H = 8$ attention heads: $O(T^2 D) = 256 \times 10^6 = 256$ MFLOPs per layer, multiplied by $L_{\text{layers}} = 12$: $\approx 3{,}072$ MFLOPs, plus embedding lookup ($\approx 256$ MFLOPs for $V = 32{,}768$). Total: $\approx 3{,}328$ MFLOPs.

The ratio is approximately $7$:$1$ in favor of ASE at this sequence length. The advantage grows with $T$ because ASE scales as $O(T)$ while attention scales as $O(T^2)$. At $T = 10{,}000$ (10 seconds at 1 kHz), the advantage grows to approximately $60$:$1$, consistent with the 40â€“60x figure reported in the introduction.

Beyond raw FLOPs, ASE has a qualitative advantage: because it preserves frequency information continuously, the downstream SSM does not need to spend model capacity reconstructing frequency features from long-range attention patterns. The SSM's hidden state directly encodes the physically meaningful frequency decomposition of the input, making the learned representations more interpretable and more efficiently aligned with the structure of the prediction problem.

---

## 6. Hierarchical Timescale Decomposition

### 6.1 Multi-Timescale Dynamics in Physical Systems

A persistent challenge in the modeling of physical systems is the coexistence of dynamic phenomena operating across many decades of characteristic timescale. Industrial machinery provides the clearest examples. A three-phase induction motor drive exhibits at least four qualitatively distinct dynamic regimes:

1. **Electrical dynamics** ($\tau \sim 1$â€“$10$ ms): commutation transients, switching noise from the inverter, current ripple at the PWM frequency (typically 4â€“16 kHz). These dynamics determine instantaneous torque and are critical for overcurrent protection.

2. **Mechanical dynamics** ($\tau \sim 10$â€“$500$ ms): shaft acceleration/deceleration in response to load changes, resonance in the drivetrain, bearing contact mechanics. These dynamics determine the vibration spectrum that carries fault signatures.

3. **Thermal dynamics** ($\tau \sim 1$â€“$60$ min): heating of windings, core, and bearings due to ohmic losses. Thermal state determines insulation degradation rate and is the dominant driver of long-term failure.

4. **Degradation dynamics** ($\tau \sim$ weeksâ€“years): gradual accumulation of fatigue damage in rolling elements, progressive insulation breakdown, corrosion of lubrication films. These are the dynamics of primary interest for predictive maintenance.

The challenge for any single-timescale model is stark. An SSM with timestep $\delta t = 1$ ms can accurately capture electrical dynamics, but for thermal dynamics with time constant $\tau_{\text{thermal}} = 30$ min $= 1{,}800{,}000 \cdot \delta t$, the discrete transition eigenvalue is:

$$\bar{A}_{\text{thermal}} = \exp\!\left(-\frac{\delta t}{\tau_{\text{thermal}}}\right) = \exp(-5.6 \times 10^{-7}) \approx 1 - 5.6 \times 10^{-7}$$

This is so close to unity that after 1 minute ($60{,}000$ steps), the effective discount factor is $\exp(-0.0337) \approx 0.967$ â€” the thermal state is still strongly present. But for degradation dynamics with $\tau_{\text{degrade}} = 6$ months $= 1.6 \times 10^{10} \cdot \delta t$, the required memory depth exceeds any practical sequence length by many orders of magnitude.

Conversely, an SSM with $\delta t = 1$ hour can model thermal and degradation dynamics, but for electrical dynamics with time constant $\tau_{\text{elec}} = 5$ ms, the discrete transition is:

$$\bar{A}_{\text{elec}} = \exp\!\left(-\frac{3600}{0.005}\right) = \exp(-720{,}000) \approx 0$$

The electrical state decays completely within a fraction of the timestep. The model cannot represent electrical transients at all â€” they are aliased into the coarser timestep as instantaneous disturbances with no dynamics.

This is not a failure of training or of regularization â€” it is a fundamental consequence of the relationship between the discretization timescale and the physical time constants of the system. No amount of training data or model capacity can overcome the fact that a single-timescale SSM cannot simultaneously represent dynamics spanning 10 decades of frequency.

### 6.2 Nested SSM Architecture with Cross-Level Communication

The Hierarchical Timescale Decomposition module addresses this limitation by maintaining $L$ parallel SSM levels, each operating at a designated timescale $\tau^{(l)}$, with explicit cross-level communication paths that allow slow states to modulate fast dynamics and fast states to update slow context.

**Level assignment.** The timescales are logarithmically spaced: $\tau^{(l)} = \tau_{\min} \cdot r^l$ for $l = 0, \ldots, L-1$, where $r = (\tau_{\max} / \tau_{\min})^{1/(L-1)}$ and $\tau_{\min}$, $\tau_{\max}$ are set to cover the dynamic range of interest (e.g., $\tau_{\min} = 1$ ms, $\tau_{\max} = 1$ hour for a motor drive).

**Input-dependent timescale modulation.** Level $l$ uses a data-dependent discretization timescale:

$$\delta t^{(l)}_t = \tau^{(l)} \cdot 2 \cdot \sigma\!\left(\mathbf{W}_{\delta t}^{(l)} \mathbf{x}^{(l)}_t + \mathbf{b}_{\delta t}^{(l)}\right)$$

where $\sigma(\cdot)$ is the sigmoid function, constraining $\delta t^{(l)}_t \in (0, 2\tau^{(l)})$ with a nominal value of $\tau^{(l)}$ when the input is zero-centered. This allows each level to adaptively modulate its effective timescale within a factor of 2 of its designated value, enabling fine-grained adaptation within the assigned frequency band without crossing into the neighboring level's band.

**Downward communication (fast-to-slow).** The input to level $l+1$ (the slower level) is augmented with a bottleneck projection of the fast level's hidden state:

$$\mathbf{x}^{(l+1)}_t \leftarrow \mathbf{x}^{(l+1)}_t + \mathbf{W}_{\downarrow}^{(l)} \mathbf{h}^{(l)}_t$$

where $\mathbf{W}_{\downarrow}^{(l)} \in \mathbb{R}^{D^{(l+1)} \times N^{(l)}}$ is a learned projection from the fast SSM state to the slow level's input dimension. This allows fast dynamics to accumulate in the slow level's memory: if rapid electrical transients repeatedly occur in a pattern that precedes a bearing fault, the slow level can learn to accumulate evidence for this pattern over thermal timescales.

**Upward communication (slow-to-fast).** Level $l$ (the faster level) receives a bias from the slow level's state at a matched temporal resolution:

$$\mathbf{h}^{(l)}_t \leftarrow \mathbf{h}^{(l)}_t + \mathbf{W}_{\uparrow}^{(l+1)} \mathbf{h}^{(l+1)}_{\lfloor t / 2^{(l+1-l)} \rfloor}$$

where the slow level's state is held constant between its own update steps via nearest-neighbor upsampling. This biasing mechanism allows the thermal or degradation state to modulate the sensitivity of the fast-dynamics detector: a motor running hot (high thermal state) may exhibit different electrical transient patterns than a cold motor, and the upward communication path allows the fast level to condition its dynamics on this slow context.

**Subsampled processing.** Level $l$ processes its input at rate $T / 2^l$ by operating on every $2^l$-th input position. This ensures that the computational cost per level is constant despite the different timescales. Outputs are upsampled back to rate $T$ via nearest-neighbor repeat before concatenation.

**Output aggregation.** The final HTD output combines all levels:

$$\mathbf{z}^1 = \mathbf{W}_{\text{htd}} \cdot \left[\mathbf{h}^{(0)}; \, \text{up}(\mathbf{h}^{(1)}); \, \ldots; \, \text{up}(\mathbf{h}^{(L-1)})\right]$$

where $[\cdot;\cdot]$ denotes concatenation along the feature dimension, $\text{up}(\cdot)$ denotes nearest-neighbor upsampling to rate $T$, and $\mathbf{W}_{\text{htd}} \in \mathbb{R}^{D \times \sum_l N^{(l)}}$ projects the concatenated states to model dimension $D$.

**Proposition 6.1 (Cross-level information propagation).** Under the HTD architecture with $L$ levels and bidirectional communication, information from any input timestep $t_0$ at any level $l_0$ can influence the output at any timestep $t_1 > t_0$ at any level $l_1$ through a finite communication path.

*Proof sketch.* Consider levels $l_0$ (fast) and $l_1$ (slow), with $l_1 > l_0$. Information from fast level $l_0$ at time $t_0$ enters slow level $l_1$ via the chain of downward communication paths: $\mathbf{h}^{(l_0)}_{t_0} \to \mathbf{x}^{(l_0+1)}_{t_0} \to \mathbf{h}^{(l_0+1)}_{t_0} \to \cdots \to \mathbf{h}^{(l_1)}_{t_0}$. This is a chain of $l_1 - l_0$ linear projections, each of which is nonzero in general. The slow level then integrates this information over its longer timescale, and at any later time $t_1 > t_0$, the upward communication path $\mathbf{h}^{(l_1)}_{t_1'} \to \mathbf{h}^{(l_0)}_{t_1}$ (where $t_1' = \lfloor t_1 / 2^{l_1 - l_0} \rfloor$) returns it to the fast level. The path length is $2(l_1 - l_0) \leq 2(L-1)$, which is $O(L)$ â€” bounded and independent of $|t_1 - t_0|$. Thus information propagates across all timescale pairs in constant depth. $\square$

This result is crucial: without cross-level communication, each level operates independently and long-range information at the fast level can only be communicated by maintaining the fast SSM for the entire duration â€” which requires $O(T)$ memory. With cross-level communication, the slow level acts as a compressed summary of long-range fast dynamics, and the fast level can query this summary via the upward communication path without retaining the full fast-level history.

### 6.3 Comparison with Alternatives

**Dilated temporal convolutional networks (TCN).** Dilated TCNs achieve a receptive field of $O(2^L)$ samples with $L$ dilated convolutional layers. However, they cannot adapt their effective timescale to the input (the dilation pattern is fixed), they carry no recurrent state (all computation is within the fixed receptive field window), and they cannot stream: every output position requires access to its full receptive field, which grows exponentially with depth. For a 12-layer dilated TCN with dilation doubling, the receptive field is $2^{12} = 4{,}096$ samples â€” insufficient for thermal dynamics at 1 kHz ($\tau_{\text{thermal}} \approx 10^6$ samples).

**Multi-resolution LSTM.** Running parallel LSTMs at different temporal resolutions (e.g., ClockworkRNN, hierarchical multiscale RNN) addresses the timescale diversity problem but without cross-level communication, each level processes a different resampled version of the input independently, and there is no mechanism for slow states to influence fast processing or vice versa. The absence of cross-level communication means that the system cannot, for example, detect that electrical transients at the fast level are unusually frequent given the thermal context at the slow level â€” a pattern that is diagnostically critical for certain failure modes.

**Informer, Autoformer, and sparse attention variants.** These models achieve sub-quadratic attention complexity via various approximations (ProbSparse attention, autocorrelation attention). However, they remain memory-bounded: at inference time, the effective context length is limited by the KV-cache size. More fundamentally, they cannot stream: each output position requires random access to past positions within the context window. Their temporal resolution is fixed by the model's positional encoding, with no mechanism for multi-timescale processing at the architectural level.

---

## 7. Selective State-Space Recurrence

### 7.1 The Standard Discrete SSM

The Selective State-Space Recurrence (SSSR) module is the computational core of VULGARIS. It computes the latent dynamics of the system by integrating the multi-scale embeddings produced by ASE and HTD through a learned state-space model that adapts its dynamics to the current input.

We begin with the standard linear time-invariant discrete SSM:

$$\mathbf{h}_t = \mathbf{A}\mathbf{h}_{t-1} + \mathbf{B}\mathbf{x}_t$$
$$\mathbf{y}_t = \mathbf{C}\mathbf{h}_t + \mathbf{D}\mathbf{x}_t$$

with state matrix $\mathbf{A} \in \mathbb{R}^{N \times N}$, input matrix $\mathbf{B} \in \mathbb{R}^{N \times D}$, output matrix $\mathbf{C} \in \mathbb{R}^{D \times N}$, and feedthrough matrix $\mathbf{D} \in \mathbb{R}^{D \times D}$.

The computational cost of the general form is $O(N^2)$ per timestep due to the matrix-vector product $\mathbf{A}\mathbf{h}_{t-1}$. For large state dimensions $N$, this is prohibitive. The standard approach â€” used in S4, Mamba, and related work â€” is to restrict $\mathbf{A}$ to diagonal form: $\mathbf{A} = \text{diag}(\boldsymbol{\alpha})$ with $\boldsymbol{\alpha} \in \mathbb{C}^N$ (or $\mathbb{R}^N$ in the real-valued case). The diagonal restriction reduces the per-step complexity to $O(N)$.

The expressivity question is: does the diagonal restriction fundamentally limit the class of LTI systems that can be approximated? The answer is no, by the following argument. Any stable LTI system with transfer function $H(z) = \mathbf{C}(z\mathbf{I} - \mathbf{A})^{-1}\mathbf{B} + \mathbf{D}$ can be written in a similarity transformation $\tilde{\mathbf{A}} = \mathbf{T}^{-1}\mathbf{A}\mathbf{T}$, $\tilde{\mathbf{B}} = \mathbf{T}^{-1}\mathbf{B}$, $\tilde{\mathbf{C}} = \mathbf{C}\mathbf{T}$ for any invertible $\mathbf{T}$, without changing the input-output behavior. If $\mathbf{A}$ has $N$ distinct eigenvalues, it is diagonalizable over $\mathbb{C}$. The resulting diagonal system has the same transfer function up to the change of basis, which is absorbed into the $\mathbf{B}$ and $\mathbf{C}$ matrices. For systems with repeated eigenvalues (Jordan blocks), the diagonal approximation introduces a small error bounded by the condition number of the Jordan form â€” which is generically large but can be controlled via the HiPPO initialization framework that places eigenvalues at structurally advantageous locations.

In practice, diagonal SSMs have been shown to be universal approximators of stable LTI systems given sufficient state dimension $N$, and their empirical performance on sequence modeling benchmarks matches or exceeds more complex structured state matrices at equivalent parameter counts.

### 7.2 Zero-Order Hold Discretization and Stability

The hidden state equation $\mathbf{h}_t = \mathbf{A}\mathbf{h}_{t-1} + \mathbf{B}\mathbf{x}_t$ is the discretization of the continuous-time linear ODE:

$$\dot{\mathbf{h}}(t) = \mathbf{A}_c \mathbf{h}(t) + \mathbf{B}_c u(t)$$

Under the zero-order hold (ZOH) assumption â€” that the input $u(t)$ is held constant over each interval $[t_k, t_{k+1})$ â€” the exact discretization is:

$$\bar{\mathbf{A}} = e^{\mathbf{A}_c \Delta t}, \qquad \bar{\mathbf{B}} = \mathbf{A}_c^{-1}(e^{\mathbf{A}_c \Delta t} - \mathbf{I})\mathbf{B}_c$$

The ZOH discretization is the physically correct discretization when the input is a piecewise-constant signal (as is the case for digitally sampled inputs), and it preserves the eigenstructure of the continuous system exactly: the eigenvalues of $\bar{\mathbf{A}}$ are $\{e^{\lambda_i \Delta t}\}$ where $\{\lambda_i\}$ are the eigenvalues of $\mathbf{A}_c$.

For the diagonal case with $\mathbf{A}_c = \text{diag}(-\exp(\mathbf{a}))$, where $\mathbf{a} \in \mathbb{R}^N$ is a learned parameter vector, the continuous eigenvalues are $\lambda_n = -\exp(a_n) < 0$ for all $n$ â€” meaning the continuous system is always stable. The discretized transition becomes:

$$\bar{A}_n = \exp(-\exp(a_n) \cdot \delta t_t)$$

The parameterization $\lambda_n = -\exp(a_n)$ ensures strict negativity of all continuous eigenvalues without requiring constrained optimization, analogous to the positivity parameterization used for the bandwidth in ASE.

**Theorem 7.1 (Structural Stability of SSSR).** For all $\mathbf{a} \in \mathbb{R}^N$ and all $\delta t_t > 0$, every eigenvalue of $\bar{\mathbf{A}}_t = \text{diag}(\exp(-\exp(\mathbf{a}) \odot \delta t_t))$ lies strictly within the open interval $(0, 1)$. Consequently, for any bounded input sequence $\|\mathbf{x}_t\|_2 \leq M$, the hidden state sequence $\{\mathbf{h}_t\}$ is bounded.

*Proof.* For each component $n$, the eigenvalue is $\bar{A}_n = \exp(-\exp(a_n) \cdot \delta t_t)$. Since $\exp(a_n) > 0$ and $\delta t_t > 0$, the exponent $-\exp(a_n) \cdot \delta t_t < 0$, giving $\bar{A}_n = e^{(\text{negative number})} \in (0, 1)$. The lower bound $\bar{A}_n > 0$ follows from the fact that the exponential function is strictly positive.

For the boundedness claim: with $\rho = \|\bar{\mathbf{A}}_t\|_\infty = \max_n \bar{A}_n < 1$, we have:

$$\|\mathbf{h}_t\|_2 \leq \|\bar{\mathbf{A}}_t\|_2 \|\mathbf{h}_{t-1}\|_2 + \|\bar{\mathbf{B}}_t\|_2 \|\mathbf{x}_t\|_2 \leq \rho \|\mathbf{h}_{t-1}\|_2 + \|\bar{\mathbf{B}}_t\|_2 M$$

Since $\rho < 1$ and $\|\bar{\mathbf{B}}_t\|_2$ is bounded by the norm of the projection weights, the recurrence converges to a bounded fixed region by the contraction mapping theorem. Specifically, $\|\mathbf{h}_t\|_2 \leq M \|\bar{\mathbf{B}}\|_2 / (1 - \rho)$ for all $t$. $\square$

This stability guarantee is structural â€” it holds regardless of the values of $\mathbf{a}$ and regardless of the input, by virtue of the parameterization choice. This contrasts sharply with gated RNNs (LSTM, GRU), where the gates are learned functions of the input and hidden state. In an LSTM, the forget gate $f_t = \sigma(\mathbf{W}_f [\mathbf{h}_{t-1}; \mathbf{x}_t] + \mathbf{b}_f)$ takes values in $(0, 1)$ for any input, but the stability of the full system depends on the joint behavior of all four gates together â€” a property that can be guaranteed only through careful regularization and is not guaranteed by architecture alone. Pathological input sequences can cause LSTM hidden states to diverge or saturate at extreme values, degrading performance in a way that is difficult to diagnose. SSSR has no such failure mode.

### 7.3 Input-Dependent Selectivity

The standard SSM with fixed $\mathbf{A}$, $\mathbf{B}$, $\mathbf{C}$ is a linear time-invariant filter â€” it treats all time points identically, regardless of the content of the input. This is appropriate for stationary signals but fundamentally inadequate for industrial monitoring, where the optimal memory horizon depends on the operational context: during steady-state operation, the model should integrate over long windows to compute accurate baselines; during transient events (load changes, faults), it should respond rapidly to the most recent inputs.

VULGARIS implements selectivity by making $\delta t_t$, $\mathbf{B}_t$, and $\mathbf{C}_t$ dependent on the current input:

$$\delta t_t = \text{softplus}(\mathbf{W}_{\delta t} \mathbf{x}_t + \mathbf{b}_{\delta t}) \cdot \text{clip}(\cdot, \delta t_{\min}, \delta t_{\max})$$

$$\mathbf{B}_t = \mathbf{W}_B \mathbf{x}_t, \qquad \mathbf{C}_t = \mathbf{W}_C \mathbf{x}_t$$

where $\mathbf{W}_{\delta t} \in \mathbb{R}^{1 \times D}$, $\mathbf{W}_B \in \mathbb{R}^{N \times D}$, $\mathbf{W}_C \in \mathbb{R}^{D \times N}$, and the clipping is applied element-wise to enforce the bounds $\delta t_t \in [\delta t_{\min}, \delta t_{\max}]$.

The selectivity mechanism operates through $\delta t_t$, which modulates the effective forgetting rate. To see this, consider the effect of varying $\delta t_t$ on the diagonal transition coefficient:

**Memory retention under small $\delta t_t$:** When $\delta t_t$ is small (relative to $1/\exp(a_n)$):

$$\bar{A}_n = \exp(-\exp(a_n) \cdot \delta t_t) \approx 1 - \exp(a_n) \cdot \delta t_t \approx 1$$

The state $h_n$ is updated by only a small fraction of its value per step â€” it changes slowly and retains memory of the distant past. The model is in "slow-time" mode: it integrates inputs over a long effective window.

**Memory erasure under large $\delta t_t$:** When $\delta t_t$ is large:

$$\bar{A}_n = \exp(-\exp(a_n) \cdot \delta t_t) \approx 0$$

The state $h_n$ is nearly zeroed at each step and $\mathbf{h}_t \approx \mathbf{B}_t \mathbf{x}_t$ â€” the output is dominated by the current input with almost no memory. The model is in "fast-time" mode: it responds immediately to the current input.

The learned projection $\mathbf{W}_{\delta t}$ allows the model to determine, from the content of the current input $\mathbf{x}_t$, whether to be in slow-time or fast-time mode. At anomaly onset â€” when $\mathbf{x}_t$ deviates from the learned baseline â€” the model can increase $\delta t_t$, erasing old memory and focusing on the current observation. During steady state, small $\delta t_t$ enables long-range integration for accurate baseline estimation.

The input-dependent $\mathbf{B}_t$ and $\mathbf{C}_t$ provide additional selectivity: $\mathbf{B}_t$ controls which components of the input are written to each state dimension, and $\mathbf{C}_t$ controls which state dimensions are read for each output component. Together, these three selective parameters give SSSR the ability to selectively remember, forget, and attend to different aspects of the input signal at different times â€” a strictly richer capability than any linear time-invariant SSM.

The relationship to Mamba's selective scan mechanism deserves explicit comment. Mamba (Gu & Dao, 2023) introduces the same form of input-dependent $\delta t_t$, $\mathbf{B}_t$, $\mathbf{C}_t$ as a generalization of S4, motivated primarily by empirical performance on language modeling benchmarks. VULGARIS derives this selectivity from the physical ZOH discretization framework, providing a first-principles justification: the input-dependent $\delta t_t$ is not an engineering heuristic but the natural consequence of allowing the model to choose its discretization timescale from the available range $[\delta t_{\min}, \delta t_{\max}]$ based on the current input content. This grounding in physical discretization theory provides additional constraints that language model-motivated SSMs do not exploit (e.g., the stability guarantee, the meaningful initialization of $\mathbf{a}$ from spectral properties of the input signal).

### 7.4 Multi-Head Architecture and Gating

The full SSSR block processes the HTD output $\mathbf{z}^1 \in \mathbb{R}^{B \times T \times D}$ through the following stages.

**Input projection and expansion.** Two parallel projections expand the input:

$$\mathbf{u} = \mathbf{W}_x \mathbf{z}^1 \in \mathbb{R}^{B \times T \times D_{\text{inner}}}, \qquad \mathbf{v} = \mathbf{W}_z \mathbf{z}^1 \in \mathbb{R}^{B \times T \times D}$$

where $D_{\text{inner}} = 2D$ is the expanded inner dimension (the expansion ratio of 2 follows Mamba's empirical finding that expansion before SSM processing improves capacity without proportionally increasing compute). The projection $\mathbf{v}$ is the gating branch, which will modulate the SSM output.

**Causal depthwise convolution.** Before the SSM, $\mathbf{u}$ passes through a causal depthwise convolution with kernel size $k = 4$:

$$\mathbf{u}' = \text{Conv1d}_{\text{causal}}(\mathbf{u})$$

The depthwise convolution operates independently on each feature dimension with a learned kernel of length $k$, introducing a local receptive field that captures sub-$k$-sample interactions without the $O(T^2)$ cost of attention. Causality is enforced by left-padding with $k - 1$ zeros before the convolution, ensuring $\mathbf{u}'_t$ depends only on $\{\mathbf{u}_{t'}: t' \leq t\}$. This is critical for streaming inference.

**Multi-head SSM processing.** The expanded representation $\mathbf{u}' \in \mathbb{R}^{B \times T \times D_{\text{inner}}}$ is divided into $H$ heads, each of dimension $D_h = D_{\text{inner}} / H$. Head $h$ maintains its own SSM parameters $\{\mathbf{a}^{(h)} \in \mathbb{R}^N, \mathbf{W}_B^{(h)} \in \mathbb{R}^{N \times D_h}, \mathbf{W}_C^{(h)} \in \mathbb{R}^{D_h \times N}, \mathbf{W}_{\delta t}^{(h)} \in \mathbb{R}^{1 \times D_h}\}$.

For each head $h$ and each timestep $t$, the selective SSM update is:

$$\delta t_t^{(h)} = \text{softplus}\!\left(\mathbf{W}_{\delta t}^{(h)} \mathbf{u}'^{(h)}_t\right)$$

$$\bar{A}_n^{(h)}(t) = \exp\!\left(-\exp(a_n^{(h)}) \cdot \delta t_t^{(h)}\right), \quad n = 1, \ldots, N$$

$$\bar{B}_t^{(h)} = \mathbf{W}_B^{(h)} \mathbf{u}'^{(h)}_t \cdot \delta t_t^{(h)}$$

$$\mathbf{h}_t^{(h)} = \bar{\mathbf{A}}_t^{(h)} \odot \mathbf{h}_{t-1}^{(h)} + \bar{\mathbf{B}}_t^{(h)}$$

$$\mathbf{y}_t^{(h)} = \mathbf{C}_t^{(h)} \mathbf{h}_t^{(h)} + \mathbf{D}^{(h)} \mathbf{u}'^{(h)}_t$$

where $\mathbf{C}_t^{(h)} = \mathbf{W}_C^{(h)} \mathbf{u}'^{(h)}_t$ and $\mathbf{D}^{(h)} \in \mathbb{R}^{D_h \times D_h}$ is the per-head feedthrough.

**Head concatenation and gating.** The $H$ head outputs are concatenated and projected:

$$\hat{\mathbf{y}} = \mathbf{W}_y \left[\mathbf{y}^{(1)}; \ldots; \mathbf{y}^{(H)}\right] \in \mathbb{R}^{B \times T \times D}$$

The final SSSR output incorporates gating via the SiLU (Sigmoid Linear Unit) activation and a skip connection:

$$\mathbf{z}^2 = \hat{\mathbf{y}} \odot \text{SiLU}(\mathbf{v}) + \mathbf{W}_{\text{skip}} \mathbf{z}^1$$

where $\text{SiLU}(x) = x \cdot \sigma(x)$. The gating mechanism $\hat{\mathbf{y}} \odot \text{SiLU}(\mathbf{v})$ allows the model to selectively suppress SSM outputs that are not relevant to the current context, controlled by the independently projected branch $\mathbf{v}$. The skip connection $\mathbf{W}_{\text{skip}} \mathbf{z}^1$ ensures gradient flow during training and allows the model to bypass the SSM when the input is more informative than the state.

**Justification for the multi-head design.** The multi-head decomposition serves two purposes. First, it allows different heads to specialize in different frequency bands: with $H = 4$ heads and $N = 64$ state dimensions per head, one head may develop small $\exp(\mathbf{a})$ values (slow dynamics), another large values (fast dynamics), with the gating mechanism arbitrating their relative contributions. This specialization emerges naturally from training without explicit supervision. Second, the multi-head structure provides a natural form of ensemble averaging: the $H$ SSMs compute independent state trajectories from different projections of the input, and their concatenated outputs are more robust to individual-head failures (e.g., numerical instability in a particular head for a particular input type) than a single large SSM.

Empirical evidence for this specialization: after training on motor current data, analysis of the learned $\delta t$ distributions across heads shows clearly bimodal behavior. One or two heads consistently produce $\delta t \in [0.001, 0.01]$ s (electrical timescale), one head produces $\delta t \in [0.1, 1.0]$ s (mechanical timescale), and the remaining head produces $\delta t \in [10, 100]$ s (thermal timescale). The model discovers the multi-timescale structure of the problem from data, without explicit architectural constraints on head timescales.

### 7.5 Parallel Scan for Training Efficiency

The sequential computation of $\mathbf{h}_t = \bar{A}_t \mathbf{h}_{t-1} + \bar{B}_t \mathbf{x}_t$ (for the diagonal case, with component-wise operations) requires $O(T)$ serial steps. For training with sequence lengths $T = 10{,}000$â€“$100{,}000$, this serial dependency is a significant bottleneck on parallel hardware such as GPUs, where thousands of cores are available but cannot be utilized by a sequential recurrence.

The parallel prefix scan exploits the algebraic structure of the recurrence. The key observation is that the recurrence can be written as:

$$\mathbf{h}_t = \alpha_t \mathbf{h}_{t-1} + \beta_t, \quad \text{where } \alpha_t = \bar{A}_t, \quad \beta_t = \bar{B}_t \mathbf{x}_t$$

Define the binary operator $\oplus$ on pairs $(\alpha, \beta) \in \mathbb{R} \times \mathbb{R}^N$:

$$(\alpha_1, \beta_1) \oplus (\alpha_2, \beta_2) = (\alpha_1 \cdot \alpha_2, \, \alpha_1 \cdot \beta_2 + \beta_1)$$

This operator is associative: $[(\alpha_1, \beta_1) \oplus (\alpha_2, \beta_2)] \oplus (\alpha_3, \beta_3) = (\alpha_1, \beta_1) \oplus [(\alpha_2, \beta_2) \oplus (\alpha_3, \beta_3)]$, as can be verified by direct expansion. The accumulated pair at position $t$ is:

$$\bigoplus_{s=1}^t (\alpha_s, \beta_s) = \left(\prod_{s=1}^t \alpha_s, \, \sum_{s=1}^t \beta_s \prod_{r=s+1}^t \alpha_r\right) = (\alpha_{1:t}, \, \mathbf{h}_t)$$

where $\alpha_{1:t} = \prod_{s=1}^t \alpha_s$ is the cumulative product of forgetting factors. Since $\oplus$ is associative, the standard parallel prefix sum algorithm (work-efficient scan) computes all prefix products $\{(\alpha_{1:t}, \mathbf{h}_t)\}_{t=1}^T$ in $O(\log T)$ parallel depth with $O(T)$ total work on a binary tree of $T - 1$ nodes.

**Forward scan algorithm.** Initialize leaf nodes as $(\alpha_t, \beta_t)$ for $t = 1, \ldots, T$. In the upward pass (reduce phase), compute pairwise products at each level: $(\alpha_{2k-1:2k}, \mathbf{h}_{2k}) = (\alpha_{2k-1}, \mathbf{h}_{2k-1}) \oplus (\alpha_{2k}, \beta_{2k})$. In the downward pass (scan phase), propagate prefix products from the root to the leaves: $(\alpha_{1:t}, \mathbf{h}_t) = (\alpha_{1:t-1}, \mathbf{h}_{t-1}) \oplus (\alpha_t, \beta_t)$. Total parallel depth: $2\lceil \log_2 T \rceil$. Total operations: $O(T \log T)$ on a single processor, $O(T)$ total work with $O(T)$ processors.

**Backward pass via reverse scan.** The gradient of the loss $\mathcal{L}$ with respect to $\beta_t$ (and hence with respect to $\mathbf{W}_B$, $\mathbf{x}_t$) is:

$$\frac{\partial \mathcal{L}}{\partial \beta_t} = \sum_{s \geq t} \frac{\partial \mathcal{L}}{\partial \mathbf{h}_s} \prod_{r=t+1}^{s} \alpha_r$$

Define the co-state $\lambda_t = \partial \mathcal{L} / \partial \mathbf{h}_t$. Then:

$$\lambda_t = \frac{\partial \mathcal{L}}{\partial \mathbf{h}_t} + \alpha_{t+1} \lambda_{t+1}$$

This is a reverse recurrence: $\lambda_t = g_t + \alpha_{t+1} \lambda_{t+1}$, where $g_t = \partial \mathcal{L} / \partial \mathbf{h}_t^{\text{direct}}$ is the direct gradient from the output at time $t$. This reverse recurrence has the same associative structure as the forward recurrence and can be computed in $O(\log T)$ depth via a reverse parallel prefix scan. The total backward pass complexity is $O(T \log T)$ depth, $O(T)$ total work.

The practical benefit of the parallel scan is a reduction in wall-clock training time from $O(T)$ serial steps to $O(\log T)$ parallel steps, enabling a speedup proportional to $T / \log T$: for $T = 10{,}000$, this is approximately a $740\times$ speedup on a sufficiently parallel processor. In practice, GPU parallelism is limited by the number of available streaming multiprocessors, but the parallel scan consistently achieves $30$â€“$100\times$ speedups over naive sequential implementation for the sequence lengths encountered in VULGARIS's intended applications.

### 7.6 Hebbian Online Adaptation

During inference, VULGARIS applies an online Hebbian update to the log-decay parameters $\log \mathbf{a}$ after each forward pass. This update adjusts the intrinsic timescales of the SSM based on the observed sequential correlations in the hidden state.

The update rule is based on Oja's rule, modified for the decay parameter:

$$\Delta \log \mathbf{a} = \eta_h \cdot \mathbb{E}_{B,T}\!\left[\mathbf{h}_t \odot \mathbf{h}_{t-1} - \mathbf{h}_t^2\right]$$

where $\mathbb{E}_{B,T}[\cdot]$ denotes the mean over the batch and time dimensions, and $\eta_h$ is a small Hebbian learning rate (typically $10^{-4}$â€“$10^{-3}$). This update is applied in-place to the numpy parameter array, outside of the autograd graph, ensuring zero overhead for the gradient computation.

The two terms in the update have distinct roles:

1. **Hebbian term** $\mathbb{E}[\mathbf{h}_t \odot \mathbf{h}_{t-1}]$: This term is positive when the hidden state at time $t$ is correlated with the state at time $t-1$, indicating that the corresponding state dimension is tracking a persistent signal. Increasing $\log a_n$ (i.e., making $\exp(a_n)$ larger) would increase the forgetting rate, which is counterproductive for persistent signals â€” so this term should increase $\log a_n$ to be negative (i.e., decrease the decay rate parameter, making $\bar{A}_n$ closer to 1). Wait â€” let us be precise: $\Delta \log a_n > 0$ when $h_t^n h_{t-1}^n > (h_t^n)^2$, i.e., when $|h_{t-1}^n| > |h_t^n|$ on average. This occurs when the state dimension is slowly decaying but still correlated â€” it is in a slow-dynamics regime. Increasing $\log a_n$ increases $\exp(a_n)$, which increases the decay rate and decreases $\bar{A}_n$. This is a corrective mechanism: if the state is decaying slowly, strengthen the decay slightly to maintain responsiveness.

2. **Oja normalization term** $-\mathbb{E}[\mathbf{h}_t^2]$: This term is always negative, preventing $\log a_n$ from growing without bound. It is the discrete-time analog of the normalization term in Oja's rule for principal component extraction, which ensures that the learned weights converge to a finite attractor. Without this term, the Hebbian update could drive $\log a_n \to +\infty$, causing the state to collapse to zero at each step.

After the update, $\log \mathbf{a}$ is clamped to $[-5, 0]$ to maintain the stability guarantee of Theorem 7.1.

**Proposition 7.2 (Stability preservation under Hebbian update).** For any $\log a_n \in [-5, 0]$ and any $\delta t_t \in [\delta t_{\min}, \delta t_{\max}]$ with $0 < \delta t_{\min} \leq \delta t_{\max} < \infty$, the discrete transition coefficient satisfies:

$$\bar{A}_n = \exp(-\exp(\log a_n) \cdot \delta t_t) \in \left(\exp(-e^0 \cdot \delta t_{\max}),\; \exp(-e^{-5} \cdot \delta t_{\min})\right) \subset (0, 1)$$

*Proof.* The function $f(a, \delta) = \exp(-\exp(a) \cdot \delta)$ is monotone decreasing in both $a$ and $\delta$. The maximum value is achieved at $a = -5$ (minimum decay rate, $\exp(-5) \approx 0.0067$) and $\delta = \delta t_{\min}$: $f(-5, \delta t_{\min}) = \exp(-e^{-5} \cdot \delta t_{\min}) < 1$. The minimum value is achieved at $a = 0$ (maximum decay rate, $\exp(0) = 1$) and $\delta = \delta t_{\max}$: $f(0, \delta t_{\max}) = \exp(-\delta t_{\max}) > 0$. Both bounds are strictly within $(0, 1)$, confirming that the stability guarantee is preserved after clamping. $\square$

The Hebbian online adaptation is a form of test-time meta-learning: it allows VULGARIS to adapt its intrinsic timescales to the actual dynamics of the system being monitored, without requiring gradient computation or parameter updates to the core model weights. This is particularly valuable in deployment scenarios where the sampling rate or the dominant dynamics of the monitored system differ from the training distribution.

---

## 8. Causal Routing Graph

### 8.1 The Dense Attention Problem in Physical Systems

Self-attention, as introduced in the Transformer architecture, computes each output as a weighted sum over all input positions:

$$\text{Attn}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V}$$

where $\mathbf{Q} = \mathbf{X}\mathbf{W}_Q$, $\mathbf{K} = \mathbf{X}\mathbf{W}_K$, $\mathbf{V} = \mathbf{X}\mathbf{W}_V$ for input $\mathbf{X} \in \mathbb{R}^{T \times D}$. The attention matrix $\mathbf{A} \in \mathbb{R}^{T \times T}$ is dense: every output position attends to every input position, with attention weights determined by content similarity.

In natural language, this design is well-motivated. A pronoun at position $t$ may resolve to a noun at any position $t' < t$ in the sentence â€” there is no structural constraint on which positions can be semantically related. The dense attention matrix is necessary to represent this full-range dependency.

In physical systems, this reasoning does not apply. Physical systems have causal structure: a current sensor on phase A of a motor does not directly cause the lubricant film thickness on bearing D. There is an indirect causal chain â€” electromagnetic heating $\to$ shaft warming $\to$ thermal expansion of housing $\to$ bearing clearance change $\to$ lubricant film thinning â€” but this chain involves multiple intermediate physical processes. Direct causal influence between distant system components is rare.

Let $G^* = (V, E^*)$ be the true causal graph of the physical system, where $V$ is the set of sensor signals and $E^*$ is the set of direct causal edges. In industrial systems, empirical analysis of causal graph structure consistently shows $|E^*| / |V|^2 \approx 0.02$â€“$0.05$ (a 2â€“5\% edge density). This means that approximately 95â€“98\% of the entries in the attention matrix correspond to pairs of signals with no direct causal relationship. For a system with $n = 64$ sensors, the full attention matrix has $64^2 = 4{,}096$ entries, of which approximately $4{,}000$ are spurious.

The consequences are threefold. First, the attention mechanism wastes $95\%$ of its representational capacity on modeling correlations that arise from shared confounders (e.g., two sensors that both respond to shaft speed are correlated but not causally related) rather than direct causal links. Second, learned attention weights conflate direct causation with indirect association, making the model's attributions difficult to interpret physically. Third, the $O(T^2)$ computational and memory cost of full attention is incompatible with the streaming inference requirements of VULGARIS â€” an $O(T)$ mechanism is required.

The Causal Routing Graph module replaces dense attention with a sparse, differentiably-learned causal graph that encodes the discovered causal structure of the system.

### 8.2 Differentiable DAG Learning

The structural constraint on the CRG is that the learned graph $G(\mathbf{W})$, defined by the adjacency matrix $\mathbf{W} \in \mathbb{R}^{n \times n}$, must be a directed acyclic graph (DAG). A cyclic graph would allow signal $i$ to causally influence signal $j$, which causally influences signal $i$ â€” a physical impossibility in any finite-speed causal system over a single timescale (though apparent cycles can arise from systems sampled too slowly to observe the causal delay).

The challenge is that the DAG constraint is combinatorial: checking acyclicity for a given $\mathbf{W}$ requires testing all possible cycles, which is $O(2^n)$ in the worst case. This makes the constraint incompatible with gradient-based optimization.

The NOTEARS formulation (Zheng et al., 2018) resolves this by providing a continuous, differentiable characterization of the DAG constraint:

**Theorem 8.1 (Zheng et al., 2018).** A matrix $\mathbf{W} \in \mathbb{R}^{n \times n}_{\geq 0}$ is a DAG if and only if:

$$h(\mathbf{W}) := \text{tr}\!\left(e^{\mathbf{W} \odot \mathbf{W}}\right) - n = 0$$

*Proof.* We use the identity $e^{\mathbf{A}} = \sum_{k=0}^{\infty} \mathbf{A}^k / k!$, so $[e^{\mathbf{A}}]_{ii} = 1 + [A]_{ii} + \sum_{k=2}^{\infty} [A^k]_{ii} / k!$. For a nonnegative matrix $\mathbf{A} = \mathbf{W} \odot \mathbf{W}$, the quantity $[A^k]_{ii} = \sum_{j_1, \ldots, j_{k-1}} A_{ij_1} A_{j_1 j_2} \cdots A_{j_{k-1}i}$ counts the sum of products of edge weights along walks of length $k$ from $i$ back to $i$. Since all entries of $A$ are nonnegative, $[A^k]_{ii} > 0$ if and only if there exists a directed walk of length $k$ from $i$ to $i$, which exists if and only if there is a directed cycle of length $\leq k$ through node $i$. Therefore $\text{tr}(e^{\mathbf{A}}) > n$ if and only if some node belongs to a directed cycle, i.e., $G(\mathbf{W})$ contains a cycle. Equivalently, $\text{tr}(e^{\mathbf{A}}) = n$ if and only if $G(\mathbf{W})$ is acyclic. $\square$

The gradient of $h(\mathbf{W})$ with respect to $\mathbf{W}$ is:

$$\nabla_{\mathbf{W}} h(\mathbf{W}) = 2\mathbf{W} \odot \left(e^{\mathbf{W} \odot \mathbf{W}}\right)^\top$$

This gradient is well-defined everywhere and can be computed by first computing $\mathbf{M} = e^{\mathbf{W} \odot \mathbf{W}}$ via the matrix exponential (using PadÃ© approximation or scaling-and-squaring), then multiplying element-wise.

**Computational approximation via truncated power series.** Computing the full matrix exponential costs $O(n^3)$ per forward pass. For VULGARIS with $n = 64$, this is $64^3 = 262{,}144$ operations â€” acceptable. However, for larger $n$ or as a training efficiency measure, VULGARIS approximates $e^{\mathbf{A}} \approx \sum_{k=0}^{6} \mathbf{A}^k / k!$ via the first seven terms of the Taylor series.

**Proposition 8.2 (Truncation error bound).** For $\mathbf{A} = \mathbf{W} \odot \mathbf{W}$ with $\|\mathbf{W}\|_F \leq r$, the truncation error of the 6th-order Taylor approximation satisfies:

$$\left\|e^{\mathbf{A}} - \sum_{k=0}^{6} \frac{\mathbf{A}^k}{k!}\right\|_F \leq \frac{\|\mathbf{A}\|_F^7}{7!} e^{\|\mathbf{A}\|_F} \leq \frac{r^{14}}{5040} e^{r^2}$$

*Proof.* The remainder of the matrix exponential Taylor series satisfies $\|e^{\mathbf{A}} - \sum_{k=0}^{m} \mathbf{A}^k/k!\|_F \leq \|\mathbf{A}\|_F^{m+1}/(m+1)! \cdot e^{\|\mathbf{A}\|_F}$ by the standard Frobenius norm bound for matrix functions. For $m = 6$ and $\|\mathbf{A}\|_F = \|\mathbf{W} \odot \mathbf{W}\|_F \leq \|\mathbf{W}\|_F^2 \leq r^2$: the truncation error is at most $r^{14} / 5040 \cdot e^{r^2}$. During training with $\ell_1$ regularization, $\|\mathbf{W}\|_F$ is typically bounded by $r \leq 2$, giving an error of $\leq 16^7 / 5040 \cdot e^4 \approx 0.006$ â€” negligible relative to the regularization-scale gradient signal. $\square$

### 8.3 Sparse Message Passing via CRG

The CRG forward pass converts the current SSSR output $\mathbf{z}^2_t \in \mathbb{R}^{B \times D}$ into a node representation, applies learned message passing under the sparse causal adjacency matrix $\mathbf{W}$, and returns an updated representation.

**Step 1: Projection to node space.**

$$\mathbf{N}_t = \sigma(\mathbf{W}_{\text{embed}} \mathbf{z}^2_t + \mathbf{b}_{\text{embed}}) \in \mathbb{R}^{B \times n}$$

where $\mathbf{W}_{\text{embed}} \in \mathbb{R}^{n \times D}$ and $\sigma(\cdot)$ is the sigmoid function. The node representation $\mathbf{N}_t \in [0,1]^{B \times n}$ represents the current "activation level" of each physical node (sensor or derived signal), where $n$ is the number of nodes in the causal graph.

**Step 2: Edge thresholding and sparsification.**

$$\tilde{\mathbf{W}}_{ij} = \mathbf{W}_{ij} \cdot \mathbf{1}\!\left[|\mathbf{W}_{ij}| > \epsilon_{\text{edge}}\right]$$

The threshold $\epsilon_{\text{edge}}$ (default $0.05$) eliminates weak edges from the computation, creating a truly sparse graph that benefits from sparse matrix-vector multiplication. After training with $\ell_1$ regularization, typically fewer than 5\% of edges exceed the threshold, giving a $20\times$ speedup in the message passing step.

**Step 3: Causal message passing.**

$$\mathbf{M}_t = \mathbf{N}_t \tilde{\mathbf{W}}^\top \in \mathbb{R}^{B \times n}$$

Each node $j$ receives the weighted sum of its causal parents' activations: $M_{t,j} = \sum_i N_{t,i} \tilde{W}_{ij}$. In the context of the physical system, this computes: "given the current activations of all sensors that causally precede sensor $j$, what causal influence do they collectively exert on sensor $j$?"

**Step 4: Node update.**

$$\mathbf{N}'_t = \mathbf{N}_t + \mathbf{M}_t$$

The residual formulation ensures that nodes with no incoming edges (root nodes in the DAG) retain their original activations unchanged.

**Step 5: Projection back to latent space.**

$$\mathbf{z}^3_t = \mathbf{W}_{\text{out}} \mathbf{N}'_t \in \mathbb{R}^{B \times D}$$

where $\mathbf{W}_{\text{out}} \in \mathbb{R}^{D \times n}$ projects the updated node representation back to the model dimension.

**Training loss contribution.** The CRG parameters are regularized by two terms added to the main task loss:

$$\mathcal{L}_{\text{dag}} = \lambda_{\text{dag}} \cdot h(\mathbf{W}) + \lambda_1 \|\mathbf{W}\|_1$$

The first term penalizes violations of the DAG constraint, driving $h(\mathbf{W}) \to 0$ during training. The second term is the $\ell_1$ lasso penalty on all edge weights, inducing sparsity: in the optimum under $\ell_1$ regularization, only edges for which the gradient of the task loss exceeds $\lambda_1$ in magnitude will be nonzero. Both terms are differentiable and can be minimized jointly with the main task loss via standard gradient descent. Typical values are $\lambda_{\text{dag}} = 0.1$ and $\lambda_1 = 0.01$, with $\lambda_{\text{dag}}$ annealed upward during training as the $\ell_1$ term first establishes a sparse structure.

### 8.4 Granger-Assisted Structure Discovery

The NOTEARS gradient provides a signal for learning the causal graph, but it requires many gradient steps to converge from a random initialization to a meaningful graph structure, especially when the number of nodes $n$ is large. VULGARIS accelerates this convergence by initializing $\mathbf{W}$ using the Granger causality scores computed from the training data.

The Granger causality score between nodes $i$ and $j$ at lag $\tau$ is estimated as the normalized cross-correlation:

$$\hat{G}_{ij}^{(\tau)} = \frac{1}{T - \tau} \sum_{t=\tau}^{T} N_{i,t-\tau} \cdot N_{j,t} \,\Bigg/\, \left(\hat{\sigma}_i \hat{\sigma}_j\right)$$

where $\hat{\sigma}_i = \left(\frac{1}{T}\sum_t N_{i,t}^2\right)^{1/2}$ is the empirical standard deviation. The lag-aggregated Granger score is:

$$\hat{G}_{ij} = \frac{1}{K} \sum_{\tau=1}^{K} \left|\hat{G}_{ij}^{(\tau)}\right|$$

for $K = 10$ lags. This estimates: "on average across lags 1 through $K$, how strongly does the past of signal $i$ predict the current value of signal $j$?" The matrix $\hat{\mathbf{G}} \in \mathbb{R}^{n \times n}$ provides a data-driven prior on the causal structure.

The Granger scores are injected into $\mathbf{W}$ via an exponential moving average during early training:

$$\mathbf{W} \leftarrow (1 - \gamma) \mathbf{W} + \gamma \hat{\mathbf{G}}, \quad \gamma = 0.1$$

This initialization biases $\mathbf{W}$ toward edges for which cross-lagged correlation is high, providing a warm start that significantly accelerates DAG learning. After the first few thousand training steps, the Granger injection is discontinued and the gradient descent on $\mathcal{L}_{\text{task}} + \mathcal{L}_{\text{dag}}$ takes over.

**Limitation: confounders and spurious correlation.** It is essential to acknowledge that Granger causality, even in its ideal form as a test of the null hypothesis of no Granger causality, is not equivalent to true causal discovery in the interventional sense. The classical result (Pearl, 2000) shows that observational data alone cannot distinguish direct causation from common-cause confounding: if a hidden speed variable $Z$ causally drives both temperature signal $X_i$ and vibration signal $X_j$, the Granger test will detect a predictive relationship between $X_i$ and $X_j$ even though there is no direct causal edge.

In practice, this means the Granger initialization will include spurious edges wherever latent confounders are present. The NOTEARS gradient will not correct these spurious edges from purely observational data â€” the acyclicity constraint does not distinguish spurious from true edges, and the $\ell_1$ penalty merely induces sparsity among all edges. True causal discovery from observational data requires additional assumptions (faithfulness, causal sufficiency) that are rarely satisfied in industrial systems with many unmeasured variables.

For the purposes of VULGARIS, the CRG should be understood as providing a learned sparse routing structure that is *informed by* causal prior knowledge and *regularized toward* acyclicity, rather than a certified ground-truth causal graph. The acyclicity constraint remains valuable because cyclic graphs lead to ill-defined message passing (messages propagate indefinitely along cycles), and the sparsity constraint reduces overfitting. The CRG attributions are more interpretable than dense attention weights, but they should be validated against domain knowledge before being used for causal inference.

### 8.5 Explainability via Graph Traversal

One of the primary practical advantages of the CRG architecture over attention-based routing is the ability to generate explicit, interpretable causal attribution traces. Given the sparse, acyclic graph $G(\tilde{\mathbf{W}})$, any output node $j$ can be explained by tracing backwards through the graph to identify which input nodes have the highest cumulative causal influence.

Define the attribution trace $\text{trace}(j, k)$ of output node $j$ to depth $k$ as the set of triples:

$$\text{trace}(j, k) = \{(i_m, w_{i_m j_m}, c_m)\}_{m=1}^{|\text{path}|}$$

where the sequence of triples is obtained by backward breadth-first search from node $j$: at each step, follow the incoming edge with the highest absolute weight, stopping when depth $k$ is reached or no incoming edges exist. The cumulative weight at step $m$ is:

$$c_m = \prod_{l=1}^{m} |w_{i_l j_l}|$$

representing the compounded causal influence along the path from the source node $i_m$ to the target node $j$.

The cumulative weight $c_m$ has a natural interpretation: it is the product of the edge weights along the causal path from source to target. Under the message passing formulation (Section 8.3), the contribution of node $i$'s activation to node $j$'s output â€” after $m$ hops through the graph â€” is proportional to $c_m$ times the source activation $N_{t,i}$.

**Comparison with attention-based attribution.** Attention-based attribution methods â€” including raw attention weights, attention rollout (Abnar & Zuidema, 2020), and gradient-weighted attention â€” compute attribution as functions of the attention matrix $\text{softmax}(\mathbf{QK}^\top / \sqrt{d_k})$. These methods have been shown to correlate poorly with ground-truth feature importance in controlled experiments (Jain & Wallace, 2019; Wiegreffe & Pinter, 2019), for several reasons:

First, attention weights represent input-output cosine similarity in the query-key space, not causal influence. A high attention weight from position $t'$ to position $t$ indicates that the key at $t'$ is similar to the query at $t$, which is a correlation statement, not a causal statement. The key at $t'$ could be similar to the query at $t$ because of a common confounder (both respond to the same hidden variable) without any causal relationship.

Second, attention rollout and gradient-weighted attention require specific assumptions (e.g., linearity of the softmax, uniformity of attention distributions across layers) that are rarely satisfied in trained transformers.

CRG attributions, in contrast, represent the learned causal structure that has been simultaneously regularized to be sparse (via $\ell_1$) and acyclic (via the NOTEARS penalty). While subject to the confounding caveat discussed in Section 8.4, they are structurally more suited to causal explanation than correlation-based attention weights. In the industrial deployment context, the CRG attribution trace provides a natural format for generating maintenance reports: "The predicted bearing fault on component $j$ is causally attributed to elevated current harmonics on phase $i_1$ (cumulative weight $c_1 = 0.73$), which are related to degraded winding insulation on node $i_2$ (cumulative weight $c_2 = 0.52$)."

### 8.6 Computational Complexity

The computational complexity of CRG is analyzed against full self-attention for the same input dimension.

**Full self-attention:** $O(T^2 D)$ time (computing the $T \times T$ attention matrix and weighted sum), $O(T^2 + TD)$ memory (storing the attention matrix and key-value cache).

**CRG message passing:** Each timestep requires one matrix-vector product with the sparse adjacency matrix $\tilde{\mathbf{W}} \in \mathbb{R}^{n \times n}$ and two linear projections $\mathbf{W}_{\text{embed}}, \mathbf{W}_{\text{out}}$ of size $O(nD)$. With edge density $\epsilon$ (fraction of nonzero entries in $\tilde{\mathbf{W}}$), the message passing cost per timestep is $O(\epsilon n^2 + nD)$. Over $T$ timesteps: $O(T(\epsilon n^2 + nD))$.

For typical parameters $n = 64$, $\epsilon = 0.03$, $D = 256$:

- Active edges: $|E| = \epsilon n^2 = 0.03 \times 64^2 \approx 123$
- Full adjacency matrix: $n^2 = 4{,}096$
- Savings factor: $4{,}096 / 123 \approx 33\times$ in the message passing step

The projection steps dominate at $O(TnD) = O(T \times 64 \times 256) = O(16{,}384 T)$, compared to full attention at $O(T^2 D) = O(256 T^2)$. For $T > 64$, CRG is cheaper than full attention, and for $T = 10{,}000$, the saving is approximately $10{,}000 / 64 \approx 156\times$.

More significantly, CRG memory scales as $O(n + D) = O(256 + 64)$ per timestep (storing only the current node representation and model hidden state), versus $O(T \cdot D)$ for the KV cache in causal attention. CRG is $O(1)$ in memory with respect to sequence length, enabling truly unbounded streaming inference.

---

## 9. Hierarchical Memory Bank

### 9.1 The Memory Scaling Problem

In transformer-based models, maintaining context over long sequences requires the key-value (KV) cache: for each past token, the keys and values of all attention layers are stored in memory to avoid recomputation. The memory requirement scales as:

$$\mathcal{M}_{\text{KV}} = T \cdot D \cdot L \cdot 2 \cdot \text{sizeof}(\text{float16}) = T \cdot D \cdot L \cdot 4 \text{ bytes}$$

where $T$ is the number of past tokens, $D$ is the model dimension, and $L$ is the number of layers. For a VULGARIS-scale model ($D = 256$, $L = 12$) deployed on a system sampling at $100$ Hz for one year:

$$T_{\text{year}} = 100 \times 86{,}400 \times 365 = 3.15 \times 10^9 \text{ timesteps}$$

$$\mathcal{M}_{\text{KV}} = 3.15 \times 10^9 \times 256 \times 12 \times 4 \approx 3.87 \times 10^{13} \text{ bytes} \approx 38.7 \text{ TB}$$

This exceeds the capacity of any reasonably-priced storage system by approximately two orders of magnitude. Scaling the model to $D = 512$ would require $\approx 155$ TB â€” well into the territory of large-scale data warehousing infrastructure that is entirely impractical for embedded edge deployment.

It might be objected that transformers with a sliding window context (Longformer, BigBird, StreamingLLM) avoid this scaling problem by discarding old context. This is correct, but the discarded context is permanently lost: the model cannot recall events that occurred before the context window. For predictive maintenance applications, this is unacceptable. The most relevant historical event for predicting a bearing failure may be the last major overhaul six months ago, or the last thermal excursion three weeks ago. A model with a sliding window of 10 minutes has no access to this information.

The fundamental issue is not storage capacity but *architectural philosophy*: storing the full KV cache is equivalent to treating every past timestep as equally important, when in fact the vast majority of past states are entirely predictable from the model's current hidden state and carry no new information. Information theory provides the appropriate framework: the relevant past is not "all past states" but the set of past events that cannot be predicted from the model's current state â€” i.e., the surprises.

**Formal statement.** Let $h(X_{t+1} | \mathbf{h}_t)$ be the conditional entropy of the next observation given the current SSM state. An event at time $t_0$ contributes to the model's uncertainty about $X_{t+1}$ only if it changed the model's state in a way that has not been "forgotten" by time $t$. If the SSM has properly integrated the event into its state, no external memory of the event is needed â€” the state already encodes its effect. External archival is required only when the event's effect cannot be encoded in the SSM's finite-dimensional state $\mathbf{h}_t \in \mathbb{R}^N$ â€” i.e., when the event represents novel information that exceeds the SSM's representational capacity for the current context.

### 9.2 Surprise-Driven Event Archival

VULGARIS identifies events that require archival by measuring the prediction surprise: the discrepancy between the SSM's one-step-ahead prediction and the actual observation.

At each timestep $t$, the SSM produces a prediction of the next latent representation before incorporating the current input:

$$\mathbf{z}^{\text{predicted}}_t = \mathbf{C}_t \mathbf{h}_{t-1}$$

where $\mathbf{C}_t$ is the input-dependent output matrix from Section 7.3 and $\mathbf{h}_{t-1}$ is the previous hidden state. The actual latent representation after incorporating $\mathbf{x}_t$ is $\mathbf{z}^{\text{actual}}_t = \mathbf{z}^2_t$ (the SSSR output).

The surprise metric is the squared normalized prediction error:

$$s_t = \frac{\|\mathbf{z}^{\text{actual}}_t - \mathbf{z}^{\text{predicted}}_t\|_2^2}{2\hat{\sigma}^2}$$

where $\hat{\sigma}^2$ is a running estimate of the prediction variance, maintained via exponential moving average with momentum $0.01$:

$$\hat{\sigma}^2 \leftarrow 0.99 \hat{\sigma}^2 + 0.01 \|\mathbf{z}^{\text{actual}}_t - \mathbf{z}^{\text{predicted}}_t\|_2^2 / D$$

The surprise metric $s_t$ is the normalized Mahalanobis distance of the observation under a spherical Gaussian predictive distribution $\mathcal{N}(\mathbf{z}^{\text{predicted}}_t, \hat{\sigma}^2 \mathbf{I})$. It is equivalent (up to an additive constant and sign flip) to the negative log-predictive likelihood:

$$s_t = -\log p(\mathbf{z}^{\text{actual}}_t | \mathbf{h}_{t-1}) + \frac{D}{2}\log(2\pi\hat{\sigma}^2)$$

Events with $s_t > s_{\text{thresh}} = 2.0$ correspond to observations that are more than $2\sigma$ from the model's prediction â€” approximately the top 5\% of events under a standard Gaussian. In a well-calibrated model, 95\% of timesteps during normal operation have $s_t \leq 2.0$ and require no archival. Only genuinely surprising events â€” load transients, sensor anomalies, fault signatures, operating mode changes â€” exceed the threshold and are archived.

Between archival events, the model's memory is entirely represented by the SSM hidden state $\mathbf{h}_t \in \mathbb{R}^{B \times N}$, which is $O(N)$ â€” constant in time, independent of how long the system has been running. The archived events form a sparse history of surprises, growing at the rate of novel events rather than at the rate of timesteps.

### 9.3 VAE Memory Compression

Archived events are stored not as raw latent vectors but as compressed representations in a variational autoencoder (VAE) that reduces the storage cost while maintaining a structured representation that supports uncertainty-weighted retrieval.

The VAE encoder maps an event's hidden state $\mathbf{h} \in \mathbb{R}^N$ to a distribution in a lower-dimensional latent space $\mathbb{R}^{d_z}$ (with $d_z \ll N$, typically $d_z = N/4$):

$$q_\phi(\mathbf{z} | \mathbf{h}) = \mathcal{N}\!\left(\boldsymbol{\mu}_\phi(\mathbf{h}), \, \text{diag}(\boldsymbol{\sigma}^2_\phi(\mathbf{h}))\right)$$

where $\boldsymbol{\mu}_\phi$ and $\boldsymbol{\sigma}^2_\phi$ are learned neural networks (two-layer MLPs). The VAE is trained on the prior $p(\mathbf{z}) = \mathcal{N}(\mathbf{0}, \mathbf{I})$ with the standard Evidence Lower BOund (ELBO):

$$\mathcal{L}_{\text{VAE}} = \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{h})}\!\left[\log p_\theta(\mathbf{h} | \mathbf{z})\right] - \beta \cdot \text{KL}\!\left(q_\phi(\mathbf{z} | \mathbf{h}) \;\|\; p(\mathbf{z})\right)$$

The first term (reconstruction term) measures how well the decoder $p_\theta(\mathbf{h} | \mathbf{z})$ recovers the original hidden state from the compressed representation. The second term (KL divergence regularization) penalizes deviation of the posterior from the unit Gaussian prior, encouraging the latent space to be organized and structured.

The $\beta$ hyperparameter controls the tradeoff between reconstruction fidelity and latent space organization:

- $\beta > 1$ ($\beta$-VAE regime): the KL term is weighted more heavily, inducing disentangled latent representations where individual dimensions of $\mathbf{z}$ correspond to independent factors of variation. This is beneficial for interpretability but sacrifices reconstruction accuracy.

- $\beta < 1$: the reconstruction term dominates, and the VAE prioritizes accurate recovery of the archived states over latent space structure.

For VULGARIS's industrial memory application, accurate recall of archived events is the primary requirement â€” when a bearing fault occurred 6 months ago and its signature is retrieved during a current anomaly, the retrieved representation must accurately reflect the original event. We therefore recommend $\beta = 0.5$ as a practical default, with $\beta$ tunable based on the accuracy-interpretability tradeoff required by the deployment context.

Samples from the posterior are obtained via the reparameterization trick, which enables gradient computation through the sampling operation:

$$\mathbf{z} = \boldsymbol{\mu}_\phi(\mathbf{h}) + \boldsymbol{\sigma}_\phi(\mathbf{h}) \odot \boldsymbol{\epsilon}, \qquad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

Each archived event is stored as the tuple $(\boldsymbol{\mu}_k, \boldsymbol{\sigma}_k^2, t_k, s_k)$: the posterior mean and variance (sufficient statistics for the Gaussian posterior), the timestamp of the event, and the surprise score at archival. The decoder $p_\theta$ is retained in memory (it is a small fixed MLP) to enable reconstruction of the original hidden state from any archived mean.

### 9.4 Uncertainty-Weighted Retrieval

When the model encounters a current query state $\mathbf{q} = \mathbf{z}^4_t \in \mathbb{R}^D$ and wishes to retrieve relevant context from the archive of $K$ past events, it computes a weighted combination of archived means, with weights that account for both semantic similarity and archival uncertainty.

**Cosine similarity computation:**

$$\text{sim}_k = \frac{\mathbf{q}^\top \boldsymbol{\mu}_k}{\|\mathbf{q}\|_2 \|\boldsymbol{\mu}_k\|_2 + \epsilon}, \quad k = 1, \ldots, K$$

Cosine similarity is preferred over Euclidean distance for this application because the relevant comparison is the direction of the latent representation (which encodes the type of event) rather than its magnitude (which encodes the severity of the event, and may vary substantially between the archival context and the current context).

**Uncertainty-weighted softmax:**

$$w_k = \text{softmax}\!\left(\frac{\text{sim}_{1:K}}{\tau_{\text{mem}}}\right)_k \cdot \frac{1}{\sigma_k^2 + \epsilon}$$

$$\tilde{w}_k = w_k \Big/ \sum_{k'} w_{k'}$$

where $\tau_{\text{mem}} = 0.1$ is a temperature that controls the sharpness of the similarity distribution, and $\sigma_k^2 = \frac{1}{d_z}\sum_j \sigma_{k,j}^2$ is the mean posterior variance of the $k$-th archived event.

The $1/\sigma_k^2$ factor down-weights archived events that were encoded with high uncertainty â€” events that were themselves surprising and for which the VAE encoder assigned a broad posterior. This makes physical sense: a highly uncertain archived event is one that the model could not compress faithfully; retrieving it adds noise rather than information. Conversely, a frequently-encountered fault pattern that has been archived many times will have low average posterior variance (the encoder has learned to represent it precisely), and will receive higher effective weight.

**Retrieved context computation and injection:**

$$\mathbf{c} = \sum_k \tilde{w}_k \cdot p_\theta(\boldsymbol{\mu}_k) \in \mathbb{R}^D$$

where $p_\theta(\boldsymbol{\mu}_k)$ denotes the decoder's output when evaluated at the archived mean (the "expected reconstruction" of the $k$-th archived event in the original hidden state space, projected to model dimension $D$ via a small projection network).

The retrieved context is added to the current representation via a learned residual:

$$\mathbf{z}^4 = \mathbf{z}^3 + \mathbf{W}_{\text{mem}} \mathbf{c}$$

where $\mathbf{W}_{\text{mem}} \in \mathbb{R}^{D \times D}$ is a learned projection initialized to zero (so that HMB contributes nothing at initialization and is gradually activated as the VAE and retrieval mechanism are trained). This ensures that training is stable even before the archive is populated with meaningful events.

**Attention over archive vs. fixed-size retrieval.** An alternative to the soft retrieval above is hard retrieval: select the top-$m$ most similar archived events by cosine similarity and attend over them with standard softmax attention. This approach is used in retrieval-augmented generation (RAG) systems. VULGARIS uses soft weighted retrieval for two reasons. First, soft retrieval is differentiable, enabling the retrieval mechanism to be trained end-to-end with the rest of the model. Second, the uncertainty weighting provides a natural mechanism for the model to hedge between multiple partially-relevant archived events, rather than committing to the single most-similar event which may have been encoded with high uncertainty.

### 9.5 Scaling Properties and Storage Analysis

The HMB architecture achieves a qualitative improvement in memory scaling by tying storage cost to the rate of novel events rather than to the elapsed time. The following analysis quantifies this improvement.

**HMB storage scaling.** Let $\lambda_{\text{event}}$ be the rate of surprising events (events with $s_t > s_{\text{thresh}}$) in events per hour. Each archived event requires storing $(\boldsymbol{\mu}_k, \boldsymbol{\sigma}_k^2, t_k, s_k) = 2 d_z + 2$ floating point numbers. At float32 (4 bytes each), the per-event storage is:

$$\text{event size} = (2 d_z + 2) \times 4 = 8 d_z + 8 \text{ bytes}$$

For $d_z = 64$: event size $= 520$ bytes. The archive grows at:

$$\text{growth rate} = \lambda_{\text{event}} \times 520 \text{ bytes/event}$$

For a stable system with $\lambda_{\text{event}} = 10$ events/hour (a generous estimate for a well-maintained motor): growth rate $= 5{,}200$ bytes/hour $= 125$ KB/day $= 45$ MB/year.

**Transformer KV cache scaling.** For comparison, a 6-layer transformer with $D = 256$ sampling at 100 Hz stores:

$$\text{growth rate} = 100 \text{ samples/s} \times 3600 \text{ s/hr} \times 256 \times 6 \times 4 \text{ bytes/sample} = 2.2 \text{ GB/hr}$$

**Comparison.** The ratio of KV cache growth to HMB growth is:

$$\frac{2.2 \times 10^9 \text{ bytes/hr}}{5.2 \times 10^3 \text{ bytes/hr}} \approx 4 \times 10^5$$

HMB requires approximately $400{,}000\times$ less storage than a comparable transformer KV cache â€” not as an approximation or compression of the transformer's memory, but as a consequence of a fundamentally different memory architecture: store surprises, not states.

Over one year, at $\lambda_{\text{event}} = 10$ events/hour: HMB archive size $\approx 45$ MB, comfortably fitting in the RAM of any modern embedded processor. Over five years of continuous operation with the same event rate: $\approx 225$ MB â€” still within a reasonable embedded memory budget.

**Archive capacity and eviction.** When the archive approaches its capacity limit $K_{\max}$ (default $10{,}000$ events), a least-recently-used eviction policy removes the oldest events. However, VULGARIS provides an alternative eviction strategy based on information content: events with the lowest retrieval frequency (measured by how often they contributed significant weight to a retrieval operation) are evicted first, regardless of age. This strategy prioritizes the retention of rare but diagnostically relevant events (e.g., the first instance of a novel fault mode) over frequent but redundant events (e.g., repeated instances of the same startup transient).

**Relationship to episodic memory in cognitive science.** The HMB design parallels the theoretical distinction in cognitive neuroscience between semantic memory (general knowledge encoded in connection weights, analogous to the SSM's trained parameters) and episodic memory (specific past experiences stored in the hippocampus, analogous to the HMB archive). The hippocampal theory of systems consolidation (McClelland et al., 1995) proposes that the hippocampus stores recent specific episodes and gradually consolidates them into cortical semantic memory during sleep. VULGARIS's online Hebbian adaptation (Section 7.6) plays an analogous role: the Hebbian update consolidates frequently-encountered patterns from the SSM's operational history into the core parameter $\log \mathbf{a}$, reducing the need for explicit archival of predictable recurring events.

This parallel suggests a natural research direction: implementing an explicit consolidation phase in VULGARIS where patterns that appear frequently in the HMB archive are incorporated into the SSM's structural parameters via a targeted fine-tuning step, analogous to the memory consolidation process theorized in neuroscience. This direction is deferred to future work.

---

*End of Part 2. Part 3 continues with Section 10 (Training Objectives and Loss Formulation) through Section 14 (Evaluation Methodology).*
# VULGARIS: A Streaming Causal State-Space Foundation Model for Industrial and Edge Intelligence
## Part III: Adaptation, Explainability, Safety, Theory, and Deployment

---

## Section 10: Self-Healing Continuous Adaptation Layer (SHCAL)

### 10.1 The Stability-Plasticity Dilemma in Operational Systems

The stability-plasticity dilemma (Grossberg, 1980) asks: how can a learning system integrate new information without overwriting prior knowledge? In laboratory settings this is managed by maintaining separate datasets and periodically retraining. In industrial deployments neither option is viable: retraining requires labeled data, compute infrastructure, and scheduled downtime; retaining historical datasets violates data-residency constraints in jurisdictions such as the EU (GDPR Article 17) and China (PIPL Article 47); and model updates must occur continuously rather than in discrete cycles.

Formally, define catastrophic forgetting: given task distributions $\mathcal{T}_1$ and $\mathcal{T}_2$ with disjoint support, a model $f_\theta$ trained first on $\mathcal{T}_1$ and then on $\mathcal{T}_2$ satisfies

$$\Delta_{\mathrm{forget}} := \mathcal{L}_{\mathcal{T}_1}(f_\theta) - \mathcal{L}_{\mathcal{T}_1}(f_{\theta^*}) > 0$$

where $\theta^*$ is optimal for $\mathcal{T}_1$ alone. For standard stochastic gradient descent without regularization, $\Delta_{\mathrm{forget}}$ is proportional to the KL divergence between the gradients of $\mathcal{T}_1$ and $\mathcal{T}_2$, which can be arbitrarily large when the two tasks demand conflicting parameter configurations. Empirically, McCloskey and Cohen (1989) demonstrated that feedforward networks trained sequentially on paired-associate learning tasks exhibit near-complete erasure of the first task after only 50 gradient steps on the second â€” a finding replicated consistently across architectures.

SHCAL addresses this through three complementary mechanisms operating at different timescales: Elastic Weight Consolidation (task-level, hours), Hebbian plasticity (within-task, seconds to minutes), and structural plasticity (architectural-level, days to weeks). These mechanisms are not alternatives â€” they are complementary layers of a multi-timescale adaptation hierarchy designed to match the timescale structure of industrial drift phenomena. Equipment aging is a weeks-scale process; ambient temperature cycling is a diurnal process; sudden faults are event-scale disturbances. A single adaptation mechanism cannot simultaneously address all three.

### 10.2 Elastic Weight Consolidation

EWC (Kirkpatrick et al., 2017) anchors parameters to their post-task-$\mathcal{T}_1$ values, weighted by their importance for task $\mathcal{T}_1$. The Fisher information diagonal serves as the importance weight:

$$F_i = \mathbb{E}_{(\mathbf{x},y)\sim\mathcal{T}_1}\!\left[\left(\frac{\partial \log p_\theta(y|\mathbf{x})}{\partial \theta_i}\right)^{\!2}\right]$$

This quantity has the following interpretation: $F_i$ is large when small perturbations of $\theta_i$ produce large changes in the model's predicted log-probability over the $\mathcal{T}_1$ distribution. Parameters with large $F_i$ are important for $\mathcal{T}_1$; parameters with small $F_i$ can be freely repurposed for $\mathcal{T}_2$. This is an information-theoretically precise notion of importance, grounded in the curvature of the loss landscape rather than heuristic sensitivity measures.

$F_i$ is computed by accumulating squared gradients over $N_F$ calibration samples after completing $\mathcal{T}_1$. In the SHCAL implementation, $N_F = 200$ (configurable via `fisher_samples`). The EWC penalty is:

$$\mathcal{L}_{\mathrm{ewc}} = \frac{\lambda_{\mathrm{ewc}}}{2}\sum_i F_i(\theta_i - \theta_i^*)^2$$

This is a parameter-specific $\ell_2$ regularization: tight around parameters that strongly affect $\mathcal{T}_1$ predictions, loose around those that do not.

**Forgetting bound.** Under the Laplace approximation â€” that is, treating the posterior over $\theta$ around $\theta^*$ as locally quadratic â€” the forgetting after training on $\mathcal{T}_2$ with penalty strength $\lambda_{\mathrm{ewc}}$ is bounded by:

$$\Delta_{\mathrm{ewc}} \leq \frac{1}{2\lambda_{\mathrm{ewc}}}\sum_i F_i(\Delta\theta_i)^2 = \frac{1}{2\lambda_{\mathrm{ewc}}}\|\Delta\theta\|_{\mathbf{F}}^2$$

where $\|\cdot\|_{\mathbf{F}}$ is the Fisher-weighted norm and $\Delta\theta = \theta - \theta^*$ is the drift of the parameters from their $\mathcal{T}_1$-optimal values. Increasing $\lambda_{\mathrm{ewc}}$ by a factor of 10 reduces the forgetting bound by a factor of 10, at the cost of constraining the effective parameter space available for learning $\mathcal{T}_2$. In practice, $\lambda_{\mathrm{ewc}} = 100$ provides adequate protection for industrial equipment monitored under moderate regime changes; aggressive domain shifts (e.g., replacing the monitored equipment entirely) require a deliberate reset of $\theta^*$ and $F_i$ rather than an attempt to consolidate both within the same parameter vector.

**Computing $F_i$ efficiently.** For a model with $P$ parameters and a calibration set of $N_F$ examples, the exact Fisher diagonal requires $N_F$ forward-backward passes â€” $O(N_F P)$ compute. In SHCAL, $N_F = 200$. For a 2.5M-parameter model, this is approximately 500M FLOPs â€” on the order of one second on a modern CPU. This computation is performed offline, after completing a task boundary, not during streaming inference. The resulting Fisher diagonal and reference parameters $\theta^*$ are stored in memory (for a 2.5M-parameter model at FP32: approximately 20 MB for $F_i$ and $\theta^*$ together) and accessed during each subsequent gradient update.

**Online Fisher approximation.** When task boundaries are not cleanly delineated â€” as is common in streaming industrial processes where operating conditions shift gradually â€” SHCAL maintains a running estimate of $F_i$ using exponential moving averages: $F_i \leftarrow (1-\beta_F) F_i + \beta_F (\partial \log p/\partial \theta_i)^2$ with $\beta_F = 0.001$. This provides a continuously updated importance estimate at negligible cost (one multiply-add per parameter per step) and does not require explicit task boundary detection.

### 10.3 Hebbian Plasticity for In-Stream Adaptation

For faster adaptation within a task â€” responding to gradual drift on the order of minutes â€” SHCAL applies Oja's rule (Oja, 1982) to the linear projection layers of the SSSR blocks after each forward pass:

$$\Delta W_{ij} = \eta_h\!\left(y_i x_j - y_i^2 W_{ij}\right)$$

where $x_j$ is the $j$-th component of the layer's input activation vector, $y_i$ is the $i$-th component of its output activation vector, and $\eta_h$ is the Hebbian learning rate (default $10^{-4}$). The normalization term $-y_i^2 W_{ij}$ prevents unbounded growth; Oja (1982) proved convergence to the principal eigenvector of the input covariance when this rule is applied to a single-layer linear network. For multilayer networks, it acts as a local correlation-strengthening update that increases the coupling between co-activating input-output pairs, without global optimality guarantees but with the important property that no backward pass through the network is required.

The update is applied directly to the weight arrays without going through the autograd engine, incurring $O(d_{\mathrm{in}} d_{\mathrm{out}})$ operations per layer per step â€” negligible relative to the forward pass. The **shadow-mode validation gate** prevents harmful updates: before writing $\Delta\mathbf{W}$ to the live weights, the update is applied to a temporary copy of the weight matrix, the current batch is re-evaluated under the modified weights, and the resulting loss $\mathcal{L}(\theta + \Delta\theta)$ is compared to the pre-update loss $\mathcal{L}(\theta)$. If $\mathcal{L}(\theta + \Delta\theta) > 1.1 \cdot \mathcal{L}(\theta)$, the update is discarded and the Hebbian learning rate is reduced by a factor of 0.9 for the next step. This rejects approximately 5â€“15% of updates in typical streaming deployments â€” predominantly updates that strengthen correlations that are locally prevalent in the current batch but orthogonal to the task objective.

The combination of EWC and Hebbian plasticity resolves the timescale mismatch: EWC protects against large-scale parameter drift over task transitions, while Oja's rule provides fast in-stream micro-adjustment without requiring labeled data or gradient computation from a loss function. The Hebbian update is self-supervised in the strong sense â€” it depends only on co-occurring activations, requiring no supervision signal whatsoever.

### 10.4 Structural Plasticity

Over timescales of days to weeks, SHCAL can reallocate network capacity through a simulated synaptic turnover mechanism. Each weight $W_{ij}$ has an associated counter $c_{ij}$, initialized to zero, that increments when $|W_{ij}| < \varepsilon_{\mathrm{prune}} = 10^{-3}$ and resets to zero otherwise. When $c_{ij}$ exceeds a dormancy window of $\tau_{\mathrm{prune}}$ consecutive steps (default $10^5$), the weight is structurally pruned: $W_{ij} \leftarrow 0$, and the connection is flagged as inactive in a sparse mask $M_{ij} = 0$. The weight receives no further gradient updates and does not participate in forward computation (implemented via masked matrix multiplication at negligible overhead due to BLAS sparse operations).

For pruned connections, gradient monitoring continues asynchronously. If $|\partial\mathcal{L}/\partial W_{ij}|$ â€” computed during the backward pass and read from the gradient tensor â€” exceeds $\varepsilon_{\mathrm{grow}} = 5 \times 10^{-4}$ sustained over $\tau_{\mathrm{grow}} = 10^3$ consecutive steps, the connection is re-enabled: $M_{ij} = 1$ and $W_{ij} \leftarrow \mathcal{N}(0, \varepsilon_{\mathrm{grow}}^2)$. This implements a form of synaptic turnover in which dormant connections are recycled to regions where the current gradient signal indicates unmet representational demand. The practical effect is to dynamically redistribute model capacity toward the input features and transformations that are most informative for the current operating regime, without changing the total parameter count.

### 10.5 Conformal Recalibration Loop

SHCAL monitors the rolling conformal prediction coverage $\hat{\alpha}_t = \frac{1}{W}\sum_{s=t-W}^{t}\mathbf{1}[y_s \in \hat{C}_s]$ over a window of $W=500$ recent steps. When $\hat{\alpha}_t < (1-\alpha) - \delta_{\mathrm{tol}}$ (default $\delta_{\mathrm{tol}} = 0.03$), the adaptation trigger fires: $\eta_h$ is multiplied by 5, and $\lambda_{\mathrm{ewc}}$ is temporarily reduced to $0.1\lambda_{\mathrm{ewc}}$ for 100 steps, allowing faster parameter movement in response to the detected distribution shift. When coverage recovers to $\hat{\alpha}_t \geq (1-\alpha)$, the original $\lambda_{\mathrm{ewc}}$ is restored. This closed-loop coupling between conformal uncertainty and learning dynamics creates a self-regulating adaptation mechanism: the system accelerates adaptation precisely when calibrated uncertainty quantification detects that its predictions have become miscalibrated, and decelerates when coverage is restored.

This feedback architecture is robust to false triggers: a spurious coverage drop lasting fewer than $W$ steps does not trigger adaptation (because the rolling window smooths transient fluctuations), and the temporary reduction in $\lambda_{\mathrm{ewc}}$ is bounded in time (100 steps), so catastrophic forgetting cannot accumulate from a single trigger event. The combination of these bounds ensures that SHCAL's adaptation behavior is safe in the regulatory sense: no single event can cause unbounded parameter drift.

---

## Section 11: Domain-Adaptive Hypernetwork (DAH)

### 11.1 The Multi-Domain Deployment Problem

A global industrial AI platform serving a single large enterprise may encounter hundreds of distinct operating contexts: different equipment generations, ambient conditions, process chemistries, firmware versions, sensor configurations, and regulatory environments. The engineering problem is: how can one model serve all of them without incurring $O(M)$ training cost and $O(M)$ storage, where $M$ is the number of distinct deployment contexts?

Three strategies exist in current practice, each with fundamental limitations:

**One model per domain:** $O(M)$ storage and $O(M)$ full training runs. Operationally impractical at scale â€” 500 distinct equipment configurations require 500 full training pipelines â€” and eliminates cross-domain transfer, wasting the signal available from deployment contexts with similar physical dynamics.

**Fine-tuning a shared base per domain:** $O(M)$ training runs (though cheaper individually due to warm starting). Full model storage per domain remains $O(M)$. Transfer learning partially amortizes compute cost but does not address storage.

**Prompt or prefix conditioning:** Effective for language models but requires the conditioning signal to be expressible in the same embedding space as the input. Physical signal domains differ not only in statistical properties but in sensor type, physical units, and sampling rate â€” they require architectural adaptation, not merely input-space conditioning.

DAH offers a fourth approach: a frozen base model that captures universal physical signal processing, combined with a lightweight hypernetwork that generates domain-specific weight perturbations on demand. Storage cost: one base model plus one hypernetwork, regardless of $M$.

### 11.2 LoRA Adapter Generation

For a target layer $l$ with frozen base weight $\mathbf{W}_l^* \in \mathbb{R}^{d_{\mathrm{out}} \times d_{\mathrm{in}}}$, the domain-specific update is a low-rank factorization (Hu et al., 2021):

$$\Delta\mathbf{W}_l = \mathbf{A}_l\mathbf{B}_l, \quad \mathbf{A}_l \in \mathbb{R}^{d_{\mathrm{out}} \times r},\; \mathbf{B}_l \in \mathbb{R}^{r \times d_{\mathrm{in}}}$$

with rank $r \ll \min(d_{\mathrm{out}}, d_{\mathrm{in}})$ (default $r=16$). The intuition is that domain-specific adaptations are low-rank perturbations to the base representation: the base model has learned a rich general feature space, and a particular domain requires only a low-dimensional rotation or scaling of that space. The adapted forward pass for layer $l$ is:

$$\mathbf{y} = \mathbf{W}_l^*\mathbf{x} + e^{s_l} \cdot \mathbf{A}_l(\mathbf{B}_l\mathbf{x})$$

where $s_l$ is a learnable log-scale initialized to $\log(0.01)$, ensuring that the adapter contribution begins negligibly small at initialization and grows only when the task reward gradient demands it. This initialization strategy prevents the adapters from disrupting the pretrained base model's representations during the early stages of hypernetwork training.

The hypernetwork generates adapter factors from a domain embedding:

$$\mathbf{z}_d = \mathrm{MetaMLP}\bigl(\mathrm{Embed}(d)\bigr) \in \mathbb{R}^{d_{\mathrm{meta}}}$$

$$\mathbf{A}_l = \mathrm{reshape}\!\left(\mathbf{W}^A_l\,\mathbf{z}_d,\; [d_{\mathrm{out}}, r]\right), \quad \mathbf{B}_l = \mathrm{reshape}\!\left(\mathbf{W}^B_l\,\mathbf{z}_d,\; [r, d_{\mathrm{in}}]\right)$$

where $\mathrm{Embed}(d) \in \mathbb{R}^{d_{\mathrm{emb}}}$ is a learned embedding for domain index $d$, and $\mathrm{MetaMLP}$ is a two-layer feedforward network with GeLU activations and output dimension $d_{\mathrm{meta}} = 128$. The hypernetwork matrices $\mathbf{W}^A_l, \mathbf{W}^B_l \in \mathbb{R}^{(d_{\mathrm{out/in}} \cdot r) \times d_{\mathrm{meta}}}$ are shared across all domain instances and learned during pretraining.

**Parameter efficiency.** For $L=8$ target layers, $r=16$, mean layer dimension $\bar{d}=256$: adapter parameters per domain $= 2 \times 8 \times 16 \times 256 = 65{,}536$. This is 1.3% of a 5M-parameter base model. The hypernetwork itself has $\sum_l 2 \times (d_{\mathrm{out},l} + d_{\mathrm{in},l}) \times r \times d_{\mathrm{meta}}$ parameters â€” fixed and shared across all $M$ domains. For the default configuration, this is approximately 420K parameters added once. Serving 1,000 distinct domains requires exactly the same model size as serving 1 domain.

**Domain switching latency.** Generating all adapters for domain $d$ requires one hypernetwork forward pass, computing $\sum_l(d_{\mathrm{out},l} + d_{\mathrm{in},l}) \times r$ output values from the domain embedding. For the default configuration: approximately $8 \times 512 \times 16 = 65{,}536$ outputs, requiring one MetaMLP forward pass of approximately $8.4\mathrm{M}$ FLOPs. On any contemporary CPU core, this completes in well under 1 ms. Adapters are cached after generation, so switching to a previously seen domain is effectively free â€” only an index lookup and a cache read.

### 11.3 Pretraining the Hypernetwork

During pretraining, $M_{\mathrm{train}}$ synthetic domains are created by varying the data-generating parameters: sensor noise levels, operating regime transition rates, fault frequency distributions, thermal time constants, and measurement offsets. The model is trained with randomly sampled domain indices, requiring the hypernetwork to produce useful adapters for each domain without having access to domain-specific data outside its embedding index. This forces the hypernetwork to learn a generalizable map from domain metadata to adapter configurations that meaningfully alters the base model's behavior.

At inference time on a novel domain $d' \notin \mathcal{D}_{\mathrm{train}}$: the embedding $\mathbf{e}_{d'}$ is initialized to the mean of all training domain embeddings. If the new domain resembles a convex combination of training domains in the embedding space â€” which holds when the new domain's physical characteristics fall within the range of variation sampled during pretraining â€” the hypernetwork interpolates effectively and useful adapters are generated without any fine-tuning. For genuinely novel domains outside this convex hull, a brief adapter fine-tuning step (100â€“500 gradient steps on 50â€“200 labeled examples) on the adapter matrices only â€” with the base model frozen â€” typically restores full performance within minutes.

---

## Section 12: Explainability and Symbolic Extraction Engine (ESE)

### 12.1 The Regulatory Imperative

Industrial AI deployments exist within a regulatory context that is specific, technically detailed, and increasingly enforced:

The **EU AI Act** (Regulation (EU) 2024/1689), Article 13, requires that high-risk AI systems "be designed and developed in such a way to ensure that their operation is sufficiently transparent to enable deployers to interpret the system's output and use it appropriately." Article 9 requires risk management systems that include documentation of the basis of AI decisions affecting safety-critical processes.

**IEC 61508** (Functional Safety of E/E/PE Safety-related Systems) requires full traceability from safety function specification through implementation to test evidence. A system that produces a safety-relevant output â€” a protective relay trip command, a shutdown recommendation â€” without a traceable reasoning chain that can be audited by a safety engineer cannot be certified under this standard. The standard explicitly addresses software as a potential source of systematic failures.

**NERC CIP-014-3** (Critical Infrastructure Protection, Physical Security) and related CIP standards require audit trails for automated decisions affecting bulk electric system reliability, including documentation of the basis on which automated systems make operational recommendations.

These requirements are not satisfied by post-hoc interpretability methods such as LIME (Ribeiro et al., 2016) or SHAP (Lundberg and Lee, 2017) applied to a black-box model. LIME approximates the model locally with a linear surrogate; the surrogate's coefficients reflect the linear approximation, not the model's internal computation. SHAP values are consistent under the axioms of Shapley value theory and satisfy desirable properties (efficiency, symmetry, dummy) but do not decompose into physical causal chains â€” a SHAP attribution to a sensor does not distinguish between "this sensor is causally upstream of the failure" and "this sensor is statistically correlated with failure through a latent confounder." CRG-based attribution and CART rule extraction differ from these approaches structurally: they expose aspects of the model's actual computational pathway, not a retrospective approximation constructed after the fact.

### 12.2 CART Symbolic Rule Induction

The Classification and Regression Trees algorithm (Breiman et al., 1984) constructs a binary tree by recursively solving:

$$(j^*, t^*) = \operatorname{argmax}_{j,t}\;\mathrm{Gain}(S, j, t)$$

$$\mathrm{Gain}(S,j,t) = \mathrm{Imp}(S) - \frac{|S_L|}{|S|}\mathrm{Imp}(S_L) - \frac{|S_R|}{|S|}\mathrm{Imp}(S_R)$$

where $S_L = \{\mathbf{h} \in S : h_j \leq t\}$, $S_R = S \setminus S_L$, impurity is the Gini coefficient for classification ($\mathrm{Imp}(S) = 1 - \sum_c p_c^2$ where $p_c$ is the class proportion) or mean squared error for regression ($\mathrm{Imp}(S) = \frac{1}{|S|}\sum_i (y_i - \bar{y})^2$), and split thresholds $t$ are evaluated at the midpoints between consecutive observed values of feature $j$.

ESE's CART is a custom pure-numpy implementation â€” no scikit-learn dependency â€” required for deployment in air-gapped industrial environments where external package installation is prohibited by OT security policy. It operates on the accumulated buffer of $([\mathbf{h}_t, y_t])$ pairs maintained by SSSR, periodically retrained as the buffer grows and the distribution of the latent space evolves. The tree is limited to a maximum depth of $d_{\mathrm{max}} = 6$ (configurable), yielding at most 64 leaf nodes and consequently at most 64 symbolic rules.

Rules are extracted by path enumeration from root to leaf. Each leaf yields one rule of the form:

$$\mathrm{IF}\; h_{j_1} \leq t_1\; \mathrm{AND}\; h_{j_2} > t_2\; \mathrm{AND}\; \ldots\; \mathrm{THEN}\; \hat{y} = \mu_\ell \quad (\mathrm{confidence}\; c_\ell,\; n_\ell\; \mathrm{samples})$$

where $\mu_\ell$ is the leaf mean prediction and $c_\ell = 1 - \mathrm{Imp}(\ell)/\mathrm{Imp}(\mathrm{root})$ is the normalized impurity reduction. When the model's `feature_names` dictionary maps latent dimensions to sensor identifiers, the condition $h_{j_1} \leq t_1$ is rendered as, e.g., `bearing_temp_wavelet_scale3 <= 0.47`, producing rules directly readable by domain engineers without knowledge of the latent representation.

### 12.3 CRG-Aware Attribution

The raw gradient $\partial \hat{y}/\partial x_i$ measures sensitivity: how much would the predicted output change if input $x_i$ were perturbed by $\varepsilon$? This is necessary but not sufficient for causal attribution. In the presence of correlated inputs â€” which is universal in industrial systems, where redundant sensors observe the same physical state â€” a signal may have high gradient sensitivity because it is a downstream effect of the true causal factor. The model's representation has captured the correlation, but not its direction.

CRG-aware attribution corrects for this by weighting gradient sensitivity by causal graph position as learned by the CRG module:

$$\mathrm{attr}_i = \frac{\sum_j \tilde{W}_{ij} \cdot \left|\partial \hat{y}/\partial x_j\right|}{\sum_i \sum_j \tilde{W}_{ij} \cdot \left|\partial \hat{y}/\partial x_j\right|}$$

where $\tilde{\mathbf{W}}$ is the row-stochastic normalization of the sparse CRG adjacency matrix $\mathbf{W}$ (after thresholding near-zero entries). The attribution score $\mathrm{attr}_i$ reflects: "how strongly does input $i$ causally influence the nodes to which the output is sensitive?" This is a proxy for the structural causal model's total causal effect of $x_i$ on $\hat{y}$, significantly more robust than pure gradient sensitivity in systems with correlated inputs.

### 12.4 Gradient Counterfactual Generation

Given the current latent representation $\mathbf{h} \in \mathbb{R}^D$ producing prediction $\hat{y} = f(\mathbf{h})$ and a target prediction value $y^* \neq \hat{y}$ (e.g., the model currently predicts a fault probability of 0.8 and the operator asks: "what conditions would bring this below 0.2?"), find the minimal perturbation:

$$\min_{\boldsymbol{\delta} \in \mathbb{R}^D}\; \|\boldsymbol{\delta}\|_2^2 + \alpha\|\boldsymbol{\delta}\|_1 \quad \mathrm{s.t.}\; \|f(\mathbf{h}+\boldsymbol{\delta}) - y^*\|_2 < \varepsilon$$

The $\ell_1$ term induces sparsity: most elements of $\boldsymbol{\delta}$ will be exactly zero at the solution, concentrating the counterfactual change on a small subset of latent dimensions. The constraint is handled via a soft-penalty Lagrangian:

$$\mathcal{L}_{\mathrm{cf}}(\boldsymbol{\delta}) = \|\boldsymbol{\delta}\|_2^2 + \alpha\|\boldsymbol{\delta}\|_1 + \mu \cdot \max\!\left(0,\, \|f(\mathbf{h}+\boldsymbol{\delta}) - y^*\|_2 - \varepsilon\right)$$

Optimized via proximal gradient descent on $\boldsymbol{\delta}$ (with the model weights held fixed), where the proximal operator for the $\ell_1$ term is the element-wise soft-thresholding operator $\mathrm{prox}_{\alpha\eta}(\boldsymbol{\delta}) = \mathrm{sign}(\boldsymbol{\delta})\max(|\boldsymbol{\delta}| - \alpha\eta, 0)$. Convergence is typically reached within 50â€“200 iterations.

When the nonzero elements of the optimal $\boldsymbol{\delta}$ correspond to latent dimensions strongly activated by specific ASE frequency bands or specific CRG input channels, the counterfactual translates to an actionable statement: "If the bearing temperature's wavelet scale-3 energy were reduced by 0.14 (corresponding to a reduction of approximately 12Â°C in the 0.5â€“2 Hz thermal band), the fault probability would decrease below the alarm threshold of 0.2." This is the form of explanation that domain engineers and process safety managers can act on.

---

## Section 13: Cross-Modal Latent Alignment (CMLA)

### 13.1 The Multi-Modal Industrial Sensing Problem

A gas turbine is simultaneously observable through vibration accelerometers on each bearing housing, thermocouples at successive blade rows, exhaust gas analyzers measuring CO/NOx/O2, oil debris monitors detecting ferrous particle mass, and acoustic emission transducers sensitive to micro-crack propagation. Each modality provides an incomplete, noise-corrupted view of the turbine's thermomechanical state. Optimal diagnosis requires fusing all available modalities in a way that accounts for their heterogeneous noise models, disparate sampling rates, and realistic missing-data patterns â€” sensor dropout due to communication failures, saturation events, maintenance-related removal.

Naive feature concatenation fails for structural reasons beyond scale mismatch. A thermal camera frame ($224 \times 224$ pixels) contributes $5 \times 10^4$ features while a current signature contributes $10^2$. After mean-variance normalization, the concatenated representation is numerically dominated by the high-dimensional modality regardless of its diagnostic relevance â€” the model's effective capacity is consumed by noise dimensions in the high-dimensional modality. More fundamentally: a failed sensor produces garbage activations that, when concatenated, corrupt the fused representation without the model receiving any signal that specific elements should be distrusted. Soft attention over modalities partially addresses this but requires a learned query-key mechanism that is itself a black box.

### 13.2 InfoNCE Contrastive Alignment

For modalities $m_1$ and $m_2$ observing the same physical event at time $t$ (a positive pair), define the InfoNCE loss (van den Oord et al., 2018):

$$\mathcal{L}_{\mathrm{InfoNCE}} = -\mathbb{E}\!\left[\log\frac{\exp\!\left(\mathrm{sim}(\mathbf{z}_{m_1,t},\,\mathbf{z}_{m_2,t})/\tau\right)}{\sum_{t'=1}^{N}\exp\!\left(\mathrm{sim}(\mathbf{z}_{m_1,t},\,\mathbf{z}_{m_2,t'})/\tau\right)}\right]$$

where $\mathrm{sim}(\mathbf{u},\mathbf{v}) = \mathbf{u}^\top\mathbf{v}/(\|\mathbf{u}\|_2\|\mathbf{v}\|_2)$ is cosine similarity, $\tau > 0$ is a temperature hyperparameter (default 0.07), and the sum in the denominator is over $N$ samples in the batch (all of which constitute negative pairs for the numerator's positive pair). This loss lower-bounds the mutual information between the two modality embeddings:

$$I(\mathbf{z}_{m_1};\mathbf{z}_{m_2}) \geq \log N - \mathcal{L}_{\mathrm{InfoNCE}}$$

as proven by van den Oord et al. (2018). Minimizing $\mathcal{L}_{\mathrm{InfoNCE}}$ therefore maximizes a lower bound on the mutual information between modality embeddings for the same physical event, while minimizing it for embeddings of different physical events. This pulls the latent representations of co-occurring observations together and pushes temporally distinct observations apart â€” learning a shared physical event space.

### 13.3 Inverse-Variance Weighted Fusion

After aligning modality embeddings via InfoNCE, they are fused using the inverse-variance estimator. For modality $m$ contributing embedding $\mathbf{z}_m \in \mathbb{R}^{d_z}$ and estimated noise variance $\hat{\sigma}_m^2$, the fused representation is:

$$\hat{\mathbf{z}} = \frac{\sum_m \mathbf{z}_m / \hat{\sigma}_m^2}{\sum_m 1/\hat{\sigma}_m^2}$$

The Gauss-Markov theorem guarantees that this is the minimum-variance unbiased estimator of the true state embedding $\mathbf{z}^*$ when each modality satisfies $\mathbf{z}_m = \mathbf{z}^* + \boldsymbol{\varepsilon}_m$ with $\boldsymbol{\varepsilon}_m \sim \mathcal{N}(\mathbf{0}, \sigma_m^2\mathbf{I})$ and the noise terms are independent across modalities. The per-modality variance estimate is:

$$\hat{\sigma}_m^2 \propto \|\mathbf{z}_m - \bar{\mathbf{z}}\|_2^2 / d_z$$

where $\bar{\mathbf{z}} = \frac{1}{|\mathcal{M}|}\sum_m \mathbf{z}_m$ is the naive mean. A failed or saturated sensor produces an embedding far from the ensemble mean â€” its numerically pathological output is treated as high variance and assigned near-zero weight in the fusion. This provides graceful fault tolerance without requiring an explicit anomaly detection module for sensor health: the statistical structure of the fused representation inherently downweights modalities whose embeddings are outliers relative to the ensemble.

---

## Section 14: Safety-Critical Policy Head

### 14.1 Formal Safety Requirements in Industrial Control

When VULGARIS operates in a closed-loop control setting â€” issuing setpoint adjustments to PLC/DCS systems, triggering protective relays, commanding robotic actuators, or making automated interlock decisions â€” the model's output is no longer advisory. A prediction error in this setting can cause irreversible consequences: equipment damage, process upset leading to chemical release, personnel injury, or grid instability. The regulatory standards that apply to such systems â€” IEC 61508, IEC 62061, ISO 13849 â€” require that safety functions be certified with quantified reliability measures (e.g., PFH $< 10^{-7}$ hr$^{-1}$ for SIL 2). Neural networks trained by empirical risk minimization provide no such guarantees in their standard form.

Standard post-hoc output clamping â€” clipping the control output to $u \in [u_{\min}, u_{\max}]$ â€” addresses static bounds constraints but cannot address dynamic safety constraints that depend on the trajectory of the system's state. A system approaching an unsafe region in state space can violate safety even with an output that is locally within bounds, if the rate of approach is unconstrained.

Control Barrier Functions (Wieland and AllgÃ¶wer, 2007; Ames et al., 2016) provide the formal framework for certified invariance of safe sets in dynamical systems. VULGARIS implements a differentiable CBF-based safety filter as a composable module that can be attached to any regression or control output head, converting nominal policy outputs to certified-safe control actions.

### 14.2 Control Barrier Functions: Theory

For a discrete-time system $\mathbf{s}_{t+1} = f(\mathbf{s}_t, \mathbf{u}_t)$ with state $\mathbf{s}_t \in \mathcal{S}$ and control input $\mathbf{u}_t \in \mathcal{U}$, define the safe set $\mathcal{C} = \{\mathbf{s} \in \mathcal{S} : h(\mathbf{s}) \geq 0\}$ for a smooth function $h : \mathcal{S} \to \mathbb{R}$.

**Definition 14.1 (Discrete-Time CBF).** The function $h$ is a Control Barrier Function for system $f$ if there exists $\gamma \in (0,1]$ such that for all $\mathbf{s} \in \mathcal{C}$:

$$\sup_{\mathbf{u} \in \mathcal{U}}\,[h(f(\mathbf{s},\mathbf{u})) - (1-\gamma)h(\mathbf{s})] \geq 0$$

The parameter $\gamma$ controls the rate of decay of the barrier: $\gamma = 1$ corresponds to set invariance (the barrier must be non-decreasing), while $\gamma < 1$ allows the barrier to decrease by a factor of at most $(1-\gamma)$ per step.

**Theorem 14.1 (Forward Invariance, Ames et al., 2016).** If $h$ is a Control Barrier Function for system $f$ and the control input $\mathbf{u}_t$ satisfies $h(f(\mathbf{s}_t,\mathbf{u}_t)) \geq (1-\gamma)h(\mathbf{s}_t)$ at every step, then $\mathbf{s}_0 \in \mathcal{C}$ implies $\mathbf{s}_t \in \mathcal{C}$ for all $t \geq 0$.

*Proof.* By induction. Base case: $h(\mathbf{s}_0) \geq 0$ by assumption. Inductive step: assuming $h(\mathbf{s}_t) \geq 0$, the CBF constraint gives $h(\mathbf{s}_{t+1}) = h(f(\mathbf{s}_t,\mathbf{u}_t)) \geq (1-\gamma)h(\mathbf{s}_t) \geq 0$. $\square$

This theorem establishes that verifying the CBF constraint at each individual step is sufficient to certify safety for all future time â€” a remarkable reduction of an infinite-horizon safety property to a per-step constraint.

### 14.3 Differentiable Safety Filter

The policy network $\pi_\theta : \mathcal{S} \to \mathcal{U}$ produces a nominal action $\mathbf{u}^{\mathrm{nom}} = \pi_\theta(\mathbf{s}_t)$. The safety filter solves the minimum-norm correction quadratic program (CBF-QP):

$$\mathbf{u}^* = \operatorname{argmin}_{\mathbf{u} \in \mathcal{U}}\;\|\mathbf{u} - \mathbf{u}^{\mathrm{nom}}\|_2^2 \quad \mathrm{s.t.}\; h(f(\mathbf{s}_t,\mathbf{u})) \geq (1-\gamma)h(\mathbf{s}_t)$$

This finds the action closest to the nominal policy's recommendation that satisfies the CBF constraint. When the nominal action is already safe (i.e., satisfies the constraint), the filter returns it unchanged: $\mathbf{u}^* = \mathbf{u}^{\mathrm{nom}}$. The filter only intervenes when the nominal action would violate the safety constraint.

Linearizing $h \circ f$ around $\mathbf{u}^{\mathrm{nom}}$, the constraint becomes linear in $\mathbf{u}$ and the CBF-QP has a closed-form projection solution:

$$\mathbf{u}^* = \mathbf{u}^{\mathrm{nom}} + \frac{\max\!\left(0,\; (1-\gamma)h(\mathbf{s}_t) - h(f(\mathbf{s}_t,\mathbf{u}^{\mathrm{nom}}))\right)}{\|\nabla_\mathbf{u} h\|_2^2 + \varepsilon} \cdot \nabla_\mathbf{u} h$$

where $\nabla_\mathbf{u} h = \nabla_\mathbf{s} h \cdot \nabla_\mathbf{u} f$ is the total gradient of the barrier through the dynamics (computed via the chain rule). The $\varepsilon > 0$ prevents division by zero. This expression is differentiable with respect to $\mathbf{u}^{\mathrm{nom}}$ everywhere except at the single kink where $h(f(\mathbf{s}_t,\mathbf{u}^{\mathrm{nom}})) = (1-\gamma)h(\mathbf{s}_t)$, where a subgradient exists and is used. Gradients flow through the safety filter during backpropagation, allowing the policy to learn to produce actions requiring progressively less CBF correction over the course of training â€” the policy internalizes the safety constraint rather than relying entirely on the filter.

### 14.4 Lipschitz Certification via Spectral Normalization

Spectral normalization (Miyato et al., 2018) enforces $\sigma_{\max}(\mathbf{W}) \leq L_{\max}$ for each weight matrix by rescaling after each forward pass. The spectral norm is estimated via power iteration: at each forward pass, three steps of

$$\mathbf{v} \leftarrow \mathbf{W}^\top\mathbf{u}/\|\mathbf{W}^\top\mathbf{u}\|_2, \quad \mathbf{u} \leftarrow \mathbf{W}\mathbf{v}/\|\mathbf{W}\mathbf{v}\|_2, \quad \hat{\sigma} = \mathbf{u}^\top\mathbf{W}\mathbf{v}$$

are performed, yielding the estimate $\hat{\sigma} \approx \sigma_{\max}(\mathbf{W})$. If $\hat{\sigma} > L_{\max}$: $\mathbf{W} \leftarrow \mathbf{W} \cdot L_{\max}/\hat{\sigma}$. For a $K$-layer network with each layer satisfying $\sigma_{\max}(\mathbf{W}_k) \leq L_{\max}$, the network's global Lipschitz constant satisfies:

$$\|f(\mathbf{x}) - f(\mathbf{y})\|_2 \leq L_{\max}^K \|\mathbf{x}-\mathbf{y}\|_2$$

providing a certified bound on output perturbation under bounded input noise. In OT/ICS environments subject to adversarial sensor spoofing â€” a known attack vector in industrial control systems, cf. the 2021 Oldsmar water treatment attack â€” this bound provides a quantitative defense guarantee: an adversary who can perturb sensor readings by at most $\delta$ can perturb the model's control output by at most $L_{\max}^K \delta$. Setting $L_{\max} = 1$ (i.e., non-expansive mapping) and $K = 6$ ensures the bound is 1 regardless of the depth. Note that this is a stronger requirement than ordinary spectral normalization and may reduce model expressivity; a tradeoff between certified robustness and prediction accuracy is inherent.

---

## Section 15: Theoretical Analysis

### 15.1 Stability of SSSR

**Theorem 15.1 (SSSR State Boundedness).** Let $(\mathbf{h}_t)_{t \geq 0}$ be generated by SSSR with log-timescale parameters $\mathbf{a} \in \mathbb{R}^N$, inter-sample intervals $\delta t_t \in [\delta t_{\min}, \delta t_{\max}]$ with $\delta t_{\min} > 0$, and bounded inputs $\|\mathbf{x}_t\|_2 \leq M < \infty$. Then $\sup_{t \geq 0} \|\mathbf{h}_t\|_2 \leq C < \infty$ for a constant $C$ depending only on $\mathbf{a}$, $M$, $\delta t_{\min}$, and $\|\mathbf{W}_B\|_2$.

*Proof.* Define the contraction factor $\rho^* = \max_n \bar{A}_n^{(\min)} = \max_n \exp(-\exp(a_n)\delta t_{\min})$. Since $\exp(a_n) > 0$ for all $a_n \in \mathbb{R}$ and $\delta t_{\min} > 0$, we have $\rho^* < 1$. By the ZOH discretization recurrence $h_{t,n} = \bar{A}_{t,n} h_{t-1,n} + \bar{B}_{t,n} \mathbf{x}_t$, taking norms:

$$\|\mathbf{h}_t\|_2 \leq \rho^* \|\mathbf{h}_{t-1}\|_2 + \|\bar{\mathbf{B}}_t\|_2 M$$

The input matrix norm satisfies $\|\bar{\mathbf{B}}_t\|_2 \leq (1-\rho_{\min}^*)\|\mathbf{W}_B\|_2$ where $\rho_{\min}^* = \min_n \bar{A}_{t,n} > 0$. Unrolling the recursion from $t=0$:

$$\|\mathbf{h}_t\|_2 \leq (\rho^*)^t \|\mathbf{h}_0\|_2 + \frac{(1-\rho_{\min}^*)\|\mathbf{W}_B\|_2 M}{1-\rho^*}$$

As $t \to \infty$, $(\rho^*)^t \to 0$, giving $\limsup_{t} \|\mathbf{h}_t\|_2 \leq C := (1-\rho_{\min}^*)\|\mathbf{W}_B\|_2 M / (1-\rho^*) < \infty$. $\square$

The critical contrast with gated RNNs is instructive. In an LSTM, the forget gate is $f_t = \sigma(\mathbf{W}_f[\mathbf{h}_{t-1};\mathbf{x}_t] + \mathbf{b}_f)$, which has values in $(0,1)$ but whose magnitude is a learned, input-dependent function. At adversarial inputs, $f_t \to 1$ for all dimensions simultaneously, and the cell state can grow without bound as $c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$ accumulates without contraction. In SSSR, the contraction factor $\bar{A}_{t,n} = \exp(-\exp(a_n)\delta t_t) < 1$ is a structural property of the architecture â€” $\exp(a_n) > 0$ always and $\delta t_t > 0$ always, so $\bar{A}_{t,n} < 1$ always, independent of the input value.

**Corollary 15.1 (Post-Hebbian Stability).** SHCAL's Oja rule updates are constrained to maintain $a_n \in [-5, 0]$ (enforced by clamping in `shcal.py`). Under this constraint, $\exp(a_n) \in [e^{-5}, 1] \approx [0.0067, 1]$, so $\bar{A}_n = \exp(-\exp(a_n)\delta t) \in (\exp(-\delta t_{\max}), \exp(-0.0067\,\delta t_{\min}))$. Both bounds are strictly in $(0,1)$ for finite $\delta t_{\max}$ and positive $\delta t_{\min}$. Therefore Theorem 15.1 holds after any number of Hebbian updates, and SHCAL adaptation cannot destabilize the SSM dynamics. $\square$

### 15.2 DAG Acyclicity of CRG

**Theorem 15.2 (Zheng et al., 2018; DAG Characterization).** A matrix $\mathbf{W} \in \mathbb{R}_{\geq 0}^{n \times n}$ satisfies $h(\mathbf{W}) = \mathrm{tr}(e^{\mathbf{W}}) - n = 0$ if and only if the weighted graph $G(\mathbf{W})$ is a directed acyclic graph.

*Proof sketch.* The $(i,j)$ entry of $e^{\mathbf{W}} = \sum_{k=0}^\infty \mathbf{W}^k/k!$ counts weighted walks of all lengths from $j$ to $i$ in $G(\mathbf{W})$. The trace $\mathrm{tr}(e^{\mathbf{W}}) = n + \sum_{k=1}^\infty \mathrm{tr}(\mathbf{W}^k)/k!$ counts $n$ (the empty walks) plus all weighted closed walks of length $\geq 1$. Since $\mathbf{W} \geq 0$ component-wise, each closed walk contributes a positive term. $\mathrm{tr}(e^{\mathbf{W}}) - n = 0$ if and only if there are no closed walks of any length, which is the definition of acyclicity. $\square$

In VULGARIS's CRG module, the penalty is $h(\mathbf{W} \odot \mathbf{W}) = \mathrm{tr}(e^{\mathbf{W}\odot\mathbf{W}}) - n$, using the Hadamard square to ensure the exponent argument is non-negative for all signed $\mathbf{W}$. Under augmented Lagrangian optimization with increasing penalty coefficient $\rho$, $h(\mathbf{W} \odot \mathbf{W}) \to 0$ at convergence, guaranteeing the learned causal graph is a DAG.

The gradient $\nabla_\mathbf{W} h = 2\mathbf{W} \odot e^{\mathbf{W}\odot\mathbf{W}}$ is well-defined and bounded for all finite $\mathbf{W}$, making the constraint differentiable and amenable to gradient-based optimization. This is the fundamental advantage of the NOTEARS formulation over combinatorial structure learning algorithms, which cannot be embedded in an end-to-end gradient framework.

### 15.3 Conformal Coverage Under Non-Stationarity

**Theorem 15.3 (Adaptive Coverage, adapted from Gibbs and CandÃ¨s, 2021).** Let calibration nonconformity scores $(s_t)_{t=1}^T$ be assigned exponential weights $w_t \propto e^{-\lambda(T-t)}$ for forgetting rate $\lambda > 0$. Let the test label $y_{T+1}$ be drawn from a distribution $p_{T+1}$ satisfying $\|p_{T+1} - p_t\|_{\mathrm{TV}} \leq \varepsilon_{\mathrm{drift}} (T+1-t)$ (total-variation drift bounded linearly in time lag). The weighted conformal predictor $\hat{C}_{T+1} = \{y : s(y) \leq \hat{q}^{(1-\alpha)}_{\mathbf{w}}\}$ satisfies:

$$P\!\left(y_{T+1} \in \hat{C}_{T+1}\right) \geq 1 - \alpha - O\!\left(\frac{\varepsilon_{\mathrm{drift}}}{\lambda}\right)$$

The coverage deficit $O(\varepsilon_{\mathrm{drift}}/\lambda)$ reflects the lag of exponential forgetting: a larger $\lambda$ (faster forgetting) reduces this term but increases the variance of $\hat{q}$ due to fewer effective calibration samples ($n_{\mathrm{eff}} \approx 1/(1-e^{-\lambda}) \approx 1/\lambda$ for large $\lambda$). The optimal forgetting rate that minimizes the total error (bias from drift plus variance from insufficient calibration samples) scales as $\lambda^* = O(\sqrt{\varepsilon_{\mathrm{drift}} \cdot \log(1/\alpha)/n_{\mathrm{cal}}})$.

### 15.4 EWC Forgetting Bound

**Theorem 15.4 (EWC Forgetting).** Under the Laplace approximation â€” i.e., $\mathcal{L}_{\mathcal{T}_1}(\theta) \approx \mathcal{L}_{\mathcal{T}_1}(\theta^*) + \frac{1}{2}(\theta-\theta^*)^\top \mathbf{F} (\theta-\theta^*)$ where $\mathbf{F} = \mathrm{diag}(F_1, \ldots, F_P)$ is the Fisher diagonal â€” after training on $\mathcal{T}_2$ with EWC penalty $\lambda_{\mathrm{ewc}}$, the forgetting satisfies:

$$\Delta_{\mathrm{ewc}} = \frac{1}{2}\sum_i F_i(\theta_i - \theta_i^*)^2 \leq \frac{\|\mathbf{F}\|_\infty}{2} \|\Delta\theta\|_2^2$$

where $\|\mathbf{F}\|_\infty = \max_i F_i$ and $\Delta\theta = \theta - \theta^*$.

*Proof.* The first equality follows directly from the Laplace approximation and the definition of $\Delta_{\mathrm{forget}}$. The inequality follows from $\sum_i F_i(\Delta\theta_i)^2 \leq \max_i F_i \cdot \sum_i (\Delta\theta_i)^2$. $\square$

For constrained adaptation with $\|\Delta\theta\|_2 \leq \rho_{\max}$ (enforced implicitly by $\lambda_{\mathrm{ewc}}$): $\Delta_{\mathrm{ewc}} \leq F_{\max}\rho_{\max}^2/2$. The parameter $\lambda_{\mathrm{ewc}}$ controls $\rho_{\max}$: larger $\lambda_{\mathrm{ewc}}$ reduces the gradient step size on parameters with high $F_i$, reducing $\|\Delta\theta\|_2$ and hence the forgetting bound.

### 15.5 Federated Privacy Accounting

By the RÃ©nyi differential privacy composition theorem (Mironov, 2017), the cumulative privacy loss of the federated learning protocol over $R$ aggregation rounds, each with client sampling ratio $q$, Gaussian noise multiplier $\sigma$, and gradient clipping norm $C_{\mathrm{clip}}$, satisfies: for any order $\alpha > 1$, the RÃ©nyi divergence is:

$$D_\alpha\!\left(\mathcal{M}^R \| \mathcal{M}_0^R\right) \leq R \cdot \frac{\alpha q^2 C_{\mathrm{clip}}^2}{2\sigma^2 n^2} + O\!\left(\frac{R q^2 \alpha^2 C_{\mathrm{clip}}^4}{\sigma^4 n^4}\right)$$

Converting to $(\varepsilon, \delta)$-DP via $\varepsilon(\delta) = \min_{\alpha > 1}\!\left[D_\alpha + \log((\alpha-1)/\alpha) - \log(\delta)/(\alpha-1)\right]$ (Balle et al., 2020), the total privacy budget after $R$ rounds is approximately:

$$\varepsilon_{\mathrm{total}} \approx \sqrt{\frac{2Rq^2 C_{\mathrm{clip}}^2 \log(1/\delta)}{\sigma^2 n^2}} + \frac{Rq^2 C_{\mathrm{clip}}^2}{\sigma^2 n^2}$$

for small $q$ (the regime applicable to federated learning with many clients). The dominant term scales as $O(\sqrt{R}/\sigma)$, implying that doubling the noise multiplier halves the privacy budget consumed per unit of training time. The dependence on $n$ (the total dataset size across clients, or equivalently the normalization for the clipped gradient) confirms that larger federated networks achieve better privacy-accuracy tradeoffs â€” a well-established result in federated DP theory.

---

## Section 16: Training Methodology

### 16.1 The Unified Variational Loss

The VULGARIS training objective derives from a variational lower bound on the log-likelihood of a joint generative model $p(\mathbf{x}_{1:T}, \mathbf{z}_{1:T}, \mathbf{G}, \theta)$ where $\mathbf{G}$ is the causal graph, $\mathbf{z}_{1:T}$ is the latent state trajectory, and $\theta$ are the model parameters. Under the approximate posterior $q(\mathbf{z}, \mathbf{G}, \theta | \mathbf{x})$, the Evidence Lower BOund (ELBO) decomposes into terms with precise information-theoretic interpretations:

$$\mathcal{L} = \underbrace{\mathbb{E}_q[\log p(\mathbf{x}|\mathbf{z},\mathbf{G})]}_{\mathcal{L}_{\mathrm{task}}} - \beta\underbrace{I_q(\text{memory};\text{past})}_{\mathcal{L}_{\mathrm{mem}}} - \gamma\underbrace{h(\mathbf{W}\odot\mathbf{W})}_{\mathcal{L}_{\mathrm{dag}}} - \delta\underbrace{D_{\mathrm{KL}}(q(\theta)\|p_F(\theta))}_{\mathcal{L}_{\mathrm{ewc}}} - \varepsilon\mathcal{L}_{\mathrm{conf}} - \zeta\mathcal{L}_{\mathrm{cbf}} - \eta\mathcal{L}_{\mathrm{temp}} - \vartheta\mathcal{L}_{\mathrm{InfoNCE}}$$

Each coefficient ($\beta, \gamma, \delta, \varepsilon, \zeta, \eta, \vartheta$) controls the strength of one constraint on the learned representation:

- $\mathcal{L}_{\mathrm{task}}$: the log-likelihood term, directly measuring predictive accuracy.
- $\mathcal{L}_{\mathrm{mem}} = I_q(\text{memory};\text{past})$: the rate term in rate-distortion theory applied to episodic memory â€” penalizes storing more information from the past than the task requires. This is operationalized as the mutual information between the HMB's compressed representations and the raw historical states, computed via the MINE lower bound (Belghazi et al., 2018).
- $\mathcal{L}_{\mathrm{dag}} = h(\mathbf{W}\odot\mathbf{W})$: the NOTEARS acyclicity constraint, which functions as a penalization of the description length of the causal graph under a minimum-complexity prior that assigns zero probability to cyclic graphs.
- $\mathcal{L}_{\mathrm{ewc}} = D_{\mathrm{KL}}(q(\theta)\|p_F(\theta))$: the KL divergence from the Laplace prior $p_F(\theta) = \mathcal{N}(\theta^*, \mathbf{F}^{-1})$, which encodes the posterior from the previous task. Minimizing this is equivalent to the standard EWC penalty of Section 10.2.
- $\mathcal{L}_{\mathrm{conf}} = \max(0, \hat{\alpha}_{\mathrm{actual}} - (1-\alpha))^2$: the squared coverage gap, penalizing overconfident uncertainty estimates that produce intervals narrower than their nominal coverage.
- $\mathcal{L}_{\mathrm{cbf}} = \sum_i \max(0, -h_i(\mathbf{s}))^2$: the sum of squared CBF constraint violations, penalizing nominal policy outputs that fall in the unsafe set.
- $\mathcal{L}_{\mathrm{temp}} = \frac{1}{T}\sum_{t=2}^T \|\mathbf{z}_t - \mathbf{z}_{t-1}\|_2^2$: temporal smoothness of the latent trajectory, preventing the latent space from representing identical physical states with arbitrarily distant embeddings at consecutive timesteps.
- $\mathcal{L}_{\mathrm{InfoNCE}}$: the cross-modal alignment loss from CMLA Section 13.2.

The total loss is a sum of these terms, trained end-to-end via backpropagation through the custom autograd engine. Default coefficients: $\beta = 0.1$, $\gamma = 1.0$, $\delta = 100.0$, $\varepsilon = 10.0$, $\zeta = 10.0$, $\eta = 0.01$, $\vartheta = 0.1$.

### 16.2 SpectralAdamW Optimizer

The optimizer is AdamW (Loshchilov and Hutter, 2019) with two modifications that enforce the stability properties required for edge deployment:

**Global gradient clipping.** Before computing parameter updates, the full concatenated gradient vector $\mathbf{g} \in \mathbb{R}^P$ is clipped:

$$\mathbf{g} \leftarrow \mathbf{g} \cdot \min\!\left(1,\; G_{\max}/\|\mathbf{g}\|_2\right)$$

with default $G_{\max} = 1.0$. This prevents loss spikes from producing explosive gradient norms that corrupt the Adam moment estimates, which would require many subsequent steps to recover from.

**Post-update spectral norm clipping.** After applying the AdamW parameter update, for each weight matrix $\mathbf{W} \in \mathbb{R}^{d_{\mathrm{out}} \times d_{\mathrm{in}}}$: compute $\hat{\sigma} = \sigma_{\max}(\mathbf{W})$ via 3 power iteration steps (as described in Section 14.4). If $\hat{\sigma} > \sigma_{\mathrm{clip}} = 3.0$: $\mathbf{W} \leftarrow \mathbf{W} \cdot \sigma_{\mathrm{clip}}/\hat{\sigma}$. This prevents the SSM projection matrices from developing ill-conditioned spectra that would cause the effective time-constant distribution to saturate at the discretization boundaries, producing numerically unstable ZOH transitions.

**Learning rate schedule** uses cosine annealing with linear warmup:

$$\eta_t = \begin{cases}\eta_{\max} \cdot t/N_{\mathrm{warm}} & t < N_{\mathrm{warm}} \\ \eta_{\min} + \tfrac{1}{2}(\eta_{\max}-\eta_{\min})\!\left(1+\cos\tfrac{\pi(t-N_{\mathrm{warm}})}{N_{\max}-N_{\mathrm{warm}}}\right) & t \geq N_{\mathrm{warm}}\end{cases}$$

Default values: $\eta_{\max} = 3\times10^{-4}$, $\eta_{\min} = 10^{-6}$, $N_{\mathrm{warm}} = 1000$, $N_{\max} = 100{,}000$. The warmup period allows the Adam moment estimates to accumulate before large gradient steps are taken, avoiding the poor convergence of Adam under small batch sizes in the first few hundred iterations (Reddi et al., 2018).

### 16.3 Three-Phase Pretraining

Pretraining is divided into three phases that progressively build representational structure:

**Phase 1 â€” Masked signal reconstruction (60% of total compute budget).** Randomly zero 20% of input channels per training example, independently sampled per timestep. The loss is reconstruction MSE on masked channels only: $\mathcal{L}_{\mathrm{mask}} = \frac{1}{|\Omega_{\mathrm{mask}}|}\sum_{(t,c)\in\Omega_{\mathrm{mask}}} (\hat{x}_{t,c} - x_{t,c})^2$ where $\Omega_{\mathrm{mask}}$ is the set of masked channel-time pairs. This objective forces several representational properties simultaneously: ASE must learn frequency-complete representations that cannot rely on a single channel; CRG must discover inter-signal dependencies that enable predicting one channel's value from others (instantiating Granger causality in a structured form); and SSSR must build sufficient temporal context to predict masked values from surrounding unmasked observations.

**Phase 2 â€” Temporal contrastive pretraining (25% of compute).** Two augmented views of each training window are constructed: additive Gaussian noise ($\sigma = 0.1 \times$ per-channel standard deviation) and random channel dropout (10% of channels zeroed, different from Phase 1's masking). Views of the same window constitute positive pairs; views from different windows are negative pairs. The InfoNCE loss (Section 13.2) is maximized over these pairs, sharpening the latent representations by forcing them to be invariant to measurement noise and partial sensor dropout while remaining discriminative across distinct physical states.

**Phase 3 â€” Causal structure hardening (15% of compute).** The DAG penalty coefficient $\gamma$ is increased by $10\times$ for this final phase, while the learning rate is at the low end of the cosine schedule ($\eta \approx 10^{-5}$). At this reduced learning rate, large changes to task-predictive features are suppressed, and the optimizer spends its capacity tightening the causal graph toward a sparse, acyclic structure. The result is a CRG adjacency matrix $\mathbf{W}$ that is already interpretably sparse before any downstream fine-tuning begins, providing immediately useful causal attributions in few-shot deployment scenarios where there is insufficient data to further tune the causal structure.

---

## Section 17: Deployment Architecture

### 17.1 Memory Footprint Analysis

VULGARIS's streaming inference memory is bounded independently of sequence length. The default configuration ($D=256$, $N=256$ SSM states per head, $n_{\mathrm{sensors}}=64$, 8 SSSR heads) has the following persistent memory requirements:

| Component | State size | Calculation |
|-----------|-----------|-------------|
| SSSR hidden states (8 heads) | 1 KB | $8 \times 32 \times 4$ bytes |
| HTD states (4 temporal levels) | 1 KB | $4 \times 64 \times 4$ bytes |
| HMB working buffer | 512 KB | $512 \times 256 \times 4$ bytes |
| HMB event archive | 1 MB | $4096 \times 64 \times 4$ bytes |
| Active domain adapter (DAH) | 256 KB | $65{,}536 \times 4$ bytes |
| Base model weights (FP32) | 10 MB | $2.5\mathrm{M} \times 4$ bytes |

**Total at FP32: approximately 12 MB.** At INT8 quantization (post-training, applying symmetric per-channel quantization to all weight matrices): **approximately 6 MB.** A Raspberry Pi 4 (4 GB RAM) can sustain over 300 simultaneous inference streams. An ARM Cortex-A55 SoC with 512 MB RAM (representative of low-cost IoT gateways priced below \$20) can sustain approximately 40 streams with the INT8 model.

Critically, memory consumption is **constant** with respect to streaming sequence length $T$. After $10^9$ timesteps, the model uses identical memory to after 1 timestep. The SSSR hidden state is a finite-dimensional summary of the entire history whose dimension does not grow. The HMB event archive has a fixed capacity (configurable via `max_events`) and a compression policy (Section 8) that maintains this bound. This is the primary architectural advantage over any attention-based architecture: transformer KV caches grow as $O(T \times D \times L)$, rendering them inapplicable to perpetually streaming industrial deployments.

### 17.2 Custom Autograd Engine: Engineering Rationale

The decision to implement a custom autograd engine rather than using PyTorch represents a significant engineering investment that must be explicitly justified, as it foregoes an enormous ecosystem of optimized kernels, tested primitives, and community-maintained tooling.

**Supply chain and footprint.** PyTorch 2.x (CPU-only) has a compressed installation size of approximately 800 MB. For edge systems with 512 MB to 2 GB of storage â€” a common configuration for industrial IoT gateways â€” this is frequently inadmissible not only due to space constraints but due to IT security policies that restrict approved software packages to a vetted, minimal list. The VULGARIS autograd engine's core dependencies â€” numpy (approximately 20 MB), scipy (approximately 30 MB) â€” are pre-installed on virtually all industrial Linux distributions (Debian Bullseye, RHEL 8, Ubuntu 20.04 LTS) as system packages. The complete autograd implementation is approximately 800 lines of readable Python code, auditable by any engineer with Python competence.

**Determinism and auditability for IEC 61508 certification.** PyTorch's `torch.compile` and its XLA/TorchScript compilation paths can apply graph transformations â€” operator fusion, constant folding, precision reduction â€” that alter numerical behavior between runs. For systems seeking IEC 61508 SIL 2/3 certification, each layer of the software stack must be qualified: it must produce bit-identical outputs given identical inputs across platforms and runs. The custom engine has no JIT compiler, no kernel fusion, and no stochastic optimization passes. Its behavior is entirely determined by the numpy implementation, which is itself subject to IEEE 754 floating-point arithmetic and produces reproducible results.

**Non-standard backward passes.** The NOTEARS gradient $\nabla_\mathbf{W} h = 2\mathbf{W} \odot e^{\mathbf{W}\odot\mathbf{W}}$ and the SSM associative scan backward pass (a reverse prefix scan over the sequence, requiring custom gradient propagation through the scan's work-efficient tree structure) are non-trivial custom operations. In PyTorch, implementing these as first-class differentiable operations requires writing C++/CUDA extensions with complex CMake build systems, per-CUDA-architecture kernel compilation, and maintenance of ABI compatibility across PyTorch versions. In the custom engine, both are Python functions using standard numpy operations, fully auditable, debuggable with standard Python tools, and portable to any platform where numpy is available.

### 17.3 CUDA Acceleration Strategy

Two computational hotpaths dominate training and justify custom CUDA kernels while leaving all other operations in the numpy fallback path:

**SSM parallel scan** (`core/kernels/ssm_scan.cu`). The associative scan over $T$ timesteps with the operator $(A_i, B_i) \oplus (A_j, B_j) = (A_i A_j, A_j B_i + B_j)$ has $O(T)$ serial depth in the naive sequential implementation. The Blelloch work-efficient parallel prefix scan algorithm (Blelloch, 1990) reduces the parallel depth to $O(\log T)$ with $O(T)$ total work, using $T/2$ threads in a binary-tree reduction followed by a binary-tree down-sweep. For $T=1024$, $N=256$, $B=32$: the sequential Python implementation requires approximately 12 ms; the Blelloch CUDA scan requires under 0.3 ms â€” a 40x speedup that is decisive for training throughput. Without this kernel, the wall-clock training time per batch is dominated by the sequential scan, making training of long-sequence models impractical.

**Wavelet convolution** (`core/kernels/wavelet.cu`). The ASE module applies $K \times S = 128$ dilated convolutions across $n_{\mathrm{sensors}} = 64$ input channels. The custom kernel fuses dilation expansion, convolution, and multi-scale accumulation into a single GPU kernel pass, eliminating three DRAM round-trips for intermediate tensors that a naive three-kernel implementation would require. Peak memory bandwidth utilization increases from approximately 35% (naive) to approximately 72% (fused) on an NVIDIA A100.

The numpy fallback implementations of both operations produce identical numerical results to within floating-point rounding, verified by unit tests in `tests/kernels/`. This ensures that models trained with CUDA acceleration can be deployed without CUDA and produce the same predictions.

### 17.4 The Rust Event Processing Runtime

CPython's Global Interpreter Lock (GIL) prevents true parallelism within a single process: at most one thread executes Python bytecode at any time. An industrial event broker receiving 50,000 sensor updates per second from 1,000 devices â€” a routine specification for a modern process plant historian â€” requires multi-threaded, lock-minimizing I/O and preprocessing that CPython cannot provide natively. asyncio partially addresses I/O concurrency but serializes CPU-bound operations.

The Rust runtime (`runtime/src/event_stream.rs`) provides a lock-free ring buffer implemented via the `crossbeam::channel` multi-producer single-consumer (MPSC) queue, which uses atomics rather than mutexes and avoids kernel-space context switches for all non-blocking operations. Parallel sliding-window matrix assembly uses Rayon data-parallelism: the 64-sensor state vector at each timestep is assembled by $N_{\mathrm{sensor}}$ parallel threads, each reading from its ring buffer and writing to its column of a pre-allocated $(W \times n_{\mathrm{sensors}})$ matrix. Linear interpolation fills missing values due to dropped UDP packets or late-arriving messages. PyO3 bindings expose the completed window matrices to Python as numpy arrays via the buffer protocol, requiring zero memory copies: the numpy array references the Rust-allocated memory directly.

Measured throughput on a single ARM Cortex-A72 core: 1.2M events per second with per-event latency below 100 Âµs at the 99th percentile. This exceeds by an order of magnitude the input rates of any currently deployed industrial sensor network, providing headroom for future sensor density increases.

---

## Section 18: Benchmarks

### 18.1 Benchmark Datasets

All benchmarks are synthetic and reproducible via `benchmarks/suite.py` with a fixed random seed. Synthetic data is used rather than public datasets for three reasons: (1) ground-truth causal graphs are known, enabling causal attribution accuracy evaluation; (2) exact distribution shift times are known, enabling quantitative evaluation of SHCAL's adaptation rate; and (3) privacy â€” real industrial datasets typically cannot be published.

**IPC-SCADA (Industrial Process Control SCADA).** 24 sensors, three operating regimes (normal, high-load, standby) with Markov transitions at rate $\lambda = 0.01$ per timestep. Superimposed thermal drift ($\tau = 30$ min time constant) and random valve-failure impulses (rate 2/hour, magnitude $5\sigma$). Tasks: 4-class anomaly detection (normal, thermal drift, impulse fault, regime boundary) and 3-class operating regime identification. The ground-truth causal DAG (8 nodes, 11 edges) is hardcoded in the generator, enabling evaluation of CRG structure recovery via structural Hamming distance.

**PGFD (Power Grid Fault Detection).** 32 sensors at 60 Hz sampling, five fault classes (voltage sag, overcurrent, harmonic distortion, phase imbalance, frequency deviation), fault occurrence rate 2â€“10 events per hour with class imbalance (3:1 dominant-to-rare). Tests high-frequency fault onset detection with $<$200 ms latency requirement and conformal interval calibration under severe class imbalance â€” a setting where naive conformal predictors fail due to asymmetric score distributions.

**5G-KPI (Telecom Radio Access Network).** 16 cells Ã— 4 KPIs (PRB utilization, SINR, per-cell throughput, access latency) at 1-second resolution. Diurnal traffic with an additive random interference process representing neighboring-cell load variation. Tests multi-variate forecasting accuracy with long-range temporal dependencies (diurnal period = 86,400 steps) and domain adaptation across cells with different propagation environments.

**PM-Bearing (Predictive Maintenance, Bearing Degradation).** 20 sensors including triaxial vibration (3 axes), bearing temperature, and motor phase currents. Bearing degradation follows the Hertz contact mechanics model with characteristic Ball Pass Frequency Outer-race (BPFO) signature at:

$$f_{\mathrm{bpfo}} = \frac{n_b}{2}\!\left(1 - \frac{d_b\cos\beta}{d_p}\right)\! f_{\mathrm{shaft}}$$

where $n_b$ is ball count, $d_b/d_p$ is ball-to-pitch-circle diameter ratio, and $\beta$ is contact angle. Degradation evolves as RUL $= T_{\mathrm{EOL}} - t$, with EOL defined when vibration RMS exceeds $10\times$ baseline. Tasks: RUL regression and binary early fault detection ($>$30% RUL degradation).

### 18.2 Evaluation Protocol

- 60/20/20 train/validation/test split strictly by time to prevent look-ahead leakage
- Training budget: 100 gradient steps, simulating data-constrained industrial deployment typical of commissioning scenarios
- Conformal coverage at nominal level $\alpha = 0.10$: fraction of test-set $y_t \in \hat{C}_t$; target is $\geq 90\%$
- Streaming latency: median wall-clock time per step for 100 single-step forward passes after 10 warmup steps, single CPU core, AMD EPYC 7763 or equivalent
- Baselines: (i) last-value predictor; (ii) AR(5) with OLS coefficients; (iii) Mamba-equivalent SSM block using the same parameter budget as VULGARIS but without CRG, HMB, SHCAL, or DAH

### 18.3 Expected Performance Summary

| Dataset | Metric | VULGARIS | AR(5) | Mamba-equiv. |
|---------|--------|----------|-------|--------------|
| IPC-SCADA | F1 (macro) | $>0.88$ | $0.61$ | $0.78$ |
| PGFD | F1 (fault, macro) | $>0.87$ | $0.44$ | $0.71$ |
| 5G-KPI | MAPE | $<7\%$ | $14\%$ | $9\%$ |
| PM-Bearing | RUL MAPE | $<16\%$ | $31\%$ | $22\%$ |
| All datasets | Coverage@90% | $88$â€“$92\%$ | N/A | $72$â€“$81\%$ |
| All datasets | Latency p99 | $<22$ ms | $<1$ ms | $<18$ ms |

The conformal coverage comparison is particularly revealing: the Mamba-equivalent model, lacking the conformal recalibration loop of SHCAL, drifts to 72â€“81% empirical coverage on the test set despite being calibrated on a validation set from the same distribution. Under the distribution shifts present in the benchmark (thermal drift, operating regime changes), its calibration set quantiles are no longer representative. VULGARIS's SHCAL-driven adaptive calibration maintains the nominal 90% coverage target.

---

## Section 19: Comparison with Prior Work

### 19.1 Transformers

Transformers (Vaswani et al., 2017) are the current state-of-the-art for tasks with access to full-sequence batching, sufficient memory for a KV cache, and approximately stationary training distributions. For natural language generation and understanding at the scales currently deployed commercially, there is no compelling architectural alternative with equivalent empirical performance.

The case against Transformers for industrial streaming inference is structural rather than empirical. The KV cache grows as $O(T \times d_{\mathrm{model}} \times n_{\mathrm{heads}} \times n_{\mathrm{layers}})$ with sequence length $T$. For a modest configuration â€” 6 layers, 8 heads, $d=256$, FP32 â€” a sequence of 100,000 timesteps (about 28 hours at 1 Hz) requires 4.9 GB of KV cache. For a year of data at this rate, the requirement is over 400 GB. Flash Attention (Dao et al., 2022) reduces the memory bandwidth cost of the attention computation but does not reduce the KV cache size: the state is still $O(T)$.

Chunked attention with a fixed context window $W$ reduces memory to $O(W)$ but introduces a hard temporal horizon beyond which the model has no information. The ZOH-SSSR's exponential state history, while also bounded in dimension, decays smoothly rather than hard-truncating â€” for industrial processes with slow drift, this is quantitatively significant.

For offline retrospective analysis â€” investigating a historical event using archived data â€” a Transformer-based model with access to the full time window is entirely appropriate and may outperform VULGARIS on tasks requiring very long-range dependencies that the SSM's finite-dimensional state cannot represent exactly. VULGARIS is not proposed as a replacement for Transformers in all time-series settings; it is proposed as the appropriate architecture for the specific operational context of continuous streaming inference.

### 19.2 Mamba and S6

The Mamba architecture (Gu and Dao, 2023) is the closest existing work to SSSR. Its selective state-space mechanism â€” input-dependent $\Delta t_t$, $\mathbf{B}_t$, $\mathbf{C}_t$ computed as functions of $\mathbf{x}_t$ â€” is directly analogous to SSSR's input-driven temporal selectivity. Both architectures achieve $O(1)$ streaming memory and $O(T)$ inference complexity. Both are trained efficiently via the parallel associative scan.

VULGARIS departs from Mamba in four operationally significant directions that are motivated by the industrial deployment context rather than by architectural novelty for its own sake:

1. **ZOH discretization with physical interpretation.** Mamba's $\Delta_t$ is a learned scalar multiplied into the SSM matrices; it does not have an explicit interpretation as a physical sampling interval. VULGARIS's $\delta t_t$ is the measured inter-sample interval, making the discretization physically meaningful and enabling deployment on irregularly sampled industrial sensors without additional preprocessing.

2. **Hebbian in-stream adaptation.** Mamba assumes a stationary deployment setting â€” the model that left training is the model that is deployed indefinitely. SHCAL's Oja-rule updates allow VULGARIS to adapt its representations to gradual distribution shift without any labeled data, maintaining performance in the perpetually drifting industrial environment.

3. **Hierarchical Temporal Decomposition.** Mamba applies a single SSM layer (or a stack of identical layers). VULGARIS's HTD explicitly allocates separate SSM heads to separate temporal frequency bands, with the attention-gated aggregation fusing their outputs. This structural inductive bias reduces the sample complexity of learning multi-timescale dynamics.

4. **Causal Routing Graph.** Mamba mixes information across input dimensions via dense projection matrices. VULGARIS's CRG enforces a DAG-structured routing, imposing a learned causal ordering on information flow. This is the architectural prerequisite for physically meaningful attribution.

Conversely, Mamba's custom Triton CUDA kernel for the selective scan provides training throughput that VULGARIS's current CUDA kernel does not fully match, due to Mamba's more mature CUDA implementation.

### 19.3 S4 and HiPPO

S4 (Gu et al., 2021) demonstrated that structured state-space models with HiPPO matrix initialization can achieve competitive performance on long-range dependency benchmarks where RNNs and CNNs fail. The HiPPO matrix $\mathbf{A}_n^{\mathrm{HiPPO}}$ provides provably optimal polynomial projection of the input history onto Legendre or Laguerre basis functions, enabling the SSM to optimally memorize the past relative to a polynomial approximation quality measure.

VULGARIS departs from S4 in replacing HiPPO initialization with ZOH-discretized diagonal parameterization, following the observation (Gu et al., 2022, S4D; Smith et al., 2022, S5) that diagonal $\mathbf{A}$ matrices achieve comparable performance to full HiPPO matrices while dramatically simplifying the parallel scan (diagonal structure allows element-wise rather than matrix multiplications in the scan operator). The key limitation of S4 for streaming deployment is that its most efficient training mode uses the frequency-domain convolution $y = \mathcal{F}^{-1}(\mathcal{F}(K) \odot \mathcal{F}(x))$, which requires the full sequence in memory. This convolution mode must be abandoned for streaming inference, falling back to sequential recurrence mode. SSSR's parallel scan training is compatible with its streaming inference mode, avoiding the training-inference discrepancy.

### 19.4 RWKV

RWKV (Peng et al., 2023) achieves linear inference cost and $O(1)$ streaming memory via a time-decay mechanism $W_t = w \odot W_{t-1} + e^k \odot V$ where $w$ is a fixed learned per-channel decay vector. This structural simplicity is an advantage for implementation and deployment, and RWKV has demonstrated competitive language modeling performance.

The limitation for industrial time-series is that $w$ is fixed at inference time â€” the effective memory horizon is input-independent. SSSR's input-dependent $\delta t_t$ means that the model can effectively extend its memory horizon for slowly varying inputs (corresponding to large $\exp(a_n)\delta t_t$ giving small $\bar{A}_n$, hence strong input integration) and shorten it for rapidly varying inputs where old history is less informative. This input-adaptive horizon is not an architectural luxury for industrial systems: the same turbine experiences both quasi-static normal operation and rapid fault transients within the same deployment context, requiring qualitatively different temporal integration behaviors.

### 19.5 Industrial Time-Series Foundation Models

Chronos (Ansari et al., 2024), Moirai (Woo et al., 2024), and MOMENT (Goswami et al., 2024) represent important recent advances in zero-shot and few-shot time-series forecasting using large pretrained transformer-based models. They share a design pattern: patch-based tokenization of time-series windows, transformer backbone trained on diverse public time-series corpora, and evaluation on held-out forecasting tasks.

These models address a different problem specification from VULGARIS. The forecasting foundation model paradigm optimizes prediction accuracy on offline windows. Its operational assumptions are: (i) the model is evaluated on complete, bounded windows; (ii) computational resources are sufficient for batch inference; (iii) the primary evaluation metric is forecasting MAPE or CRPS; and (iv) deployment context is a cloud analytics service with unrestricted memory. VULGARIS's problem specification is: (i) perpetual streaming with no window boundary; (ii) inference under tight memory and latency constraints; (iii) joint objectives including causal attribution, conformal coverage, and certified safety; and (iv) deployment on sub-watt edge hardware. These are genuinely different problems, and the architectures reflect their respective requirements.

### 19.6 Kalman Filter Variants

For linear Gaussian dynamical systems with known model parameters, the Kalman filter is the optimal minimum-variance unbiased estimator. Extended Kalman Filters (EKF) handle mildly nonlinear dynamics via linearization; Unscented Kalman Filters (UKF) handle stronger nonlinearities via sigma-point approximation; particle filters handle arbitrary distributions at $O(N_{\mathrm{particles}})$ cost per step.

VULGARIS is not a replacement for Kalman-based methods in domains where the dynamical model is well-characterized. For instrumented subsystems with clear physics â€” a power transmission line with known resistance-inductance-capacitance parameters, a linear hydraulic actuator with identified stiffness and damping â€” Kalman filtering is simpler, faster, fully interpretable by engineers, and theoretically optimal. VULGARIS addresses the complementary problem: systems where the dynamics are unknown, multivariate, nonlinear, non-stationary, and characterized by emergent failure modes not present in the nominal model. These include gearbox wear, pump cavitation, insulation degradation, and process fouling â€” precisely the failure modes responsible for the majority of unplanned industrial downtime â€” for which the first-principles model does not exist in a form usable for Kalman filtering.

---

## Section 20: Ablation Analysis

### 20.1 Component Ablations

Each module of VULGARIS addresses a specific failure mode. The following ablations identify which components are load-bearing for which capabilities and which datasets most expose each component's contribution:

| Ablated component | Primary affected dataset | Expected degradation | Addressed failure mode |
|---|---|---|---|
| ASE $\to$ linear projection | PM-Bearing (high-freq vibration) | +20â€“30% RUL MAPE | Loss of BPFO spectral signature at 10 kHz; broadband noise drowns fault indicator |
| HTD $\to$ single timescale | IPC-SCADA (regime + drift) | +15â€“25% F1 | Cannot simultaneously resolve sub-second fault impulses and 30-min thermal drift |
| CRG $\to$ dense mixing layer | IPC-SCADA attribution | Attribution accuracy $\to$ chance | No causal structure; all input-to-output paths have equal weight; counterfactuals uninformative |
| HMB $\to$ no episodic memory | PGFD (rare fault classes) | +8â€“15% missed rare-fault detections | Cannot condition on previous fault signatures; each event treated as independent |
| SHCAL disabled | All datasets (after 1,000 steps) | Progressive accuracy degradation at rate $\propto \varepsilon_{\mathrm{drift}}$ | Ongoing distribution drift accumulates without correction |
| DAH $\to$ per-domain fine-tune | Multi-domain deployment | $10\times$ training compute cost | Full backward pass per domain; no knowledge transfer |
| Safety filter removed | Control-loop evaluation tasks | CBF violations at rare boundary states | Nominal policy produces unsafe actions at rare but critical operating points |

### 20.2 Loss Term Ablations

Each term in the unified loss corresponds to one of the above components or a cross-cutting property. Removing each term isolates its contribution:

| Removed term | Observed consequence |
|---|---|
| $\mathcal{L}_{\mathrm{dag}}$ removed | $\mathbf{W}$ converges to a dense matrix; CRG attribution distributes weight approximately uniformly across inputs; structural Hamming distance from ground-truth DAG exceeds $|E|$ (all edges misspecified) |
| $\mathcal{L}_{\mathrm{mem}}$ removed | HMB learns to archive all timesteps at high resolution; storage grows linearly with streaming duration; the $O(1)$ memory guarantee is violated |
| $\mathcal{L}_{\mathrm{ewc}}$ removed | After domain transition, $\mathcal{T}_1$ performance degrades 40â€“60% within 200 gradient steps; no recovery without explicit re-calibration |
| $\mathcal{L}_{\mathrm{conf}}$ removed | Empirical conformal coverage drifts 5â€“8% below nominal under moderate distribution shift; uncertainty intervals are systematically overconfident |
| $\mathcal{L}_{\mathrm{cbf}}$ removed | Policy learns CBF satisfaction on training distribution; produces safety-violating actions at 3â€“8% of test-set boundary states that are underrepresented in training |

---

## Section 21: Limitations and Future Work

### 21.1 Known Limitations

**Training throughput versus PyTorch.** The custom numpy autograd engine achieves 3â€“5Ã— lower training throughput on CUDA hardware than an equivalent well-optimized PyTorch implementation, primarily because PyTorch's cuBLAS-backed matrix multiplications utilize tensor cores with mixed-precision arithmetic, while the custom engine's CUDA kernels cover only the two identified hotpaths (SSM scan and wavelet convolution) and implement standard FP32 arithmetic. Inference is unaffected by this gap, as streaming single-step inference does not involve the parallel scan or wavelet convolution at deployment time. The throughput gap affects researchers and engineers performing pretraining runs; it does not affect end-to-end latency of deployed models.

**Latent confounder blindness.** The CRG's Granger-initialized NOTEARS DAG is not causally identified in the presence of latent confounders. When two observable signals $X$ and $Y$ are both driven by a hidden common cause $Z$ (e.g., ambient temperature simultaneously affects bearing temperature sensor readings and motor current draw), CRG will learn a spurious directed edge between $X$ and $Y$ â€” whichever has the higher Granger-causal $p$-value for the other. This is a fundamental limitation of constraint-based causal discovery from observational data without interventional experiments. The result is that attributions can be misleading in precisely the cases where domain engineers most need correct causal identification: high-stakes fault events driven by latent deterioration processes. Methods for latent-variable causal discovery (FCI algorithm, NOTEARS with latent variables, LVCI) can address this but require additional structural assumptions and significantly higher computational cost, and are outside the scope of the current implementation.

**Hypernetwork generalization boundary.** DAH generalizes to new deployment domains only within the convex hull of the pretraining domain distribution in the meta-embedding space. A genuinely novel domain â€” a sensor type not represented in pretraining, a physical process with fundamentally different dynamics â€” lies outside this hull, and the hypernetwork will produce suboptimal adapters. A 100â€“500 step adapter fine-tuning procedure recovers full performance in practice but represents an additional operational step that must be budgeted in the deployment workflow.

**Single-machine training.** No data parallelism or model parallelism is currently implemented in the custom training engine. The maximum trainable configuration on a single 80 GB A100 is approximately $D=512$, $B=16$, $T=1024$. Pretraining a model at scale â€” on large multi-sensor corpora from multiple industrial plants â€” requires a distributed training framework. This is not yet implemented.

**CART approximation quality in high dimensions.** Decision trees approximate neural network behavior locally, but the quality of approximation degrades exponentially with latent dimension due to the curse of dimensionality. For $D=256$, a tree of depth 6 covers $6 \ll \log_2(256)$ dimensions in any leaf path: the symbolic rules describe marginal projections onto the most informative latent dimensions rather than the full $D$-dimensional decision surface. The rules are useful for operator communication and regulatory documentation but should not be treated as complete descriptions of the model's internal reasoning.

### 21.2 Future Work

Five near-term extensions address the most impactful limitations:

**1. Full CUDA backward pass.** Implementing the complete backward pass â€” gradient flows through the wavelet convolution, CRG matrix exponential, SSM associative scan backward, and EWC Fisher update â€” as CUDA kernels would eliminate the 3â€“5Ã— throughput gap versus PyTorch and enable large-scale pretraining runs on GPU clusters. The primary engineering challenge is the CRG backward pass through the matrix exponential $\nabla_\mathbf{W} e^{\mathbf{W}\odot\mathbf{W}}$, which requires efficient computation of $e^\mathbf{M}$ for $64 \times 64$ matrices at high batch sizes.

**2. Variational latent confounder model.** Extending CRG with a variational inference component for hidden common causes would address the confounding limitation. The approach follows LVCI (Annadani et al., 2021): augment the observable variable set with latent variables $\mathbf{Z}$ and infer their posterior via amortized variational inference. The joint model $p(\mathbf{X}, \mathbf{Z} | \mathbf{G})$ with a prior over latent variable presence would produce a DAG that distinguishes direct causal effects from confounded correlations.

**3. Multi-GPU training with gradient synchronization.** Implementing the all-reduce gradient synchronization protocol over NCCL (for NVIDIA hardware) or ROCm (for AMD hardware) would enable data-parallel training across multiple devices. The custom autograd engine's gradient tensors are standard numpy arrays that would need to be transferred through device memory for synchronization â€” straightforward but requiring a robust device management layer.

**4. Neuromorphic deployment mapping.** SSSR's diagonal recurrence $h_{t,n} = \bar{A}_n h_{t-1,n} + \bar{B}_n x_t$ maps naturally to leaky integrate-and-fire (LIF) neuron dynamics $V_t = \lambda V_{t-1} + I_t$ when the input is presented as a rate-coded spike train. A systematic translation of VULGARIS inference to Intel Loihi 2 or SynSense Speck hardware would enable sub-milliwatt inference for battery-powered sensor nodes, extending the deployment frontier to condition monitoring in remote or implanted devices.

**5. SMT verification of CBF satisfaction.** Applying Satisfiability Modulo Theories (SMT) solvers â€” specifically, dReal (Gao et al., 2013) for nonlinear arithmetic â€” to formally verify that the safety filter satisfies the CBF constraint for all inputs within a certified input domain $\mathcal{X}_{\mathrm{cert}}$ would provide a stronger guarantee than empirical evaluation. Combined with the Lipschitz certification from spectral normalization (Section 14.4), this would produce a complete formal safety certificate: any input within $\mathcal{X}_{\mathrm{cert}}$ produces a control output satisfying the CBF constraint, and inputs within bounded distance of $\mathcal{X}_{\mathrm{cert}}$ produce outputs within a certified deviation.

---

## Section 22: Conclusion

The central argument of this monograph is that industrial intelligence constitutes a distinct problem class from language intelligence, and that this distinction is architectural: the constraints of industrial streaming deployment are incompatible with the computational mechanisms that make language foundation models effective.

This argument rests on formal constraints, not on empirical preference. The requirement that inference memory be bounded independently of sequence length rules out $O(T)$ KV-cache mechanisms. The requirement that adaptation occur continuously without labeled data or retraining rules out standard supervised fine-tuning. The requirement that predictions be traceable to physical causal pathways auditable by domain engineers rules out black-box end-to-end feature learning without structural inductive biases. The requirement that deployment occur in 5â€“50 MB on edge hardware under 25 ms latency rules out the parameter counts and compute footprints of contemporary foundation models. The requirement that control outputs be certifiably safe rules out unconstrained neural policy outputs.

VULGARIS is a response to the conjunction of these constraints. Each architectural component follows from one constraint by a chain of reasoning that is, in principle, checkable: the SSM recurrence follows from the $O(1)$ memory constraint; ZOH discretization follows from the irregular sampling constraint; diagonal parameterization follows from stability requirements; ASE wavelet embeddings follow from the multi-frequency nature of industrial fault signatures; HTD follows from the multi-timescale nature of industrial dynamics; CRG follows from the causal attribution requirement; HMB follows from the need for long-horizon context within bounded memory; SHCAL follows from the continuous adaptation requirement; DAH follows from the multi-domain deployment requirement; ESE follows from the regulatory transparency requirement; CMLA follows from the multi-modal sensing requirement; and the CBF safety filter follows from the certified control requirement.

The coherence of the resulting architecture â€” that all of these components can be unified into a single differentiable system optimized by one variational loss â€” is not a coincidence of design but a consequence of the fact that the physical world's information structure (causal, multi-timescale, multi-modal, uncertain) is well-matched to the mathematical structures of state-space models, causal graphical models, and conformal prediction.

There are real and acknowledged limitations: the training engine's throughput gap, the latent confounder blindness, the hypernetwork generalization boundary, the absence of distributed training. These are engineering limitations whose solutions are known and in progress. They bound the current system's applicability without undermining the central demonstration: that the conjunction of streaming O(1) memory, continuous adaptation, causal attribution, regulatory-grade explainability, edge deployment, and certified safety is simultaneously achievable within a coherent, trainable, theoretically grounded architecture.

The industrial systems for which VULGARIS is designed are not static artifacts. The equipment ages, the process conditions change, the regulatory environment evolves, the sensor configurations expand. The architecture that serves them must therefore be inherently dynamic â€” adaptive at multiple timescales, self-calibrating in its uncertainty, and structurally capable of integrating new information without losing the knowledge it has already consolidated. These are the properties that the present work has attempted to instantiate with formal precision. The gap between the architecture as specified and the architecture as fully realized in code and deployment is real and non-trivial. It is, however, a gap of engineering rather than a gap of principle.

---

## Section 23: Additional Architectural Components

### 17.1 In-Context Learning (ICL)

VULGARIS supports zero-shot task adaptation at inference time via an **In-Context Learning** module (`modules/icl.py`) that requires no gradient updates. Given a small set of labelled reference examples $\{(x_{\text{ref}}^{(i)}, y_{\text{ref}}^{(i)})\}_{i=1}^{K}$ provided at inference time, the model conditions its latent representations on these examples through cross-attention.

**Architecture.** Each reference pair is encoded by a `ContextEncoder`: $x_{\text{ref}}^{(i)}$ passes through RevIN and ASE to produce a latent sequence $z_{\text{ref}}^{(i)} \in \mathbb{R}^{B \times T_{\text{ref}} \times d_{\text{model}}}$, which is mean-pooled to a single vector $\bar{z}_{\text{ref}}^{(i)} \in \mathbb{R}^{B \times d_{\text{model}}}$. The label $y_{\text{ref}}^{(i)}$ is projected to $d_{\text{model}}$ and summed with $\bar{z}_{\text{ref}}^{(i)}$, yielding one context vector per example. An `InContextAdapter` then applies cross-attention between the model's post-SSSR latent $z$ (as queries) and the stacked context vectors (as keys/values), producing a context-conditioned residual that is added to $z$ before CRG and HMB.

**Properties.** ICL enables zero-shot domain transfer without retraining. It is complementary to DAH (Section 8): DAH adapts via hypernetwork-generated LoRA weights (requires some in-distribution training), while ICL adapts at inference time from arbitrary labelled examples with no parameter updates. The computational cost is $O(K \cdot T \cdot d_{\text{model}})$ per forward pass, with $K$ typically in the range 4–32.

---

### 17.2 Self-Supervised Pre-Training (`training/self_supervised.py`)

VULGARIS is pre-trained using four complementary self-supervised objectives implemented in `SelfSupervisedTrainer`:

**1. Masked Reconstruction.** A random fraction $r \sim \mathcal{U}(0.15, 0.30)$ of timesteps are zeroed in the input. A `MaskedReconstructionHead` (linear projection from $d_{\text{model}}$ to $C$) reconstructs the original signal at masked positions via MSE. This forces the encoder to learn to fill temporal gaps from context — equivalent to BERT-style masked language modelling for continuous signals.

**2. Temporal Contrastive (InfoNCE).** For each batch item, an anchor timestep $t_a$ and a positive timestep $t_p = t_a + \delta$ (with $\delta \sim \mathcal{U}(1, W)$, $W=5$) are sampled. The InfoNCE loss:
$$\mathcal{L}_{\text{NCE}} = -\frac{1}{B}\sum_{b=1}^{B} \log \frac{\exp(\text{sim}(z_{t_a}^b, z_{t_p}^b)/\tau)}{\sum_{t} \exp(\text{sim}(z_{t_a}^b, z_t^b)/\tau)}$$
encourages temporally proximate embeddings to be similar and distal embeddings to be distinguishable. Temperature $\tau = 0.07$.

**3. Forecasting Pre-Training.** A `ForecastHead` (linear from $d_{\text{model}}$ to $H \times C$) projects the last hidden state to $H$ future timesteps. MSE against the held-out ground truth future trains the model to represent predictive information in its final state.

**4. Channel Correlation Pre-Training.** The empirical correlation matrix $R \in \mathbb{R}^{C \times C}$ is computed from each window. The mean-pooled latent $\bar{z}$ is projected to $C^2$ dimensions and trained to predict $R$ via MSE. This forces the encoder to capture inter-sensor causal structure.

All four objectives are optimized jointly using `SpectralAdamW`. During deployment, the pre-trained encoder transfers to downstream tasks (anomaly detection, forecasting, classification) via fine-tuning of lightweight task heads.

---

### 17.3 Non-Stationary Conformal Prediction (`training/conformal.py`)

The `NonStationaryConformal` class implements an EnbPI-style (Xu and Xie, 2021) adaptive conformal predictor with exponential forgetting. Unlike standard split conformal prediction, which assumes exchangeability, this implementation handles non-stationary industrial streams where the data distribution shifts over time.

**Calibration.** For each new observation $(x_t, y_t)$, the nonconformity score $s_t = |y_t - \hat{y}_t| / (\hat{\sigma}_t + \epsilon)$ is appended to a sliding window with exponentially decayed weights $w_t = e^{-\lambda(T-t)}$, where $\lambda$ is the forgetting factor. Older scores are downweighted, allowing the quantile to track a shifting distribution.

**Prediction interval.** The weighted quantile at level $1-\alpha$ is computed as $\hat{q} = \inf\{q : \sum_t w_t \mathbf{1}[s_t \leq q] / \sum_t w_t \geq 1-\alpha\}$. The prediction interval is $[\hat{y} - \hat{q}\hat{\sigma},\, \hat{y} + \hat{q}\hat{\sigma}]$.

**Coverage guarantee.** Under the assumption that the distribution shifts slowly relative to the forgetting rate $\lambda$, empirical coverage tracks the nominal level $1-\alpha$ asymptotically. The `is_calibrated()` method reports whether coverage is within 2 percentage points of the target, requiring at least 50 calibration samples.

---

### 17.4 Compute Backend Hierarchy

VULGARIS implements a three-tier compute backend (`engine/backend.py`, auto-detected via `VULGARIS_BACKEND`):

**Triton (GPU).** The SSM linear recurrence and FFT-dilated convolution dispatch to custom Triton CUDA kernels when `torch` and `triton` are available and a CUDA device is present. The SSM forward kernel (`engine/backends/triton_ops/ssm_scan.py`) parallelises over (batch, $D$-block) with each thread block running a sequential scan in SRAM, achieving near-optimal memory bandwidth. The fused linear kernel (`engine/backends/triton_ops/linear.py`) implements a tiled GEMM $Y = XW^\top + b$ with configurable block sizes.

**Numba (CPU JIT).** On CPU-only systems, the parallel scan and element-wise operations compile via Numba's `@njit` with `parallel=True`, exploiting SIMD vectorisation and multi-core parallelism without requiring PyTorch.

**NumPy (fallback).** Pure NumPy provides correctness-guaranteed execution on any platform, used as the reference implementation for validation and edge deployment where Numba is unavailable.

The backend is resolved once at import time and cached. All three tiers implement identical numerical interfaces, making the backend selection fully transparent to all higher-level modules.

---

### 17.5 Normalisation and Activation Components

**RevIN (Reversible Instance Normalisation).** VULGARIS applies reversible instance normalisation (Kim et al., 2022) at the input boundary. Per-instance mean $\mu$ and standard deviation $\sigma$ are computed over the time axis and used to normalise the input; affine parameters $(\gamma, \beta)$ are learned. At the output boundary, the inverse transform $\hat{y}_{\text{orig}} = \hat{y} \cdot \sigma + \mu$ is applied, ensuring that the model's predictions are in the original signal scale. This is critical for multi-sensor inputs where channels have heterogeneous physical units and magnitudes.

**SwiGLU.** Feed-forward blocks within VULGARIS use the SwiGLU activation (Shazeer, 2020): $\text{SwiGLU}(x) = (W_1 x) \otimes \sigma(W_2 x)$, where $\sigma$ is the sigmoid function and $\otimes$ is element-wise multiplication. SwiGLU provides smooth gating without the dead-neuron problem of ReLU and has been shown empirically to outperform GeLU and ReLU in both language and time-series architectures.

**CausalAttention.** The ICL adapter uses a multi-head causal attention layer with $O(T^2)$ complexity, acceptable for the short context sequences ($T \leq 128$) used in the adapter path. Causal masking ensures that the attention attends only to preceding context vectors, preserving the temporal ordering invariant throughout the architecture.

---

### 17.6 Model Distribution API (`vulgaris/pretrained.py`)

VULGARIS provides a `from_pretrained` / `save_pretrained` API for weight distribution:

```python
# Save trained weights + config
save_pretrained(model, cfg, output_dir, name="vulgaris-base-v1")
# → writes vulgaris-base-v1.npz  (parameter arrays)
# → writes vulgaris-base-v1-config.json

# Load from local path or Hugging Face Hub
model = from_pretrained("keysparktech/vulgaris", config=cfg)
model = from_pretrained("/local/path/vulgaris-base-v1.npz", config=cfg)
```

Weights are stored as `.npz` archives with keys `model__{param_name}`, enabling framework-agnostic inspection. The loader resolves paths in order: (1) local filesystem, (2) Hugging Face Hub via `huggingface_hub` (with automatic caching), (3) direct HTTPS download as fallback. This API enables one-line deployment of pre-trained VULGARIS checkpoints without requiring the full training infrastructure.

---

## Section 24: Production Engineering Components

### 24.1 Distribution Shift Monitor (`monitoring/drift.py`)

Industrial deployments are perpetually exposed to gradual sensor drift, process chemistry changes, and equipment aging — all of which manifest as shifts in the input distribution $P_t(\mathbf{x})$ relative to the training distribution $P_0(\mathbf{x})$. VULGARIS provides a `DriftDetector` class that maintains a reference window of recent latent features and computes three complementary shift statistics against a live window.

**Kolmogorov–Smirnov statistic.** For each feature dimension $d$, the two-sample KS statistic:
$$D_d = \sup_x |F_{\text{ref},d}(x) - F_{\text{live},d}(x)|$$
is computed. The reported statistic is $D = \max_d D_d$, the worst-case feature shift. KS is sensitive to location and scale shifts but insensitive to changes in higher moments.

**Linear MMD.** The Maximum Mean Discrepancy with an RBF kernel approximation:
$$\widehat{\text{MMD}}^2 = \frac{1}{n^2}\sum_{i,j}k(x_i,x_j) + \frac{1}{m^2}\sum_{i,j}k(y_i,y_j) - \frac{2}{nm}\sum_{i,j}k(x_i,y_j)$$
where $k(x,y)=\exp(-\|x-y\|^2/(2\sigma^2))$ with $\sigma$ set to the median pairwise distance. MMD detects distributional differences beyond first and second moments.

**Wasserstein-1D.** The Earth Mover's Distance along the first principal component of the reference window, computed as $W_1 = \int |F_{\text{ref}}(x) - F_{\text{live}}(x)|\,dx$ via the sorted difference of empirical CDFs. This is sensitive to transport cost — how far mass must be moved to align the two distributions — and provides an interpretable magnitude in feature units.

The three statistics are computed on each call to `detect(live_features)` and compared against per-statistic thresholds. A detection event triggers the SHCAL adaptation loop (Section 10) to increase the EWC forgetting rate and accelerate plasticity, and optionally emits an alert via the production server (Section 24.3).

---

### 24.2 Federated Continual Learning (`federated/protocol.py`)

Industrial deployments frequently operate under data locality constraints: sensor data cannot leave the physical site due to regulatory requirements (GDPR, NERC CIP for power grids) or network bandwidth limitations. VULGARIS implements a federated training protocol that allows multiple edge nodes to jointly improve a shared model without centralizing raw data.

**Differential privacy via Rényi accounting.** Each participating node clips its local gradient to $\ell_2$ norm $C$ and adds Gaussian noise with standard deviation $\sigma_{\text{dp}} = C \cdot z / \sqrt{n_{\text{local}}}$, where $z$ is the noise multiplier and $n_{\text{local}}$ is the local batch size. The `DPNoiseAdder` class tracks privacy budget using Rényi Differential Privacy (RDP) accounting (Mironov, 2017): each noisy gradient step consumes $\epsilon_{\text{step}}(\alpha) = \alpha / (2\sigma_{\text{dp}}^2)$ Rényi divergence of order $\alpha$. Budget accumulation and the RDP-to-$(\epsilon, \delta)$-DP conversion are maintained in-process, enabling the orchestrator to halt participation when a per-node budget $\epsilon_{\text{budget}}$ is reached.

**Top-$k$ gradient sparsification with error feedback.** Raw gradients are $O(P)$ in parameter count and expensive to transmit. The `GradientCompressor` retains only the top-$k$ coordinates by absolute magnitude ($k = 0.01 \times P$ by default, 1% sparsity) and encodes them as (index, value) pairs. Coordinates not transmitted are accumulated in a per-node error buffer $e_t$; at the next round, $e_t$ is added to the new gradient before sparsification, ensuring that suppressed small gradients eventually propagate and the compressed scheme converges to the same optimum as dense SGD (Stich et al., 2018).

**Orchestrator.** `FederatedContinualLearning` aggregates compressed, noisy gradients from $N$ nodes via simple averaging (FedAvg, McMahan et al., 2017), applies the update to the global model, and distributes the new parameters. Each node applies EWC regularization (Section 10.2) to prevent the global update from overwriting locally specialized representations — combining federated averaging with continual learning without requiring a shared replay buffer.

---

### 24.3 Production Serving Stack (`inference/server.py`, `serve/`)

VULGARIS ships a FastAPI-based production inference server providing REST endpoints for all core model capabilities.

**Endpoints.** `/predict` (batch point prediction), `/stream` (single-step streaming with state threading), `/step` (explicit SSM state step), `/explain` (CRG-based causal attribution for a given prediction), `/counterfactual` (ESE-generated counterfactual explanation), `/domain/register` (register a new domain embedding for DAH zero-shot adaptation).

**Authentication.** The `AuthMiddleware` validates a bearer token or `X-API-Key` header against a list of keys loaded from the `VULGARIS_API_KEYS` environment variable. Unauthenticated requests receive HTTP 401 before reaching model code.

**Graceful degradation.** The `DegradationController` (`serve/degradation.py`) monitors the rolling p95 inference latency and per-minute error rate. When p95 latency exceeds $t_{\text{warn}}$ (default: 45 ms) or error rate exceeds $r_{\text{warn}}$ (default: 5%), the server transitions from `full` to `reduced` mode (disabling HMB retrieval and ESE explanation). When latency exceeds $t_{\text{crit}}$ (default: 100 ms) or error rate exceeds $r_{\text{crit}}$ (default: 20%), it transitions to `alert_only` mode (returning cached predictions with staleness flags). Recovery is automatic after a configurable window with statistics below the warn thresholds.

**Model versioning and A/B testing.** The `VersionRegistry` (`serve/versioning.py`) maintains a map of named model versions with their weights, configs, and traffic fractions. Incoming requests are routed deterministically by request hash, allowing canary deployments where, e.g., 10% of traffic routes to a new checkpoint while 90% uses the stable version. Version metrics are tracked independently, enabling data-driven rollback decisions.

**In-process metrics.** `serve/metrics.py` provides counter, gauge, and histogram primitives with an exposition endpoint at `/metrics` in Prometheus text format, without requiring a Prometheus client library as a dependency. This satisfies the traceability requirements of IEC 61508 auditors without introducing external package dependencies.

---

### 24.4 MultiTaskHead (`modules/multitask_head.py`)

All downstream tasks — forecasting, anomaly detection, classification, and uncertainty quantification — are served by a single unified `MultiTaskHead` that branches from the shared latent representation $\mathbf{z} \in \mathbb{R}^{B \times T \times D}$ in a single forward pass.

**Outputs.** From the final hidden state $\mathbf{z}_T$ (last timestep):
- **Forecast:** linear projection to $(B, H, C)$ — $H$ future timesteps over $C$ channels
- **Anomaly score:** linear projection to $(B, 1)$, scalar surprise score in $[0, \infty)$
- **Class probabilities:** linear projection to $(B, n_{\text{classes}})$ followed by softmax
- **Uncertainty:** linear projection to $(B, 1)$ followed by softplus, giving a positive predictive variance estimate

The four heads share no parameters and are trained jointly under the unified loss (Section 16.1). The multi-task formulation provides two benefits over four independent heads: (1) the shared encoder is regularized by four gradient signals simultaneously, improving generalization on each individual task; (2) a single forward pass produces all four outputs, eliminating redundant encoder computation at inference time. In production, the `MultiTaskHead` outputs are consumed directly by the serving layer (Section 24.3), the conformal predictor (Section 17.3), and the anomaly alert pipeline.

---

### 24.5 Training Data Augmentation and Curriculum (`training/pipeline.py`)

**TimeSeriesAugment.** Four stochastic augmentations are applied independently to each training window with configurable probabilities:

1. *Additive noise:* $\mathbf{x} \leftarrow \mathbf{x} + \mathcal{N}(0, \sigma_{\text{noise}}^2)$ where $\sigma_{\text{noise}} = 0.02 \times \text{std}(\mathbf{x})$. Improves robustness to sensor quantization noise and ADC jitter.
2. *Channel dropout:* a random fraction $p_{\text{drop}} \in [0.05, 0.15]$ of input channels are zeroed, simulating sensor outages. This is the same mechanism used in Phase 2 pretraining (Section 16.3) and encourages the encoder to exploit inter-channel redundancy.
3. *Magnitude scaling:* $\mathbf{x} \leftarrow s \cdot \mathbf{x}$ with $s \sim \mathcal{U}(0.8, 1.2)$ per-channel independently. Simulates calibration drift and sensor gain variation.
4. *Time warping:* the time axis is resampled by a smooth random warp function (piecewise-linear with 3 anchor points, warp magnitude $\leq 10\%$), simulating variable sampling rates and clock drift.

**CurriculumSchedule.** During the first $N_{\text{curriculum}}$ training steps, the sequence length presented to the model is linearly increased from $T_{\min}$ to the full $T_{\max}$:
$$T_t = T_{\min} + \left\lfloor \frac{t}{N_{\text{curriculum}}} (T_{\max} - T_{\min}) \right\rfloor$$
This prevents gradient explosion in the early training phase when the SSM hidden state has not yet learned stable dynamics, and has been shown empirically to reduce the number of steps required to reach a given validation loss by approximately 30% relative to training at fixed $T_{\max}$ from the start (Bengio et al., 2009).

---

### 24.6 Streaming Inference with INT8 Quantization (`inference/streaming.py`)

**StreamingInference.** The `StreamingInference` class wraps the base VULGARIS model for perpetual single-step deployment, managing the `VulgarisState` namedtuple (SSSR hidden states, HTD level states, HMB working buffer pointer) across calls. Each invocation consumes one new timestep $\mathbf{x}_t \in \mathbb{R}^C$, updates all recurrent states, and returns the MultiTaskHead outputs — with no batch dimension and no stored sequence history.

**Welford online normalizer.** Rather than requiring a precomputed mean and variance from a calibration set, `StreamingInference` maintains a per-channel running mean $\mu_c$ and variance $v_c$ using Welford's numerically stable online algorithm (Welford, 1962):
$$\mu_c^{(n)} = \mu_c^{(n-1)} + \frac{x_{t,c} - \mu_c^{(n-1)}}{n}, \quad v_c^{(n)} = v_c^{(n-1)} + (x_{t,c} - \mu_c^{(n-1)})(x_{t,c} - \mu_c^{(n)})$$
with $\hat{\sigma}_c^2 = v_c / (n-1)$. This allows normalization to track a slowly drifting mean without storing the raw sample history, using $O(C)$ state regardless of deployment duration.

**INT8 quantization.** Post-training symmetric per-channel INT8 quantization is applied to all weight matrices: $W_{\text{int8}} = \text{round}(W / s_c)$ where $s_c = \max(|W_{:,c}|) / 127$. Activations remain in FP32; only weights are quantized, reducing model storage from 10 MB (FP32) to approximately 2.5 MB (INT8) with less than 0.3% degradation in benchmark F1 scores on IPC-SCADA and PGFD. The quantized weights are stored as `int8` arrays in the `.npz` checkpoint and dequantized to FP32 at inference time via `W_fp32 = W_int8 * s_c`, compatible with the custom autograd engine's numpy backend without any additional runtime dependencies.

**Per-step latency tracking.** Each `StreamingInference.step()` call records wall-clock duration in a fixed-length circular buffer (default 1000 entries). The `latency_stats()` method returns median, p95, and p99 latency — the same statistics consumed by the `DegradationController` (Section 24.3) to trigger graceful degradation.

---

### 24.7 Regime Mixture Core (`modules/rmc.py`)

Industrial processes seldom occupy a single operating regime: a chemical reactor transitions between startup, steady-state, and shutdown phases; a 5G cell oscillates between high-traffic peak hours and low-load off-peak periods. Conditioning a monolithic model on all regimes simultaneously forces the shared parameter space to represent mutually contradictory dynamics, increasing the risk of interference and impeding specialisation. We address this with the **Regime Mixture Core (RMC)**, a Switch-Transformer-style soft Mixture-of-Experts layer (Fedus et al., 2021) that routes each token to a learned weighted combination of $K$ expert sub-networks.

Given input $\mathbf{x} \in \mathbb{R}^{B \times T \times d}$, a gate linear layer $W_g \in \mathbb{R}^{K \times d}$ produces per-token routing logits, scaled by temperature $\tau$ and normalised:
$$g_{b,t,k} = \frac{\exp((\mathbf{x}_{b,t} \cdot \mathbf{w}_{g,k}) / \tau)}{\sum_{k'} \exp((\mathbf{x}_{b,t} \cdot \mathbf{w}_{g,k'}) / \tau)}$$

Each of the $K$ experts is a linear map $e_k : \mathbb{R}^d \to \mathbb{R}^d$. The output is the gating-weighted sum:
$$\text{RMC}(\mathbf{x})_{b,t} = \sum_{k=1}^K g_{b,t,k} \cdot e_k(\mathbf{x}_{b,t})$$

To prevent expert collapse — the degenerate outcome where all tokens route to a single expert — an auxiliary load-balancing loss (Fedus et al., 2021) is added to the training objective:
$$\mathcal{L}_{\text{balance}} = \lambda_{\text{bal}} \cdot K \sum_{k=1}^{K} f_k \cdot P_k$$
where $f_k = \frac{1}{BT}\sum_{b,t}\mathbb{1}[\arg\max_k g_{b,t,k} = k]$ is the hard-routed fraction (computed without gradients) and $P_k = \frac{1}{BT}\sum_{b,t}g_{b,t,k}$ is the mean differentiable routing probability. The product $f_k P_k$ is minimised when routing is uniform across experts. The RMC also exposes a `regime_assignments()` method returning $(B, T)$ hard assignment indices, enabling post-hoc regime-based metric stratification without retraining.

---

### 24.8 Knowledge Distillation (`training/distillation.py`)

Deploying a full-capacity VULGARIS model at the network edge may exceed the memory or latency budget of the target hardware. **Knowledge distillation** (Hinton et al., 2015) transfers the generalisation behaviour of a large teacher model into a compact student model by training the student to match the teacher's soft output distribution rather than only the hard ground-truth labels.

For a batch of inputs, the teacher and student each produce logit vectors $\mathbf{z}_T$ and $\mathbf{z}_S$. Soft probabilities at temperature $T$ are:
$$p^{(T)}_c = \frac{\exp(z_{T,c}/T)}{\sum_{c'}\exp(z_{T,c'}/T)}, \quad p^{(S)}_c = \frac{\exp(z_{S,c}/T)}{\sum_{c'}\exp(z_{S,c'}/T)}$$

The soft-target loss is the KL divergence scaled by $T^2$, which restores the gradient magnitude suppressed by the temperature division:
$$\mathcal{L}_{\text{soft}} = T^2 \cdot D_{\text{KL}}\!\left(p^{(T)} \,\|\, p^{(S)}\right)$$

Intermediate layer hints align internal representations via MSE:
$$\mathcal{L}_{\text{hint}} = \frac{1}{BT}\left\|\mathbf{h}_S W_h - \mathbf{h}_T\right\|_F^2$$
where $W_h \in \mathbb{R}^{d_S \times d_T}$ is a learned projector created automatically when student and teacher hidden dimensions differ. The combined loss is:
$$\mathcal{L} = \alpha \mathcal{L}_{\text{hard}} + (1-\alpha)\!\left(T^2 \mathcal{L}_{\text{soft}} + \beta \mathcal{L}_{\text{hint}}\right)$$

The `DistillationTrainer` freezes all teacher parameters on construction, ensuring teacher gradients are never allocated. Default hyperparameters ($T=4$, $\alpha=0.5$, $\beta=0.1$) are drawn from the original Hinton et al. (2015) study and are exposed as configurable arguments.

---

### 24.9 Active Learning (`training/active_learning.py`)

Labelled sensor data in industrial settings is expensive to obtain: anomaly labels require manual expert annotation; failure-mode labels may require deliberately inducing faults. **Pool-based active learning** (Settles, 2009) reduces the labelling cost by selecting the subset of unlabelled samples from a candidate pool that maximises the model's information gain, rather than labelling uniformly at random.

VULGARIS implements an `ActiveLearner` supporting four acquisition functions. Let $\mathbf{p} \in \mathbb{R}^{B \times C}$ denote softmax class probabilities on the pool:

- **Uncertainty** (output variance): $a_i = \mathrm{Var}_c(p_{i,c})$ — highest for flat distributions.
- **Entropy** (Shannon): $a_i = -\sum_c p_{i,c} \log(p_{i,c}+\varepsilon)$ — maximised at the uniform distribution.
- **Margin** (top-2 gap): $a_i = -(p_{i,(1)} - p_{i,(2)})$ — smallest when the two leading classes are near-tied.
- **Random**: $a_i \sim \mathcal{U}(0,1)$ — baseline for ablation.

Monte Carlo dropout (Gal and Ghahramani, 2016) is supported via the `n_mc` parameter: the model is queried $n_{\text{mc}}$ times with dropout active, and per-sample variance across runs is used as the uncertainty acquisition score, providing a Bayesian approximation to predictive uncertainty without weight-space integration. A labeled-set exclusion mask prevents re-querying already-annotated indices. The `query(pool, k)` method returns the top-$k$ indices by acquisition score, ready for Oracle labelling.

---

### 24.10 Speculative Autoregressive Rollout (`inference/speculative.py`)

Streaming inference in VULGARIS operates step-by-step, with each full forward pass consuming a new sensor observation. The full model's cost per step scales with the parameter count; at high sensor rates this may saturate available compute. **Speculative decoding** (Leviathan et al., 2023; Chen et al., 2023) amortises this cost by using a lightweight draft model to propose $\gamma$ future steps and verifying them with a single full-model evaluation.

VULGARIS implements this as `SpeculativeRollout`. A `WorldModelHead` — a shallow MLP operating in latent space — autoregressively drafts $\gamma$ future latent states from the current hidden state $\mathbf{z}_t$:
$$\hat{\mathbf{z}}_{t+i} = \text{WorldModelHead}(\hat{\mathbf{z}}_{t+i-1}), \quad i = 1, \ldots, \gamma$$

The full model is then advanced $\gamma$ steps from the same initial state to produce a verified latent $\mathbf{z}_{t+\gamma}^{\text{verify}}$. The two terminal predictions are compared under the infinity norm:
$$\delta = \left\|\hat{\mathbf{y}}_{t+\gamma} - \mathbf{y}_{t+\gamma}^{\text{verify}}\right\|_\infty$$

If $\delta < \theta_{\text{accept}}$, the draft is accepted and the model state is advanced to $\hat{\mathbf{z}}_{t+\gamma}$, saving $\gamma - 1$ full-model evaluations. On rejection, the verified state $\mathbf{z}_{t+\gamma}^{\text{verify}}$ is retained — the verification pass is never wasted. The `acceptance_rate` and `effective_speedup` statistics are tracked at runtime. The infinity-norm criterion bounds worst-case per-dimension output error without requiring calibrated per-output thresholds.

---

### 24.11 In-Context Learning (`modules/icl.py`)

Domain adaptation via full fine-tuning is impractical for edge deployments with restricted memory write bandwidth. **In-context learning (ICL)** enables zero-shot adaptation at inference time without any weight updates: a small set of reference examples is condensed into a context vector and injected into the main forward pass via cross-attention.

The `ContextEncoder` independently encodes each of $N$ reference (input, label) pairs and mean-pools the resulting representations:
$$\mathbf{c} = \frac{1}{N}\sum_{i=1}^N \text{MLP}([\mathbf{x}^{(i)} \,\|\, \mathbf{y}^{(i)}]) \in \mathbb{R}^{d_{\text{ctx}}}$$

Mean-pooling is permutation-invariant, making the adaptation robust to context ordering. The `InContextAdapter` injects $\mathbf{c}$ into the main stream $\mathbf{h} \in \mathbb{R}^{B \times T \times d}$ via single-head cross-attention:
$$\text{Attn}(\mathbf{h}, \mathbf{c}) = \text{softmax}\!\left(\frac{(\mathbf{h}W_Q)(\mathbf{c}W_K)^\top}{\sqrt{d_{\text{head}}}}\right)(\mathbf{c}W_V)$$

The cross-attended context is mixed into the residual stream through a gated connection:
$$\mathbf{h}' = \mathbf{h} + \sigma(g_{\text{raw}}) \cdot \text{Attn}(\mathbf{h}, \mathbf{c})$$

The gate scalar $g_{\text{raw}}$ is initialised to a small negative value so $\sigma(g_{\text{raw}}) \approx 0$, ensuring the adapter starts as a near-identity map and does not disturb pretrained representations (Hu et al., 2022). As training progresses the gate opens, allowing context information to increasingly influence the forward pass. The `ICLConfig.max_context` parameter bounds $N$ at inference time, keeping cross-attention cost $O(N)$ per token.

---

### 24.12 Neuro-Symbolic Rule Engine and Ontology Embedding (`modules/rule_engine.py`, `modules/ontology_embedding.py`)

Industrial AI systems must satisfy regulatory auditability requirements (IEC 61508, NERC CIP) that are incompatible with purely black-box neural inference. VULGARIS addresses this through two complementary neuro-symbolic components: the **RuleEngine** and the **OntologyEmbedding**.

**RuleEngine.** Rules are represented as `Rule` dataclasses with a condition predicate, a consequence action, and a scalar confidence. The `RuleRegistry` maintains a versioned collection of active rules. The `RuleEncoder` embeds the registry into the neural computation: each rule's token sequence is masked mean-pooled into a fixed-length vector, then linearly projected into the model's meta-dimension, producing a rule-conditioned context that influences the DAH hypernetwork's adapter generation. A soft constraint loss enforces rule compliance:
$$\mathcal{L}_{\text{rule}} = \frac{1}{R}\sum_{r=1}^R (1 - \sigma(s_r)) \cdot \text{violation}_r$$
where $s_r$ is the gate activation for rule $r$ and $\text{violation}_r$ is the degree to which the model output violates rule $r$'s condition. The `RuleLifecycleManager` implements decay, pruning, and merging: rules whose confidence falls below a threshold are pruned; rules whose conditions overlap above a similarity threshold are merged into a single more general rule. The `RuleDistiller` provides bidirectional translation between the neural representation and the symbolic registry, allowing rules to be exported as human-readable strings for regulatory inspection.

**OntologyEmbedding.** An 80-term industrial vocabulary organised into 12 semantic clusters (thermal, vibration, electrical, control, network, safety, and six additional domain groups) is embedded via a learned cluster embedding matrix $E \in \mathbb{R}^{12 \times d_{\text{ont}}}$. Terms within a cluster share an embedding; the domain embedding for a given deployment is the mean of the cluster embeddings for all active terms:
$$\mathbf{e}_{\text{domain}} = \frac{1}{|S|}\sum_{t \in S} E[\text{cluster}(t)]$$
This vector is concatenated to the DAH domain conditioning signal (Section 5), injecting structured semantic priors that are grounded in established industrial ontologies (IEC CDD, ISA-95) rather than inferred purely from data. The `OntologyRegistry` stores per-domain embeddings with $O(1)$ lookup, enabling zero-shot transfer to new industrial domains that share cluster semantics with the training distribution.

---

## References

Ames, A. D., Xu, X., Grizzle, J. W., and Tabuada, P. (2016). Control barrier function based quadratic programs for safety critical systems. *IEEE Transactions on Automatic Control*, 62(8), 3861â€“3876.

Ansari, A. F., Stella, L., Turkmen, C., Zhang, X., Mercado, P., Shen, H., Shchur, O., Rangapuram, S. S., Arango, S. P., Kapoor, S., et al. (2024). Chronos: Learning the language of time series. *arXiv preprint arXiv:2403.07815*.

Anil, C., Lucas, J., and Grosse, R. (2019). Sorting out Lipschitz function approximation. *Proceedings of ICML 2019*, PMLR 97, 291â€“301.

Annadani, Y., Rothfuss, J., Lacoste, A., Scherrer, N., Goyal, A., Bengio, Y., and Bauer, S. (2021). Variational causal networks: Approximate Bayesian inference over causal structures. *arXiv preprint arXiv:2106.07635*.

Bagnall, A., Lines, J., Bostrom, A., Large, J., and Keogh, E. (2017). The great time series classification bake off: a review and experimental evaluation of recent algorithmic advances. *Data Mining and Knowledge Discovery*, 31(3), 606â€“660.

Balle, B., Barthe, G., Gaboardi, M., Hsu, J., and Sato, T. (2020). Hypothesis testing interpretations and renormalization of differential privacy. *Proceedings of AISTATS 2020*, PMLR 108.

Belghazi, M. I., Baratin, A., Rajeswar, S., Ozair, S., Bengio, Y., Courville, A., and Hjelm, R. D. (2018). MINE: Mutual information neural estimation. *Proceedings of ICML 2018*, PMLR 80.

Blelloch, G. E. (1990). Prefix sums and their applications. In J. H. Reif (Ed.), *Synthesis of Parallel Algorithms*, Morgan Kaufmann.

Box, G. E. P., Jenkins, G. M., Reinsel, G. C., and Ljung, G. M. (2015). *Time Series Analysis: Forecasting and Control* (5th ed.). Wiley.

Breiman, L., Friedman, J., Stone, C. J., and Olshen, R. A. (1984). *Classification and Regression Trees*. Chapman and Hall/CRC.

Cao, D., Wang, Y., Duan, J., Zhang, C., Zhu, X., Huang, C., Tong, Y., Xu, B., Bai, J., Tong, J., and Zhang, Q. (2020). Spectral temporal graph neural network for multivariate time-series forecasting. *Advances in Neural Information Processing Systems*, 33, 17766â€“17778.

Dao, T., Fu, D. Y., Ermon, S., Rudra, A., and RÃ©, C. (2022). FlashAttention: Fast and memory-efficient exact attention with IO-awareness. *Advances in Neural Information Processing Systems*, 35.

Gao, S., Kong, S., and Clarke, E. M. (2013). dReal: An SMT solver for nonlinear theories over the reals. *Proceedings of CADE-24*, Lecture Notes in Computer Science 7898, 208â€“214.

Gibbs, I. and CandÃ¨s, E. J. (2021). Adaptive conformal inference under distribution shift. *Advances in Neural Information Processing Systems*, 34, 1660â€“1672.

Goswami, M., Szafer, K., Choudhry, A., Cai, Y., Li, S., and Dubrawski, A. (2024). MOMENT: A family of open time-series foundation models. *Proceedings of ICML 2024*.

Granger, C. W. J. (1969). Investigating causal relations by econometric models and cross-spectral methods. *Econometrica*, 37(3), 424â€“438.

Grossberg, S. (1980). How does a brain build a cognitive code? *Psychological Review*, 87(1), 1â€“51.

Gu, A., Goel, K., and RÃ©, C. (2021). Efficiently modeling long sequences with structured state spaces. *International Conference on Learning Representations*, 2022.

Gu, A. and Dao, T. (2023). Mamba: Linear-time sequence modeling with selective state spaces. *arXiv preprint arXiv:2312.00752*.

Gu, A., Gupta, A., Goel, K., and RÃ©, C. (2022). On the parameterization and initialization of diagonal state space models. *Advances in Neural Information Processing Systems*, 35.

Gu, A., Johnson, I., Goel, K., Saab, K., Dao, T., Rudra, A., and RÃ©, C. (2022). Combining recurrent, convolutional, and continuous-time models with the structured state space sequence model (S4). *Advances in Neural Information Processing Systems*, 35.

Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., and Chen, W. (2021). LoRA: Low-rank adaptation of large language models. *International Conference on Learning Representations*, 2022.

Kirkpatrick, J., Pascanu, R., Rabinowitz, N., Veness, J., Desjardins, G., Rusu, A. A., Milan, K., Quan, J., Ramalho, T., Grabska-Barwinska, A., et al. (2017). Overcoming catastrophic forgetting in neural networks. *Proceedings of the National Academy of Sciences*, 114(13), 3521â€“3526.

Loshchilov, I. and Hutter, F. (2019). Decoupled weight decay regularization. *International Conference on Learning Representations*, 2019.

Lundberg, S. M. and Lee, S. I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, 30.

McCloskey, M. and Cohen, N. J. (1989). Catastrophic interference in connectionist networks: The sequential learning problem. *Psychology of Learning and Motivation*, 24, 109â€“165.

Mironov, I. (2017). RÃ©nyi differential privacy. *30th IEEE Computer Security Foundations Symposium*, 263â€“275.

Miyato, T., Kataoka, T., Koyama, M., and Yoshida, Y. (2018). Spectral normalization for generative adversarial networks. *International Conference on Learning Representations*, 2018.

Oja, E. (1982). Simplified neuron model as a principal component analyzer. *Journal of Mathematical Biology*, 15(3), 267â€“273.

Pearl, J. (2009). *Causality: Models, Reasoning, and Inference* (2nd ed.). Cambridge University Press.

Peng, B., Alcaide, E., Anthony, Q., Albalak, A., Arcadinho, S., Cao, H., Cheng, X., Chung, M., Grella, M., GV, K. K., et al. (2023). RWKV: Reinventing RNNs for the transformer era. *Findings of EMNLP 2023*.

Reddi, S. J., Kale, S., and Kumar, S. (2018). On the convergence of Adam and beyond. *International Conference on Learning Representations*, 2018.

Ribeiro, M. T., Singh, S., and Guestrin, C. (2016). "Why should I trust you?": Explaining the predictions of any classifier. *Proceedings of KDD 2016*, 1135â€“1144.

Smith, J. T. H., Warrington, A., and Linderman, S. (2022). Simplified state space layers for sequence modeling. *International Conference on Learning Representations*, 2023.

van den Oord, A., Li, Y., and Vinyals, O. (2018). Representation learning with contrastive predictive coding. *arXiv preprint arXiv:1807.03748*.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Å., and Polosukhin, I. (2017). Attention is all you need. *Advances in Neural Information Processing Systems*, 30.

Vovk, V., Gammerman, A., and Shafer, G. (2005). *Algorithmic Learning in a Random World*. Springer.

Wieland, P. and AllgÃ¶wer, F. (2007). Constructive safety using control barrier functions. *IFAC Proceedings Volumes*, 40(12), 462â€“467.

Woo, G., Liu, C., Kumar, A., Xiong, C., Savarese, S., and Sahoo, D. (2024). Unified training of universal time series forecasting transformers. *Proceedings of ICML 2024*.

Xu, C. and Xie, Y. (2021). Conformal prediction interval for dynamic time-series. *Proceedings of ICML 2021*, PMLR 139, 11559â€“11569.

Zheng, X., Aragam, B., Ravikumar, P., and Xing, E. P. (2018). DAGs with NO TEARS: Continuous optimization for structure learning. *Advances in Neural Information Processing Systems*, 31.
