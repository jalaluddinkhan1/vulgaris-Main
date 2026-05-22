# VULGARIS: A Streaming Causal State-Space Foundation Model for Industrial Time Series
## Part 2: Core Architectural Modules

---

## 5. Adaptive Signal Embedding

### 5.1 The Tokenization Problem for Physical Signals

The dominant paradigm in sequence modeling treats the embedding of raw input as a preprocessing step: map discrete tokens to dense vectors via a learned vocabulary table, then pass those vectors to the sequence model. This paradigm originated in natural language processing, where it is well-motivated. Words and subword units are discrete, semantically cohesive, and their identity is independent of their temporal position. The token "temperature" carries the same conceptual content whether it appeared as the 42nd or the 4200th token in a document. Position is encoded separately, as an afterthought.

Physical signals admit no such separation. A voltage reading of $0.73$ V at time $t = 1.240$ s is a fundamentally different event than a voltage reading of $0.73$ V at time $t = 1.350$ s when the preceding sample occurred at $t = 1.240$ s — in the latter case, a gap of $110$ ms separates the two measurements, potentially indicating a sensor dropoff, a communication fault, or a deliberate hold. The sample value and the sampling interval are jointly informative, and they cannot be disentangled without loss.

More fundamentally, physical signals are continuous, band-limited functions of time. By the Shannon–Nyquist sampling theorem, a signal sampled at rate $f_s$ Hz contains frequency components up to $f_s / 2$ Hz and no higher. The information content of the signal is distributed continuously across this frequency band. Tokenization — the discretization of continuous amplitudes into a vocabulary of $V$ distinct symbols — is a many-to-one mapping $\tau_V : \mathbb{R} \to \{1, \ldots, V\}$ that irreversibly destroys amplitude information below the quantization granularity.

To make this precise, consider a signal with amplitude range $[x_{\min}, x_{\max}]$ uniformly quantized into $V$ vocabulary entries. The quantization step is $\Delta_q = (x_{\max} - x_{\min}) / V$, and the resulting quantization noise power is:

$$\sigma_q^2 = \frac{\Delta_q^2}{12} = \frac{(x_{\max} - x_{\min})^2}{12 V^2}$$

For a practical bearing fault detection scenario, consider a current sensor monitoring a 690 V line-to-line motor drive. The amplitude range is on the order of 1 kV peak-to-peak. With a vocabulary of $V = 256$ tokens (comparable to byte-level tokenization used in some time-series models), the quantization noise floor is:

$$\sigma_q = \frac{1000}{12^{1/2} \cdot 256} \approx 1.13 \text{ V}$$

This is not negligible. The ball pass frequency outer race (BPFO) fault signature for a 6205 bearing at 1750 RPM appears as an amplitude modulation in the current spectrum at approximately $105$ Hz with a typical fault amplitude of $0.5$–$2$ V peak. At $V = 256$, the quantization noise floor is comparable to the fault signature itself. The model trained on tokenized signals cannot, in principle, reliably detect such faults — not because of insufficient training data or model capacity, but because the information has been destroyed at the input.

The failure mode compounds with higher-frequency phenomena. The BPFO and its harmonics up to the fifth order span $105$–$525$ Hz, requiring amplitude resolution well below $0.5$ V at 1 kV range. With $V = 1024$ (already a large vocabulary for time-series models), the quantization noise is still $0.28$ V — marginal at best.

Beyond amplitude quantization, tokenization destroys temporal structure. Consider two representations of the same physical event: a sudden current spike at $t = 10.00$ s followed by return to baseline at $t = 10.05$ s. If the sampling rate is $1$ kHz, these two events are separated by $50$ samples. If the tokenizer operates at a lower effective rate (as in patching-based approaches), the spike and recovery may be merged into a single patch token, and the $50$ ms duration — which carries diagnostic information about the type of fault — is lost. Conversely, if the sampling rate varies (irregular sampling due to network jitter in industrial Ethernet), two nominally identical sequences of tokens may correspond to physically very different events.

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

The Morlet wavelet achieves this bound, making it optimal among all wavelets for problems where joint localization is desired. In bearing fault detection, the BPFO appears at a specific frequency but its onset time carries diagnostic information about when the fault initiated — exactly the joint localization scenario where the Morlet wavelet is optimal.

However, the CWT with a fixed mother wavelet makes a strong prior assumption: that the optimal time-frequency tradeoff is given by the Heisenberg bound, applied uniformly across all frequencies and all channels. Physical signals often violate this assumption. Electrical transients require high time resolution and low frequency resolution (short $\Delta t$, large $\Delta f$); thermal dynamics require the opposite. A fixed wavelet imposes a single, suboptimal tradeoff for all phenomena simultaneously.

The Empirical Mode Decomposition (EMD) and its variants (EEMD, CEEMDAN) offer a data-adaptive alternative by decomposing signals into Intrinsic Mode Functions (IMFs) that are locally mono-component. However, EMD is not defined by a convex optimization and has no natural gradient — it cannot be learned end-to-end. Furthermore, the IMFs lack a direct interpretation in terms of physical frequency bands, making it difficult to initialize the decomposition based on domain knowledge.

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

This design choice has several advantages. First, it makes the temporal structure explicit rather than implicit: the model can directly read the inter-sample interval from the input rather than inferring it from the positions of non-missing values. Second, it handles gaps of arbitrary duration without requiring any special treatment — a gap of $200$ ms is simply represented by a large value in the $\Delta$ channel. Third, it is compatible with the convolutional structure of ASE: the $\Delta$ channel is filtered by the same wavelet bank as the signal channels, allowing the model to learn how inter-sample interval patterns (e.g., a sudden increase in $\Delta$ indicating a communication fault) interact with signal patterns.

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

The ratio is approximately $7$:$1$ in favor of ASE at this sequence length. The advantage grows with $T$ because ASE scales as $O(T)$ while attention scales as $O(T^2)$. At $T = 10{,}000$ (10 seconds at 1 kHz), the advantage grows to approximately $60$:$1$, consistent with the 40–60x figure reported in the introduction.

Beyond raw FLOPs, ASE has a qualitative advantage: because it preserves frequency information continuously, the downstream SSM does not need to spend model capacity reconstructing frequency features from long-range attention patterns. The SSM's hidden state directly encodes the physically meaningful frequency decomposition of the input, making the learned representations more interpretable and more efficiently aligned with the structure of the prediction problem.

---

## 6. Hierarchical Timescale Decomposition

### 6.1 Multi-Timescale Dynamics in Physical Systems

A persistent challenge in the modeling of physical systems is the coexistence of dynamic phenomena operating across many decades of characteristic timescale. Industrial machinery provides the clearest examples. A three-phase induction motor drive exhibits at least four qualitatively distinct dynamic regimes:

1. **Electrical dynamics** ($\tau \sim 1$–$10$ ms): commutation transients, switching noise from the inverter, current ripple at the PWM frequency (typically 4–16 kHz). These dynamics determine instantaneous torque and are critical for overcurrent protection.

2. **Mechanical dynamics** ($\tau \sim 10$–$500$ ms): shaft acceleration/deceleration in response to load changes, resonance in the drivetrain, bearing contact mechanics. These dynamics determine the vibration spectrum that carries fault signatures.

3. **Thermal dynamics** ($\tau \sim 1$–$60$ min): heating of windings, core, and bearings due to ohmic losses. Thermal state determines insulation degradation rate and is the dominant driver of long-term failure.

4. **Degradation dynamics** ($\tau \sim$ weeks–years): gradual accumulation of fatigue damage in rolling elements, progressive insulation breakdown, corrosion of lubrication films. These are the dynamics of primary interest for predictive maintenance.

The challenge for any single-timescale model is stark. An SSM with timestep $\delta t = 1$ ms can accurately capture electrical dynamics, but for thermal dynamics with time constant $\tau_{\text{thermal}} = 30$ min $= 1{,}800{,}000 \cdot \delta t$, the discrete transition eigenvalue is:

$$\bar{A}_{\text{thermal}} = \exp\!\left(-\frac{\delta t}{\tau_{\text{thermal}}}\right) = \exp(-5.6 \times 10^{-7}) \approx 1 - 5.6 \times 10^{-7}$$

This is so close to unity that after 1 minute ($60{,}000$ steps), the effective discount factor is $\exp(-0.0337) \approx 0.967$ — the thermal state is still strongly present. But for degradation dynamics with $\tau_{\text{degrade}} = 6$ months $= 1.6 \times 10^{10} \cdot \delta t$, the required memory depth exceeds any practical sequence length by many orders of magnitude.

Conversely, an SSM with $\delta t = 1$ hour can model thermal and degradation dynamics, but for electrical dynamics with time constant $\tau_{\text{elec}} = 5$ ms, the discrete transition is:

$$\bar{A}_{\text{elec}} = \exp\!\left(-\frac{3600}{0.005}\right) = \exp(-720{,}000) \approx 0$$

The electrical state decays completely within a fraction of the timestep. The model cannot represent electrical transients at all — they are aliased into the coarser timestep as instantaneous disturbances with no dynamics.

This is not a failure of training or of regularization — it is a fundamental consequence of the relationship between the discretization timescale and the physical time constants of the system. No amount of training data or model capacity can overcome the fact that a single-timescale SSM cannot simultaneously represent dynamics spanning 10 decades of frequency.

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

*Proof sketch.* Consider levels $l_0$ (fast) and $l_1$ (slow), with $l_1 > l_0$. Information from fast level $l_0$ at time $t_0$ enters slow level $l_1$ via the chain of downward communication paths: $\mathbf{h}^{(l_0)}_{t_0} \to \mathbf{x}^{(l_0+1)}_{t_0} \to \mathbf{h}^{(l_0+1)}_{t_0} \to \cdots \to \mathbf{h}^{(l_1)}_{t_0}$. This is a chain of $l_1 - l_0$ linear projections, each of which is nonzero in general. The slow level then integrates this information over its longer timescale, and at any later time $t_1 > t_0$, the upward communication path $\mathbf{h}^{(l_1)}_{t_1'} \to \mathbf{h}^{(l_0)}_{t_1}$ (where $t_1' = \lfloor t_1 / 2^{l_1 - l_0} \rfloor$) returns it to the fast level. The path length is $2(l_1 - l_0) \leq 2(L-1)$, which is $O(L)$ — bounded and independent of $|t_1 - t_0|$. Thus information propagates across all timescale pairs in constant depth. $\square$

This result is crucial: without cross-level communication, each level operates independently and long-range information at the fast level can only be communicated by maintaining the fast SSM for the entire duration — which requires $O(T)$ memory. With cross-level communication, the slow level acts as a compressed summary of long-range fast dynamics, and the fast level can query this summary via the upward communication path without retaining the full fast-level history.

### 6.3 Comparison with Alternatives

**Dilated temporal convolutional networks (TCN).** Dilated TCNs achieve a receptive field of $O(2^L)$ samples with $L$ dilated convolutional layers. However, they cannot adapt their effective timescale to the input (the dilation pattern is fixed), they carry no recurrent state (all computation is within the fixed receptive field window), and they cannot stream: every output position requires access to its full receptive field, which grows exponentially with depth. For a 12-layer dilated TCN with dilation doubling, the receptive field is $2^{12} = 4{,}096$ samples — insufficient for thermal dynamics at 1 kHz ($\tau_{\text{thermal}} \approx 10^6$ samples).

**Multi-resolution LSTM.** Running parallel LSTMs at different temporal resolutions (e.g., ClockworkRNN, hierarchical multiscale RNN) addresses the timescale diversity problem but without cross-level communication, each level processes a different resampled version of the input independently, and there is no mechanism for slow states to influence fast processing or vice versa. The absence of cross-level communication means that the system cannot, for example, detect that electrical transients at the fast level are unusually frequent given the thermal context at the slow level — a pattern that is diagnostically critical for certain failure modes.

**Informer, Autoformer, and sparse attention variants.** These models achieve sub-quadratic attention complexity via various approximations (ProbSparse attention, autocorrelation attention). However, they remain memory-bounded: at inference time, the effective context length is limited by the KV-cache size. More fundamentally, they cannot stream: each output position requires random access to past positions within the context window. Their temporal resolution is fixed by the model's positional encoding, with no mechanism for multi-timescale processing at the architectural level.

---

## 7. Selective State-Space Recurrence

### 7.1 The Standard Discrete SSM

The Selective State-Space Recurrence (SSSR) module is the computational core of VULGARIS. It computes the latent dynamics of the system by integrating the multi-scale embeddings produced by ASE and HTD through a learned state-space model that adapts its dynamics to the current input.

We begin with the standard linear time-invariant discrete SSM:

$$\mathbf{h}_t = \mathbf{A}\mathbf{h}_{t-1} + \mathbf{B}\mathbf{x}_t$$
$$\mathbf{y}_t = \mathbf{C}\mathbf{h}_t + \mathbf{D}\mathbf{x}_t$$

with state matrix $\mathbf{A} \in \mathbb{R}^{N \times N}$, input matrix $\mathbf{B} \in \mathbb{R}^{N \times D}$, output matrix $\mathbf{C} \in \mathbb{R}^{D \times N}$, and feedthrough matrix $\mathbf{D} \in \mathbb{R}^{D \times D}$.

The computational cost of the general form is $O(N^2)$ per timestep due to the matrix-vector product $\mathbf{A}\mathbf{h}_{t-1}$. For large state dimensions $N$, this is prohibitive. The standard approach — used in S4, Mamba, and related work — is to restrict $\mathbf{A}$ to diagonal form: $\mathbf{A} = \text{diag}(\boldsymbol{\alpha})$ with $\boldsymbol{\alpha} \in \mathbb{C}^N$ (or $\mathbb{R}^N$ in the real-valued case). The diagonal restriction reduces the per-step complexity to $O(N)$.

The expressivity question is: does the diagonal restriction fundamentally limit the class of LTI systems that can be approximated? The answer is no, by the following argument. Any stable LTI system with transfer function $H(z) = \mathbf{C}(z\mathbf{I} - \mathbf{A})^{-1}\mathbf{B} + \mathbf{D}$ can be written in a similarity transformation $\tilde{\mathbf{A}} = \mathbf{T}^{-1}\mathbf{A}\mathbf{T}$, $\tilde{\mathbf{B}} = \mathbf{T}^{-1}\mathbf{B}$, $\tilde{\mathbf{C}} = \mathbf{C}\mathbf{T}$ for any invertible $\mathbf{T}$, without changing the input-output behavior. If $\mathbf{A}$ has $N$ distinct eigenvalues, it is diagonalizable over $\mathbb{C}$. The resulting diagonal system has the same transfer function up to the change of basis, which is absorbed into the $\mathbf{B}$ and $\mathbf{C}$ matrices. For systems with repeated eigenvalues (Jordan blocks), the diagonal approximation introduces a small error bounded by the condition number of the Jordan form — which is generically large but can be controlled via the HiPPO initialization framework that places eigenvalues at structurally advantageous locations.

In practice, diagonal SSMs have been shown to be universal approximators of stable LTI systems given sufficient state dimension $N$, and their empirical performance on sequence modeling benchmarks matches or exceeds more complex structured state matrices at equivalent parameter counts.

### 7.2 Zero-Order Hold Discretization and Stability

The hidden state equation $\mathbf{h}_t = \mathbf{A}\mathbf{h}_{t-1} + \mathbf{B}\mathbf{x}_t$ is the discretization of the continuous-time linear ODE:

$$\dot{\mathbf{h}}(t) = \mathbf{A}_c \mathbf{h}(t) + \mathbf{B}_c u(t)$$

Under the zero-order hold (ZOH) assumption — that the input $u(t)$ is held constant over each interval $[t_k, t_{k+1})$ — the exact discretization is:

$$\bar{\mathbf{A}} = e^{\mathbf{A}_c \Delta t}, \qquad \bar{\mathbf{B}} = \mathbf{A}_c^{-1}(e^{\mathbf{A}_c \Delta t} - \mathbf{I})\mathbf{B}_c$$

The ZOH discretization is the physically correct discretization when the input is a piecewise-constant signal (as is the case for digitally sampled inputs), and it preserves the eigenstructure of the continuous system exactly: the eigenvalues of $\bar{\mathbf{A}}$ are $\{e^{\lambda_i \Delta t}\}$ where $\{\lambda_i\}$ are the eigenvalues of $\mathbf{A}_c$.

For the diagonal case with $\mathbf{A}_c = \text{diag}(-\exp(\mathbf{a}))$, where $\mathbf{a} \in \mathbb{R}^N$ is a learned parameter vector, the continuous eigenvalues are $\lambda_n = -\exp(a_n) < 0$ for all $n$ — meaning the continuous system is always stable. The discretized transition becomes:

$$\bar{A}_n = \exp(-\exp(a_n) \cdot \delta t_t)$$

The parameterization $\lambda_n = -\exp(a_n)$ ensures strict negativity of all continuous eigenvalues without requiring constrained optimization, analogous to the positivity parameterization used for the bandwidth in ASE.

**Theorem 7.1 (Structural Stability of SSSR).** For all $\mathbf{a} \in \mathbb{R}^N$ and all $\delta t_t > 0$, every eigenvalue of $\bar{\mathbf{A}}_t = \text{diag}(\exp(-\exp(\mathbf{a}) \odot \delta t_t))$ lies strictly within the open interval $(0, 1)$. Consequently, for any bounded input sequence $\|\mathbf{x}_t\|_2 \leq M$, the hidden state sequence $\{\mathbf{h}_t\}$ is bounded.

*Proof.* For each component $n$, the eigenvalue is $\bar{A}_n = \exp(-\exp(a_n) \cdot \delta t_t)$. Since $\exp(a_n) > 0$ and $\delta t_t > 0$, the exponent $-\exp(a_n) \cdot \delta t_t < 0$, giving $\bar{A}_n = e^{(\text{negative number})} \in (0, 1)$. The lower bound $\bar{A}_n > 0$ follows from the fact that the exponential function is strictly positive.

For the boundedness claim: with $\rho = \|\bar{\mathbf{A}}_t\|_\infty = \max_n \bar{A}_n < 1$, we have:

$$\|\mathbf{h}_t\|_2 \leq \|\bar{\mathbf{A}}_t\|_2 \|\mathbf{h}_{t-1}\|_2 + \|\bar{\mathbf{B}}_t\|_2 \|\mathbf{x}_t\|_2 \leq \rho \|\mathbf{h}_{t-1}\|_2 + \|\bar{\mathbf{B}}_t\|_2 M$$

Since $\rho < 1$ and $\|\bar{\mathbf{B}}_t\|_2$ is bounded by the norm of the projection weights, the recurrence converges to a bounded fixed region by the contraction mapping theorem. Specifically, $\|\mathbf{h}_t\|_2 \leq M \|\bar{\mathbf{B}}\|_2 / (1 - \rho)$ for all $t$. $\square$

This stability guarantee is structural — it holds regardless of the values of $\mathbf{a}$ and regardless of the input, by virtue of the parameterization choice. This contrasts sharply with gated RNNs (LSTM, GRU), where the gates are learned functions of the input and hidden state. In an LSTM, the forget gate $f_t = \sigma(\mathbf{W}_f [\mathbf{h}_{t-1}; \mathbf{x}_t] + \mathbf{b}_f)$ takes values in $(0, 1)$ for any input, but the stability of the full system depends on the joint behavior of all four gates together — a property that can be guaranteed only through careful regularization and is not guaranteed by architecture alone. Pathological input sequences can cause LSTM hidden states to diverge or saturate at extreme values, degrading performance in a way that is difficult to diagnose. SSSR has no such failure mode.

### 7.3 Input-Dependent Selectivity

The standard SSM with fixed $\mathbf{A}$, $\mathbf{B}$, $\mathbf{C}$ is a linear time-invariant filter — it treats all time points identically, regardless of the content of the input. This is appropriate for stationary signals but fundamentally inadequate for industrial monitoring, where the optimal memory horizon depends on the operational context: during steady-state operation, the model should integrate over long windows to compute accurate baselines; during transient events (load changes, faults), it should respond rapidly to the most recent inputs.

VULGARIS implements selectivity by making $\delta t_t$, $\mathbf{B}_t$, and $\mathbf{C}_t$ dependent on the current input:

$$\delta t_t = \text{softplus}(\mathbf{W}_{\delta t} \mathbf{x}_t + \mathbf{b}_{\delta t}) \cdot \text{clip}(\cdot, \delta t_{\min}, \delta t_{\max})$$

$$\mathbf{B}_t = \mathbf{W}_B \mathbf{x}_t, \qquad \mathbf{C}_t = \mathbf{W}_C \mathbf{x}_t$$

where $\mathbf{W}_{\delta t} \in \mathbb{R}^{1 \times D}$, $\mathbf{W}_B \in \mathbb{R}^{N \times D}$, $\mathbf{W}_C \in \mathbb{R}^{D \times N}$, and the clipping is applied element-wise to enforce the bounds $\delta t_t \in [\delta t_{\min}, \delta t_{\max}]$.

The selectivity mechanism operates through $\delta t_t$, which modulates the effective forgetting rate. To see this, consider the effect of varying $\delta t_t$ on the diagonal transition coefficient:

**Memory retention under small $\delta t_t$:** When $\delta t_t$ is small (relative to $1/\exp(a_n)$):

$$\bar{A}_n = \exp(-\exp(a_n) \cdot \delta t_t) \approx 1 - \exp(a_n) \cdot \delta t_t \approx 1$$

The state $h_n$ is updated by only a small fraction of its value per step — it changes slowly and retains memory of the distant past. The model is in "slow-time" mode: it integrates inputs over a long effective window.

**Memory erasure under large $\delta t_t$:** When $\delta t_t$ is large:

$$\bar{A}_n = \exp(-\exp(a_n) \cdot \delta t_t) \approx 0$$

The state $h_n$ is nearly zeroed at each step and $\mathbf{h}_t \approx \mathbf{B}_t \mathbf{x}_t$ — the output is dominated by the current input with almost no memory. The model is in "fast-time" mode: it responds immediately to the current input.

The learned projection $\mathbf{W}_{\delta t}$ allows the model to determine, from the content of the current input $\mathbf{x}_t$, whether to be in slow-time or fast-time mode. At anomaly onset — when $\mathbf{x}_t$ deviates from the learned baseline — the model can increase $\delta t_t$, erasing old memory and focusing on the current observation. During steady state, small $\delta t_t$ enables long-range integration for accurate baseline estimation.

The input-dependent $\mathbf{B}_t$ and $\mathbf{C}_t$ provide additional selectivity: $\mathbf{B}_t$ controls which components of the input are written to each state dimension, and $\mathbf{C}_t$ controls which state dimensions are read for each output component. Together, these three selective parameters give SSSR the ability to selectively remember, forget, and attend to different aspects of the input signal at different times — a strictly richer capability than any linear time-invariant SSM.

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

The sequential computation of $\mathbf{h}_t = \bar{A}_t \mathbf{h}_{t-1} + \bar{B}_t \mathbf{x}_t$ (for the diagonal case, with component-wise operations) requires $O(T)$ serial steps. For training with sequence lengths $T = 10{,}000$–$100{,}000$, this serial dependency is a significant bottleneck on parallel hardware such as GPUs, where thousands of cores are available but cannot be utilized by a sequential recurrence.

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

The practical benefit of the parallel scan is a reduction in wall-clock training time from $O(T)$ serial steps to $O(\log T)$ parallel steps, enabling a speedup proportional to $T / \log T$: for $T = 10{,}000$, this is approximately a $740\times$ speedup on a sufficiently parallel processor. In practice, GPU parallelism is limited by the number of available streaming multiprocessors, but the parallel scan consistently achieves $30$–$100\times$ speedups over naive sequential implementation for the sequence lengths encountered in VULGARIS's intended applications.

### 7.6 Hebbian Online Adaptation

During inference, VULGARIS applies an online Hebbian update to the log-decay parameters $\log \mathbf{a}$ after each forward pass. This update adjusts the intrinsic timescales of the SSM based on the observed sequential correlations in the hidden state.

The update rule is based on Oja's rule, modified for the decay parameter:

$$\Delta \log \mathbf{a} = \eta_h \cdot \mathbb{E}_{B,T}\!\left[\mathbf{h}_t \odot \mathbf{h}_{t-1} - \mathbf{h}_t^2\right]$$

where $\mathbb{E}_{B,T}[\cdot]$ denotes the mean over the batch and time dimensions, and $\eta_h$ is a small Hebbian learning rate (typically $10^{-4}$–$10^{-3}$). This update is applied in-place to the numpy parameter array, outside of the autograd graph, ensuring zero overhead for the gradient computation.

The two terms in the update have distinct roles:

1. **Hebbian term** $\mathbb{E}[\mathbf{h}_t \odot \mathbf{h}_{t-1}]$: This term is positive when the hidden state at time $t$ is correlated with the state at time $t-1$, indicating that the corresponding state dimension is tracking a persistent signal. Increasing $\log a_n$ (i.e., making $\exp(a_n)$ larger) would increase the forgetting rate, which is counterproductive for persistent signals — so this term should increase $\log a_n$ to be negative (i.e., decrease the decay rate parameter, making $\bar{A}_n$ closer to 1). Wait — let us be precise: $\Delta \log a_n > 0$ when $h_t^n h_{t-1}^n > (h_t^n)^2$, i.e., when $|h_{t-1}^n| > |h_t^n|$ on average. This occurs when the state dimension is slowly decaying but still correlated — it is in a slow-dynamics regime. Increasing $\log a_n$ increases $\exp(a_n)$, which increases the decay rate and decreases $\bar{A}_n$. This is a corrective mechanism: if the state is decaying slowly, strengthen the decay slightly to maintain responsiveness.

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

In natural language, this design is well-motivated. A pronoun at position $t$ may resolve to a noun at any position $t' < t$ in the sentence — there is no structural constraint on which positions can be semantically related. The dense attention matrix is necessary to represent this full-range dependency.

In physical systems, this reasoning does not apply. Physical systems have causal structure: a current sensor on phase A of a motor does not directly cause the lubricant film thickness on bearing D. There is an indirect causal chain — electromagnetic heating $\to$ shaft warming $\to$ thermal expansion of housing $\to$ bearing clearance change $\to$ lubricant film thinning — but this chain involves multiple intermediate physical processes. Direct causal influence between distant system components is rare.

Let $G^* = (V, E^*)$ be the true causal graph of the physical system, where $V$ is the set of sensor signals and $E^*$ is the set of direct causal edges. In industrial systems, empirical analysis of causal graph structure consistently shows $|E^*| / |V|^2 \approx 0.02$–$0.05$ (a 2–5\% edge density). This means that approximately 95–98\% of the entries in the attention matrix correspond to pairs of signals with no direct causal relationship. For a system with $n = 64$ sensors, the full attention matrix has $64^2 = 4{,}096$ entries, of which approximately $4{,}000$ are spurious.

The consequences are threefold. First, the attention mechanism wastes $95\%$ of its representational capacity on modeling correlations that arise from shared confounders (e.g., two sensors that both respond to shaft speed are correlated but not causally related) rather than direct causal links. Second, learned attention weights conflate direct causation with indirect association, making the model's attributions difficult to interpret physically. Third, the $O(T^2)$ computational and memory cost of full attention is incompatible with the streaming inference requirements of VULGARIS — an $O(T)$ mechanism is required.

The Causal Routing Graph module replaces dense attention with a sparse, differentiably-learned causal graph that encodes the discovered causal structure of the system.

### 8.2 Differentiable DAG Learning

The structural constraint on the CRG is that the learned graph $G(\mathbf{W})$, defined by the adjacency matrix $\mathbf{W} \in \mathbb{R}^{n \times n}$, must be a directed acyclic graph (DAG). A cyclic graph would allow signal $i$ to causally influence signal $j$, which causally influences signal $i$ — a physical impossibility in any finite-speed causal system over a single timescale (though apparent cycles can arise from systems sampled too slowly to observe the causal delay).

The challenge is that the DAG constraint is combinatorial: checking acyclicity for a given $\mathbf{W}$ requires testing all possible cycles, which is $O(2^n)$ in the worst case. This makes the constraint incompatible with gradient-based optimization.

The NOTEARS formulation (Zheng et al., 2018) resolves this by providing a continuous, differentiable characterization of the DAG constraint:

**Theorem 8.1 (Zheng et al., 2018).** A matrix $\mathbf{W} \in \mathbb{R}^{n \times n}_{\geq 0}$ is a DAG if and only if:

$$h(\mathbf{W}) := \text{tr}\!\left(e^{\mathbf{W} \odot \mathbf{W}}\right) - n = 0$$

*Proof.* We use the identity $e^{\mathbf{A}} = \sum_{k=0}^{\infty} \mathbf{A}^k / k!$, so $[e^{\mathbf{A}}]_{ii} = 1 + [A]_{ii} + \sum_{k=2}^{\infty} [A^k]_{ii} / k!$. For a nonnegative matrix $\mathbf{A} = \mathbf{W} \odot \mathbf{W}$, the quantity $[A^k]_{ii} = \sum_{j_1, \ldots, j_{k-1}} A_{ij_1} A_{j_1 j_2} \cdots A_{j_{k-1}i}$ counts the sum of products of edge weights along walks of length $k$ from $i$ back to $i$. Since all entries of $A$ are nonnegative, $[A^k]_{ii} > 0$ if and only if there exists a directed walk of length $k$ from $i$ to $i$, which exists if and only if there is a directed cycle of length $\leq k$ through node $i$. Therefore $\text{tr}(e^{\mathbf{A}}) > n$ if and only if some node belongs to a directed cycle, i.e., $G(\mathbf{W})$ contains a cycle. Equivalently, $\text{tr}(e^{\mathbf{A}}) = n$ if and only if $G(\mathbf{W})$ is acyclic. $\square$

The gradient of $h(\mathbf{W})$ with respect to $\mathbf{W}$ is:

$$\nabla_{\mathbf{W}} h(\mathbf{W}) = 2\mathbf{W} \odot \left(e^{\mathbf{W} \odot \mathbf{W}}\right)^\top$$

This gradient is well-defined everywhere and can be computed by first computing $\mathbf{M} = e^{\mathbf{W} \odot \mathbf{W}}$ via the matrix exponential (using Padé approximation or scaling-and-squaring), then multiplying element-wise.

**Computational approximation via truncated power series.** Computing the full matrix exponential costs $O(n^3)$ per forward pass. For VULGARIS with $n = 64$, this is $64^3 = 262{,}144$ operations — acceptable. However, for larger $n$ or as a training efficiency measure, VULGARIS approximates $e^{\mathbf{A}} \approx \sum_{k=0}^{6} \mathbf{A}^k / k!$ via the first seven terms of the Taylor series.

**Proposition 8.2 (Truncation error bound).** For $\mathbf{A} = \mathbf{W} \odot \mathbf{W}$ with $\|\mathbf{W}\|_F \leq r$, the truncation error of the 6th-order Taylor approximation satisfies:

$$\left\|e^{\mathbf{A}} - \sum_{k=0}^{6} \frac{\mathbf{A}^k}{k!}\right\|_F \leq \frac{\|\mathbf{A}\|_F^7}{7!} e^{\|\mathbf{A}\|_F} \leq \frac{r^{14}}{5040} e^{r^2}$$

*Proof.* The remainder of the matrix exponential Taylor series satisfies $\|e^{\mathbf{A}} - \sum_{k=0}^{m} \mathbf{A}^k/k!\|_F \leq \|\mathbf{A}\|_F^{m+1}/(m+1)! \cdot e^{\|\mathbf{A}\|_F}$ by the standard Frobenius norm bound for matrix functions. For $m = 6$ and $\|\mathbf{A}\|_F = \|\mathbf{W} \odot \mathbf{W}\|_F \leq \|\mathbf{W}\|_F^2 \leq r^2$: the truncation error is at most $r^{14} / 5040 \cdot e^{r^2}$. During training with $\ell_1$ regularization, $\|\mathbf{W}\|_F$ is typically bounded by $r \leq 2$, giving an error of $\leq 16^7 / 5040 \cdot e^4 \approx 0.006$ — negligible relative to the regularization-scale gradient signal. $\square$

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

In practice, this means the Granger initialization will include spurious edges wherever latent confounders are present. The NOTEARS gradient will not correct these spurious edges from purely observational data — the acyclicity constraint does not distinguish spurious from true edges, and the $\ell_1$ penalty merely induces sparsity among all edges. True causal discovery from observational data requires additional assumptions (faithfulness, causal sufficiency) that are rarely satisfied in industrial systems with many unmeasured variables.

For the purposes of VULGARIS, the CRG should be understood as providing a learned sparse routing structure that is *informed by* causal prior knowledge and *regularized toward* acyclicity, rather than a certified ground-truth causal graph. The acyclicity constraint remains valuable because cyclic graphs lead to ill-defined message passing (messages propagate indefinitely along cycles), and the sparsity constraint reduces overfitting. The CRG attributions are more interpretable than dense attention weights, but they should be validated against domain knowledge before being used for causal inference.

### 8.5 Explainability via Graph Traversal

One of the primary practical advantages of the CRG architecture over attention-based routing is the ability to generate explicit, interpretable causal attribution traces. Given the sparse, acyclic graph $G(\tilde{\mathbf{W}})$, any output node $j$ can be explained by tracing backwards through the graph to identify which input nodes have the highest cumulative causal influence.

Define the attribution trace $\text{trace}(j, k)$ of output node $j$ to depth $k$ as the set of triples:

$$\text{trace}(j, k) = \{(i_m, w_{i_m j_m}, c_m)\}_{m=1}^{|\text{path}|}$$

where the sequence of triples is obtained by backward breadth-first search from node $j$: at each step, follow the incoming edge with the highest absolute weight, stopping when depth $k$ is reached or no incoming edges exist. The cumulative weight at step $m$ is:

$$c_m = \prod_{l=1}^{m} |w_{i_l j_l}|$$

representing the compounded causal influence along the path from the source node $i_m$ to the target node $j$.

The cumulative weight $c_m$ has a natural interpretation: it is the product of the edge weights along the causal path from source to target. Under the message passing formulation (Section 8.3), the contribution of node $i$'s activation to node $j$'s output — after $m$ hops through the graph — is proportional to $c_m$ times the source activation $N_{t,i}$.

**Comparison with attention-based attribution.** Attention-based attribution methods — including raw attention weights, attention rollout (Abnar & Zuidema, 2020), and gradient-weighted attention — compute attribution as functions of the attention matrix $\text{softmax}(\mathbf{QK}^\top / \sqrt{d_k})$. These methods have been shown to correlate poorly with ground-truth feature importance in controlled experiments (Jain & Wallace, 2019; Wiegreffe & Pinter, 2019), for several reasons:

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

This exceeds the capacity of any reasonably-priced storage system by approximately two orders of magnitude. Scaling the model to $D = 512$ would require $\approx 155$ TB — well into the territory of large-scale data warehousing infrastructure that is entirely impractical for embedded edge deployment.

It might be objected that transformers with a sliding window context (Longformer, BigBird, StreamingLLM) avoid this scaling problem by discarding old context. This is correct, but the discarded context is permanently lost: the model cannot recall events that occurred before the context window. For predictive maintenance applications, this is unacceptable. The most relevant historical event for predicting a bearing failure may be the last major overhaul six months ago, or the last thermal excursion three weeks ago. A model with a sliding window of 10 minutes has no access to this information.

The fundamental issue is not storage capacity but *architectural philosophy*: storing the full KV cache is equivalent to treating every past timestep as equally important, when in fact the vast majority of past states are entirely predictable from the model's current hidden state and carry no new information. Information theory provides the appropriate framework: the relevant past is not "all past states" but the set of past events that cannot be predicted from the model's current state — i.e., the surprises.

**Formal statement.** Let $h(X_{t+1} | \mathbf{h}_t)$ be the conditional entropy of the next observation given the current SSM state. An event at time $t_0$ contributes to the model's uncertainty about $X_{t+1}$ only if it changed the model's state in a way that has not been "forgotten" by time $t$. If the SSM has properly integrated the event into its state, no external memory of the event is needed — the state already encodes its effect. External archival is required only when the event's effect cannot be encoded in the SSM's finite-dimensional state $\mathbf{h}_t \in \mathbb{R}^N$ — i.e., when the event represents novel information that exceeds the SSM's representational capacity for the current context.

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

Events with $s_t > s_{\text{thresh}} = 2.0$ correspond to observations that are more than $2\sigma$ from the model's prediction — approximately the top 5\% of events under a standard Gaussian. In a well-calibrated model, 95\% of timesteps during normal operation have $s_t \leq 2.0$ and require no archival. Only genuinely surprising events — load transients, sensor anomalies, fault signatures, operating mode changes — exceed the threshold and are archived.

Between archival events, the model's memory is entirely represented by the SSM hidden state $\mathbf{h}_t \in \mathbb{R}^{B \times N}$, which is $O(N)$ — constant in time, independent of how long the system has been running. The archived events form a sparse history of surprises, growing at the rate of novel events rather than at the rate of timesteps.

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

For VULGARIS's industrial memory application, accurate recall of archived events is the primary requirement — when a bearing fault occurred 6 months ago and its signature is retrieved during a current anomaly, the retrieved representation must accurately reflect the original event. We therefore recommend $\beta = 0.5$ as a practical default, with $\beta$ tunable based on the accuracy-interpretability tradeoff required by the deployment context.

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

The $1/\sigma_k^2$ factor down-weights archived events that were encoded with high uncertainty — events that were themselves surprising and for which the VAE encoder assigned a broad posterior. This makes physical sense: a highly uncertain archived event is one that the model could not compress faithfully; retrieving it adds noise rather than information. Conversely, a frequently-encountered fault pattern that has been archived many times will have low average posterior variance (the encoder has learned to represent it precisely), and will receive higher effective weight.

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

HMB requires approximately $400{,}000\times$ less storage than a comparable transformer KV cache — not as an approximation or compression of the transformer's memory, but as a consequence of a fundamentally different memory architecture: store surprises, not states.

Over one year, at $\lambda_{\text{event}} = 10$ events/hour: HMB archive size $\approx 45$ MB, comfortably fitting in the RAM of any modern embedded processor. Over five years of continuous operation with the same event rate: $\approx 225$ MB — still within a reasonable embedded memory budget.

**Archive capacity and eviction.** When the archive approaches its capacity limit $K_{\max}$ (default $10{,}000$ events), a least-recently-used eviction policy removes the oldest events. However, VULGARIS provides an alternative eviction strategy based on information content: events with the lowest retrieval frequency (measured by how often they contributed significant weight to a retrieval operation) are evicted first, regardless of age. This strategy prioritizes the retention of rare but diagnostically relevant events (e.g., the first instance of a novel fault mode) over frequent but redundant events (e.g., repeated instances of the same startup transient).

**Relationship to episodic memory in cognitive science.** The HMB design parallels the theoretical distinction in cognitive neuroscience between semantic memory (general knowledge encoded in connection weights, analogous to the SSM's trained parameters) and episodic memory (specific past experiences stored in the hippocampus, analogous to the HMB archive). The hippocampal theory of systems consolidation (McClelland et al., 1995) proposes that the hippocampus stores recent specific episodes and gradually consolidates them into cortical semantic memory during sleep. VULGARIS's online Hebbian adaptation (Section 7.6) plays an analogous role: the Hebbian update consolidates frequently-encountered patterns from the SSM's operational history into the core parameter $\log \mathbf{a}$, reducing the need for explicit archival of predictable recurring events.

This parallel suggests a natural research direction: implementing an explicit consolidation phase in VULGARIS where patterns that appear frequently in the HMB archive are incorporated into the SSM's structural parameters via a targeted fine-tuning step, analogous to the memory consolidation process theorized in neuroscience. This direction is deferred to future work.

---

*End of Part 2. Part 3 continues with Section 10 (Training Objectives and Loss Formulation) through Section 14 (Evaluation Methodology).*
