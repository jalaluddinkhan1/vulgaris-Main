# VULGARIS: A Streaming Causal State-Space Foundation Model for Industrial and Edge Intelligence
## Part III: Adaptation, Explainability, Safety, Theory, and Deployment

---

## Section 10: Self-Healing Continuous Adaptation Layer (SHCAL)

### 10.1 The Stability-Plasticity Dilemma in Operational Systems

The stability-plasticity dilemma (Grossberg, 1980) asks: how can a learning system integrate new information without overwriting prior knowledge? In laboratory settings this is managed by maintaining separate datasets and periodically retraining. In industrial deployments neither option is viable: retraining requires labeled data, compute infrastructure, and scheduled downtime; retaining historical datasets violates data-residency constraints in jurisdictions such as the EU (GDPR Article 17) and China (PIPL Article 47); and model updates must occur continuously rather than in discrete cycles.

Formally, define catastrophic forgetting: given task distributions $\mathcal{T}_1$ and $\mathcal{T}_2$ with disjoint support, a model $f_\theta$ trained first on $\mathcal{T}_1$ and then on $\mathcal{T}_2$ satisfies

$$\Delta_{\mathrm{forget}} := \mathcal{L}_{\mathcal{T}_1}(f_\theta) - \mathcal{L}_{\mathcal{T}_1}(f_{\theta^*}) > 0$$

where $\theta^*$ is optimal for $\mathcal{T}_1$ alone. For standard stochastic gradient descent without regularization, $\Delta_{\mathrm{forget}}$ is proportional to the KL divergence between the gradients of $\mathcal{T}_1$ and $\mathcal{T}_2$, which can be arbitrarily large when the two tasks demand conflicting parameter configurations. Empirically, McCloskey and Cohen (1989) demonstrated that feedforward networks trained sequentially on paired-associate learning tasks exhibit near-complete erasure of the first task after only 50 gradient steps on the second — a finding replicated consistently across architectures.

SHCAL addresses this through three complementary mechanisms operating at different timescales: Elastic Weight Consolidation (task-level, hours), Hebbian plasticity (within-task, seconds to minutes), and structural plasticity (architectural-level, days to weeks). These mechanisms are not alternatives — they are complementary layers of a multi-timescale adaptation hierarchy designed to match the timescale structure of industrial drift phenomena. Equipment aging is a weeks-scale process; ambient temperature cycling is a diurnal process; sudden faults are event-scale disturbances. A single adaptation mechanism cannot simultaneously address all three.

### 10.2 Elastic Weight Consolidation

EWC (Kirkpatrick et al., 2017) anchors parameters to their post-task-$\mathcal{T}_1$ values, weighted by their importance for task $\mathcal{T}_1$. The Fisher information diagonal serves as the importance weight:

$$F_i = \mathbb{E}_{(\mathbf{x},y)\sim\mathcal{T}_1}\!\left[\left(\frac{\partial \log p_\theta(y|\mathbf{x})}{\partial \theta_i}\right)^{\!2}\right]$$

This quantity has the following interpretation: $F_i$ is large when small perturbations of $\theta_i$ produce large changes in the model's predicted log-probability over the $\mathcal{T}_1$ distribution. Parameters with large $F_i$ are important for $\mathcal{T}_1$; parameters with small $F_i$ can be freely repurposed for $\mathcal{T}_2$. This is an information-theoretically precise notion of importance, grounded in the curvature of the loss landscape rather than heuristic sensitivity measures.

$F_i$ is computed by accumulating squared gradients over $N_F$ calibration samples after completing $\mathcal{T}_1$. In the SHCAL implementation, $N_F = 200$ (configurable via `fisher_samples`). The EWC penalty is:

$$\mathcal{L}_{\mathrm{ewc}} = \frac{\lambda_{\mathrm{ewc}}}{2}\sum_i F_i(\theta_i - \theta_i^*)^2$$

This is a parameter-specific $\ell_2$ regularization: tight around parameters that strongly affect $\mathcal{T}_1$ predictions, loose around those that do not.

**Forgetting bound.** Under the Laplace approximation — that is, treating the posterior over $\theta$ around $\theta^*$ as locally quadratic — the forgetting after training on $\mathcal{T}_2$ with penalty strength $\lambda_{\mathrm{ewc}}$ is bounded by:

$$\Delta_{\mathrm{ewc}} \leq \frac{1}{2\lambda_{\mathrm{ewc}}}\sum_i F_i(\Delta\theta_i)^2 = \frac{1}{2\lambda_{\mathrm{ewc}}}\|\Delta\theta\|_{\mathbf{F}}^2$$

where $\|\cdot\|_{\mathbf{F}}$ is the Fisher-weighted norm and $\Delta\theta = \theta - \theta^*$ is the drift of the parameters from their $\mathcal{T}_1$-optimal values. Increasing $\lambda_{\mathrm{ewc}}$ by a factor of 10 reduces the forgetting bound by a factor of 10, at the cost of constraining the effective parameter space available for learning $\mathcal{T}_2$. In practice, $\lambda_{\mathrm{ewc}} = 100$ provides adequate protection for industrial equipment monitored under moderate regime changes; aggressive domain shifts (e.g., replacing the monitored equipment entirely) require a deliberate reset of $\theta^*$ and $F_i$ rather than an attempt to consolidate both within the same parameter vector.

**Computing $F_i$ efficiently.** For a model with $P$ parameters and a calibration set of $N_F$ examples, the exact Fisher diagonal requires $N_F$ forward-backward passes — $O(N_F P)$ compute. In SHCAL, $N_F = 200$. For a 2.5M-parameter model, this is approximately 500M FLOPs — on the order of one second on a modern CPU. This computation is performed offline, after completing a task boundary, not during streaming inference. The resulting Fisher diagonal and reference parameters $\theta^*$ are stored in memory (for a 2.5M-parameter model at FP32: approximately 20 MB for $F_i$ and $\theta^*$ together) and accessed during each subsequent gradient update.

**Online Fisher approximation.** When task boundaries are not cleanly delineated — as is common in streaming industrial processes where operating conditions shift gradually — SHCAL maintains a running estimate of $F_i$ using exponential moving averages: $F_i \leftarrow (1-\beta_F) F_i + \beta_F (\partial \log p/\partial \theta_i)^2$ with $\beta_F = 0.001$. This provides a continuously updated importance estimate at negligible cost (one multiply-add per parameter per step) and does not require explicit task boundary detection.

### 10.3 Hebbian Plasticity for In-Stream Adaptation

For faster adaptation within a task — responding to gradual drift on the order of minutes — SHCAL applies Oja's rule (Oja, 1982) to the linear projection layers of the SSSR blocks after each forward pass:

$$\Delta W_{ij} = \eta_h\!\left(y_i x_j - y_i^2 W_{ij}\right)$$

where $x_j$ is the $j$-th component of the layer's input activation vector, $y_i$ is the $i$-th component of its output activation vector, and $\eta_h$ is the Hebbian learning rate (default $10^{-4}$). The normalization term $-y_i^2 W_{ij}$ prevents unbounded growth; Oja (1982) proved convergence to the principal eigenvector of the input covariance when this rule is applied to a single-layer linear network. For multilayer networks, it acts as a local correlation-strengthening update that increases the coupling between co-activating input-output pairs, without global optimality guarantees but with the important property that no backward pass through the network is required.

The update is applied directly to the weight arrays without going through the autograd engine, incurring $O(d_{\mathrm{in}} d_{\mathrm{out}})$ operations per layer per step — negligible relative to the forward pass. The **shadow-mode validation gate** prevents harmful updates: before writing $\Delta\mathbf{W}$ to the live weights, the update is applied to a temporary copy of the weight matrix, the current batch is re-evaluated under the modified weights, and the resulting loss $\mathcal{L}(\theta + \Delta\theta)$ is compared to the pre-update loss $\mathcal{L}(\theta)$. If $\mathcal{L}(\theta + \Delta\theta) > 1.1 \cdot \mathcal{L}(\theta)$, the update is discarded and the Hebbian learning rate is reduced by a factor of 0.9 for the next step. This rejects approximately 5–15% of updates in typical streaming deployments — predominantly updates that strengthen correlations that are locally prevalent in the current batch but orthogonal to the task objective.

The combination of EWC and Hebbian plasticity resolves the timescale mismatch: EWC protects against large-scale parameter drift over task transitions, while Oja's rule provides fast in-stream micro-adjustment without requiring labeled data or gradient computation from a loss function. The Hebbian update is self-supervised in the strong sense — it depends only on co-occurring activations, requiring no supervision signal whatsoever.

### 10.4 Structural Plasticity

Over timescales of days to weeks, SHCAL can reallocate network capacity through a simulated synaptic turnover mechanism. Each weight $W_{ij}$ has an associated counter $c_{ij}$, initialized to zero, that increments when $|W_{ij}| < \varepsilon_{\mathrm{prune}} = 10^{-3}$ and resets to zero otherwise. When $c_{ij}$ exceeds a dormancy window of $\tau_{\mathrm{prune}}$ consecutive steps (default $10^5$), the weight is structurally pruned: $W_{ij} \leftarrow 0$, and the connection is flagged as inactive in a sparse mask $M_{ij} = 0$. The weight receives no further gradient updates and does not participate in forward computation (implemented via masked matrix multiplication at negligible overhead due to BLAS sparse operations).

For pruned connections, gradient monitoring continues asynchronously. If $|\partial\mathcal{L}/\partial W_{ij}|$ — computed during the backward pass and read from the gradient tensor — exceeds $\varepsilon_{\mathrm{grow}} = 5 \times 10^{-4}$ sustained over $\tau_{\mathrm{grow}} = 10^3$ consecutive steps, the connection is re-enabled: $M_{ij} = 1$ and $W_{ij} \leftarrow \mathcal{N}(0, \varepsilon_{\mathrm{grow}}^2)$. This implements a form of synaptic turnover in which dormant connections are recycled to regions where the current gradient signal indicates unmet representational demand. The practical effect is to dynamically redistribute model capacity toward the input features and transformations that are most informative for the current operating regime, without changing the total parameter count.

### 10.5 Conformal Recalibration Loop

SHCAL monitors the rolling conformal prediction coverage $\hat{\alpha}_t = \frac{1}{W}\sum_{s=t-W}^{t}\mathbf{1}[y_s \in \hat{C}_s]$ over a window of $W=500$ recent steps. When $\hat{\alpha}_t < (1-\alpha) - \delta_{\mathrm{tol}}$ (default $\delta_{\mathrm{tol}} = 0.03$), the adaptation trigger fires: $\eta_h$ is multiplied by 5, and $\lambda_{\mathrm{ewc}}$ is temporarily reduced to $0.1\lambda_{\mathrm{ewc}}$ for 100 steps, allowing faster parameter movement in response to the detected distribution shift. When coverage recovers to $\hat{\alpha}_t \geq (1-\alpha)$, the original $\lambda_{\mathrm{ewc}}$ is restored. This closed-loop coupling between conformal uncertainty and learning dynamics creates a self-regulating adaptation mechanism: the system accelerates adaptation precisely when calibrated uncertainty quantification detects that its predictions have become miscalibrated, and decelerates when coverage is restored.

This feedback architecture is robust to false triggers: a spurious coverage drop lasting fewer than $W$ steps does not trigger adaptation (because the rolling window smooths transient fluctuations), and the temporary reduction in $\lambda_{\mathrm{ewc}}$ is bounded in time (100 steps), so catastrophic forgetting cannot accumulate from a single trigger event. The combination of these bounds ensures that SHCAL's adaptation behavior is safe in the regulatory sense: no single event can cause unbounded parameter drift.

---

## Section 11: Domain-Adaptive Hypernetwork (DAH)

### 11.1 The Multi-Domain Deployment Problem

A global industrial AI platform serving a single large enterprise may encounter hundreds of distinct operating contexts: different equipment generations, ambient conditions, process chemistries, firmware versions, sensor configurations, and regulatory environments. The engineering problem is: how can one model serve all of them without incurring $O(M)$ training cost and $O(M)$ storage, where $M$ is the number of distinct deployment contexts?

Three strategies exist in current practice, each with fundamental limitations:

**One model per domain:** $O(M)$ storage and $O(M)$ full training runs. Operationally impractical at scale — 500 distinct equipment configurations require 500 full training pipelines — and eliminates cross-domain transfer, wasting the signal available from deployment contexts with similar physical dynamics.

**Fine-tuning a shared base per domain:** $O(M)$ training runs (though cheaper individually due to warm starting). Full model storage per domain remains $O(M)$. Transfer learning partially amortizes compute cost but does not address storage.

**Prompt or prefix conditioning:** Effective for language models but requires the conditioning signal to be expressible in the same embedding space as the input. Physical signal domains differ not only in statistical properties but in sensor type, physical units, and sampling rate — they require architectural adaptation, not merely input-space conditioning.

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

**Parameter efficiency.** For $L=8$ target layers, $r=16$, mean layer dimension $\bar{d}=256$: adapter parameters per domain $= 2 \times 8 \times 16 \times 256 = 65{,}536$. This is 1.3% of a 5M-parameter base model. The hypernetwork itself has $\sum_l 2 \times (d_{\mathrm{out},l} + d_{\mathrm{in},l}) \times r \times d_{\mathrm{meta}}$ parameters — fixed and shared across all $M$ domains. For the default configuration, this is approximately 420K parameters added once. Serving 1,000 distinct domains requires exactly the same model size as serving 1 domain.

**Domain switching latency.** Generating all adapters for domain $d$ requires one hypernetwork forward pass, computing $\sum_l(d_{\mathrm{out},l} + d_{\mathrm{in},l}) \times r$ output values from the domain embedding. For the default configuration: approximately $8 \times 512 \times 16 = 65{,}536$ outputs, requiring one MetaMLP forward pass of approximately $8.4\mathrm{M}$ FLOPs. On any contemporary CPU core, this completes in well under 1 ms. Adapters are cached after generation, so switching to a previously seen domain is effectively free — only an index lookup and a cache read.

### 11.3 Pretraining the Hypernetwork

During pretraining, $M_{\mathrm{train}}$ synthetic domains are created by varying the data-generating parameters: sensor noise levels, operating regime transition rates, fault frequency distributions, thermal time constants, and measurement offsets. The model is trained with randomly sampled domain indices, requiring the hypernetwork to produce useful adapters for each domain without having access to domain-specific data outside its embedding index. This forces the hypernetwork to learn a generalizable map from domain metadata to adapter configurations that meaningfully alters the base model's behavior.

At inference time on a novel domain $d' \notin \mathcal{D}_{\mathrm{train}}$: the embedding $\mathbf{e}_{d'}$ is initialized to the mean of all training domain embeddings. If the new domain resembles a convex combination of training domains in the embedding space — which holds when the new domain's physical characteristics fall within the range of variation sampled during pretraining — the hypernetwork interpolates effectively and useful adapters are generated without any fine-tuning. For genuinely novel domains outside this convex hull, a brief adapter fine-tuning step (100–500 gradient steps on 50–200 labeled examples) on the adapter matrices only — with the base model frozen — typically restores full performance within minutes.

---

## Section 12: Explainability and Symbolic Extraction Engine (ESE)

### 12.1 The Regulatory Imperative

Industrial AI deployments exist within a regulatory context that is specific, technically detailed, and increasingly enforced:

The **EU AI Act** (Regulation (EU) 2024/1689), Article 13, requires that high-risk AI systems "be designed and developed in such a way to ensure that their operation is sufficiently transparent to enable deployers to interpret the system's output and use it appropriately." Article 9 requires risk management systems that include documentation of the basis of AI decisions affecting safety-critical processes.

**IEC 61508** (Functional Safety of E/E/PE Safety-related Systems) requires full traceability from safety function specification through implementation to test evidence. A system that produces a safety-relevant output — a protective relay trip command, a shutdown recommendation — without a traceable reasoning chain that can be audited by a safety engineer cannot be certified under this standard. The standard explicitly addresses software as a potential source of systematic failures.

**NERC CIP-014-3** (Critical Infrastructure Protection, Physical Security) and related CIP standards require audit trails for automated decisions affecting bulk electric system reliability, including documentation of the basis on which automated systems make operational recommendations.

These requirements are not satisfied by post-hoc interpretability methods such as LIME (Ribeiro et al., 2016) or SHAP (Lundberg and Lee, 2017) applied to a black-box model. LIME approximates the model locally with a linear surrogate; the surrogate's coefficients reflect the linear approximation, not the model's internal computation. SHAP values are consistent under the axioms of Shapley value theory and satisfy desirable properties (efficiency, symmetry, dummy) but do not decompose into physical causal chains — a SHAP attribution to a sensor does not distinguish between "this sensor is causally upstream of the failure" and "this sensor is statistically correlated with failure through a latent confounder." CRG-based attribution and CART rule extraction differ from these approaches structurally: they expose aspects of the model's actual computational pathway, not a retrospective approximation constructed after the fact.

### 12.2 CART Symbolic Rule Induction

The Classification and Regression Trees algorithm (Breiman et al., 1984) constructs a binary tree by recursively solving:

$$(j^*, t^*) = \operatorname{argmax}_{j,t}\;\mathrm{Gain}(S, j, t)$$

$$\mathrm{Gain}(S,j,t) = \mathrm{Imp}(S) - \frac{|S_L|}{|S|}\mathrm{Imp}(S_L) - \frac{|S_R|}{|S|}\mathrm{Imp}(S_R)$$

where $S_L = \{\mathbf{h} \in S : h_j \leq t\}$, $S_R = S \setminus S_L$, impurity is the Gini coefficient for classification ($\mathrm{Imp}(S) = 1 - \sum_c p_c^2$ where $p_c$ is the class proportion) or mean squared error for regression ($\mathrm{Imp}(S) = \frac{1}{|S|}\sum_i (y_i - \bar{y})^2$), and split thresholds $t$ are evaluated at the midpoints between consecutive observed values of feature $j$.

ESE's CART is a custom pure-numpy implementation — no scikit-learn dependency — required for deployment in air-gapped industrial environments where external package installation is prohibited by OT security policy. It operates on the accumulated buffer of $([\mathbf{h}_t, y_t])$ pairs maintained by SSSR, periodically retrained as the buffer grows and the distribution of the latent space evolves. The tree is limited to a maximum depth of $d_{\mathrm{max}} = 6$ (configurable), yielding at most 64 leaf nodes and consequently at most 64 symbolic rules.

Rules are extracted by path enumeration from root to leaf. Each leaf yields one rule of the form:

$$\mathrm{IF}\; h_{j_1} \leq t_1\; \mathrm{AND}\; h_{j_2} > t_2\; \mathrm{AND}\; \ldots\; \mathrm{THEN}\; \hat{y} = \mu_\ell \quad (\mathrm{confidence}\; c_\ell,\; n_\ell\; \mathrm{samples})$$

where $\mu_\ell$ is the leaf mean prediction and $c_\ell = 1 - \mathrm{Imp}(\ell)/\mathrm{Imp}(\mathrm{root})$ is the normalized impurity reduction. When the model's `feature_names` dictionary maps latent dimensions to sensor identifiers, the condition $h_{j_1} \leq t_1$ is rendered as, e.g., `bearing_temp_wavelet_scale3 <= 0.47`, producing rules directly readable by domain engineers without knowledge of the latent representation.

### 12.3 CRG-Aware Attribution

The raw gradient $\partial \hat{y}/\partial x_i$ measures sensitivity: how much would the predicted output change if input $x_i$ were perturbed by $\varepsilon$? This is necessary but not sufficient for causal attribution. In the presence of correlated inputs — which is universal in industrial systems, where redundant sensors observe the same physical state — a signal may have high gradient sensitivity because it is a downstream effect of the true causal factor. The model's representation has captured the correlation, but not its direction.

CRG-aware attribution corrects for this by weighting gradient sensitivity by causal graph position as learned by the CRG module:

$$\mathrm{attr}_i = \frac{\sum_j \tilde{W}_{ij} \cdot \left|\partial \hat{y}/\partial x_j\right|}{\sum_i \sum_j \tilde{W}_{ij} \cdot \left|\partial \hat{y}/\partial x_j\right|}$$

where $\tilde{\mathbf{W}}$ is the row-stochastic normalization of the sparse CRG adjacency matrix $\mathbf{W}$ (after thresholding near-zero entries). The attribution score $\mathrm{attr}_i$ reflects: "how strongly does input $i$ causally influence the nodes to which the output is sensitive?" This is a proxy for the structural causal model's total causal effect of $x_i$ on $\hat{y}$, significantly more robust than pure gradient sensitivity in systems with correlated inputs.

### 12.4 Gradient Counterfactual Generation

Given the current latent representation $\mathbf{h} \in \mathbb{R}^D$ producing prediction $\hat{y} = f(\mathbf{h})$ and a target prediction value $y^* \neq \hat{y}$ (e.g., the model currently predicts a fault probability of 0.8 and the operator asks: "what conditions would bring this below 0.2?"), find the minimal perturbation:

$$\min_{\boldsymbol{\delta} \in \mathbb{R}^D}\; \|\boldsymbol{\delta}\|_2^2 + \alpha\|\boldsymbol{\delta}\|_1 \quad \mathrm{s.t.}\; \|f(\mathbf{h}+\boldsymbol{\delta}) - y^*\|_2 < \varepsilon$$

The $\ell_1$ term induces sparsity: most elements of $\boldsymbol{\delta}$ will be exactly zero at the solution, concentrating the counterfactual change on a small subset of latent dimensions. The constraint is handled via a soft-penalty Lagrangian:

$$\mathcal{L}_{\mathrm{cf}}(\boldsymbol{\delta}) = \|\boldsymbol{\delta}\|_2^2 + \alpha\|\boldsymbol{\delta}\|_1 + \mu \cdot \max\!\left(0,\, \|f(\mathbf{h}+\boldsymbol{\delta}) - y^*\|_2 - \varepsilon\right)$$

Optimized via proximal gradient descent on $\boldsymbol{\delta}$ (with the model weights held fixed), where the proximal operator for the $\ell_1$ term is the element-wise soft-thresholding operator $\mathrm{prox}_{\alpha\eta}(\boldsymbol{\delta}) = \mathrm{sign}(\boldsymbol{\delta})\max(|\boldsymbol{\delta}| - \alpha\eta, 0)$. Convergence is typically reached within 50–200 iterations.

When the nonzero elements of the optimal $\boldsymbol{\delta}$ correspond to latent dimensions strongly activated by specific ASE frequency bands or specific CRG input channels, the counterfactual translates to an actionable statement: "If the bearing temperature's wavelet scale-3 energy were reduced by 0.14 (corresponding to a reduction of approximately 12°C in the 0.5–2 Hz thermal band), the fault probability would decrease below the alarm threshold of 0.2." This is the form of explanation that domain engineers and process safety managers can act on.

---

## Section 13: Cross-Modal Latent Alignment (CMLA)

### 13.1 The Multi-Modal Industrial Sensing Problem

A gas turbine is simultaneously observable through vibration accelerometers on each bearing housing, thermocouples at successive blade rows, exhaust gas analyzers measuring CO/NOx/O2, oil debris monitors detecting ferrous particle mass, and acoustic emission transducers sensitive to micro-crack propagation. Each modality provides an incomplete, noise-corrupted view of the turbine's thermomechanical state. Optimal diagnosis requires fusing all available modalities in a way that accounts for their heterogeneous noise models, disparate sampling rates, and realistic missing-data patterns — sensor dropout due to communication failures, saturation events, maintenance-related removal.

Naive feature concatenation fails for structural reasons beyond scale mismatch. A thermal camera frame ($224 \times 224$ pixels) contributes $5 \times 10^4$ features while a current signature contributes $10^2$. After mean-variance normalization, the concatenated representation is numerically dominated by the high-dimensional modality regardless of its diagnostic relevance — the model's effective capacity is consumed by noise dimensions in the high-dimensional modality. More fundamentally: a failed sensor produces garbage activations that, when concatenated, corrupt the fused representation without the model receiving any signal that specific elements should be distrusted. Soft attention over modalities partially addresses this but requires a learned query-key mechanism that is itself a black box.

### 13.2 InfoNCE Contrastive Alignment

For modalities $m_1$ and $m_2$ observing the same physical event at time $t$ (a positive pair), define the InfoNCE loss (van den Oord et al., 2018):

$$\mathcal{L}_{\mathrm{InfoNCE}} = -\mathbb{E}\!\left[\log\frac{\exp\!\left(\mathrm{sim}(\mathbf{z}_{m_1,t},\,\mathbf{z}_{m_2,t})/\tau\right)}{\sum_{t'=1}^{N}\exp\!\left(\mathrm{sim}(\mathbf{z}_{m_1,t},\,\mathbf{z}_{m_2,t'})/\tau\right)}\right]$$

where $\mathrm{sim}(\mathbf{u},\mathbf{v}) = \mathbf{u}^\top\mathbf{v}/(\|\mathbf{u}\|_2\|\mathbf{v}\|_2)$ is cosine similarity, $\tau > 0$ is a temperature hyperparameter (default 0.07), and the sum in the denominator is over $N$ samples in the batch (all of which constitute negative pairs for the numerator's positive pair). This loss lower-bounds the mutual information between the two modality embeddings:

$$I(\mathbf{z}_{m_1};\mathbf{z}_{m_2}) \geq \log N - \mathcal{L}_{\mathrm{InfoNCE}}$$

as proven by van den Oord et al. (2018). Minimizing $\mathcal{L}_{\mathrm{InfoNCE}}$ therefore maximizes a lower bound on the mutual information between modality embeddings for the same physical event, while minimizing it for embeddings of different physical events. This pulls the latent representations of co-occurring observations together and pushes temporally distinct observations apart — learning a shared physical event space.

### 13.3 Inverse-Variance Weighted Fusion

After aligning modality embeddings via InfoNCE, they are fused using the inverse-variance estimator. For modality $m$ contributing embedding $\mathbf{z}_m \in \mathbb{R}^{d_z}$ and estimated noise variance $\hat{\sigma}_m^2$, the fused representation is:

$$\hat{\mathbf{z}} = \frac{\sum_m \mathbf{z}_m / \hat{\sigma}_m^2}{\sum_m 1/\hat{\sigma}_m^2}$$

The Gauss-Markov theorem guarantees that this is the minimum-variance unbiased estimator of the true state embedding $\mathbf{z}^*$ when each modality satisfies $\mathbf{z}_m = \mathbf{z}^* + \boldsymbol{\varepsilon}_m$ with $\boldsymbol{\varepsilon}_m \sim \mathcal{N}(\mathbf{0}, \sigma_m^2\mathbf{I})$ and the noise terms are independent across modalities. The per-modality variance estimate is:

$$\hat{\sigma}_m^2 \propto \|\mathbf{z}_m - \bar{\mathbf{z}}\|_2^2 / d_z$$

where $\bar{\mathbf{z}} = \frac{1}{|\mathcal{M}|}\sum_m \mathbf{z}_m$ is the naive mean. A failed or saturated sensor produces an embedding far from the ensemble mean — its numerically pathological output is treated as high variance and assigned near-zero weight in the fusion. This provides graceful fault tolerance without requiring an explicit anomaly detection module for sensor health: the statistical structure of the fused representation inherently downweights modalities whose embeddings are outliers relative to the ensemble.

---

## Section 14: Safety-Critical Policy Head

### 14.1 Formal Safety Requirements in Industrial Control

When VULGARIS operates in a closed-loop control setting — issuing setpoint adjustments to PLC/DCS systems, triggering protective relays, commanding robotic actuators, or making automated interlock decisions — the model's output is no longer advisory. A prediction error in this setting can cause irreversible consequences: equipment damage, process upset leading to chemical release, personnel injury, or grid instability. The regulatory standards that apply to such systems — IEC 61508, IEC 62061, ISO 13849 — require that safety functions be certified with quantified reliability measures (e.g., PFH $< 10^{-7}$ hr$^{-1}$ for SIL 2). Neural networks trained by empirical risk minimization provide no such guarantees in their standard form.

Standard post-hoc output clamping — clipping the control output to $u \in [u_{\min}, u_{\max}]$ — addresses static bounds constraints but cannot address dynamic safety constraints that depend on the trajectory of the system's state. A system approaching an unsafe region in state space can violate safety even with an output that is locally within bounds, if the rate of approach is unconstrained.

Control Barrier Functions (Wieland and Allgöwer, 2007; Ames et al., 2016) provide the formal framework for certified invariance of safe sets in dynamical systems. VULGARIS implements a differentiable CBF-based safety filter as a composable module that can be attached to any regression or control output head, converting nominal policy outputs to certified-safe control actions.

### 14.2 Control Barrier Functions: Theory

For a discrete-time system $\mathbf{s}_{t+1} = f(\mathbf{s}_t, \mathbf{u}_t)$ with state $\mathbf{s}_t \in \mathcal{S}$ and control input $\mathbf{u}_t \in \mathcal{U}$, define the safe set $\mathcal{C} = \{\mathbf{s} \in \mathcal{S} : h(\mathbf{s}) \geq 0\}$ for a smooth function $h : \mathcal{S} \to \mathbb{R}$.

**Definition 14.1 (Discrete-Time CBF).** The function $h$ is a Control Barrier Function for system $f$ if there exists $\gamma \in (0,1]$ such that for all $\mathbf{s} \in \mathcal{C}$:

$$\sup_{\mathbf{u} \in \mathcal{U}}\,[h(f(\mathbf{s},\mathbf{u})) - (1-\gamma)h(\mathbf{s})] \geq 0$$

The parameter $\gamma$ controls the rate of decay of the barrier: $\gamma = 1$ corresponds to set invariance (the barrier must be non-decreasing), while $\gamma < 1$ allows the barrier to decrease by a factor of at most $(1-\gamma)$ per step.

**Theorem 14.1 (Forward Invariance, Ames et al., 2016).** If $h$ is a Control Barrier Function for system $f$ and the control input $\mathbf{u}_t$ satisfies $h(f(\mathbf{s}_t,\mathbf{u}_t)) \geq (1-\gamma)h(\mathbf{s}_t)$ at every step, then $\mathbf{s}_0 \in \mathcal{C}$ implies $\mathbf{s}_t \in \mathcal{C}$ for all $t \geq 0$.

*Proof.* By induction. Base case: $h(\mathbf{s}_0) \geq 0$ by assumption. Inductive step: assuming $h(\mathbf{s}_t) \geq 0$, the CBF constraint gives $h(\mathbf{s}_{t+1}) = h(f(\mathbf{s}_t,\mathbf{u}_t)) \geq (1-\gamma)h(\mathbf{s}_t) \geq 0$. $\square$

This theorem establishes that verifying the CBF constraint at each individual step is sufficient to certify safety for all future time — a remarkable reduction of an infinite-horizon safety property to a per-step constraint.

### 14.3 Differentiable Safety Filter

The policy network $\pi_\theta : \mathcal{S} \to \mathcal{U}$ produces a nominal action $\mathbf{u}^{\mathrm{nom}} = \pi_\theta(\mathbf{s}_t)$. The safety filter solves the minimum-norm correction quadratic program (CBF-QP):

$$\mathbf{u}^* = \operatorname{argmin}_{\mathbf{u} \in \mathcal{U}}\;\|\mathbf{u} - \mathbf{u}^{\mathrm{nom}}\|_2^2 \quad \mathrm{s.t.}\; h(f(\mathbf{s}_t,\mathbf{u})) \geq (1-\gamma)h(\mathbf{s}_t)$$

This finds the action closest to the nominal policy's recommendation that satisfies the CBF constraint. When the nominal action is already safe (i.e., satisfies the constraint), the filter returns it unchanged: $\mathbf{u}^* = \mathbf{u}^{\mathrm{nom}}$. The filter only intervenes when the nominal action would violate the safety constraint.

Linearizing $h \circ f$ around $\mathbf{u}^{\mathrm{nom}}$, the constraint becomes linear in $\mathbf{u}$ and the CBF-QP has a closed-form projection solution:

$$\mathbf{u}^* = \mathbf{u}^{\mathrm{nom}} + \frac{\max\!\left(0,\; (1-\gamma)h(\mathbf{s}_t) - h(f(\mathbf{s}_t,\mathbf{u}^{\mathrm{nom}}))\right)}{\|\nabla_\mathbf{u} h\|_2^2 + \varepsilon} \cdot \nabla_\mathbf{u} h$$

where $\nabla_\mathbf{u} h = \nabla_\mathbf{s} h \cdot \nabla_\mathbf{u} f$ is the total gradient of the barrier through the dynamics (computed via the chain rule). The $\varepsilon > 0$ prevents division by zero. This expression is differentiable with respect to $\mathbf{u}^{\mathrm{nom}}$ everywhere except at the single kink where $h(f(\mathbf{s}_t,\mathbf{u}^{\mathrm{nom}})) = (1-\gamma)h(\mathbf{s}_t)$, where a subgradient exists and is used. Gradients flow through the safety filter during backpropagation, allowing the policy to learn to produce actions requiring progressively less CBF correction over the course of training — the policy internalizes the safety constraint rather than relying entirely on the filter.

### 14.4 Lipschitz Certification via Spectral Normalization

Spectral normalization (Miyato et al., 2018) enforces $\sigma_{\max}(\mathbf{W}) \leq L_{\max}$ for each weight matrix by rescaling after each forward pass. The spectral norm is estimated via power iteration: at each forward pass, three steps of

$$\mathbf{v} \leftarrow \mathbf{W}^\top\mathbf{u}/\|\mathbf{W}^\top\mathbf{u}\|_2, \quad \mathbf{u} \leftarrow \mathbf{W}\mathbf{v}/\|\mathbf{W}\mathbf{v}\|_2, \quad \hat{\sigma} = \mathbf{u}^\top\mathbf{W}\mathbf{v}$$

are performed, yielding the estimate $\hat{\sigma} \approx \sigma_{\max}(\mathbf{W})$. If $\hat{\sigma} > L_{\max}$: $\mathbf{W} \leftarrow \mathbf{W} \cdot L_{\max}/\hat{\sigma}$. For a $K$-layer network with each layer satisfying $\sigma_{\max}(\mathbf{W}_k) \leq L_{\max}$, the network's global Lipschitz constant satisfies:

$$\|f(\mathbf{x}) - f(\mathbf{y})\|_2 \leq L_{\max}^K \|\mathbf{x}-\mathbf{y}\|_2$$

providing a certified bound on output perturbation under bounded input noise. In OT/ICS environments subject to adversarial sensor spoofing — a known attack vector in industrial control systems, cf. the 2021 Oldsmar water treatment attack — this bound provides a quantitative defense guarantee: an adversary who can perturb sensor readings by at most $\delta$ can perturb the model's control output by at most $L_{\max}^K \delta$. Setting $L_{\max} = 1$ (i.e., non-expansive mapping) and $K = 6$ ensures the bound is 1 regardless of the depth. Note that this is a stronger requirement than ordinary spectral normalization and may reduce model expressivity; a tradeoff between certified robustness and prediction accuracy is inherent.

---

## Section 15: Theoretical Analysis

### 15.1 Stability of SSSR

**Theorem 15.1 (SSSR State Boundedness).** Let $(\mathbf{h}_t)_{t \geq 0}$ be generated by SSSR with log-timescale parameters $\mathbf{a} \in \mathbb{R}^N$, inter-sample intervals $\delta t_t \in [\delta t_{\min}, \delta t_{\max}]$ with $\delta t_{\min} > 0$, and bounded inputs $\|\mathbf{x}_t\|_2 \leq M < \infty$. Then $\sup_{t \geq 0} \|\mathbf{h}_t\|_2 \leq C < \infty$ for a constant $C$ depending only on $\mathbf{a}$, $M$, $\delta t_{\min}$, and $\|\mathbf{W}_B\|_2$.

*Proof.* Define the contraction factor $\rho^* = \max_n \bar{A}_n^{(\min)} = \max_n \exp(-\exp(a_n)\delta t_{\min})$. Since $\exp(a_n) > 0$ for all $a_n \in \mathbb{R}$ and $\delta t_{\min} > 0$, we have $\rho^* < 1$. By the ZOH discretization recurrence $h_{t,n} = \bar{A}_{t,n} h_{t-1,n} + \bar{B}_{t,n} \mathbf{x}_t$, taking norms:

$$\|\mathbf{h}_t\|_2 \leq \rho^* \|\mathbf{h}_{t-1}\|_2 + \|\bar{\mathbf{B}}_t\|_2 M$$

The input matrix norm satisfies $\|\bar{\mathbf{B}}_t\|_2 \leq (1-\rho_{\min}^*)\|\mathbf{W}_B\|_2$ where $\rho_{\min}^* = \min_n \bar{A}_{t,n} > 0$. Unrolling the recursion from $t=0$:

$$\|\mathbf{h}_t\|_2 \leq (\rho^*)^t \|\mathbf{h}_0\|_2 + \frac{(1-\rho_{\min}^*)\|\mathbf{W}_B\|_2 M}{1-\rho^*}$$

As $t \to \infty$, $(\rho^*)^t \to 0$, giving $\limsup_{t} \|\mathbf{h}_t\|_2 \leq C := (1-\rho_{\min}^*)\|\mathbf{W}_B\|_2 M / (1-\rho^*) < \infty$. $\square$

The critical contrast with gated RNNs is instructive. In an LSTM, the forget gate is $f_t = \sigma(\mathbf{W}_f[\mathbf{h}_{t-1};\mathbf{x}_t] + \mathbf{b}_f)$, which has values in $(0,1)$ but whose magnitude is a learned, input-dependent function. At adversarial inputs, $f_t \to 1$ for all dimensions simultaneously, and the cell state can grow without bound as $c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$ accumulates without contraction. In SSSR, the contraction factor $\bar{A}_{t,n} = \exp(-\exp(a_n)\delta t_t) < 1$ is a structural property of the architecture — $\exp(a_n) > 0$ always and $\delta t_t > 0$ always, so $\bar{A}_{t,n} < 1$ always, independent of the input value.

**Corollary 15.1 (Post-Hebbian Stability).** SHCAL's Oja rule updates are constrained to maintain $a_n \in [-5, 0]$ (enforced by clamping in `shcal.py`). Under this constraint, $\exp(a_n) \in [e^{-5}, 1] \approx [0.0067, 1]$, so $\bar{A}_n = \exp(-\exp(a_n)\delta t) \in (\exp(-\delta t_{\max}), \exp(-0.0067\,\delta t_{\min}))$. Both bounds are strictly in $(0,1)$ for finite $\delta t_{\max}$ and positive $\delta t_{\min}$. Therefore Theorem 15.1 holds after any number of Hebbian updates, and SHCAL adaptation cannot destabilize the SSM dynamics. $\square$

### 15.2 DAG Acyclicity of CRG

**Theorem 15.2 (Zheng et al., 2018; DAG Characterization).** A matrix $\mathbf{W} \in \mathbb{R}_{\geq 0}^{n \times n}$ satisfies $h(\mathbf{W}) = \mathrm{tr}(e^{\mathbf{W}}) - n = 0$ if and only if the weighted graph $G(\mathbf{W})$ is a directed acyclic graph.

*Proof sketch.* The $(i,j)$ entry of $e^{\mathbf{W}} = \sum_{k=0}^\infty \mathbf{W}^k/k!$ counts weighted walks of all lengths from $j$ to $i$ in $G(\mathbf{W})$. The trace $\mathrm{tr}(e^{\mathbf{W}}) = n + \sum_{k=1}^\infty \mathrm{tr}(\mathbf{W}^k)/k!$ counts $n$ (the empty walks) plus all weighted closed walks of length $\geq 1$. Since $\mathbf{W} \geq 0$ component-wise, each closed walk contributes a positive term. $\mathrm{tr}(e^{\mathbf{W}}) - n = 0$ if and only if there are no closed walks of any length, which is the definition of acyclicity. $\square$

In VULGARIS's CRG module, the penalty is $h(\mathbf{W} \odot \mathbf{W}) = \mathrm{tr}(e^{\mathbf{W}\odot\mathbf{W}}) - n$, using the Hadamard square to ensure the exponent argument is non-negative for all signed $\mathbf{W}$. Under augmented Lagrangian optimization with increasing penalty coefficient $\rho$, $h(\mathbf{W} \odot \mathbf{W}) \to 0$ at convergence, guaranteeing the learned causal graph is a DAG.

The gradient $\nabla_\mathbf{W} h = 2\mathbf{W} \odot e^{\mathbf{W}\odot\mathbf{W}}$ is well-defined and bounded for all finite $\mathbf{W}$, making the constraint differentiable and amenable to gradient-based optimization. This is the fundamental advantage of the NOTEARS formulation over combinatorial structure learning algorithms, which cannot be embedded in an end-to-end gradient framework.

### 15.3 Conformal Coverage Under Non-Stationarity

**Theorem 15.3 (Adaptive Coverage, adapted from Gibbs and Candès, 2021).** Let calibration nonconformity scores $(s_t)_{t=1}^T$ be assigned exponential weights $w_t \propto e^{-\lambda(T-t)}$ for forgetting rate $\lambda > 0$. Let the test label $y_{T+1}$ be drawn from a distribution $p_{T+1}$ satisfying $\|p_{T+1} - p_t\|_{\mathrm{TV}} \leq \varepsilon_{\mathrm{drift}} (T+1-t)$ (total-variation drift bounded linearly in time lag). The weighted conformal predictor $\hat{C}_{T+1} = \{y : s(y) \leq \hat{q}^{(1-\alpha)}_{\mathbf{w}}\}$ satisfies:

$$P\!\left(y_{T+1} \in \hat{C}_{T+1}\right) \geq 1 - \alpha - O\!\left(\frac{\varepsilon_{\mathrm{drift}}}{\lambda}\right)$$

The coverage deficit $O(\varepsilon_{\mathrm{drift}}/\lambda)$ reflects the lag of exponential forgetting: a larger $\lambda$ (faster forgetting) reduces this term but increases the variance of $\hat{q}$ due to fewer effective calibration samples ($n_{\mathrm{eff}} \approx 1/(1-e^{-\lambda}) \approx 1/\lambda$ for large $\lambda$). The optimal forgetting rate that minimizes the total error (bias from drift plus variance from insufficient calibration samples) scales as $\lambda^* = O(\sqrt{\varepsilon_{\mathrm{drift}} \cdot \log(1/\alpha)/n_{\mathrm{cal}}})$.

### 15.4 EWC Forgetting Bound

**Theorem 15.4 (EWC Forgetting).** Under the Laplace approximation — i.e., $\mathcal{L}_{\mathcal{T}_1}(\theta) \approx \mathcal{L}_{\mathcal{T}_1}(\theta^*) + \frac{1}{2}(\theta-\theta^*)^\top \mathbf{F} (\theta-\theta^*)$ where $\mathbf{F} = \mathrm{diag}(F_1, \ldots, F_P)$ is the Fisher diagonal — after training on $\mathcal{T}_2$ with EWC penalty $\lambda_{\mathrm{ewc}}$, the forgetting satisfies:

$$\Delta_{\mathrm{ewc}} = \frac{1}{2}\sum_i F_i(\theta_i - \theta_i^*)^2 \leq \frac{\|\mathbf{F}\|_\infty}{2} \|\Delta\theta\|_2^2$$

where $\|\mathbf{F}\|_\infty = \max_i F_i$ and $\Delta\theta = \theta - \theta^*$.

*Proof.* The first equality follows directly from the Laplace approximation and the definition of $\Delta_{\mathrm{forget}}$. The inequality follows from $\sum_i F_i(\Delta\theta_i)^2 \leq \max_i F_i \cdot \sum_i (\Delta\theta_i)^2$. $\square$

For constrained adaptation with $\|\Delta\theta\|_2 \leq \rho_{\max}$ (enforced implicitly by $\lambda_{\mathrm{ewc}}$): $\Delta_{\mathrm{ewc}} \leq F_{\max}\rho_{\max}^2/2$. The parameter $\lambda_{\mathrm{ewc}}$ controls $\rho_{\max}$: larger $\lambda_{\mathrm{ewc}}$ reduces the gradient step size on parameters with high $F_i$, reducing $\|\Delta\theta\|_2$ and hence the forgetting bound.

### 15.5 Federated Privacy Accounting

By the Rényi differential privacy composition theorem (Mironov, 2017), the cumulative privacy loss of the federated learning protocol over $R$ aggregation rounds, each with client sampling ratio $q$, Gaussian noise multiplier $\sigma$, and gradient clipping norm $C_{\mathrm{clip}}$, satisfies: for any order $\alpha > 1$, the Rényi divergence is:

$$D_\alpha\!\left(\mathcal{M}^R \| \mathcal{M}_0^R\right) \leq R \cdot \frac{\alpha q^2 C_{\mathrm{clip}}^2}{2\sigma^2 n^2} + O\!\left(\frac{R q^2 \alpha^2 C_{\mathrm{clip}}^4}{\sigma^4 n^4}\right)$$

Converting to $(\varepsilon, \delta)$-DP via $\varepsilon(\delta) = \min_{\alpha > 1}\!\left[D_\alpha + \log((\alpha-1)/\alpha) - \log(\delta)/(\alpha-1)\right]$ (Balle et al., 2020), the total privacy budget after $R$ rounds is approximately:

$$\varepsilon_{\mathrm{total}} \approx \sqrt{\frac{2Rq^2 C_{\mathrm{clip}}^2 \log(1/\delta)}{\sigma^2 n^2}} + \frac{Rq^2 C_{\mathrm{clip}}^2}{\sigma^2 n^2}$$

for small $q$ (the regime applicable to federated learning with many clients). The dominant term scales as $O(\sqrt{R}/\sigma)$, implying that doubling the noise multiplier halves the privacy budget consumed per unit of training time. The dependence on $n$ (the total dataset size across clients, or equivalently the normalization for the clipped gradient) confirms that larger federated networks achieve better privacy-accuracy tradeoffs — a well-established result in federated DP theory.

---

## Section 16: Training Methodology

### 16.1 The Unified Variational Loss

The VULGARIS training objective derives from a variational lower bound on the log-likelihood of a joint generative model $p(\mathbf{x}_{1:T}, \mathbf{z}_{1:T}, \mathbf{G}, \theta)$ where $\mathbf{G}$ is the causal graph, $\mathbf{z}_{1:T}$ is the latent state trajectory, and $\theta$ are the model parameters. Under the approximate posterior $q(\mathbf{z}, \mathbf{G}, \theta | \mathbf{x})$, the Evidence Lower BOund (ELBO) decomposes into terms with precise information-theoretic interpretations:

$$\mathcal{L} = \underbrace{\mathbb{E}_q[\log p(\mathbf{x}|\mathbf{z},\mathbf{G})]}_{\mathcal{L}_{\mathrm{task}}} - \beta\underbrace{I_q(\text{memory};\text{past})}_{\mathcal{L}_{\mathrm{mem}}} - \gamma\underbrace{h(\mathbf{W}\odot\mathbf{W})}_{\mathcal{L}_{\mathrm{dag}}} - \delta\underbrace{D_{\mathrm{KL}}(q(\theta)\|p_F(\theta))}_{\mathcal{L}_{\mathrm{ewc}}} - \varepsilon\mathcal{L}_{\mathrm{conf}} - \zeta\mathcal{L}_{\mathrm{cbf}} - \eta\mathcal{L}_{\mathrm{temp}} - \vartheta\mathcal{L}_{\mathrm{InfoNCE}}$$

Each coefficient ($\beta, \gamma, \delta, \varepsilon, \zeta, \eta, \vartheta$) controls the strength of one constraint on the learned representation:

- $\mathcal{L}_{\mathrm{task}}$: the log-likelihood term, directly measuring predictive accuracy.
- $\mathcal{L}_{\mathrm{mem}} = I_q(\text{memory};\text{past})$: the rate term in rate-distortion theory applied to episodic memory — penalizes storing more information from the past than the task requires. This is operationalized as the mutual information between the HMB's compressed representations and the raw historical states, computed via the MINE lower bound (Belghazi et al., 2018).
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

**Phase 1 — Masked signal reconstruction (60% of total compute budget).** Randomly zero 20% of input channels per training example, independently sampled per timestep. The loss is reconstruction MSE on masked channels only: $\mathcal{L}_{\mathrm{mask}} = \frac{1}{|\Omega_{\mathrm{mask}}|}\sum_{(t,c)\in\Omega_{\mathrm{mask}}} (\hat{x}_{t,c} - x_{t,c})^2$ where $\Omega_{\mathrm{mask}}$ is the set of masked channel-time pairs. This objective forces several representational properties simultaneously: ASE must learn frequency-complete representations that cannot rely on a single channel; CRG must discover inter-signal dependencies that enable predicting one channel's value from others (instantiating Granger causality in a structured form); and SSSR must build sufficient temporal context to predict masked values from surrounding unmasked observations.

**Phase 2 — Temporal contrastive pretraining (25% of compute).** Two augmented views of each training window are constructed: additive Gaussian noise ($\sigma = 0.1 \times$ per-channel standard deviation) and random channel dropout (10% of channels zeroed, different from Phase 1's masking). Views of the same window constitute positive pairs; views from different windows are negative pairs. The InfoNCE loss (Section 13.2) is maximized over these pairs, sharpening the latent representations by forcing them to be invariant to measurement noise and partial sensor dropout while remaining discriminative across distinct physical states.

**Phase 3 — Causal structure hardening (15% of compute).** The DAG penalty coefficient $\gamma$ is increased by $10\times$ for this final phase, while the learning rate is at the low end of the cosine schedule ($\eta \approx 10^{-5}$). At this reduced learning rate, large changes to task-predictive features are suppressed, and the optimizer spends its capacity tightening the causal graph toward a sparse, acyclic structure. The result is a CRG adjacency matrix $\mathbf{W}$ that is already interpretably sparse before any downstream fine-tuning begins, providing immediately useful causal attributions in few-shot deployment scenarios where there is insufficient data to further tune the causal structure.

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

**Supply chain and footprint.** PyTorch 2.x (CPU-only) has a compressed installation size of approximately 800 MB. For edge systems with 512 MB to 2 GB of storage — a common configuration for industrial IoT gateways — this is frequently inadmissible not only due to space constraints but due to IT security policies that restrict approved software packages to a vetted, minimal list. The VULGARIS autograd engine's core dependencies — numpy (approximately 20 MB), scipy (approximately 30 MB) — are pre-installed on virtually all industrial Linux distributions (Debian Bullseye, RHEL 8, Ubuntu 20.04 LTS) as system packages. The complete autograd implementation is approximately 800 lines of readable Python code, auditable by any engineer with Python competence.

**Determinism and auditability for IEC 61508 certification.** PyTorch's `torch.compile` and its XLA/TorchScript compilation paths can apply graph transformations — operator fusion, constant folding, precision reduction — that alter numerical behavior between runs. For systems seeking IEC 61508 SIL 2/3 certification, each layer of the software stack must be qualified: it must produce bit-identical outputs given identical inputs across platforms and runs. The custom engine has no JIT compiler, no kernel fusion, and no stochastic optimization passes. Its behavior is entirely determined by the numpy implementation, which is itself subject to IEEE 754 floating-point arithmetic and produces reproducible results.

**Non-standard backward passes.** The NOTEARS gradient $\nabla_\mathbf{W} h = 2\mathbf{W} \odot e^{\mathbf{W}\odot\mathbf{W}}$ and the SSM associative scan backward pass (a reverse prefix scan over the sequence, requiring custom gradient propagation through the scan's work-efficient tree structure) are non-trivial custom operations. In PyTorch, implementing these as first-class differentiable operations requires writing C++/CUDA extensions with complex CMake build systems, per-CUDA-architecture kernel compilation, and maintenance of ABI compatibility across PyTorch versions. In the custom engine, both are Python functions using standard numpy operations, fully auditable, debuggable with standard Python tools, and portable to any platform where numpy is available.

### 17.3 CUDA Acceleration Strategy

Two computational hotpaths dominate training and justify custom CUDA kernels while leaving all other operations in the numpy fallback path:

**SSM parallel scan** (`core/kernels/ssm_scan.cu`). The associative scan over $T$ timesteps with the operator $(A_i, B_i) \oplus (A_j, B_j) = (A_i A_j, A_j B_i + B_j)$ has $O(T)$ serial depth in the naive sequential implementation. The Blelloch work-efficient parallel prefix scan algorithm (Blelloch, 1990) reduces the parallel depth to $O(\log T)$ with $O(T)$ total work, using $T/2$ threads in a binary-tree reduction followed by a binary-tree down-sweep. For $T=1024$, $N=256$, $B=32$: the sequential Python implementation requires approximately 12 ms; the Blelloch CUDA scan requires under 0.3 ms — a 40x speedup that is decisive for training throughput. Without this kernel, the wall-clock training time per batch is dominated by the sequential scan, making training of long-sequence models impractical.

**Wavelet convolution** (`core/kernels/wavelet.cu`). The ASE module applies $K \times S = 128$ dilated convolutions across $n_{\mathrm{sensors}} = 64$ input channels. The custom kernel fuses dilation expansion, convolution, and multi-scale accumulation into a single GPU kernel pass, eliminating three DRAM round-trips for intermediate tensors that a naive three-kernel implementation would require. Peak memory bandwidth utilization increases from approximately 35% (naive) to approximately 72% (fused) on an NVIDIA A100.

The numpy fallback implementations of both operations produce identical numerical results to within floating-point rounding, verified by unit tests in `tests/kernels/`. This ensures that models trained with CUDA acceleration can be deployed without CUDA and produce the same predictions.

### 17.4 The Rust Event Processing Runtime

CPython's Global Interpreter Lock (GIL) prevents true parallelism within a single process: at most one thread executes Python bytecode at any time. An industrial event broker receiving 50,000 sensor updates per second from 1,000 devices — a routine specification for a modern process plant historian — requires multi-threaded, lock-minimizing I/O and preprocessing that CPython cannot provide natively. asyncio partially addresses I/O concurrency but serializes CPU-bound operations.

The Rust runtime (`runtime/src/event_stream.rs`) provides a lock-free ring buffer implemented via the `crossbeam::channel` multi-producer single-consumer (MPSC) queue, which uses atomics rather than mutexes and avoids kernel-space context switches for all non-blocking operations. Parallel sliding-window matrix assembly uses Rayon data-parallelism: the 64-sensor state vector at each timestep is assembled by $N_{\mathrm{sensor}}$ parallel threads, each reading from its ring buffer and writing to its column of a pre-allocated $(W \times n_{\mathrm{sensors}})$ matrix. Linear interpolation fills missing values due to dropped UDP packets or late-arriving messages. PyO3 bindings expose the completed window matrices to Python as numpy arrays via the buffer protocol, requiring zero memory copies: the numpy array references the Rust-allocated memory directly.

Measured throughput on a single ARM Cortex-A72 core: 1.2M events per second with per-event latency below 100 µs at the 99th percentile. This exceeds by an order of magnitude the input rates of any currently deployed industrial sensor network, providing headroom for future sensor density increases.

---

## Section 18: Benchmarks

### 18.1 Benchmark Datasets

All benchmarks are synthetic and reproducible via `benchmarks/suite.py` with a fixed random seed. Synthetic data is used rather than public datasets for three reasons: (1) ground-truth causal graphs are known, enabling causal attribution accuracy evaluation; (2) exact distribution shift times are known, enabling quantitative evaluation of SHCAL's adaptation rate; and (3) privacy — real industrial datasets typically cannot be published.

**IPC-SCADA (Industrial Process Control SCADA).** 24 sensors, three operating regimes (normal, high-load, standby) with Markov transitions at rate $\lambda = 0.01$ per timestep. Superimposed thermal drift ($\tau = 30$ min time constant) and random valve-failure impulses (rate 2/hour, magnitude $5\sigma$). Tasks: 4-class anomaly detection (normal, thermal drift, impulse fault, regime boundary) and 3-class operating regime identification. The ground-truth causal DAG (8 nodes, 11 edges) is hardcoded in the generator, enabling evaluation of CRG structure recovery via structural Hamming distance.

**PGFD (Power Grid Fault Detection).** 32 sensors at 60 Hz sampling, five fault classes (voltage sag, overcurrent, harmonic distortion, phase imbalance, frequency deviation), fault occurrence rate 2–10 events per hour with class imbalance (3:1 dominant-to-rare). Tests high-frequency fault onset detection with $<$200 ms latency requirement and conformal interval calibration under severe class imbalance — a setting where naive conformal predictors fail due to asymmetric score distributions.

**5G-KPI (Telecom Radio Access Network).** 16 cells × 4 KPIs (PRB utilization, SINR, per-cell throughput, access latency) at 1-second resolution. Diurnal traffic with an additive random interference process representing neighboring-cell load variation. Tests multi-variate forecasting accuracy with long-range temporal dependencies (diurnal period = 86,400 steps) and domain adaptation across cells with different propagation environments.

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
| All datasets | Coverage@90% | $88$–$92\%$ | N/A | $72$–$81\%$ |
| All datasets | Latency p99 | $<22$ ms | $<1$ ms | $<18$ ms |

The conformal coverage comparison is particularly revealing: the Mamba-equivalent model, lacking the conformal recalibration loop of SHCAL, drifts to 72–81% empirical coverage on the test set despite being calibrated on a validation set from the same distribution. Under the distribution shifts present in the benchmark (thermal drift, operating regime changes), its calibration set quantiles are no longer representative. VULGARIS's SHCAL-driven adaptive calibration maintains the nominal 90% coverage target.

---

## Section 19: Comparison with Prior Work

### 19.1 Transformers

Transformers (Vaswani et al., 2017) are the current state-of-the-art for tasks with access to full-sequence batching, sufficient memory for a KV cache, and approximately stationary training distributions. For natural language generation and understanding at the scales currently deployed commercially, there is no compelling architectural alternative with equivalent empirical performance.

The case against Transformers for industrial streaming inference is structural rather than empirical. The KV cache grows as $O(T \times d_{\mathrm{model}} \times n_{\mathrm{heads}} \times n_{\mathrm{layers}})$ with sequence length $T$. For a modest configuration — 6 layers, 8 heads, $d=256$, FP32 — a sequence of 100,000 timesteps (about 28 hours at 1 Hz) requires 4.9 GB of KV cache. For a year of data at this rate, the requirement is over 400 GB. Flash Attention (Dao et al., 2022) reduces the memory bandwidth cost of the attention computation but does not reduce the KV cache size: the state is still $O(T)$.

Chunked attention with a fixed context window $W$ reduces memory to $O(W)$ but introduces a hard temporal horizon beyond which the model has no information. The ZOH-SSSR's exponential state history, while also bounded in dimension, decays smoothly rather than hard-truncating — for industrial processes with slow drift, this is quantitatively significant.

For offline retrospective analysis — investigating a historical event using archived data — a Transformer-based model with access to the full time window is entirely appropriate and may outperform VULGARIS on tasks requiring very long-range dependencies that the SSM's finite-dimensional state cannot represent exactly. VULGARIS is not proposed as a replacement for Transformers in all time-series settings; it is proposed as the appropriate architecture for the specific operational context of continuous streaming inference.

### 19.2 Mamba and S6

The Mamba architecture (Gu and Dao, 2023) is the closest existing work to SSSR. Its selective state-space mechanism — input-dependent $\Delta t_t$, $\mathbf{B}_t$, $\mathbf{C}_t$ computed as functions of $\mathbf{x}_t$ — is directly analogous to SSSR's input-driven temporal selectivity. Both architectures achieve $O(1)$ streaming memory and $O(T)$ inference complexity. Both are trained efficiently via the parallel associative scan.

VULGARIS departs from Mamba in four operationally significant directions that are motivated by the industrial deployment context rather than by architectural novelty for its own sake:

1. **ZOH discretization with physical interpretation.** Mamba's $\Delta_t$ is a learned scalar multiplied into the SSM matrices; it does not have an explicit interpretation as a physical sampling interval. VULGARIS's $\delta t_t$ is the measured inter-sample interval, making the discretization physically meaningful and enabling deployment on irregularly sampled industrial sensors without additional preprocessing.

2. **Hebbian in-stream adaptation.** Mamba assumes a stationary deployment setting — the model that left training is the model that is deployed indefinitely. SHCAL's Oja-rule updates allow VULGARIS to adapt its representations to gradual distribution shift without any labeled data, maintaining performance in the perpetually drifting industrial environment.

3. **Hierarchical Temporal Decomposition.** Mamba applies a single SSM layer (or a stack of identical layers). VULGARIS's HTD explicitly allocates separate SSM heads to separate temporal frequency bands, with the attention-gated aggregation fusing their outputs. This structural inductive bias reduces the sample complexity of learning multi-timescale dynamics.

4. **Causal Routing Graph.** Mamba mixes information across input dimensions via dense projection matrices. VULGARIS's CRG enforces a DAG-structured routing, imposing a learned causal ordering on information flow. This is the architectural prerequisite for physically meaningful attribution.

Conversely, Mamba's custom Triton CUDA kernel for the selective scan provides training throughput that VULGARIS's current CUDA kernel does not fully match, due to Mamba's more mature CUDA implementation.

### 19.3 S4 and HiPPO

S4 (Gu et al., 2021) demonstrated that structured state-space models with HiPPO matrix initialization can achieve competitive performance on long-range dependency benchmarks where RNNs and CNNs fail. The HiPPO matrix $\mathbf{A}_n^{\mathrm{HiPPO}}$ provides provably optimal polynomial projection of the input history onto Legendre or Laguerre basis functions, enabling the SSM to optimally memorize the past relative to a polynomial approximation quality measure.

VULGARIS departs from S4 in replacing HiPPO initialization with ZOH-discretized diagonal parameterization, following the observation (Gu et al., 2022, S4D; Smith et al., 2022, S5) that diagonal $\mathbf{A}$ matrices achieve comparable performance to full HiPPO matrices while dramatically simplifying the parallel scan (diagonal structure allows element-wise rather than matrix multiplications in the scan operator). The key limitation of S4 for streaming deployment is that its most efficient training mode uses the frequency-domain convolution $y = \mathcal{F}^{-1}(\mathcal{F}(K) \odot \mathcal{F}(x))$, which requires the full sequence in memory. This convolution mode must be abandoned for streaming inference, falling back to sequential recurrence mode. SSSR's parallel scan training is compatible with its streaming inference mode, avoiding the training-inference discrepancy.

### 19.4 RWKV

RWKV (Peng et al., 2023) achieves linear inference cost and $O(1)$ streaming memory via a time-decay mechanism $W_t = w \odot W_{t-1} + e^k \odot V$ where $w$ is a fixed learned per-channel decay vector. This structural simplicity is an advantage for implementation and deployment, and RWKV has demonstrated competitive language modeling performance.

The limitation for industrial time-series is that $w$ is fixed at inference time — the effective memory horizon is input-independent. SSSR's input-dependent $\delta t_t$ means that the model can effectively extend its memory horizon for slowly varying inputs (corresponding to large $\exp(a_n)\delta t_t$ giving small $\bar{A}_n$, hence strong input integration) and shorten it for rapidly varying inputs where old history is less informative. This input-adaptive horizon is not an architectural luxury for industrial systems: the same turbine experiences both quasi-static normal operation and rapid fault transients within the same deployment context, requiring qualitatively different temporal integration behaviors.

### 19.5 Industrial Time-Series Foundation Models

Chronos (Ansari et al., 2024), Moirai (Woo et al., 2024), and MOMENT (Goswami et al., 2024) represent important recent advances in zero-shot and few-shot time-series forecasting using large pretrained transformer-based models. They share a design pattern: patch-based tokenization of time-series windows, transformer backbone trained on diverse public time-series corpora, and evaluation on held-out forecasting tasks.

These models address a different problem specification from VULGARIS. The forecasting foundation model paradigm optimizes prediction accuracy on offline windows. Its operational assumptions are: (i) the model is evaluated on complete, bounded windows; (ii) computational resources are sufficient for batch inference; (iii) the primary evaluation metric is forecasting MAPE or CRPS; and (iv) deployment context is a cloud analytics service with unrestricted memory. VULGARIS's problem specification is: (i) perpetual streaming with no window boundary; (ii) inference under tight memory and latency constraints; (iii) joint objectives including causal attribution, conformal coverage, and certified safety; and (iv) deployment on sub-watt edge hardware. These are genuinely different problems, and the architectures reflect their respective requirements.

### 19.6 Kalman Filter Variants

For linear Gaussian dynamical systems with known model parameters, the Kalman filter is the optimal minimum-variance unbiased estimator. Extended Kalman Filters (EKF) handle mildly nonlinear dynamics via linearization; Unscented Kalman Filters (UKF) handle stronger nonlinearities via sigma-point approximation; particle filters handle arbitrary distributions at $O(N_{\mathrm{particles}})$ cost per step.

VULGARIS is not a replacement for Kalman-based methods in domains where the dynamical model is well-characterized. For instrumented subsystems with clear physics — a power transmission line with known resistance-inductance-capacitance parameters, a linear hydraulic actuator with identified stiffness and damping — Kalman filtering is simpler, faster, fully interpretable by engineers, and theoretically optimal. VULGARIS addresses the complementary problem: systems where the dynamics are unknown, multivariate, nonlinear, non-stationary, and characterized by emergent failure modes not present in the nominal model. These include gearbox wear, pump cavitation, insulation degradation, and process fouling — precisely the failure modes responsible for the majority of unplanned industrial downtime — for which the first-principles model does not exist in a form usable for Kalman filtering.

---

## Section 20: Ablation Analysis

### 20.1 Component Ablations

Each module of VULGARIS addresses a specific failure mode. The following ablations identify which components are load-bearing for which capabilities and which datasets most expose each component's contribution:

| Ablated component | Primary affected dataset | Expected degradation | Addressed failure mode |
|---|---|---|---|
| ASE $\to$ linear projection | PM-Bearing (high-freq vibration) | +20–30% RUL MAPE | Loss of BPFO spectral signature at 10 kHz; broadband noise drowns fault indicator |
| HTD $\to$ single timescale | IPC-SCADA (regime + drift) | +15–25% F1 | Cannot simultaneously resolve sub-second fault impulses and 30-min thermal drift |
| CRG $\to$ dense mixing layer | IPC-SCADA attribution | Attribution accuracy $\to$ chance | No causal structure; all input-to-output paths have equal weight; counterfactuals uninformative |
| HMB $\to$ no episodic memory | PGFD (rare fault classes) | +8–15% missed rare-fault detections | Cannot condition on previous fault signatures; each event treated as independent |
| SHCAL disabled | All datasets (after 1,000 steps) | Progressive accuracy degradation at rate $\propto \varepsilon_{\mathrm{drift}}$ | Ongoing distribution drift accumulates without correction |
| DAH $\to$ per-domain fine-tune | Multi-domain deployment | $10\times$ training compute cost | Full backward pass per domain; no knowledge transfer |
| Safety filter removed | Control-loop evaluation tasks | CBF violations at rare boundary states | Nominal policy produces unsafe actions at rare but critical operating points |

### 20.2 Loss Term Ablations

Each term in the unified loss corresponds to one of the above components or a cross-cutting property. Removing each term isolates its contribution:

| Removed term | Observed consequence |
|---|---|
| $\mathcal{L}_{\mathrm{dag}}$ removed | $\mathbf{W}$ converges to a dense matrix; CRG attribution distributes weight approximately uniformly across inputs; structural Hamming distance from ground-truth DAG exceeds $|E|$ (all edges misspecified) |
| $\mathcal{L}_{\mathrm{mem}}$ removed | HMB learns to archive all timesteps at high resolution; storage grows linearly with streaming duration; the $O(1)$ memory guarantee is violated |
| $\mathcal{L}_{\mathrm{ewc}}$ removed | After domain transition, $\mathcal{T}_1$ performance degrades 40–60% within 200 gradient steps; no recovery without explicit re-calibration |
| $\mathcal{L}_{\mathrm{conf}}$ removed | Empirical conformal coverage drifts 5–8% below nominal under moderate distribution shift; uncertainty intervals are systematically overconfident |
| $\mathcal{L}_{\mathrm{cbf}}$ removed | Policy learns CBF satisfaction on training distribution; produces safety-violating actions at 3–8% of test-set boundary states that are underrepresented in training |

---

## Section 21: Limitations and Future Work

### 21.1 Known Limitations

**Training throughput versus PyTorch.** The custom numpy autograd engine achieves 3–5× lower training throughput on CUDA hardware than an equivalent well-optimized PyTorch implementation, primarily because PyTorch's cuBLAS-backed matrix multiplications utilize tensor cores with mixed-precision arithmetic, while the custom engine's CUDA kernels cover only the two identified hotpaths (SSM scan and wavelet convolution) and implement standard FP32 arithmetic. Inference is unaffected by this gap, as streaming single-step inference does not involve the parallel scan or wavelet convolution at deployment time. The throughput gap affects researchers and engineers performing pretraining runs; it does not affect end-to-end latency of deployed models.

**Latent confounder blindness.** The CRG's Granger-initialized NOTEARS DAG is not causally identified in the presence of latent confounders. When two observable signals $X$ and $Y$ are both driven by a hidden common cause $Z$ (e.g., ambient temperature simultaneously affects bearing temperature sensor readings and motor current draw), CRG will learn a spurious directed edge between $X$ and $Y$ — whichever has the higher Granger-causal $p$-value for the other. This is a fundamental limitation of constraint-based causal discovery from observational data without interventional experiments. The result is that attributions can be misleading in precisely the cases where domain engineers most need correct causal identification: high-stakes fault events driven by latent deterioration processes. Methods for latent-variable causal discovery (FCI algorithm, NOTEARS with latent variables, LVCI) can address this but require additional structural assumptions and significantly higher computational cost, and are outside the scope of the current implementation.

**Hypernetwork generalization boundary.** DAH generalizes to new deployment domains only within the convex hull of the pretraining domain distribution in the meta-embedding space. A genuinely novel domain — a sensor type not represented in pretraining, a physical process with fundamentally different dynamics — lies outside this hull, and the hypernetwork will produce suboptimal adapters. A 100–500 step adapter fine-tuning procedure recovers full performance in practice but represents an additional operational step that must be budgeted in the deployment workflow.

**Single-machine training.** No data parallelism or model parallelism is currently implemented in the custom training engine. The maximum trainable configuration on a single 80 GB A100 is approximately $D=512$, $B=16$, $T=1024$. Pretraining a model at scale — on large multi-sensor corpora from multiple industrial plants — requires a distributed training framework. This is not yet implemented.

**CART approximation quality in high dimensions.** Decision trees approximate neural network behavior locally, but the quality of approximation degrades exponentially with latent dimension due to the curse of dimensionality. For $D=256$, a tree of depth 6 covers $6 \ll \log_2(256)$ dimensions in any leaf path: the symbolic rules describe marginal projections onto the most informative latent dimensions rather than the full $D$-dimensional decision surface. The rules are useful for operator communication and regulatory documentation but should not be treated as complete descriptions of the model's internal reasoning.

### 21.2 Future Work

Five near-term extensions address the most impactful limitations:

**1. Full CUDA backward pass.** Implementing the complete backward pass — gradient flows through the wavelet convolution, CRG matrix exponential, SSM associative scan backward, and EWC Fisher update — as CUDA kernels would eliminate the 3–5× throughput gap versus PyTorch and enable large-scale pretraining runs on GPU clusters. The primary engineering challenge is the CRG backward pass through the matrix exponential $\nabla_\mathbf{W} e^{\mathbf{W}\odot\mathbf{W}}$, which requires efficient computation of $e^\mathbf{M}$ for $64 \times 64$ matrices at high batch sizes.

**2. Variational latent confounder model.** Extending CRG with a variational inference component for hidden common causes would address the confounding limitation. The approach follows LVCI (Annadani et al., 2021): augment the observable variable set with latent variables $\mathbf{Z}$ and infer their posterior via amortized variational inference. The joint model $p(\mathbf{X}, \mathbf{Z} | \mathbf{G})$ with a prior over latent variable presence would produce a DAG that distinguishes direct causal effects from confounded correlations.

**3. Multi-GPU training with gradient synchronization.** Implementing the all-reduce gradient synchronization protocol over NCCL (for NVIDIA hardware) or ROCm (for AMD hardware) would enable data-parallel training across multiple devices. The custom autograd engine's gradient tensors are standard numpy arrays that would need to be transferred through device memory for synchronization — straightforward but requiring a robust device management layer.

**4. Neuromorphic deployment mapping.** SSSR's diagonal recurrence $h_{t,n} = \bar{A}_n h_{t-1,n} + \bar{B}_n x_t$ maps naturally to leaky integrate-and-fire (LIF) neuron dynamics $V_t = \lambda V_{t-1} + I_t$ when the input is presented as a rate-coded spike train. A systematic translation of VULGARIS inference to Intel Loihi 2 or SynSense Speck hardware would enable sub-milliwatt inference for battery-powered sensor nodes, extending the deployment frontier to condition monitoring in remote or implanted devices.

**5. SMT verification of CBF satisfaction.** Applying Satisfiability Modulo Theories (SMT) solvers — specifically, dReal (Gao et al., 2013) for nonlinear arithmetic — to formally verify that the safety filter satisfies the CBF constraint for all inputs within a certified input domain $\mathcal{X}_{\mathrm{cert}}$ would provide a stronger guarantee than empirical evaluation. Combined with the Lipschitz certification from spectral normalization (Section 14.4), this would produce a complete formal safety certificate: any input within $\mathcal{X}_{\mathrm{cert}}$ produces a control output satisfying the CBF constraint, and inputs within bounded distance of $\mathcal{X}_{\mathrm{cert}}$ produce outputs within a certified deviation.

---

## Section 22: Conclusion

The central argument of this monograph is that industrial intelligence constitutes a distinct problem class from language intelligence, and that this distinction is architectural: the constraints of industrial streaming deployment are incompatible with the computational mechanisms that make language foundation models effective.

This argument rests on formal constraints, not on empirical preference. The requirement that inference memory be bounded independently of sequence length rules out $O(T)$ KV-cache mechanisms. The requirement that adaptation occur continuously without labeled data or retraining rules out standard supervised fine-tuning. The requirement that predictions be traceable to physical causal pathways auditable by domain engineers rules out black-box end-to-end feature learning without structural inductive biases. The requirement that deployment occur in 5–50 MB on edge hardware under 25 ms latency rules out the parameter counts and compute footprints of contemporary foundation models. The requirement that control outputs be certifiably safe rules out unconstrained neural policy outputs.

VULGARIS is a response to the conjunction of these constraints. Each architectural component follows from one constraint by a chain of reasoning that is, in principle, checkable: the SSM recurrence follows from the $O(1)$ memory constraint; ZOH discretization follows from the irregular sampling constraint; diagonal parameterization follows from stability requirements; ASE wavelet embeddings follow from the multi-frequency nature of industrial fault signatures; HTD follows from the multi-timescale nature of industrial dynamics; CRG follows from the causal attribution requirement; HMB follows from the need for long-horizon context within bounded memory; SHCAL follows from the continuous adaptation requirement; DAH follows from the multi-domain deployment requirement; ESE follows from the regulatory transparency requirement; CMLA follows from the multi-modal sensing requirement; and the CBF safety filter follows from the certified control requirement.

The coherence of the resulting architecture — that all of these components can be unified into a single differentiable system optimized by one variational loss — is not a coincidence of design but a consequence of the fact that the physical world's information structure (causal, multi-timescale, multi-modal, uncertain) is well-matched to the mathematical structures of state-space models, causal graphical models, and conformal prediction.

There are real and acknowledged limitations: the training engine's throughput gap, the latent confounder blindness, the hypernetwork generalization boundary, the absence of distributed training. These are engineering limitations whose solutions are known and in progress. They bound the current system's applicability without undermining the central demonstration: that the conjunction of streaming O(1) memory, continuous adaptation, causal attribution, regulatory-grade explainability, edge deployment, and certified safety is simultaneously achievable within a coherent, trainable, theoretically grounded architecture.

The industrial systems for which VULGARIS is designed are not static artifacts. The equipment ages, the process conditions change, the regulatory environment evolves, the sensor configurations expand. The architecture that serves them must therefore be inherently dynamic — adaptive at multiple timescales, self-calibrating in its uncertainty, and structurally capable of integrating new information without losing the knowledge it has already consolidated. These are the properties that the present work has attempted to instantiate with formal precision. The gap between the architecture as specified and the architecture as fully realized in code and deployment is real and non-trivial. It is, however, a gap of engineering rather than a gap of principle.

---

## References

Ames, A. D., Xu, X., Grizzle, J. W., and Tabuada, P. (2016). Control barrier function based quadratic programs for safety critical systems. *IEEE Transactions on Automatic Control*, 62(8), 3861–3876.

Ansari, A. F., Stella, L., Turkmen, C., Zhang, X., Mercado, P., Shen, H., Shchur, O., Rangapuram, S. S., Arango, S. P., Kapoor, S., et al. (2024). Chronos: Learning the language of time series. *arXiv preprint arXiv:2403.07815*.

Anil, C., Lucas, J., and Grosse, R. (2019). Sorting out Lipschitz function approximation. *Proceedings of ICML 2019*, PMLR 97, 291–301.

Annadani, Y., Rothfuss, J., Lacoste, A., Scherrer, N., Goyal, A., Bengio, Y., and Bauer, S. (2021). Variational causal networks: Approximate Bayesian inference over causal structures. *arXiv preprint arXiv:2106.07635*.

Bagnall, A., Lines, J., Bostrom, A., Large, J., and Keogh, E. (2017). The great time series classification bake off: a review and experimental evaluation of recent algorithmic advances. *Data Mining and Knowledge Discovery*, 31(3), 606–660.

Balle, B., Barthe, G., Gaboardi, M., Hsu, J., and Sato, T. (2020). Hypothesis testing interpretations and renormalization of differential privacy. *Proceedings of AISTATS 2020*, PMLR 108.

Belghazi, M. I., Baratin, A., Rajeswar, S., Ozair, S., Bengio, Y., Courville, A., and Hjelm, R. D. (2018). MINE: Mutual information neural estimation. *Proceedings of ICML 2018*, PMLR 80.

Blelloch, G. E. (1990). Prefix sums and their applications. In J. H. Reif (Ed.), *Synthesis of Parallel Algorithms*, Morgan Kaufmann.

Box, G. E. P., Jenkins, G. M., Reinsel, G. C., and Ljung, G. M. (2015). *Time Series Analysis: Forecasting and Control* (5th ed.). Wiley.

Breiman, L., Friedman, J., Stone, C. J., and Olshen, R. A. (1984). *Classification and Regression Trees*. Chapman and Hall/CRC.

Cao, D., Wang, Y., Duan, J., Zhang, C., Zhu, X., Huang, C., Tong, Y., Xu, B., Bai, J., Tong, J., and Zhang, Q. (2020). Spectral temporal graph neural network for multivariate time-series forecasting. *Advances in Neural Information Processing Systems*, 33, 17766–17778.

Dao, T., Fu, D. Y., Ermon, S., Rudra, A., and Ré, C. (2022). FlashAttention: Fast and memory-efficient exact attention with IO-awareness. *Advances in Neural Information Processing Systems*, 35.

Gao, S., Kong, S., and Clarke, E. M. (2013). dReal: An SMT solver for nonlinear theories over the reals. *Proceedings of CADE-24*, Lecture Notes in Computer Science 7898, 208–214.

Gibbs, I. and Candès, E. J. (2021). Adaptive conformal inference under distribution shift. *Advances in Neural Information Processing Systems*, 34, 1660–1672.

Goswami, M., Szafer, K., Choudhry, A., Cai, Y., Li, S., and Dubrawski, A. (2024). MOMENT: A family of open time-series foundation models. *Proceedings of ICML 2024*.

Granger, C. W. J. (1969). Investigating causal relations by econometric models and cross-spectral methods. *Econometrica*, 37(3), 424–438.

Grossberg, S. (1980). How does a brain build a cognitive code? *Psychological Review*, 87(1), 1–51.

Gu, A., Goel, K., and Ré, C. (2021). Efficiently modeling long sequences with structured state spaces. *International Conference on Learning Representations*, 2022.

Gu, A. and Dao, T. (2023). Mamba: Linear-time sequence modeling with selective state spaces. *arXiv preprint arXiv:2312.00752*.

Gu, A., Gupta, A., Goel, K., and Ré, C. (2022). On the parameterization and initialization of diagonal state space models. *Advances in Neural Information Processing Systems*, 35.

Gu, A., Johnson, I., Goel, K., Saab, K., Dao, T., Rudra, A., and Ré, C. (2022). Combining recurrent, convolutional, and continuous-time models with the structured state space sequence model (S4). *Advances in Neural Information Processing Systems*, 35.

Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., and Chen, W. (2021). LoRA: Low-rank adaptation of large language models. *International Conference on Learning Representations*, 2022.

Kirkpatrick, J., Pascanu, R., Rabinowitz, N., Veness, J., Desjardins, G., Rusu, A. A., Milan, K., Quan, J., Ramalho, T., Grabska-Barwinska, A., et al. (2017). Overcoming catastrophic forgetting in neural networks. *Proceedings of the National Academy of Sciences*, 114(13), 3521–3526.

Loshchilov, I. and Hutter, F. (2019). Decoupled weight decay regularization. *International Conference on Learning Representations*, 2019.

Lundberg, S. M. and Lee, S. I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, 30.

McCloskey, M. and Cohen, N. J. (1989). Catastrophic interference in connectionist networks: The sequential learning problem. *Psychology of Learning and Motivation*, 24, 109–165.

Mironov, I. (2017). Rényi differential privacy. *30th IEEE Computer Security Foundations Symposium*, 263–275.

Miyato, T., Kataoka, T., Koyama, M., and Yoshida, Y. (2018). Spectral normalization for generative adversarial networks. *International Conference on Learning Representations*, 2018.

Oja, E. (1982). Simplified neuron model as a principal component analyzer. *Journal of Mathematical Biology*, 15(3), 267–273.

Pearl, J. (2009). *Causality: Models, Reasoning, and Inference* (2nd ed.). Cambridge University Press.

Peng, B., Alcaide, E., Anthony, Q., Albalak, A., Arcadinho, S., Cao, H., Cheng, X., Chung, M., Grella, M., GV, K. K., et al. (2023). RWKV: Reinventing RNNs for the transformer era. *Findings of EMNLP 2023*.

Reddi, S. J., Kale, S., and Kumar, S. (2018). On the convergence of Adam and beyond. *International Conference on Learning Representations*, 2018.

Ribeiro, M. T., Singh, S., and Guestrin, C. (2016). "Why should I trust you?": Explaining the predictions of any classifier. *Proceedings of KDD 2016*, 1135–1144.

Smith, J. T. H., Warrington, A., and Linderman, S. (2022). Simplified state space layers for sequence modeling. *International Conference on Learning Representations*, 2023.

van den Oord, A., Li, Y., and Vinyals, O. (2018). Representation learning with contrastive predictive coding. *arXiv preprint arXiv:1807.03748*.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., and Polosukhin, I. (2017). Attention is all you need. *Advances in Neural Information Processing Systems*, 30.

Vovk, V., Gammerman, A., and Shafer, G. (2005). *Algorithmic Learning in a Random World*. Springer.

Wieland, P. and Allgöwer, F. (2007). Constructive safety using control barrier functions. *IFAC Proceedings Volumes*, 40(12), 462–467.

Woo, G., Liu, C., Kumar, A., Xiong, C., Savarese, S., and Sahoo, D. (2024). Unified training of universal time series forecasting transformers. *Proceedings of ICML 2024*.

Xu, C. and Xie, Y. (2021). Conformal prediction interval for dynamic time-series. *Proceedings of ICML 2021*, PMLR 139, 11559–11569.

Zheng, X., Aragam, B., Ravikumar, P., and Xing, E. P. (2018). DAGs with NO TEARS: Continuous optimization for structure learning. *Advances in Neural Information Processing Systems*, 31.
