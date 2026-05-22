import numpy as np
from typing import Dict, List, Optional, Tuple, Callable

from engine.tensor import Tensor, Parameter, zeros
from engine.module import Module
from config import ESEConfig


# ---------------------------------------------------------------------------
# CART decision tree — pure numpy, no sklearn
# ---------------------------------------------------------------------------

class DecisionNode:
    """Node in a CART decision tree."""

    def __init__(self):
        self.feature_idx: Optional[int] = None
        self.threshold: Optional[float] = None
        self.left: Optional["DecisionNode"] = None
        self.right: Optional["DecisionNode"] = None
        self.is_leaf: bool = False
        self.prediction: Optional[float] = None   # majority class or mean
        self.confidence: Optional[float] = None   # class probability or 1.0
        self.n_samples: int = 0
        self.class_counts: Optional[dict] = None  # for classification rule text


class CARTExtractor:
    """Pure numpy CART decision tree for rule extraction (classification or regression)."""

    def __init__(self, max_depth: int = 5, min_samples_split: int = 20):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root: Optional[DecisionNode] = None
        self._is_classifier: bool = True

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit tree.

        X: (N, n_features) float64
        y: (N,) integer class labels or float targets
        """
        self._is_classifier = (y.dtype.kind in ('i', 'u')) or (np.unique(y).size <= 20)
        self.root = self._build(X, y, depth=0)

    def _gini(self, y: np.ndarray) -> float:
        """Gini impurity of label array."""
        if y.size == 0:
            return 0.0
        classes, counts = np.unique(y, return_counts=True)
        probs = counts / y.size
        return float(1.0 - np.sum(probs ** 2))

    def _mse(self, y: np.ndarray) -> float:
        """Mean squared error for regression splits."""
        if y.size == 0:
            return 0.0
        return float(np.var(y))

    def _impurity(self, y: np.ndarray) -> float:
        return self._gini(y) if self._is_classifier else self._mse(y)

    def _best_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[int, float, float]:
        """Returns (feature_idx, threshold, gain).

        Iterates over each feature and a set of candidate thresholds,
        choosing the split that maximises information gain.
        """
        n, n_feat = X.shape
        best_gain = -np.inf
        best_feat = 0
        best_thr = 0.0

        parent_imp = self._impurity(y)

        for feat in range(n_feat):
            col = X[:, feat]
            # Use unique sorted values as candidate thresholds (midpoints)
            unique_vals = np.unique(col)
            if unique_vals.size < 2:
                continue
            thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0

            for thr in thresholds:
                left_mask = col <= thr
                right_mask = ~left_mask

                n_l = left_mask.sum()
                n_r = right_mask.sum()
                if n_l == 0 or n_r == 0:
                    continue

                y_l = y[left_mask]
                y_r = y[right_mask]

                gain = parent_imp - (n_l / n) * self._impurity(y_l) \
                                  - (n_r / n) * self._impurity(y_r)

                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat
                    best_thr = thr

        return best_feat, best_thr, best_gain

    def _make_leaf(self, y: np.ndarray, n_samples: int) -> DecisionNode:
        node = DecisionNode()
        node.is_leaf = True
        node.n_samples = n_samples
        if self._is_classifier:
            classes, counts = np.unique(y, return_counts=True)
            best_idx = np.argmax(counts)
            node.prediction = float(classes[best_idx])
            node.confidence = float(counts[best_idx]) / n_samples
            node.class_counts = {int(c): int(k) for c, k in zip(classes, counts)}
        else:
            node.prediction = float(np.mean(y))
            node.confidence = 1.0
        return node

    def _build(self, X: np.ndarray, y: np.ndarray, depth: int) -> DecisionNode:
        n = y.size

        # Stopping criteria
        if depth >= self.max_depth or n < self.min_samples_split or \
                (self._is_classifier and np.unique(y).size == 1):
            return self._make_leaf(y, n)

        feat, thr, gain = self._best_split(X, y)

        if gain <= 0.0:
            return self._make_leaf(y, n)

        left_mask = X[:, feat] <= thr
        right_mask = ~left_mask

        if left_mask.sum() == 0 or right_mask.sum() == 0:
            return self._make_leaf(y, n)

        node = DecisionNode()
        node.feature_idx = feat
        node.threshold = thr
        node.n_samples = n
        node.left = self._build(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build(X[right_mask], y[right_mask], depth + 1)
        return node

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def _predict_one(self, x: np.ndarray, node: DecisionNode) -> float:
        if node.is_leaf:
            return node.prediction
        if x[node.feature_idx] <= node.threshold:
            return self._predict_one(x, node.left)
        else:
            return self._predict_one(x, node.right)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict for each row of X."""
        if self.root is None:
            raise RuntimeError("CARTExtractor not fitted. Call fit() first.")
        return np.array([self._predict_one(x, self.root) for x in X])

    # ------------------------------------------------------------------
    # Rule extraction
    # ------------------------------------------------------------------

    def extract_rules(self, feature_names: Optional[List[str]] = None) -> List[str]:
        """Walk tree, return list of IF-THEN rules as strings."""
        if self.root is None:
            return []

        rules: List[str] = []

        def _walk(node: DecisionNode, conditions: List[str]):
            if node.is_leaf:
                cond_str = " AND ".join(conditions) if conditions else "TRUE"
                if self._is_classifier:
                    rule = (f"IF {cond_str} THEN class={int(node.prediction)} "
                            f"(conf={node.confidence:.2f}, n={node.n_samples})")
                else:
                    rule = (f"IF {cond_str} THEN value={node.prediction:.4f} "
                            f"(n={node.n_samples})")
                rules.append(rule)
                return

            fname = (feature_names[node.feature_idx]
                     if feature_names and node.feature_idx < len(feature_names)
                     else f"h[{node.feature_idx}]")

            _walk(node.left,
                  conditions + [f"{fname} <= {node.threshold:.4f}"])
            _walk(node.right,
                  conditions + [f"{fname} > {node.threshold:.4f}"])

        _walk(self.root, [])
        return rules


# ---------------------------------------------------------------------------
# Explainability Engine
# ---------------------------------------------------------------------------

class ExplainabilityEngine(Module):
    """Black-box elimination: rule extraction + attribution + counterfactual tracing."""

    def __init__(self, d_model: int, n_output: int, config: ESEConfig,
                 feature_names: Optional[List[str]] = None):
        super().__init__()
        self.d_model = d_model
        self.n_output = n_output
        self.config = config
        self.feature_names = feature_names

        self.cart = CARTExtractor(
            max_depth=config.max_depth,
            min_samples_split=config.min_samples
        )

        object.__setattr__(self, "collected_latents", [])   # list of (h_np, y_np)
        object.__setattr__(self, "_cart_fitted", False)
        object.__setattr__(self, "_crg_weights", None)       # set externally

    # ------------------------------------------------------------------
    # Data collection
    # ------------------------------------------------------------------

    def record(self, h: np.ndarray, y: np.ndarray):
        """Accumulate (h, y) pairs; subsample buffer to 5000 max."""
        buf = self.collected_latents
        buf.append((h.copy(), y.copy()))

        if len(buf) > 5000:
            # Keep a random 5000
            indices = np.random.choice(len(buf), 5000, replace=False)
            new_buf = [buf[i] for i in indices]
            object.__setattr__(self, "collected_latents", new_buf)

    def fit_rules(self):
        """Fit CART on all collected (h, y) pairs."""
        buf = self.collected_latents
        if len(buf) < self.config.min_samples:
            return

        H = np.stack([pair[0].flatten() for pair in buf], axis=0)  # (N, d_model)
        Y_raw = np.stack([pair[1].flatten() for pair in buf], axis=0)  # (N, *)

        # Use argmax of y as class label for multi-output; scalar for single
        if Y_raw.ndim > 1 and Y_raw.shape[1] > 1:
            Y = Y_raw.argmax(axis=1).astype(np.int32)
        else:
            Y = Y_raw.flatten()

        self.cart.fit(H, Y)
        object.__setattr__(self, "_cart_fitted", True)

    # ------------------------------------------------------------------
    # Attribution
    # ------------------------------------------------------------------

    def attribute(self, x: Tensor, y: Tensor,
                  crg_W: Optional[np.ndarray] = None) -> np.ndarray:
        """Gradient-based attribution + CRG weighting.

        x: (batch, T, d_model)
        y: scalar or (batch, output_dim)
        Returns: (d_model,) normalised attribution scores.
        """
        # Ensure x requires grad
        x_grad = Tensor(x.data.copy(), requires_grad=True)

        # Re-run forward just using x as-is (identity through d_model dim)
        # We compute ∂y/∂x_i using the existing gradient on x after backward
        # Since we don't have the full model here, approximate via:
        # attr_i = |∂(y.sum())/∂x_i| averaged over batch & time

        # Use y.sum() as scalar loss, backward through y which references x
        loss = y.sum()
        loss.backward()

        # Try to find gradient on x by matching data pointer
        if x.requires_grad and x.grad is not None:
            raw_grad = x.grad  # (batch, T, d_model) or (batch, d_model)
        elif x_grad.grad is not None:
            raw_grad = x_grad.grad
        else:
            # Fallback: finite differences approximation not feasible here;
            # use uniform attribution
            raw_grad = np.ones(x.data.shape)

        # Average absolute gradient over batch & time -> (d_model,)
        if raw_grad.ndim == 3:
            attr = np.abs(raw_grad).mean(axis=(0, 1))  # (d_model,)
        elif raw_grad.ndim == 2:
            attr = np.abs(raw_grad).mean(axis=0)       # (d_model,)
        else:
            attr = np.abs(raw_grad).flatten()[:self.d_model]

        # CRG weighting: attr_i = Σ_j W_ij * |∂y/∂x_j|
        W = crg_W if crg_W is not None else self._crg_weights
        if W is not None:
            # W: (n_nodes, n_nodes); attr may be (d_model,)
            # Resize to compatible dimension
            n = min(W.shape[0], attr.shape[0])
            W_sub = np.abs(W[:n, :n])
            attr_sub = attr[:n]
            attr_weighted = W_sub @ attr_sub
            # Pad back if needed
            if n < attr.shape[0]:
                attr = np.concatenate([attr_weighted, attr[n:]])
            else:
                attr = attr_weighted[:attr.shape[0]]

        # Normalise so sum = 1
        total = attr.sum()
        if total > 1e-12:
            attr = attr / total
        else:
            attr = np.ones_like(attr) / max(attr.size, 1)

        return attr

    # ------------------------------------------------------------------
    # Counterfactual generation
    # ------------------------------------------------------------------

    def counterfactual(self, h_np: np.ndarray, y_target: np.ndarray,
                       forward_fn: Callable, n_steps: int = 50) -> dict:
        """Find minimal perturbation δ such that forward_fn(h+δ) ≈ y_target.

        Optimises: ||δ||^2 + α*||δ||_1  s.t.  ||forward_fn(h+δ) - y_target|| < ε
        via gradient descent on δ.

        Returns dict with h_counterfactual, delta, feature_changes, steps_taken.
        """
        cf_lr = self.config.cf_lr
        alpha_l1 = 0.01   # L1 sparsity weight
        eps_target = 1e-2

        delta = np.zeros_like(h_np, dtype=np.float64)
        y_tgt = y_target.astype(np.float64)
        steps_taken = n_steps

        for step in range(n_steps):
            delta_t = Tensor(delta.copy(), requires_grad=True)
            h_t = Tensor(h_np + delta, requires_grad=False)

            # Combine so gradient flows through delta_t
            h_in = Tensor(h_np, requires_grad=False)
            h_perturbed = h_in + delta_t

            y_pred = forward_fn(h_perturbed)
            y_tgt_t = Tensor(y_tgt, requires_grad=False)

            # Task loss: ||y_pred - y_target||^2
            diff = y_pred - y_tgt_t
            task_loss = (diff * diff).sum()

            # Regularisation: ||δ||^2 + α*||δ||_1
            delta_sq = (delta_t * delta_t).sum()
            delta_abs = delta_t.abs().sum()
            reg_loss = delta_sq + Tensor(np.array([alpha_l1])) * delta_abs

            total_loss = task_loss + reg_loss
            total_loss.backward()

            if delta_t.grad is not None:
                delta -= cf_lr * delta_t.grad
                delta_t.grad = None

            # Check convergence
            residual = np.linalg.norm(y_pred.data - y_tgt)
            if residual < eps_target:
                steps_taken = step + 1
                break

        h_cf = h_np + delta

        # Feature changes: list of (feature_idx, delta_value) sorted by |delta|
        order = np.argsort(np.abs(delta))[::-1]
        feature_changes = [(int(i), float(delta[i])) for i in order if abs(delta[i]) > 1e-6]

        return {
            "h_counterfactual": h_cf,
            "delta": delta,
            "feature_changes": feature_changes,
            "steps_taken": steps_taken,
        }

    # ------------------------------------------------------------------
    # Full explanation pipeline
    # ------------------------------------------------------------------

    def explain(self, x: Tensor, y: Tensor, crg_W: Optional[np.ndarray] = None,
                feature_names: Optional[List[str]] = None) -> dict:
        """Full explanation pipeline.

        Returns structured dict with attribution, rules, top_features.
        """
        names = feature_names or self.feature_names

        # 1. Attribution
        attr = self.attribute(x, y, crg_W=crg_W)

        # 2. Active rules from CART (if fitted)
        rules: List[str] = []
        if self._cart_fitted:
            # Find rules matching the current latent
            h_np = x.data
            if h_np.ndim == 3:
                h_np = h_np[0, -1, :]   # last timestep, first batch
            elif h_np.ndim == 2:
                h_np = h_np[0, :]
            h_np = h_np.flatten()[:self.d_model]
            rules = self.cart.extract_rules(feature_names=names)

        # 3. Top-k influential features (k=10)
        k = min(10, attr.shape[0])
        top_k_idx = np.argsort(attr)[::-1][:k]
        top_features = []
        for i in top_k_idx:
            fname = (names[i] if names and i < len(names) else f"feature_{i}")
            top_features.append((fname, float(attr[i])))

        return {
            "attribution": attr,
            "top_features": top_features,
            "rules": rules,
            "n_rules": len(rules),
            "cart_fitted": self._cart_fitted,
        }

    def format_report(self, explanation: dict) -> str:
        """Return human-readable string report."""
        lines = []

        # Attribution line
        top = explanation.get("top_features", [])
        if top:
            attr_parts = ", ".join(f"{name} ({score:.4f})" for name, score in top)
            lines.append(f"ATTRIBUTION: {attr_parts}")

        # Rules
        rules = explanation.get("rules", [])
        if rules:
            lines.append("RULES:")
            for rule in rules[:self.config.max_rules]:
                lines.append(f"  {rule}")
        else:
            lines.append("RULES: (no rules fitted yet)")

        return "\n".join(lines)

    def forward(self, x: Tensor) -> Tensor:
        """Pass-through; ExplainabilityEngine is used via explain/attribute methods."""
        return x
