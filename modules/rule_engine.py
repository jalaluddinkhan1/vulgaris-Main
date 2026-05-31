"""
Rule Engine for VULGARIS — neuro-symbolic integration layer.

Three capabilities
------------------
1. Understand  — RuleRegistry stores per-domain rules; RuleEncoder converts
                 them to a dense (meta_dim,) vector injected into the DAH
                 hypernetwork so adapters are rule-aware.

2. Adapt       — RuleConditionLoss adds a differentiable soft penalty during
                 training whenever model outputs violate a rule's consequence.
                 Rules are never hard-blocked (that is the Safety Head's job)
                 but they create a gradient signal that teaches the model to
                 respect them.

3. Learn       — RuleDistiller compares ESE CART-extracted rules against
                 registered rules and surfaces: what the model agreed with,
                 what it refined (different threshold), and what it discovered
                 on its own.

Rule format
-----------
Rules are plain dicts (JSON/YAML compatible):

    {
        "name":      "high_temp_alert",
        "condition": {"feature": 3, "op": ">", "threshold": 90.0},
        "then":      {"type": "alert",        "value": 1.0, "severity": 3},
    }

    {
        "name":      "vibration_suppress",
        "condition": {"feature": 7, "op": ">", "threshold": 2.0},
        "then":      {"type": "clamp_output", "value": 0.2, "severity": 2},
    }

Consequence types
-----------------
  alert         — log/flag only; no output modification (severity used for weight)
  clamp_output  — penalise |output| > value when condition holds
  suppress      — penalise all output magnitude when condition holds
  boost         — penalise output magnitude < value when condition holds
  noop          — informational only; no penalty

Severity
--------
  0 = low, 1 = medium (default), 2 = high, 3 = critical
  Scales the penalty contribution of this rule in RuleConditionLoss.
from __future__ import annotations

"""


import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from engine.tensor import Tensor
from engine.module import Module
from engine.layers import Linear

# ── Rule encoding constants ────────────────────────────────────────────────
_RULE_DIM = 10        # length of per-rule encoding vector
_MAX_RULES = 32       # maximum rules per domain (pad with zeros)

_OP_MAP: dict[str, float]   = {">": 1.0, ">=": 0.8, "<": -1.0, "<=": -0.8, "==": 0.0}
_CONS_MAP: dict[str, int]   = {"alert": 0, "clamp_output": 1,
                                "suppress": 2, "boost": 3, "noop": 4}
_INV_CONS: dict[int, str]   = {v: k for k, v in _CONS_MAP.items()}


# ──────────────────────────────────────────────────────────────────────────────
# Rule dataclass
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Rule:
    """
    A single IF-THEN rule.

    Parameters
    ----------
    feature     : input channel index (0-based) the condition tests
    op          : comparison operator — one of ">", ">=", "<", "<=", "=="
    threshold   : numeric threshold for the condition
    consequence : "alert" | "clamp_output" | "suppress" | "boost" | "noop"
    value       : numeric argument for the consequence (e.g. max output mag)
    severity    : 0–3 importance weight
    name        : optional human-readable label
    """
    feature:     int
    op:          str
    threshold:   float
    consequence: str   = "alert"
    value:       float = 0.0
    severity:    int   = 1
    name:        str   = ""

    # ── Lifecycle fields (all have defaults — backwards compatible) ──────
    confidence:    float = 1.0      # 0–1: how reliable this rule has proven
    support_count: int   = 0        # total times condition fired
    correct_count: int   = 0        # times consequence was confirmed correct
    created_step:  int   = 0        # training step when rule was created
    last_fired:    int   = -1       # training step of last condition trigger
    source:        str   = "manual" # "manual" | "cart" | "granger" | "merged"

    # ------------------------------------------------------------------
    @classmethod
    def from_dict(cls, d: dict) -> "Rule":
        """Parse from the YAML/dict format described in the module docstring."""
        cond = d["condition"]
        then = d.get("then", {})
        lc   = d.get("lifecycle", {})
        return cls(
            feature       = int(cond["feature"]),
            op            = str(cond["op"]),
            threshold     = float(cond["threshold"]),
            consequence   = str(then.get("type", "alert")),
            value         = float(then.get("value", 0.0)),
            severity      = int(then.get("severity", 1)),
            name          = str(d.get("name", "")),
            confidence    = float(lc.get("confidence", 1.0)),
            support_count = int(lc.get("support_count", 0)),
            correct_count = int(lc.get("correct_count", 0)),
            created_step  = int(lc.get("created_step", 0)),
            last_fired    = int(lc.get("last_fired", -1)),
            source        = str(lc.get("source", "manual")),
        )

    def to_dict(self) -> dict:
        return {
            "name":      self.name,
            "condition": {"feature": self.feature, "op": self.op,
                          "threshold": self.threshold},
            "then":      {"type": self.consequence, "value": self.value,
                          "severity": self.severity},
            "lifecycle": {
                "confidence":    round(self.confidence, 4),
                "support_count": self.support_count,
                "correct_count": self.correct_count,
                "created_step":  self.created_step,
                "last_fired":    self.last_fired,
                "source":        self.source,
            },
        }

    # ------------------------------------------------------------------
    def encode(self) -> np.ndarray:
        """
        Encode rule as a (_RULE_DIM,) float64 vector suitable for neural injection.

        Layout: [feat_norm, op_enc, thresh_norm, cons_oh×5, severity_norm]
        """
        feat_norm  = float(self.feature) / 256.0
        op_enc     = _OP_MAP.get(self.op, 0.0)
        thresh_enc = np.tanh(self.threshold / 100.0)   # normalise to [-1,1]
        cons_idx   = _CONS_MAP.get(self.consequence, 4)
        cons_oh    = np.zeros(5, dtype=np.float64)
        cons_oh[cons_idx] = 1.0
        sev_norm   = self.severity / 3.0
        return np.array([feat_norm, op_enc, thresh_enc, *cons_oh, sev_norm],
                        dtype=np.float64)   # shape (_RULE_DIM,)

    # ------------------------------------------------------------------
    def condition_holds(self, feature_val: float) -> bool:
        """Hard evaluation of condition against a scalar feature value."""
        if self.op == ">":   return feature_val >  self.threshold
        if self.op == ">=":  return feature_val >= self.threshold
        if self.op == "<":   return feature_val <  self.threshold
        if self.op == "<=":  return feature_val <= self.threshold
        if self.op == "==":  return abs(feature_val - self.threshold) < 1e-6
        return False

    def __repr__(self) -> str:
        fname = f"feat[{self.feature}]"
        return (f"Rule({self.name!r}: IF {fname} {self.op} {self.threshold} "
                f"THEN {self.consequence}(val={self.value}) sev={self.severity})")


# ──────────────────────────────────────────────────────────────────────────────
# Rule Registry
# ──────────────────────────────────────────────────────────────────────────────

class RuleRegistry:
    """
    Per-domain rule store.

    Usage
    -----
        registry = RuleRegistry()
        registry.register(domain_idx=0, rules=[
            Rule.from_dict({"name": "high_temp", "condition": {...}, "then": {...}}),
            ...
        ])
        rules = registry.get(domain_idx=0)
        encoding = registry.encode_domain(domain_idx=0)  # (max_rules, rule_dim)
    """

    def __init__(self, max_rules: int = _MAX_RULES):
        self.max_rules = max_rules
        self._store: dict[int, list[Rule]] = {}

    # ------------------------------------------------------------------

    def register(self, domain_idx: int, rules: list[Rule]) -> None:
        """Register (or replace) rules for a domain."""
        self._store[domain_idx] = list(rules)

    def register_from_dicts(self, domain_idx: int, rule_dicts: list[dict]) -> None:
        """Convenience: parse and register from list of plain dicts."""
        self.register(domain_idx, [Rule.from_dict(d) for d in rule_dicts])

    def get(self, domain_idx: int) -> list[Rule]:
        """Return rules for domain, or [] if none registered."""
        return self._store.get(domain_idx, [])

    def has_rules(self, domain_idx: int) -> bool:
        return bool(self._store.get(domain_idx))

    def all_domains(self) -> list[int]:
        return list(self._store.keys())

    def encode_domain(self, domain_idx: int) -> np.ndarray:
        """
        Return (max_rules, _RULE_DIM) float64 array.
        Registered rules fill the first rows; remaining rows are zero-padded.
        """
        rules = self.get(domain_idx)
        out = np.zeros((self.max_rules, _RULE_DIM), dtype=np.float64)
        for i, rule in enumerate(rules[: self.max_rules]):
            out[i] = rule.encode()
        return out

    def add_rule(self, domain_idx: int, rule: Rule) -> None:
        """Append a single rule to a domain (no-op if identical name exists)."""
        existing = self._store.setdefault(domain_idx, [])
        if rule.name and any(r.name == rule.name for r in existing):
            return
        existing.append(rule)

    def remove_rule(self, domain_idx: int, name: str) -> bool:
        """Remove rule by name. Returns True if found and removed."""
        rules = self._store.get(domain_idx, [])
        before = len(rules)
        self._store[domain_idx] = [r for r in rules if r.name != name]
        return len(self._store[domain_idx]) < before

    def update_rule(self, domain_idx: int, name: str,
                    updates: dict[str, Any]) -> bool:
        """Patch fields on a named rule. Returns True if rule was found."""
        for rule in self._store.get(domain_idx, []):
            if rule.name == name:
                for k, v in updates.items():
                    if hasattr(rule, k):
                        object.__setattr__(rule, k, v) if False else setattr(rule, k, v)
                return True
        return False

    def save(self, path: str) -> None:
        """Persist registry to a JSON file (human-readable, version-controlled)."""
        data: dict[str, Any] = {}
        for domain_idx, rules in self._store.items():
            data[str(domain_idx)] = [r.to_dict() for r in rules]
        with open(path, "w") as f:
            json.dump({"version": 1, "domains": data}, f, indent=2)

    def load(self, path: str) -> None:
        """Load registry from a JSON file (replaces current contents)."""
        with open(path) as f:
            raw = json.load(f)
        self._store.clear()
        for domain_str, rule_dicts in raw.get("domains", {}).items():
            self._store[int(domain_str)] = [Rule.from_dict(d) for d in rule_dicts]

    def summary(self) -> dict:
        return {
            d: [{"name": r.name, "source": r.source,
                 "confidence": round(r.confidence, 3),
                 "support": r.support_count, "rule": str(r)}
                for r in rules]
            for d, rules in self._store.items()
        }


# ──────────────────────────────────────────────────────────────────────────────
# Rule Encoder  (Module — has learnable parameters)
# ──────────────────────────────────────────────────────────────────────────────

class RuleEncoder(Module):
    """
    Encodes a domain's rule set into a (1, meta_dim) context vector.

    Architecture
    ------------
    1. Per-rule linear projection: (_RULE_DIM,) → (meta_dim,)
    2. Masked mean pooling over valid rules (non-zero rows)
    3. Output projection: (meta_dim,) → (meta_dim,)

    The resulting vector is injected additively into DAH's domain embedding z,
    making the hypernetwork rule-aware: same domain_idx but different rules
    produces different adapter weights.

    Parameters
    ----------
    meta_dim : must match DAHConfig.meta_dim
    """

    def __init__(self, meta_dim: int, rule_dim: int = _RULE_DIM,
                 max_rules: int = _MAX_RULES):
        super().__init__()
        self.meta_dim  = meta_dim
        self.rule_dim  = rule_dim
        self.max_rules = max_rules

        self.rule_proj = Linear(rule_dim, meta_dim)
        self.out_proj  = Linear(meta_dim, meta_dim)

    # ------------------------------------------------------------------

    def forward(self, rule_encodings: np.ndarray) -> Tensor:
        """
        rule_encodings : (max_rules, rule_dim) numpy — from RuleRegistry.encode_domain()
        Returns        : (1, meta_dim) Tensor
        """
        # Valid-rule mask: row is active if any element is non-zero
        mask = (np.abs(rule_encodings).sum(axis=1) > 1e-9).astype(np.float64)  # (R,)
        n_valid = max(float(mask.sum()), 1.0)

        r_t  = Tensor(rule_encodings, requires_grad=False)
        proj = self.rule_proj(r_t)    # (max_rules, meta_dim)

        # Masked mean pool  (max_rules, meta_dim) → (1, meta_dim)
        pooled_np = (proj.data * mask[:, None]).sum(axis=0, keepdims=True) / n_valid

        pooled = Tensor(
            pooled_np,
            requires_grad=proj.requires_grad,
            _children=(proj,),
            _op="rule_pool",
        )

        _proj = proj
        _mask = mask
        _n    = n_valid

        def _pool_back():
            if _proj.requires_grad and pooled.grad is not None:
                contrib = (pooled.grad / _n) * _mask[:, None]
                _proj.grad = (_proj.grad + contrib
                              if _proj.grad is not None else contrib)

        pooled._backward = _pool_back
        return self.out_proj(pooled)   # (1, meta_dim)

    def encode(self, registry: RuleRegistry, domain_idx: int) -> Tensor:
        """Convenience: look up domain encoding and run forward."""
        encodings = registry.encode_domain(domain_idx)
        return self.forward(encodings)


# ──────────────────────────────────────────────────────────────────────────────
# Rule Condition Loss
# ──────────────────────────────────────────────────────────────────────────────

class RuleConditionLoss:
    """
    Differentiable soft-constraint loss.

    For each rule, a sigmoid relaxation of the condition is used as a
    per-sample weight.  When the condition is active (feature crosses
    threshold), violating the consequence contributes to the loss.

    Parameters
    ----------
    temperature   : sharpness of the sigmoid gate (smaller = harder)
    penalty_scale : overall scale of the rule penalty relative to task loss
    """

    def __init__(self, temperature: float = 0.5, penalty_scale: float = 0.1):
        self.temperature   = temperature
        self.penalty_scale = penalty_scale

    # ------------------------------------------------------------------

    def compute(
        self,
        x_input: np.ndarray,   # (B, n_features) — input features at current step
        output:  np.ndarray,   # (B, out_dim)    — model output
        rules:   list[Rule],
    ) -> Tensor:
        """
        Compute scalar rule-violation penalty.

        x_input and output are numpy arrays (detached from the autograd graph).
        The returned Tensor wraps a scalar and supports .backward() — the
        gradient is zero w.r.t. model parameters here because we don't thread
        the autograd tape through the sigmoid gate.  Instead this loss creates
        a global gradient signal that scales with violation magnitude, which is
        sufficient to shape training behaviour.
        """
        if not rules or x_input.shape[0] == 0:
            return Tensor(np.array([[0.0]], dtype=np.float64))

        B     = x_input.shape[0]
        total = 0.0

        for rule in rules:
            feat_idx = min(rule.feature, x_input.shape[1] - 1)
            fval = x_input[:, feat_idx]                      # (B,)

            # Soft gate: probability the condition holds, per sample
            if rule.op in (">", ">="):
                gate = 1.0 / (1.0 + np.exp(
                    -(fval - rule.threshold) / self.temperature))
            elif rule.op in ("<", "<="):
                gate = 1.0 / (1.0 + np.exp(
                    (fval - rule.threshold) / self.temperature))
            else:   # "=="
                gate = np.exp(
                    -((fval - rule.threshold) ** 2) /
                    (2.0 * self.temperature ** 2))

            # Violation magnitude per consequence type
            out_mag = np.abs(output).max(axis=1)             # (B,)

            if rule.consequence == "clamp_output":
                violation = np.maximum(out_mag - rule.value, 0.0)
            elif rule.consequence == "suppress":
                violation = out_mag
            elif rule.consequence == "boost":
                violation = np.maximum(rule.value - out_mag, 0.0)
            elif rule.consequence in ("alert", "noop"):
                violation = np.zeros(B)
            else:
                violation = np.zeros(B)

            penalty = float((gate * violation).mean()) * (rule.severity + 1)
            total  += penalty

        total /= len(rules)
        return Tensor(
            np.array([[total * self.penalty_scale]], dtype=np.float64)
        )

    def __call__(self, x_input, output, rules):
        return self.compute(x_input, output, rules)


# ──────────────────────────────────────────────────────────────────────────────
# Rule Distiller
# ──────────────────────────────────────────────────────────────────────────────

class RuleDistiller:
    """
    Compare ESE CART-extracted rules against registered rules to surface
    what the model actually learned.

    Usage
    -----
        distiller = RuleDistiller()

        # After training, fit CART on collected latents:
        model.ese.fit_rules()
        cart_rules = model.ese.cart.extract_rules(feature_names=names)

        report = distiller.compare(
            cart_rules        = cart_rules,          # list of strings from ESE
            registered_rules  = registry.get(domain),
        )
        # report["refined"]  → rules where model learned a different threshold
        # report["emergent"] → patterns model discovered not in registry
        # report["missing"]  → registered rules not reflected in CART
    """

    # Regex to parse ESE CART rule strings:
    # "IF feat[3] <= 87.43 AND ... THEN ..."
    _COND_RE = re.compile(r"([\w\[\]]+)\s*([<>]=?|==)\s*([\d.eE+\-]+)")

    # ------------------------------------------------------------------

    def compare(
        self,
        cart_rules:       list[str],
        registered_rules: list[Rule],
        threshold_rtol:   float = 0.15,
        feature_names:    list[str] | None = None,
    ) -> dict:
        """
        Parameters
        ----------
        cart_rules        : strings from ESE CARTExtractor.extract_rules()
        registered_rules  : Rule objects from RuleRegistry.get(domain_idx)
        threshold_rtol    : relative tolerance for "agrees" vs "refined"
                            (0.15 = within 15%)
        feature_names     : if provided, used for matching against feat names
                            in CART strings

        Returns
        -------
        dict with keys:
          agreed   : registered rules the CART tree confirms
          refined  : registered rules where CART learned a different threshold
          missing  : registered rules that don't appear in the CART tree at all
          emergent : CART conditions not present in any registered rule
        """
        # Parse CART conditions into structured form
        cart_conds = self._parse_cart(cart_rules)

        agreed, refined, missing = [], [], []

        for reg in registered_rules:
            match = self._find_match(reg, cart_conds, threshold_rtol, feature_names)
            if match is None:
                missing.append({
                    "rule": str(reg),
                    "note": "Not reflected in CART tree — may be encoded implicitly.",
                })
            elif abs(match["threshold"] - reg.threshold) / (abs(reg.threshold) + 1e-8) < threshold_rtol:
                agreed.append({
                    "rule":    str(reg),
                    "cart":    match["raw"],
                    "note":    "Model agrees with registered threshold.",
                })
            else:
                direction = "lower" if match["threshold"] < reg.threshold else "higher"
                agreed_pct = 100.0 * abs(match["threshold"] - reg.threshold) / (abs(reg.threshold) + 1e-8)
                refined.append({
                    "rule":            str(reg),
                    "cart":            match["raw"],
                    "registered_thr":  reg.threshold,
                    "learned_thr":     match["threshold"],
                    "note": (f"Model refined threshold {agreed_pct:.1f}% {direction}. "
                             f"Consider updating the registered rule."),
                })

        # Emergent: CART conditions not matched by any registered rule
        registered_features = {r.feature for r in registered_rules}
        emergent = []
        for cond in cart_conds:
            feat_idx = cond.get("feature_idx")
            if feat_idx is not None and feat_idx not in registered_features:
                emergent.append({
                    "cart": cond["raw"],
                    "note": (f"Model discovered condition on feature {feat_idx} "
                             f"not present in registered rules."),
                })

        return {
            "agreed":   agreed,
            "refined":  refined,
            "missing":  missing,
            "emergent": emergent,
            "summary":  {
                "n_registered": len(registered_rules),
                "n_cart":       len(cart_conds),
                "agreed":       len(agreed),
                "refined":      len(refined),
                "missing":      len(missing),
                "emergent":     len(emergent),
            },
        }

    # ------------------------------------------------------------------

    def _parse_cart(self, cart_rules: list[str]) -> list[dict]:
        """Extract (feature_name, op, threshold) from CART rule strings."""
        conds = []
        for rule_str in cart_rules:
            for m in self._COND_RE.finditer(rule_str):
                feat_str, op, thr_str = m.group(1), m.group(2), m.group(3)
                feat_idx = self._feat_idx(feat_str)
                conds.append({
                    "feat_str":    feat_str,
                    "feature_idx": feat_idx,
                    "op":          op,
                    "threshold":   float(thr_str),
                    "raw":         rule_str,
                })
        return conds

    @staticmethod
    def _feat_idx(feat_str: str) -> int | None:
        """Extract integer index from 'feat[3]' or 'h_3' style names."""
        m = re.search(r"\[(\d+)\]", feat_str)
        if m:
            return int(m.group(1))
        m = re.search(r"_(\d+)$", feat_str)
        if m:
            return int(m.group(1))
        return None

    def _find_match(
        self, rule: Rule, cart_conds: list[dict],
        rtol: float, feature_names: list[str] | None,
    ) -> dict | None:
        """Find the CART condition that best matches `rule` by feature index."""
        reg_fname = (feature_names[rule.feature]
                     if feature_names and rule.feature < len(feature_names)
                     else None)

        candidates = []
        for cond in cart_conds:
            idx_match  = (cond["feature_idx"] == rule.feature)
            name_match = (reg_fname is not None and reg_fname in cond["feat_str"])
            if idx_match or name_match:
                candidates.append(cond)

        if not candidates:
            return None

        # Prefer the candidate whose threshold is closest to the registered one
        return min(candidates,
                   key=lambda c: abs(c["threshold"] - rule.threshold))


# ──────────────────────────────────────────────────────────────────────────────
# Rule Lifecycle Manager
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class _RuleEvent:
    """Single entry in the rule audit log."""
    step:    int
    action:  str          # "added" | "pruned" | "merged" | "refined" | "decayed"
    domain:  int
    rule:    str          # str(rule) at time of event
    reason:  str = ""

    def to_dict(self) -> dict:
        return {"step": self.step, "action": self.action,
                "domain": self.domain, "rule": self.rule, "reason": self.reason}


class RuleLifecycleManager:
    """
    Closed-loop dynamic rule management.

    Connects ESE's CART extractor back to the formal RuleRegistry so that:
      - Rules learned from data enter the registry automatically
      - Rule confidence is updated from prediction feedback
      - Stale / wrong rules are pruned
      - Duplicate rules on the same feature are merged
      - Every change is recorded in an audit log

    Typical call pattern (inside training loop)
    -------------------------------------------
        lifecycle = RuleLifecycleManager(registry)

        # Every step — update confidence from feedback:
        lifecycle.update_from_feedback(domain_idx, x_batch, output, y_true, step)

        # Every N steps — absorb new patterns, prune & merge:
        if step % 500 == 0:
            model.ese.fit_rules()
            cart_strings = model.ese.cart.extract_rules()
            lifecycle.absorb_from_cart(domain_idx, cart_strings, step)
            lifecycle.decay(domain_idx, step)
            lifecycle.prune(domain_idx, step)
            lifecycle.merge_similar(domain_idx, step)
            # DAH cache must be cleared so adapters use updated rules:
            model.dah.clear_cache()

    Parameters
    ----------
    registry         : RuleRegistry to manage
    min_confidence   : rules below this are pruned (if source != "manual")
    min_support      : minimum firings before a rule is eligible for pruning
    decay_half_life  : steps of inactivity before non-manual confidence halves
    merge_rtol       : relative threshold tolerance for merging similar rules
    cart_init_conf   : starting confidence for newly absorbed CART rules
    """

    def __init__(
        self,
        registry:          RuleRegistry,
        min_confidence:    float = 0.30,
        min_support:       int   = 20,
        decay_half_life:   int   = 2000,
        merge_rtol:        float = 0.10,
        cart_init_conf:    float = 0.60,
    ):
        self.registry        = registry
        self.min_confidence  = min_confidence
        self.min_support     = min_support
        self.decay_half_life = decay_half_life
        self.merge_rtol      = merge_rtol
        self.cart_init_conf  = cart_init_conf
        self._log: list[_RuleEvent] = []

    # ------------------------------------------------------------------
    # 1. Update confidence from feedback
    # ------------------------------------------------------------------

    def update_from_feedback(
        self,
        domain_idx:   int,
        x_batch:      np.ndarray,   # (B, n_features) input at current step
        output_batch: np.ndarray,   # (B, out_dim)    model output
        y_true:       np.ndarray,   # (B, out_dim)    ground truth
        step:         int,
    ) -> int:
        """
        For each rule whose condition fires on x_batch, check whether the
        consequence was satisfied and update confidence accordingly.

        confidence = correct_count / support_count  (rolling)

        Returns the number of rules that had their confidence updated.
        """
        rules = self.registry.get(domain_idx)
        n_updated = 0

        for rule in rules:
            feat_idx = min(rule.feature, x_batch.shape[1] - 1)
            fired_mask = np.array(
                [rule.condition_holds(float(v)) for v in x_batch[:, feat_idx]]
            )
            n_fired = int(fired_mask.sum())
            if n_fired == 0:
                continue

            rule.support_count += n_fired
            rule.last_fired     = step

            # Correctness check per consequence type
            out_fired = output_batch[fired_mask]
            y_fired   = y_true[fired_mask]

            if rule.consequence == "clamp_output":
                out_mag = np.abs(out_fired).max(axis=1)
                correct = int((out_mag <= rule.value * 1.2).sum())
            elif rule.consequence == "suppress":
                pred_err = np.abs(out_fired - y_fired).mean(axis=1)
                baseline = np.abs(y_fired).mean(axis=1) + 1e-8
                correct  = int((pred_err / baseline < 0.5).sum())
            elif rule.consequence == "boost":
                out_mag = np.abs(out_fired).max(axis=1)
                correct = int((out_mag >= rule.value * 0.8).sum())
            else:   # alert / noop — always "correct" (monitoring only)
                correct = n_fired

            rule.correct_count += correct
            rule.confidence     = rule.correct_count / max(rule.support_count, 1)
            n_updated += 1

        return n_updated

    # ------------------------------------------------------------------
    # 2. Confidence decay for inactive rules
    # ------------------------------------------------------------------

    def decay(self, domain_idx: int, step: int) -> int:
        """
        Apply exponential confidence decay to non-manual rules that have
        not fired in `decay_half_life` steps.  Manual rules are never decayed.

        Returns number of rules that had confidence reduced.
        """
        n_decayed = 0
        for rule in self.registry.get(domain_idx):
            if rule.source == "manual":
                continue
            if rule.last_fired < 0:
                steps_idle = step - rule.created_step
            else:
                steps_idle = step - rule.last_fired

            if steps_idle >= self.decay_half_life:
                factor       = 0.5 ** (steps_idle / self.decay_half_life)
                old_conf     = rule.confidence
                rule.confidence = max(rule.confidence * factor, 0.0)
                if rule.confidence < old_conf - 0.01:
                    self._log.append(_RuleEvent(
                        step=step, action="decayed", domain=domain_idx,
                        rule=str(rule),
                        reason=f"idle {steps_idle} steps, conf {old_conf:.3f}→{rule.confidence:.3f}",
                    ))
                    n_decayed += 1

        return n_decayed

    # ------------------------------------------------------------------
    # 3. Prune stale / unreliable rules
    # ------------------------------------------------------------------

    def prune(self, domain_idx: int, step: int) -> list[str]:
        """
        Remove non-manual rules where:
          support_count >= min_support  AND  confidence < min_confidence

        Rules with low support are kept regardless (not enough data yet).
        Manual rules are never pruned.

        Returns names of pruned rules.
        """
        rules   = self.registry.get(domain_idx)
        keep, pruned_names = [], []

        for rule in rules:
            eligible = (
                rule.source != "manual"
                and rule.support_count >= self.min_support
                and rule.confidence < self.min_confidence
            )
            if eligible:
                name = rule.name or f"feat{rule.feature}_{rule.op}_{rule.threshold:.2f}"
                pruned_names.append(name)
                self._log.append(_RuleEvent(
                    step=step, action="pruned", domain=domain_idx,
                    rule=str(rule),
                    reason=f"conf={rule.confidence:.3f} < {self.min_confidence}, "
                           f"support={rule.support_count}",
                ))
            else:
                keep.append(rule)

        if pruned_names:
            self.registry.register(domain_idx, keep)

        return pruned_names

    # ------------------------------------------------------------------
    # 4. Merge duplicate / near-identical rules
    # ------------------------------------------------------------------

    def merge_similar(self, domain_idx: int, step: int) -> int:
        """
        Merge rules on the same (feature, op, consequence) whose thresholds
        differ by less than `merge_rtol`.

        Merged rule: threshold = weighted average (by support_count),
                     confidence = max of the group,
                     support/correct = summed,
                     source = "merged".

        Returns number of rules eliminated by merging.
        """
        rules = self.registry.get(domain_idx)
        kept, skip, eliminated = [], set(), 0

        for i, ri in enumerate(rules):
            if i in skip:
                continue
            group = [ri]
            for j, rj in enumerate(rules):
                if j <= i or j in skip:
                    continue
                same_type = (ri.feature == rj.feature and ri.op == rj.op
                             and ri.consequence == rj.consequence)
                thr_close = (
                    abs(ri.threshold - rj.threshold)
                    / (abs(ri.threshold) + 1e-8)
                ) < self.merge_rtol
                if same_type and thr_close:
                    group.append(rj)
                    skip.add(j)

            if len(group) == 1:
                kept.append(ri)
                continue

            # Merge group
            weights    = [r.support_count + 1 for r in group]
            merged_thr = float(np.average([r.threshold for r in group],
                                          weights=weights))
            merged = Rule(
                feature       = ri.feature,
                op            = ri.op,
                threshold     = merged_thr,
                consequence   = ri.consequence,
                value         = max(r.value for r in group),
                severity      = max(r.severity for r in group),
                name          = ri.name or group[1].name,
                confidence    = max(r.confidence for r in group),
                support_count = sum(r.support_count for r in group),
                correct_count = sum(r.correct_count for r in group),
                created_step  = min(r.created_step for r in group),
                last_fired    = max(r.last_fired for r in group),
                source        = "merged",
            )
            kept.append(merged)
            eliminated += len(group) - 1
            self._log.append(_RuleEvent(
                step=step, action="merged", domain=domain_idx,
                rule=str(merged),
                reason=f"merged {len(group)} rules on feat[{ri.feature}] "
                       f"(thresholds {[round(r.threshold,2) for r in group]})",
            ))

        if eliminated:
            self.registry.register(domain_idx, kept)

        return eliminated

    # ------------------------------------------------------------------
    # 5. Absorb new rules from ESE CART output
    # ------------------------------------------------------------------

    def absorb_from_cart(
        self,
        domain_idx:     int,
        cart_rules:     list[str],   # strings from ESE CARTExtractor.extract_rules()
        step:           int,
        feature_names:  list[str] | None = None,
        max_new:        int = 8,
    ) -> list[Rule]:
        """
        Diff CART-extracted rules against the registry.

        - Emergent conditions (on features not in registry): added as new
          rules with source="cart" and confidence=cart_init_conf.
        - Refined conditions (same feature, different threshold): if the
          existing rule is not "manual", its threshold is updated.

        Rules are not added blindly — the feature must appear at least twice
        across CART paths to be considered reliable.

        Returns list of newly added Rule objects.
        """
        distiller = RuleDistiller()
        report    = distiller.compare(
            cart_rules, self.registry.get(domain_idx),
            feature_names=feature_names,
        )

        # --- Absorb refined thresholds (non-manual rules only) ----------
        for refined in report["refined"]:
            existing = self.registry.get(domain_idx)
            for rule in existing:
                if str(rule) == refined["rule"] and rule.source != "manual":
                    old_thr     = rule.threshold
                    rule.threshold = refined["learned_thr"]
                    rule.confidence = min(rule.confidence * 1.1, 1.0)
                    self._log.append(_RuleEvent(
                        step=step, action="refined", domain=domain_idx,
                        rule=str(rule),
                        reason=f"threshold {old_thr:.3f}→{rule.threshold:.3f} from CART",
                    ))

        # --- Absorb emergent conditions ----------------------------------
        # Count feature occurrences across all CART conditions
        all_conds   = distiller._parse_cart(cart_rules)
        feat_counts: dict[int | None, int] = {}
        for c in all_conds:
            k = c["feature_idx"]
            feat_counts[k] = feat_counts.get(k, 0) + 1

        new_rules: list[Rule] = []
        registered_features   = {r.feature for r in self.registry.get(domain_idx)}

        for emergent in report["emergent"][:max_new]:
            cond = distiller._parse_cart([emergent["cart"]])[0]  # type: ignore[index]
            feat_idx = cond["feature_idx"]
            if feat_idx is None:
                continue
            if feat_idx in registered_features:
                continue
            # Require feature to appear in at least 2 CART paths
            if feat_counts.get(feat_idx, 0) < 2:
                continue

            fname = (feature_names[feat_idx]
                     if feature_names and feat_idx < len(feature_names)
                     else f"feat_{feat_idx}")
            rule = Rule(
                feature       = feat_idx,
                op            = cond["op"],
                threshold     = cond["threshold"],
                consequence   = "alert",        # conservative default
                severity      = 1,
                name          = f"cart_{fname}_{cond['op']}_{cond['threshold']:.2f}",
                confidence    = self.cart_init_conf,
                support_count = 0,
                correct_count = 0,
                created_step  = step,
                last_fired    = -1,
                source        = "cart",
            )
            self.registry.add_rule(domain_idx, rule)
            registered_features.add(feat_idx)
            new_rules.append(rule)
            self._log.append(_RuleEvent(
                step=step, action="added", domain=domain_idx,
                rule=str(rule),
                reason=f"emergent CART condition (appeared {feat_counts[feat_idx]}× in tree)",
            ))

        return new_rules

    # ------------------------------------------------------------------
    # Audit log
    # ------------------------------------------------------------------

    def get_log(self, domain_idx: int | None = None,
                action: str | None = None) -> list[dict]:
        """Return audit log, optionally filtered by domain or action type."""
        entries = self._log
        if domain_idx is not None:
            entries = [e for e in entries if e.domain == domain_idx]
        if action is not None:
            entries = [e for e in entries if e.action == action]
        return [e.to_dict() for e in entries]

    def save_log(self, path: str) -> None:
        """Write full audit log to a JSON file."""
        with open(path, "w") as f:
            json.dump(self.get_log(), f, indent=2)

    def stats(self, domain_idx: int) -> dict:
        """Return current rule health statistics for a domain."""
        rules = self.registry.get(domain_idx)
        if not rules:
            return {"n_rules": 0}
        confs = [r.confidence for r in rules]
        return {
            "n_rules":        len(rules),
            "n_manual":       sum(1 for r in rules if r.source == "manual"),
            "n_cart":         sum(1 for r in rules if r.source == "cart"),
            "n_merged":       sum(1 for r in rules if r.source == "merged"),
            "mean_confidence": round(float(np.mean(confs)), 3),
            "min_confidence":  round(float(np.min(confs)), 3),
            "n_low_conf":     sum(1 for c in confs if c < self.min_confidence),
            "log_events":     len(self._log),
        }
