"""
O(1) amortised log-line → numeric feature encoder.

Uses a Drain3-style fixed-depth prefix tree to cluster log templates,
mapping each unique template to an integer code.  Two output channels
per event:
  channel 0 — template_id  (integer, cast to float)
  channel 1 — severity      (0–4: DEBUG/INFO/WARN/ERROR/CRITICAL)

Tree lookup is O(depth) ≈ O(1) in practice (depth=4, branching=100).
"""

from __future__ import annotations

import re
import numpy as np
from typing import Dict, List, Optional, Tuple

_SEVERITY_MAP: Dict[str, int] = {
    "debug": 0, "trace": 0,
    "info": 1, "notice": 1,
    "warn": 2, "warning": 2,
    "error": 3, "err": 3,
    "critical": 4, "fatal": 4, "crit": 4,
}

_NUM_RE = re.compile(r"\b\d+(?:[.,]\d+)*\b")
_HEX_RE = re.compile(r"\b0x[0-9a-fA-F]+\b")
_IP_RE  = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
_UUID_RE = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I)


def _tokenize(line: str) -> List[str]:
    line = _UUID_RE.sub("<UUID>", line)
    line = _IP_RE.sub("<IP>", line)
    line = _HEX_RE.sub("<HEX>", line)
    line = _NUM_RE.sub("<NUM>", line)
    return line.split()


class _TrieNode:
    __slots__ = ("children", "template_id", "count")

    def __init__(self):
        self.children: Dict[str, "_TrieNode"] = {}
        self.template_id: Optional[int] = None
        self.count: int = 0


class LogEncoder:
    """
    Streaming log encoder.  Call `encode(line)` per log line; returns a
    length-2 float64 array [template_id, severity].

    Parameters
    ----------
    depth : int
        Number of token levels in the prefix trie.  Default 4.
    sim_threshold : float
        Fraction of tokens that must match to merge into existing template.
    max_templates : int
        Hard cap; once reached, new patterns map to template 0 ("unknown").
    """

    def __init__(
        self,
        depth: int = 4,
        sim_threshold: float = 0.5,
        max_templates: int = 4096,
    ):
        self.depth = depth
        self.sim_threshold = sim_threshold
        self.max_templates = max_templates

        self._root = _TrieNode()
        self._templates: List[List[str]] = []   # list of token sequences
        self._next_id: int = 1                   # 0 reserved for "unknown"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(self, line: str) -> np.ndarray:
        """
        Encode one log line to [template_id, severity] float64 array.
        Thread-safe reads; writes are protected by the GIL (CPython).
        """
        severity = self._parse_severity(line)
        tokens = _tokenize(line)
        tid = self._get_or_create_template(tokens)
        return np.array([float(tid), float(severity)], dtype=np.float64)

    def encode_batch(self, lines: List[str]) -> np.ndarray:
        """Returns (N, 2) float64 array."""
        return np.array([self.encode(l) for l in lines], dtype=np.float64)

    @property
    def n_templates(self) -> int:
        return self._next_id

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_severity(line: str) -> int:
        lower = line.lower()
        for token, level in _SEVERITY_MAP.items():
            if token in lower:
                return level
        return 1   # default INFO

    def _get_or_create_template(self, tokens: List[str]) -> int:
        if not tokens:
            return 0

        # Walk prefix trie to depth
        node = self._root
        prefix_tokens = tokens[: self.depth]
        for tok in prefix_tokens:
            key = tok if not _NUM_RE.fullmatch(tok) and tok not in ("<NUM>", "<IP>", "<HEX>", "<UUID>") else "<*>"
            if key not in node.children:
                node.children[key] = _TrieNode()
            node = node.children[key]

        node.count += 1

        # If this leaf already has a template, check similarity
        if node.template_id is not None:
            existing = self._templates[node.template_id - 1]  # id is 1-based
            sim = self._token_sim(tokens, existing)
            if sim >= self.sim_threshold:
                # Optionally update template to wildcard diverging positions
                self._templates[node.template_id - 1] = self._merge(tokens, existing)
                return node.template_id

        # New template
        if self._next_id >= self.max_templates:
            return 0

        tid = self._next_id
        self._next_id += 1
        self._templates.append(tokens[:])
        node.template_id = tid
        return tid

    @staticmethod
    def _token_sim(a: List[str], b: List[str]) -> float:
        if not a or not b:
            return 0.0
        length = max(len(a), len(b))
        matches = sum(
            1 for x, y in zip(a, b) if x == y or x == "<*>" or y == "<*>"
        )
        return matches / length

    @staticmethod
    def _merge(new_tokens: List[str], template: List[str]) -> List[str]:
        merged = []
        for i, (n, t) in enumerate(zip(new_tokens, template)):
            merged.append(t if n == t or t == "<*>" else "<*>")
        # Append remaining tokens from longer sequence
        if len(new_tokens) > len(template):
            merged.extend(["<*>"] * (len(new_tokens) - len(template)))
        return merged
