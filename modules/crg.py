from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np

from engine.tensor import Tensor, Parameter
from engine.module import Module
from engine.layers import Linear
from config import CRGConfig

if TYPE_CHECKING:
    from memory.causal import CausalMemory

# Edge type labels stored alongside active edges
EDGE_CAUSAL      = "causal"
EDGE_ASSOCIATIVE = "associative"


class CausalRoutingGraph(Module):
    """
    Sparse DAG routing: O(E) compute with Granger-based online structure discovery.
    Differentiable message passing + NOTEARS penalty for acyclicity.

    Spurious-causality guard: after each `update_structure` call, edges whose
    partial correlation (conditioning on the top-2 common drivers) drops below
    `ci_threshold` are relabelled as "associative" and zeroed from W so they
    cannot influence message passing.
    """

    def __init__(self, d_model: int, config: CRGConfig):
        super().__init__()
        self.d_model = d_model
        self.n_nodes = config.n_nodes
        self.sparsity_lambda = config.sparsity_lambda
        self.dag_lambda = config.dag_lambda
        self.n_lags = config.n_lags
        self.update_interval = config.update_interval
        self.edge_threshold = 0.01
        self.ci_threshold: float = config.ci_threshold
        self.n_regimes: int      = config.n_regimes
        self.max_ci_edges: int   = getattr(config, "max_ci_edges", 64)

        # Base adjacency matrix: W_ij = edge strength from node i to node j
        w_init = np.random.randn(config.n_nodes, config.n_nodes).astype(np.float64) * 0.01
        np.fill_diagonal(w_init, 0.0)
        self.W = Parameter(w_init, name="W")

        # Neural Granger mask: M_ij learned logits; sigmoid(M_ij) gates each edge.
        # Initialized to 0 → sigmoid(0)=0.5 (neutral); L1 penalty drives unused
        # edges toward −∞ (sigmoid→0) during training.
        m_init = np.zeros((config.n_nodes, config.n_nodes), dtype=np.float32)
        np.fill_diagonal(m_init, -10.0)   # hard-zero self-loops from the start
        self.M = Parameter(m_init, name="M")

        # Regime-conditioned bias: one additive W adjustment per regime.
        # W_eff = (W + sum_k(w_k * W_bias_k)) * sigmoid(M)
        # Allows different causal structures per operating regime.
        self.W_regime_bias = Parameter(
            np.zeros((self.n_regimes, config.n_nodes, config.n_nodes), dtype=np.float32),
            name="W_regime_bias"
        )

        self.node_embed = Linear(d_model, config.n_nodes)
        self.node_out = Linear(config.n_nodes, d_model)

        object.__setattr__(self, "step_counter", 0)
        object.__setattr__(self, "granger_accumulator",
                           np.zeros((config.n_nodes, config.n_nodes), dtype=np.float32))
        object.__setattr__(self, "edge_labels", {})

    # ------------------------------------------------------------------
    # Conditional independence / spurious causality
    # ------------------------------------------------------------------

    def _partial_correlation(
        self, x: np.ndarray, i: int, j: int, cond: list[int]
    ) -> float:
        """
        Partial correlation of columns i and j conditioned on `cond` columns.

        Uses the residual approach: regress i and j on `cond`, then correlate
        the residuals.  Returns a value in [-1, 1]; |value| near 0 means
        conditional independence.

        x : (T, N) float64
        """
        T = x.shape[0]
        if T < len(cond) + 4:
            return 1.0   # too little data — conservatively keep edge

        xi = x[:, i]
        xj = x[:, j]

        if not cond:
            # Marginal correlation
            xi_dm = xi - xi.mean()
            xj_dm = xj - xj.mean()
            denom = (np.std(xi_dm) * np.std(xj_dm) + 1e-8)
            return float(np.mean(xi_dm * xj_dm) / denom)

        Z = x[:, cond]                        # (T, k)
        ZtZ = Z.T @ Z + 1e-6 * np.eye(len(cond))
        ZtZi = np.linalg.inv(ZtZ)

        # Residualise i and j on Z
        beta_i = ZtZi @ (Z.T @ xi)
        beta_j = ZtZi @ (Z.T @ xj)
        ri = xi - Z @ beta_i
        rj = xj - Z @ beta_j

        ri_dm = ri - ri.mean()
        rj_dm = rj - rj.mean()
        denom = np.std(ri_dm) * np.std(rj_dm) + 1e-8
        return float(np.mean(ri_dm * rj_dm) / denom)

    def _top_confounders(self, x: np.ndarray, i: int, j: int, k: int = 2) -> list[int]:
        """
        Return the k node indices (excluding i, j) whose time-series are most
        correlated with both x[:, i] and x[:, j].  These are the conditioning
        set used for the CI test.
        """
        N = x.shape[1]
        exclude = {i, j}
        scores = []
        for c in range(N):
            if c in exclude:
                continue
            xi = x[:, i] - x[:, i].mean()
            xj = x[:, j] - x[:, j].mean()
            xc = x[:, c] - x[:, c].mean()
            std_c = np.std(xc) + 1e-8
            std_i = np.std(xi) + 1e-8
            std_j = np.std(xj) + 1e-8
            r_ic = float(np.mean(xi * xc)) / (std_i * std_c)
            r_jc = float(np.mean(xj * xc)) / (std_j * std_c)
            # Score = harmonic mean of |r_ic| and |r_jc|: high if correlated with both
            score = 2 * abs(r_ic) * abs(r_jc) / (abs(r_ic) + abs(r_jc) + 1e-8)
            scores.append((c, score))
        scores.sort(key=lambda t: t[1], reverse=True)
        return [c for c, _ in scores[:k]]

    def _prune_spurious_edges(self, x_history: np.ndarray):
        """
        For every active edge (i→j), run a conditional independence test
        conditioning on the top-2 confounders.  If the partial correlation
        |r_ij|z| < ci_threshold, schedule the edge for pruning.

        Pruning is DEFERRED — edges are collected into `_pending_prune` and
        applied only when `apply_pending_prune()` is called (from the training
        loop, after backward + optimizer step).  This prevents mid-graph W.data
        mutation corrupting a pending backward pass on the same step.

        x_history : (T, N) float64
        """
        W_data = self.W.data
        labels: dict[tuple[int, int], str] = {}
        pending: list[tuple[int, int]] = []

        rows, cols = np.where(np.abs(W_data) > self.edge_threshold)
        all_edges  = [(int(i), int(j)) for i, j in zip(rows, cols) if i != j]

        # Cap edges checked per call — _top_confounders is O(N) per edge →
        # O(N²) total. Default max_ci_edges=64 bounds cost to 64 CI tests.
        max_check = getattr(self, "max_ci_edges", 64)
        if len(all_edges) > max_check:
            rng_idx  = np.random.choice(len(all_edges), size=max_check, replace=False)
            all_edges = [all_edges[k] for k in rng_idx]

        for i, j in all_edges:
            confounders = self._top_confounders(x_history, i, j, k=2)
            pcorr = self._partial_correlation(x_history, i, j, confounders)
            if abs(pcorr) < self.ci_threshold:
                labels[(i, j)] = EDGE_ASSOCIATIVE
                pending.append((i, j))
            else:
                labels[(i, j)] = EDGE_CAUSAL

        object.__setattr__(self, "edge_labels", labels)
        # Accumulate (don't replace) so multiple update cycles don't lose edges
        existing = list(getattr(self, "_pending_prune", []))
        object.__setattr__(self, "_pending_prune", existing + pending)

    def init_from_graph(
        self,
        W_init: np.ndarray,
        scale: float = 0.1,
        set_mask: bool = True,
    ) -> None:
        """
        Warm-start the CRG adjacency matrix from a known topology.

        Instead of learning the causal graph from scratch (requires thousands
        of training steps), provide prior knowledge from:
          - OT     : P&ID diagram adjacency matrix
          - IT     : service mesh / call graph (Istio, Consul)
          - Telecom: RAN cell adjacency (geographic or backhaul topology)
          - Any domain with a known graph structure

        CRG then REFINES the initialised graph from data (via Granger EMA
        and DAGMA penalty).  This cuts cold-start time from O(10k steps) to
        near-zero.

        Parameters
        ----------
        W_init   : (n_nodes, n_nodes) float array — known adjacency weights.
                   Use binary {0, 1} for unweighted topologies or positive
                   floats for confidence-weighted edges.
        scale    : float — multiplier applied to W_init before storing.
                   Default 0.1 keeps initial weights small so gradient
                   updates dominate quickly.
        set_mask : bool — if True, also initialise the Neural Granger mask M
                   so edges present in W_init start with a positive logit
                   (sigmoid → ~0.73) and absent edges start negative (-3).

        Example
        -------
            # From a service mesh call graph
            adj = np.array([[0,1,0],[1,0,1],[0,0,0]], dtype=np.float32)
            model.crg.init_from_graph(adj)

            # From a pandas DataFrame of known P&ID connections
            adj = nx.to_numpy_array(pid_graph, nodelist=sensor_names)
            model.crg.init_from_graph(adj, scale=0.2)
        """
        assert W_init.shape == (self.n_nodes, self.n_nodes), (
            f"W_init shape {W_init.shape} must match "
            f"(n_nodes={self.n_nodes}, n_nodes={self.n_nodes})"
        )
        W = (np.asarray(W_init, dtype=np.float32) * scale)
        np.fill_diagonal(W, 0.0)
        self.W.data = W

        if set_mask:
            # Edges present in W_init → logit +3 (sigmoid ≈ 0.95)
            # Absent edges          → logit -3 (sigmoid ≈ 0.05)
            M = np.where(np.abs(W_init) > 0, 3.0, -3.0).astype(np.float32)
            np.fill_diagonal(M, -10.0)
            self.M.data = M

    def apply_pending_prune(self) -> int:
        """
        Apply deferred edge pruning to W.data and M.data.

        Call this AFTER backward() + optimizer.step() — never inside a forward
        pass — so the in-place mutation doesn't corrupt a pending backward graph.

        Returns the number of edges pruned.
        """
        pending: list[tuple[int, int]] = list(getattr(self, "_pending_prune", []))
        if not pending:
            return 0
        for i, j in pending:
            self.W.data[i, j] = 0.0
            self.M.data[i, j] = -10.0   # drive sigmoid(M) → 0 for pruned edges
        np.fill_diagonal(self.W.data, 0.0)
        object.__setattr__(self, "_pending_prune", [])
        return len(pending)

    # ------------------------------------------------------------------

    def _dag_penalty(self) -> Tensor:
        """
        DAGMA acyclicity penalty (Yu et al. 2023):
            h(W) = -log det(s·I - W⊙W) - n·log(s)

        Strictly acyclic iff h(W) = 0. Unlike NOTEARS' tr(exp(W²))-n this is:
          - Strictly convex in a neighbourhood of the DAG solution
          - O(n³) via Cholesky instead of matrix exponential (same cost, better numerics)
          - Better-conditioned: no exponential blow-up of eigenvalues

        Gradient: d(h)/d(W_ij) = 2·W_ij·[(s·I - W⊙W)^{-1}]_ji
        """
        n  = self.n_nodes
        s  = float(n) / 2.0 + 1.0      # s > spectral_radius(W⊙W); n/2+1 is a safe default
        W  = self.W.data                # (n, n)
        M  = s * np.eye(n) - W * W      # s·I - W⊙W  (must be positive definite for DAG)

        # Clamp to ensure positive definiteness even if W is large early in training
        M  = M + np.eye(n) * 1e-6
        try:
            sign, log_abs_det = np.linalg.slogdet(M)
        except np.linalg.LinAlgError:
            sign, log_abs_det = 1.0, 0.0

        # h = -log|det(M)| - n·log(s)
        h_val = float(-sign * log_abs_det - n * np.log(s + 1e-8))

        out = Tensor(
            np.array([[h_val]], dtype=np.float32),
            requires_grad=self.W.requires_grad,
            _children=(self.W,),
            _op="dagma_penalty"
        )
        _W  = self.W
        _M  = M.copy()
        _s  = s

        def _back():
            if _W.requires_grad and out.grad is not None:
                g_scalar = float(out.grad.sum())
                try:
                    M_inv = np.linalg.inv(_M)
                except np.linalg.LinAlgError:
                    return
                # d(h)/d(W_ij) = 2·W_ij·(M^{-1})_ji
                contrib = (2.0 * _W.data * M_inv.T * g_scalar).astype(np.float32)
                _W.grad = _W.grad + contrib if _W.grad is not None else contrib

        out._backward = _back
        return out

    def _message_pass(self, node_states: Tensor, W_sparse: np.ndarray) -> Tensor:
        """
        node_states : (batch, T, n_nodes)
        W_sparse    : (n_nodes, n_nodes) numpy — masked adjacency
        Computes updated_j = node_states_j + Σ_i W_ij * node_states_i
        = node_states + node_states @ W_sparse   (batched matmul)
        Returns (batch, T, n_nodes).
        """
        # node_states @ W_sparse: (B, T, n_nodes) @ (n_nodes, n_nodes) -> (B, T, n_nodes)
        W_t = Tensor(W_sparse, requires_grad=self.W.requires_grad,
                     _children=(self.W,), _op="W_sparse")

        # Backward for W_t: gradient flows back to W only through unmasked entries
        _W = self.W
        _mask = (np.abs(W_sparse) > 0).astype(np.float64)
        _W_t = W_t

        def _w_sparse_back():
            if _W.requires_grad and _W_t.grad is not None:
                contrib = _W_t.grad * _mask
                _W.grad = _W.grad + contrib if _W.grad is not None else contrib

        W_t._backward = _w_sparse_back

        messages = node_states @ W_t       # (B, T, n_nodes) @ (n_nodes, n_nodes)
        return node_states + messages

    def _effective_W(self, regime_weights: np.ndarray | None = None) -> np.ndarray:
        """
        Compute the effective adjacency matrix:
          W_eff = (W_base + regime_adjustment) * sigmoid(M)

        regime_weights : (K,) mean routing weights per regime, or None.
        Returns (n_nodes, n_nodes) float64.
        """
        W_base = self.W.data.copy()

        # Regime-conditioned adjustment: blend K bias matrices by routing weights
        if regime_weights is not None and len(regime_weights) == self.n_regimes:
            rw = np.asarray(regime_weights, dtype=np.float32)
            rw = rw / (rw.sum() + 1e-8)     # normalise to sum=1
            regime_adj = np.einsum("k,kij->ij", rw, self.W_regime_bias.data)
            W_base = W_base + regime_adj

        # Neural Granger gate: sigmoid(M) masks each edge independently
        gate = 1.0 / (1.0 + np.exp(-np.clip(self.M.data, -20, 20)))
        W_eff = W_base * gate
        np.fill_diagonal(W_eff, 0.0)
        return W_eff

    def forward(self, x: Tensor,
                regime_weights: np.ndarray | None = None) -> tuple[Tensor, Tensor]:
        """
        x              : (batch, T, d_model)
        regime_weights : optional (K,) mean per-regime routing weights from RMC;
                         when provided, blends regime-specific causal graph biases.
        Returns (output, dag_penalty):
            output     : (batch, T, d_model)
            dag_penalty: scalar Tensor (DAG + L1 + Granger mask sparsity)
        """
        # Project to node space
        node_states = self.node_embed(x)    # (B, T, n_nodes)

        # Effective adjacency: base W gated by learned Granger mask, regime-adjusted
        W_eff = self._effective_W(regime_weights)
        W_sparse = np.where(np.abs(W_eff) > self.edge_threshold, W_eff, 0.0)

        # Message passing (differentiable)
        updated = self._message_pass(node_states, W_sparse)   # (B, T, n_nodes)

        # Project back to d_model
        output = self.node_out(updated)     # (B, T, d_model)

        # DAG penalty on W_eff
        dag_pen = self._dag_penalty()       # scalar Tensor

        # L1 on W
        l1_W = self.W.abs().sum() * self.sparsity_lambda
        # L1 on sigmoid(M): encourages sparse Granger mask (drives unused edges to 0)
        gate_t = Tensor(
            1.0 / (1.0 + np.exp(-np.clip(self.M.data, -20, 20))),
            requires_grad=self.M.requires_grad,
            _children=(self.M,), _op="granger_gate"
        )
        _M = self.M
        _gate_np = gate_t.data.copy()

        def _gate_back():
            if _M.requires_grad and gate_t.grad is not None:
                dsig = _gate_np * (1.0 - _gate_np)
                contrib = gate_t.grad * dsig
                _M.grad = _M.grad + contrib if _M.grad is not None else contrib

        gate_t._backward = _gate_back
        l1_M = gate_t.abs().sum() * self.sparsity_lambda

        total_penalty = dag_pen * self.dag_lambda + l1_W + l1_M

        # CRG collapse guard: reset W if NaN/Inf contamination detected
        if not np.all(np.isfinite(self.W.data)):
            w_reset = np.random.randn(self.n_nodes, self.n_nodes) * 0.01
            np.fill_diagonal(w_reset, 0.0)
            self.W.data[:] = w_reset
            if self.W.grad is not None:
                self.W.grad[:] = 0.0
            object.__setattr__(self, "granger_accumulator",
                               np.zeros((self.n_nodes, self.n_nodes), dtype=np.float32))

        # Online structure discovery (every update_interval steps, training only)
        if self.training:
            object.__setattr__(self, "step_counter", self.step_counter + 1)
            if self.step_counter % self.update_interval == 0:
                # Use node_states numpy as history proxy
                ns_np = node_states.data             # (B, T, n_nodes)
                # Average over batch
                ns_avg = ns_np.mean(axis=0)          # (T, n_nodes)
                self.update_structure(ns_avg)

        return output, total_penalty

    def update_structure(self, x_history: np.ndarray):
        """
        x_history: (T, n_nodes)
        Granger-like: G_ij = mean over lags of |corr(x_i[:-lag], x_j[lag:])|
        Updates granger_accumulator with EMA (α=0.9), injects into W.data.
        """
        T, N = x_history.shape
        if T < self.n_lags + 2:
            return

        G = np.zeros((N, N), dtype=np.float32)
        for lag in range(1, self.n_lags + 1):
            x_past = x_history[:-lag, :]    # (T-lag, N)
            x_fut = x_history[lag:, :]      # (T-lag, N)

            # Compute correlation matrix: G[i,j] = corr(x_past[:,i], x_fut[:,j])
            past_demean = x_past - x_past.mean(axis=0, keepdims=True)
            fut_demean = x_fut - x_fut.mean(axis=0, keepdims=True)
            past_std = np.std(x_past, axis=0) + 1e-8   # (N,)
            fut_std = np.std(x_fut, axis=0) + 1e-8      # (N,)

            # corr_ij = mean_t(past_demean_t_i * fut_demean_t_j) / (std_i * std_j)
            cross = (past_demean.T @ fut_demean) / past_demean.shape[0]  # (N, N)
            corr = cross / (past_std[:, None] * fut_std[None, :])
            G += np.abs(corr)

        G /= self.n_lags
        np.fill_diagonal(G, 0.0)

        # EMA update of accumulator
        alpha = 0.9
        acc = self.granger_accumulator
        new_acc = alpha * acc + (1.0 - alpha) * G
        object.__setattr__(self, "granger_accumulator", new_acc)

        # Inject into W with small learning rate
        lr_struct = 1e-3
        self.W.data += lr_struct * (new_acc - np.abs(self.W.data))
        np.fill_diagonal(self.W.data, 0.0)

        # Update Neural Granger mask M: edges with strong Granger signal get
        # positive logits (gate → 1); edges with weak signal get negative logits
        # (gate → 0). Uses a slow learning rate so gradient training dominates.
        lr_mask = 5e-4
        granger_signal = new_acc / (new_acc.max() + 1e-8)  # normalise to [0,1]
        # Push M toward +3 for strong edges, -3 for weak edges
        target_M = (granger_signal - 0.5) * 6.0
        np.fill_diagonal(target_M, -10.0)
        self.M.data += lr_mask * (target_M - self.M.data)

        # Conditional independence pruning: remove spurious correlational edges
        self._prune_spurious_edges(x_history)

        # Write strong causal edges to attached CausalMemory (if any)
        if getattr(self, "_causal_memory", None) is not None:
            rows, cols = np.where(new_acc > 0.1)
            for i, j in zip(rows.tolist(), cols.tolist()):
                if i != j:
                    self._causal_memory.record(
                        cause=int(i), effect=int(j),
                        confidence=float(new_acc[i, j])
                    )

    def attach_causal_memory(self, causal_memory: "CausalMemory") -> None:
        """Attach a CausalMemory instance; CRG writes discovered edges into it."""
        object.__setattr__(self, "_causal_memory", causal_memory)

    def explain(self, query_node_idx: int) -> list[tuple[int, float]]:
        """
        Trace influential predecessors of query_node via BFS over strongest edges.
        Returns list of (node_idx, cumulative_weight) sorted by descending influence.
        """
        W_data = self.W.data.copy()
        np.fill_diagonal(W_data, 0.0)

        # BFS from query_node following strongest incoming edges
        visited = {query_node_idx: 1.0}
        frontier = [(query_node_idx, 1.0)]
        results = []

        max_depth = 6
        for _ in range(max_depth):
            next_frontier = []
            for node, cum_w in frontier:
                # Incoming edges to node: column node in W (W_ij = i->j)
                incoming = W_data[:, node]   # (n_nodes,)
                # Sort by absolute strength descending; take top-5
                order = np.argsort(np.abs(incoming))[::-1][:5]
                for src in order:
                    w = float(incoming[src])
                    if abs(w) < self.edge_threshold:
                        break
                    if src in visited:
                        continue
                    new_cum = cum_w * abs(w)
                    visited[src] = new_cum
                    results.append((src, new_cum))
                    next_frontier.append((src, new_cum))
            if not next_frontier:
                break
            frontier = next_frontier

        results.sort(key=lambda t: t[1], reverse=True)
        return results

    def propagate_failure(
        self,
        triggered_nodes: list[int],
        edge_threshold: float | None = None,
        max_hops: int = 4,
        decay: float = 0.85,
    ) -> list[tuple[int, float]]:
        """
        Forward-propagate failure signals through the causal graph.

        Given a set of nodes where an anomaly was detected, follows causal edges
        forward to find downstream nodes likely to be affected.  Failure
        probability decays with each hop and with edge weight.

        Parameters
        ----------
        triggered_nodes : node indices where the anomaly fired
        edge_threshold  : minimum |W_eff| to follow; default self.edge_threshold
        max_hops        : maximum propagation depth
        decay           : per-hop probability decay factor (0 < decay ≤ 1)

        Returns
        -------
        list of (node_idx, failure_probability) sorted descending, excluding
        the trigger nodes themselves.
        """
        thr = edge_threshold if edge_threshold is not None else self.edge_threshold
        W_eff = self._effective_W()
        W_sparse = np.where(np.abs(W_eff) > thr, W_eff, 0.0)

        # Seed failure probabilities at trigger nodes
        failure_prob: dict[int, float] = {n: 1.0 for n in triggered_nodes}
        frontier = list(set(triggered_nodes))

        for hop in range(max_hops):
            next_frontier: list[int] = []
            for src in frontier:
                for dst in range(self.n_nodes):
                    if dst == src:
                        continue
                    w = float(W_sparse[src, dst])
                    if abs(w) < thr:
                        continue
                    # Probability = parent_prob × |edge_weight| × decay^hop
                    new_prob = failure_prob[src] * abs(w) * (decay ** hop)
                    if dst not in failure_prob or failure_prob[dst] < new_prob:
                        failure_prob[dst] = new_prob
                        next_frontier.append(dst)
            frontier = list(set(next_frontier))
            if not frontier:
                break

        triggered_set = set(triggered_nodes)
        results = [(node, prob) for node, prob in failure_prob.items()
                   if node not in triggered_set]
        results.sort(key=lambda t: t[1], reverse=True)
        return results

    # ------------------------------------------------------------------
    # do-calculus intervention API
    # ------------------------------------------------------------------

    def intervene(
        self,
        node_interventions: dict[int, float],
        x: Tensor,
    ) -> Tensor:
        """
        Pearl do-calculus intervention: do(X_i = v_i for i in node_interventions).

        Per do-calculus, setting X_i = v_i means:
          1. All incoming edges to node i are severed (the value is externally
             forced, not caused by its parents).
          2. Outgoing edges from i are kept — the intervention propagates forward.

        This is implemented by:
          - Projecting x to node space.
          - Overwriting the intervened node values with the given constants.
          - Running message passing with a modified adjacency that has the
            columns corresponding to intervened nodes zeroed (no incoming flow).
          - Projecting back to d_model.

        Args:
            node_interventions : {node_idx: value} — nodes to force and their values
            x                  : (B, T, d_model) input Tensor

        Returns:
            (B, T, d_model) output under the intervention — the counterfactual
            predicted representation if the intervened nodes had been as specified.
        """
        # Project to node space (no grad needed for intervention analysis)
        node_states_np = self.node_embed(x).data.copy()   # (B, T, n_nodes)

        # Sever incoming edges for intervened nodes
        W_do = self.W.data.copy()
        for node_idx in node_interventions:
            W_do[:, node_idx] = 0.0          # zero the column → no incoming
        np.fill_diagonal(W_do, 0.0)
        W_do_sparse = np.where(np.abs(W_do) > self.edge_threshold, W_do, 0.0)

        # Overwrite intervened node values across all (batch, time) positions
        for node_idx, value in node_interventions.items():
            node_states_np[:, :, node_idx] = float(value)

        # Message passing under intervention
        node_states_do = Tensor(node_states_np, requires_grad=False)
        updated = node_states_do + Tensor(node_states_np @ W_do_sparse,
                                          requires_grad=False)

        # Project back to d_model
        return self.node_out(updated)   # (B, T, d_model)

    def counterfactual_root_cause(
        self,
        x: Tensor,
        target_node: int,
        candidate_nodes: list[int] | None = None,
        intervention_values: dict[int, float] | None = None,
        top_k: int = 5,
    ) -> dict:
        """
        Root-cause analysis via do-calculus: identify which upstream nodes,
        when intervened upon, most change the activation of `target_node`.

        For each candidate upstream node i, we:
          1. Set node i to 0 (ablation) and measure the change in target_node.
          2. Set node i to its observed mean + 2σ (activation) and measure the change.
        The node with the largest |Δtarget| is the strongest root cause.

        Args:
            x                   : (B, T, d_model) input Tensor
            target_node         : index of the node whose activation we study
            candidate_nodes     : nodes to test (default: all predecessors of target)
            intervention_values : custom intervention values per node
                                  (default: ablation to 0)
            top_k               : number of top causes to return

        Returns:
            dict with keys:
              "ranked_causes" : list of {node, delta_activation, direction} sorted
                                by |delta_activation| descending
              "target_node"   : target_node index
              "n_candidates"  : number of nodes tested
        """
        # Observed activation of target node under no intervention
        node_states_obs = self.node_embed(x).data   # (B, T, n_nodes)
        W_sparse = np.where(np.abs(self.W.data) > self.edge_threshold,
                            self.W.data, 0.0)
        updated_obs = node_states_obs + node_states_obs @ W_sparse
        target_obs = float(updated_obs[:, :, target_node].mean())

        # Determine candidate nodes (predecessors in graph)
        if candidate_nodes is None:
            # All nodes with a path to target_node (incoming edges)
            ancestors = set()
            queue = [target_node]
            visited = {target_node}
            while queue:
                node = queue.pop()
                for src in range(self.n_nodes):
                    if src != node and abs(W_sparse[src, node]) > self.edge_threshold:
                        if src not in visited:
                            ancestors.add(src)
                            visited.add(src)
                            queue.append(src)
            candidate_nodes = list(ancestors) if ancestors else list(range(self.n_nodes))
            candidate_nodes = [n for n in candidate_nodes if n != target_node]

        # For each candidate, ablate (set to 0) and measure delta
        results = []
        for node_idx in candidate_nodes:
            iv = intervention_values.get(node_idx, 0.0) if intervention_values else 0.0
            # Re-project to node space to read target_node under intervention
            node_states_do = self.node_embed(x).data.copy()
            node_states_do[:, :, node_idx] = iv
            W_do = W_sparse.copy()
            W_do[:, node_idx] = 0.0
            updated_do = node_states_do + node_states_do @ W_do
            target_do = float(updated_do[:, :, target_node].mean())
            delta = target_do - target_obs
            results.append({
                "node":             node_idx,
                "delta_activation": round(delta, 6),
                "direction":        "up" if delta > 0 else "down",
            })

        results.sort(key=lambda r: abs(r["delta_activation"]), reverse=True)

        return {
            "ranked_causes": results[:top_k],
            "target_node":   target_node,
            "target_obs_activation": round(target_obs, 6),
            "n_candidates":  len(candidate_nodes),
        }

    def get_active_edges(self) -> list[tuple[int, int, float, str]]:
        """Returns list of (i, j, weight, label) for |W_ij| > edge_threshold.

        `label` is 'causal' or 'associative' (set after update_structure).
        Newly added edges before the first CI test carry label 'causal'.
        """
        W_data = self.W.data
        labels: dict[tuple[int, int], str] = self.edge_labels
        edges = []
        rows, cols = np.where(np.abs(W_data) > self.edge_threshold)
        for i, j in zip(rows.tolist(), cols.tolist()):
            if i != j:
                lbl = labels.get((i, j), EDGE_CAUSAL)
                edges.append((int(i), int(j), float(W_data[i, j]), lbl))
        edges.sort(key=lambda t: abs(t[2]), reverse=True)
        return edges
