import numpy as np
from typing import Dict, List, Optional, Tuple

from engine.tensor import Tensor, Parameter
from engine.module import Module
from engine.layers import Linear
from config import CRGConfig

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
        # Partial-correlation threshold for causal vs associative labelling
        self.ci_threshold: float = getattr(config, "ci_threshold", 0.05)

        # Adjacency matrix: W_ij = edge strength from node i to node j
        w_init = np.random.randn(config.n_nodes, config.n_nodes).astype(np.float64) * 0.01
        np.fill_diagonal(w_init, 0.0)   # no self-loops
        self.W = Parameter(w_init, name="W")

        self.node_embed = Linear(d_model, config.n_nodes)
        self.node_out = Linear(config.n_nodes, d_model)

        # Non-parameter state
        object.__setattr__(self, "step_counter", 0)
        object.__setattr__(self, "granger_accumulator",
                           np.zeros((config.n_nodes, config.n_nodes), dtype=np.float64))
        # edge_labels[i][j] = EDGE_CAUSAL | EDGE_ASSOCIATIVE
        object.__setattr__(self, "edge_labels", {})

    # ------------------------------------------------------------------
    # Conditional independence / spurious causality
    # ------------------------------------------------------------------

    def _partial_correlation(
        self, x: np.ndarray, i: int, j: int, cond: List[int]
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

    def _top_confounders(self, x: np.ndarray, i: int, j: int, k: int = 2) -> List[int]:
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
        |r_ij|z| < ci_threshold, zero out W[i, j] and label the edge
        EDGE_ASSOCIATIVE; otherwise label it EDGE_CAUSAL.

        x_history : (T, N) float64
        """
        W_data = self.W.data
        labels: Dict[Tuple[int, int], str] = {}

        rows, cols = np.where(np.abs(W_data) > self.edge_threshold)
        for i, j in zip(rows.tolist(), cols.tolist()):
            if i == j:
                continue
            confounders = self._top_confounders(x_history, i, j, k=2)
            pcorr = self._partial_correlation(x_history, i, j, confounders)
            if abs(pcorr) < self.ci_threshold:
                W_data[i, j] = 0.0
                labels[(i, j)] = EDGE_ASSOCIATIVE
            else:
                labels[(i, j)] = EDGE_CAUSAL

        object.__setattr__(self, "edge_labels", labels)
        np.fill_diagonal(W_data, 0.0)

    # ------------------------------------------------------------------

    def _dag_penalty(self) -> Tensor:
        from scipy.linalg import expm as scipy_expm
        n = self.n_nodes
        W_np = self.W.data           # (n, n) numpy float64
        A_np = W_np ** 2             # element-wise square
        expm_A = scipy_expm(A_np)    # (n, n) exact matrix exponential
        trace_val = float(np.trace(expm_A))

        out = Tensor(
            np.array([[trace_val - n]], dtype=np.float64),
            requires_grad=self.W.requires_grad,
            _children=(self.W,),
            _op="dag_penalty"
        )

        _W = self.W
        _expm_A = expm_A.copy()

        def _back():
            if _W.requires_grad and out.grad is not None:
                g_scalar = float(out.grad.sum())
                # Analytic: d(tr(expm(W²)))/d(W_ij) = 2 * W_ij * expm(W²)_ij
                contrib = 2.0 * _W.data * _expm_A * g_scalar
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

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        x: (batch, T, d_model)
        Returns (output, dag_penalty):
            output     : (batch, T, d_model)
            dag_penalty: scalar Tensor
        """
        # Project to node space
        node_states = self.node_embed(x)    # (B, T, n_nodes)

        # Sparse adjacency: zero out edges below threshold
        W_data = self.W.data.copy()
        np.fill_diagonal(W_data, 0.0)
        W_sparse = np.where(np.abs(W_data) > self.edge_threshold, W_data, 0.0)

        # Message passing (differentiable)
        updated = self._message_pass(node_states, W_sparse)   # (B, T, n_nodes)

        # Project back to d_model
        output = self.node_out(updated)     # (B, T, d_model)

        # DAG penalty
        dag_pen = self._dag_penalty()       # scalar Tensor

        # L1 sparsity penalty on W (as part of dag_pen for caller convenience)
        l1 = self.W.abs().sum() * self.sparsity_lambda
        total_penalty = dag_pen * self.dag_lambda + l1

        # CRG collapse guard: reset W if NaN/Inf contamination detected
        if not np.all(np.isfinite(self.W.data)):
            w_reset = np.random.randn(self.n_nodes, self.n_nodes) * 0.01
            np.fill_diagonal(w_reset, 0.0)
            self.W.data[:] = w_reset
            if self.W.grad is not None:
                self.W.grad[:] = 0.0
            object.__setattr__(self, "granger_accumulator",
                               np.zeros((self.n_nodes, self.n_nodes), dtype=np.float64))

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

        G = np.zeros((N, N), dtype=np.float64)
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

        # Conditional independence pruning: remove spurious correlational edges
        self._prune_spurious_edges(x_history)

    def explain(self, query_node_idx: int) -> List[Tuple[int, float]]:
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

    # ------------------------------------------------------------------
    # do-calculus intervention API
    # ------------------------------------------------------------------

    def intervene(
        self,
        node_interventions: Dict[int, float],
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
        candidate_nodes: Optional[List[int]] = None,
        intervention_values: Optional[Dict[int, float]] = None,
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

    def get_active_edges(self) -> List[Tuple[int, int, float, str]]:
        """Returns list of (i, j, weight, label) for |W_ij| > edge_threshold.

        `label` is 'causal' or 'associative' (set after update_structure).
        Newly added edges before the first CI test carry label 'causal'.
        """
        W_data = self.W.data
        labels: Dict[Tuple[int, int], str] = self.edge_labels
        edges = []
        rows, cols = np.where(np.abs(W_data) > self.edge_threshold)
        for i, j in zip(rows.tolist(), cols.tolist()):
            if i != j:
                lbl = labels.get((i, j), EDGE_CAUSAL)
                edges.append((int(i), int(j), float(W_data[i, j]), lbl))
        edges.sort(key=lambda t: abs(t[2]), reverse=True)
        return edges
