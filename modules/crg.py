import numpy as np
from typing import List, Tuple

from engine.tensor import Tensor, Parameter
from engine.module import Module
from engine.layers import Linear
from config import CRGConfig


class CausalRoutingGraph(Module):
    """
    Sparse DAG routing: O(E) compute with Granger-based online structure discovery.
    Differentiable message passing + NOTEARS penalty for acyclicity.
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

    def get_active_edges(self) -> List[Tuple[int, int, float]]:
        """Returns list of (i, j, weight) for |W_ij| > edge_threshold."""
        W_data = self.W.data
        edges = []
        rows, cols = np.where(np.abs(W_data) > self.edge_threshold)
        for i, j in zip(rows.tolist(), cols.tolist()):
            if i != j:
                edges.append((int(i), int(j), float(W_data[i, j])))
        edges.sort(key=lambda t: abs(t[2]), reverse=True)
        return edges
