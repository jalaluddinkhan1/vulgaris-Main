import numpy as np
import hashlib
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

from model.vulgaris import Vulgaris


class DPNoiseAdder:
    """Differential privacy noise for gradient updates."""

    def __init__(self, noise_multiplier: float = 1.0, max_grad_norm: float = 1.0):
        self.sigma = noise_multiplier
        self.C = max_grad_norm
        self.privacy_budget_used = 0.0
        self._step_count = 0

    def clip_and_noise(self, gradients: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        noisy = {}
        for name, g in gradients.items():
            g = np.asarray(g, dtype=np.float64)
            # Per-parameter gradient clipping to norm C
            gnorm = np.linalg.norm(g)
            if gnorm > self.C:
                g = g * (self.C / (gnorm + 1e-8))
            # Add Gaussian noise N(0, (sigma * C)^2)
            noise = np.random.normal(0.0, self.sigma * self.C, size=g.shape)
            noisy[name] = g + noise
        self._step_count += 1
        return noisy

    def privacy_cost(self, n_steps: int, n_samples: int, batch_size: int,
                     delta: float = 1e-5) -> float:
        # Rényi DP accounting via moments method (simplified Abadi et al. 2016)
        # Sampling ratio
        q = batch_size / max(n_samples, 1)
        sigma = self.sigma
        # For each order alpha, compute Rényi divergence bound
        # Use alpha in range [2, 64] and find tightest epsilon
        best_eps = float('inf')
        for alpha in range(2, 65):
            # Rényi divergence for Gaussian mechanism with subsampling
            # D_alpha(M(S) || M(S')) ≤ (1/alpha-1) * log(1 + q^2 * alpha*(alpha-1)/(2*sigma^2) + ...)
            # Simplified bound:
            rda = (q ** 2) * alpha / (2.0 * sigma ** 2)
            # Total over n_steps
            total_rda = n_steps * rda
            # Convert to (eps, delta)-DP: eps = total_rda + log(1/delta) / (alpha - 1)
            eps = total_rda + np.log(1.0 / (delta + 1e-300)) / max(alpha - 1, 1)
            if eps < best_eps:
                best_eps = eps
        self.privacy_budget_used = best_eps
        return best_eps


class GradientCompressor:
    """Top-k sparse gradient compression with error feedback."""

    def __init__(self, compression_ratio: float = 0.01):
        self.k_ratio = compression_ratio
        self.error_buffers: Dict[str, np.ndarray] = {}

    def compress(self, gradients: Dict[str, np.ndarray], client_id: str) -> dict:
        sparse = {}
        key_prefix = client_id + ':'
        for name, g in gradients.items():
            g = np.asarray(g, dtype=np.float64)
            buf_key = key_prefix + name
            # Add accumulated error to gradient
            if buf_key not in self.error_buffers:
                self.error_buffers[buf_key] = np.zeros_like(g)
            g_with_error = g + self.error_buffers[buf_key]
            # Select top-k by magnitude
            flat = g_with_error.flatten()
            k = max(1, int(len(flat) * self.k_ratio))
            topk_idx = np.argpartition(np.abs(flat), -k)[-k:]
            topk_vals = flat[topk_idx]
            # Compute residual (error feedback)
            residual = g_with_error.copy()
            residual.flat[topk_idx] = 0.0
            self.error_buffers[buf_key] = residual
            sparse[name] = (topk_idx, topk_vals, g.shape)
        return sparse

    def decompress(self, sparse_grads: dict, shapes: Dict[str, tuple]) -> Dict[str, np.ndarray]:
        dense = {}
        for name, payload in sparse_grads.items():
            indices, values, shape = payload
            flat = np.zeros(int(np.prod(shape)), dtype=np.float64)
            flat[indices] = values
            dense[name] = flat.reshape(shape)
        return dense


class FederatedContinualLearning:
    def __init__(self, global_model: Vulgaris, config,
                 dp_noise: float = 1.0, compression: float = 0.01):
        self.global_model = global_model
        self.config = config
        self.dp = DPNoiseAdder(noise_multiplier=dp_noise, max_grad_norm=1.0)
        self.compressor = GradientCompressor(compression_ratio=compression)
        self.client_registry: Dict[str, dict] = {}
        self.round_history: deque = deque(maxlen=100)
        self.fedprox_mu: float = 0.01
        # Accumulate updates per round
        self._round_updates: Dict[str, List[np.ndarray]] = {}
        self._round_id: int = 0

    def register_client(self, client_id: str, metadata: dict) -> str:
        if client_id in self.client_registry:
            return str(self.client_registry[client_id]['domain_idx'])
        # Assign domain index cycling through available domains
        n_domains = self.config.dah.n_domains
        domain_idx = len(self.client_registry) % n_domains
        self.client_registry[client_id] = {
            'domain_idx': domain_idx,
            'metadata': metadata,
            'registered_at': time.time(),
            'n_updates': 0,
            'n_samples_total': 0,
            'flagged_byzantine': False,
        }
        return str(domain_idx)

    def _get_global_param_shapes(self) -> Dict[str, tuple]:
        shapes = {}
        for i, p in enumerate(self.global_model.parameters()):
            shapes[str(i)] = p.data.shape
        return shapes

    def _get_global_params(self) -> Dict[str, np.ndarray]:
        params = {}
        for i, p in enumerate(self.global_model.parameters()):
            params[str(i)] = p.data.copy()
        return params

    def client_update(self, client_id: str, local_gradients: Dict[str, np.ndarray],
                      n_samples: int) -> dict:
        if client_id not in self.client_registry:
            return {'accepted': False, 'reason': 'client_not_registered'}

        info = self.client_registry[client_id]
        if info['flagged_byzantine']:
            return {'accepted': False, 'reason': 'client_flagged_byzantine'}

        # Decompress sparse gradients
        shapes = self._get_global_param_shapes()
        # Check if already compressed (tuple payload) or raw
        is_sparse = any(
            isinstance(v, (tuple, list)) and len(v) == 3
            for v in local_gradients.values()
        )
        if is_sparse:
            dense_grads = self.compressor.decompress(local_gradients, shapes)
        else:
            # Apply DP noise + compression
            dp_grads = self.dp.clip_and_noise(local_gradients)
            compressed = self.compressor.compress(dp_grads, client_id)
            dense_grads = self.compressor.decompress(compressed, shapes)

        # Byzantine check: compare to current round aggregate if available
        if self._round_updates:
            for param_key, grad in dense_grads.items():
                if param_key in self._round_updates and self._round_updates[param_key]:
                    existing = np.stack(self._round_updates[param_key], axis=0)
                    median = np.median(existing, axis=0)
                    std = existing.std(axis=0).mean() + 1e-8
                    dist = np.linalg.norm(grad - median)
                    threshold = 2.0 * std * np.sqrt(grad.size)
                    if dist > threshold:
                        info['flagged_byzantine'] = True
                        return {'accepted': False, 'reason': 'byzantine_detected'}

        # Accumulate
        for param_key, grad in dense_grads.items():
            if param_key not in self._round_updates:
                self._round_updates[param_key] = []
            self._round_updates[param_key].append(grad)

        info['n_updates'] += 1
        info['n_samples_total'] += n_samples
        return {'accepted': True, 'reason': 'ok'}

    def aggregate(self, round_id: int) -> Dict[str, np.ndarray]:
        if not self._round_updates:
            return {}

        aggregated_deltas = {}
        delta_norms = {}

        params = list(self.global_model.parameters())

        for param_key, grad_list in self._round_updates.items():
            if not grad_list:
                continue
            stacked = np.stack(grad_list, axis=0)  # (n_clients, ...)
            n = stacked.shape[0]
            # Byzantine-robust trimmed mean: remove top and bottom 10%
            k_trim = max(1, int(n * 0.1))
            if n > 2 * k_trim:
                sorted_idx = np.argsort(
                    np.linalg.norm(stacked.reshape(n, -1), axis=1)
                )
                keep_idx = sorted_idx[k_trim: n - k_trim]
                stacked = stacked[keep_idx]
            agg = np.mean(stacked, axis=0)
            aggregated_deltas[param_key] = agg
            delta_norms[param_key] = float(np.linalg.norm(agg))

        # Apply to global model
        for i, p in enumerate(params):
            key = str(i)
            if key in aggregated_deltas:
                p.data -= aggregated_deltas[key]

        self.round_history.append({
            'round_id': round_id,
            'n_clients': len(self.client_registry),
            'delta_norms': delta_norms,
            'timestamp': time.time(),
        })
        self._round_updates = {}
        self._round_id = round_id + 1
        return delta_norms

    def get_global_update(self) -> Dict[str, np.ndarray]:
        params = self._get_global_params()
        # Compress for transmission (use a dummy client_id for server side)
        compressed = self.compressor.compress(params, '__server__')
        shapes = {k: v.shape for k, v in params.items()}
        return self.compressor.decompress(compressed, shapes)

    def fedprox_penalty(self, local_params: Dict[str, np.ndarray]) -> float:
        # ||theta_local - theta_global||^2 * mu/2
        global_params = self._get_global_params()
        total = 0.0
        for key in local_params:
            if key in global_params:
                diff = local_params[key] - global_params[key]
                total += float(np.sum(diff ** 2))
        return total * self.fedprox_mu / 2.0

    def privacy_report(self) -> dict:
        n_clients = len(self.client_registry)
        total_samples = sum(
            info['n_samples_total']
            for info in self.client_registry.values()
        )
        # Estimate consumed budget across all rounds
        if total_samples > 0 and n_clients > 0:
            avg_samples_per_client = total_samples / max(n_clients, 1)
            eps = self.dp.privacy_cost(
                n_steps=self._round_id,
                n_samples=int(avg_samples_per_client),
                batch_size=max(1, int(avg_samples_per_client // 10)),
                delta=1e-5,
            )
        else:
            eps = 0.0
        return {
            'epsilon': eps,
            'delta': 1e-5,
            'noise_multiplier': self.dp.sigma,
            'max_grad_norm': self.dp.C,
            'rounds_completed': self._round_id,
            'n_clients': n_clients,
        }
