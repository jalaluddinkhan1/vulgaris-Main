# -*- coding: utf-8 -*-
"""Unit tests for VULGARIS modules — shape, dtype, gradient flow."""
import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from engine.tensor import Tensor
from config import ModelConfig


def _default_config(d_model=32, n_classes=3, input_dim=4):
    """Minimal config for fast testing."""
    cfg = ModelConfig(
        input_dim=input_dim,
        output_dim=1,
        n_classes=n_classes,
    )
    cfg.ase.latent_dim = d_model
    cfg.ase.n_filters = 8
    cfg.ase.n_scales = 2
    cfg.sssr.state_dim = 16
    cfg.sssr.d_inner = d_model * 2
    cfg.crg.n_nodes = 4
    cfg.hmb.n_slots = 8
    cfg.hmb.latent_dim = d_model
    return cfg


class TestASE(unittest.TestCase):

    def test_output_shape(self):
        from modules.ase import AdaptiveSignalEmbedding
        B, C, T = 2, 4, 20
        ase = AdaptiveSignalEmbedding(in_channels=C, n_filters=8, n_scales=2,
                                      filter_len=8, latent_dim=32)
        x = Tensor(np.random.randn(B, C, T).astype(np.float32))
        out = ase(x)
        self.assertEqual(out.data.shape, (B, T, 32))

    def test_output_dtype(self):
        from modules.ase import AdaptiveSignalEmbedding
        ase = AdaptiveSignalEmbedding(in_channels=4, n_filters=8, n_scales=2,
                                      filter_len=8, latent_dim=32)
        x = Tensor(np.random.randn(2, 4, 20).astype(np.float32))
        out = ase(x)
        self.assertEqual(out.data.dtype, np.float32)

    def test_output_finite(self):
        from modules.ase import AdaptiveSignalEmbedding
        ase = AdaptiveSignalEmbedding(in_channels=4, n_filters=8, n_scales=2,
                                      filter_len=8, latent_dim=32)
        x = Tensor(np.random.randn(2, 4, 20).astype(np.float32))
        out = ase(x)
        self.assertTrue(np.all(np.isfinite(out.data)), "ASE output has inf/nan")


class TestSSSR(unittest.TestCase):

    def _make_sssr(self, d_model=32):
        from modules.sssr import SelectiveSSR
        cfg = _default_config(d_model=d_model)
        return SelectiveSSR(d_model=d_model, config=cfg.sssr)

    def test_output_shape(self):
        sssr = self._make_sssr(32)
        B, T, D = 2, 10, 32
        z = Tensor(np.random.randn(B, T, D).astype(np.float32))
        out, states = sssr(z)
        self.assertEqual(out.data.shape, (B, T, D))

    def test_finite(self):
        sssr = self._make_sssr(32)
        z = Tensor(np.random.randn(2, 10, 32).astype(np.float32))
        out, _ = sssr(z)
        self.assertTrue(np.all(np.isfinite(out.data)))

    def test_streaming_shape(self):
        sssr = self._make_sssr(32)
        z_single = Tensor(np.random.randn(2, 1, 32).astype(np.float32))
        out, states = sssr(z_single, states=None)
        self.assertEqual(out.data.shape, (2, 1, 32))


class TestCRG(unittest.TestCase):

    def test_output_shape_and_finite(self):
        from modules.crg import CausalRoutingGraph
        cfg = _default_config(d_model=32)
        crg = CausalRoutingGraph(d_model=32, config=cfg.crg)
        B, T, D = 2, 8, 32
        z = Tensor(np.random.randn(B, T, D).astype(np.float32))
        out, penalty = crg(z)
        self.assertEqual(out.data.shape, (B, T, D))
        self.assertTrue(np.isfinite(float(penalty.data.sum())))


class TestVulgaris(unittest.TestCase):

    def setUp(self):
        from model.vulgaris import Vulgaris
        self.cfg = _default_config(d_model=32, n_classes=3, input_dim=4)
        self.model = Vulgaris(self.cfg)

    def test_forward_shape(self):
        B, C, T = 2, 4, 20
        x = Tensor(np.random.randn(B, C, T).astype(np.float32))
        out, aux = self.model(x)
        self.assertEqual(out.data.shape, (B, 3))  # n_classes=3

    def test_output_float32(self):
        x = Tensor(np.random.randn(2, 4, 20).astype(np.float32))
        out, _ = self.model(x)
        self.assertEqual(out.data.dtype, np.float32)

    def test_output_finite(self):
        x = Tensor(np.random.randn(2, 4, 20).astype(np.float32))
        out, aux = self.model(x)
        self.assertTrue(np.all(np.isfinite(out.data)))
        h = aux["h_states"]
        self.assertTrue(np.all(np.isfinite(h.data)))

    def test_softmax_sums_to_one(self):
        x = Tensor(np.random.randn(2, 4, 20).astype(np.float32))
        out, _ = self.model(x)
        row_sums = out.data.sum(axis=-1)
        np.testing.assert_allclose(row_sums, np.ones_like(row_sums), atol=1e-5)

    def test_streaming_step(self):
        B = 2
        state = self.model.init_state(B)
        x_t = Tensor(np.random.randn(B, 4).astype(np.float32))
        out_t, new_state = self.model.step(x_t, state)
        self.assertEqual(out_t.data.shape, (B, 3))
        self.assertEqual(new_state.step, 1)

    def test_multi_domain(self):
        """DAH domain switching shouldn't crash."""
        x = Tensor(np.random.randn(2, 4, 20).astype(np.float32))
        for domain in range(3):
            out, _ = self.model(x, domain_idx=domain)
            self.assertTrue(np.all(np.isfinite(out.data)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
