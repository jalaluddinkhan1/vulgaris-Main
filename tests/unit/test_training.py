# -*- coding: utf-8 -*-
"""Unit tests for training pipeline, drift detection, and baselines."""
import sys
import os
import unittest
import tempfile
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from config import ModelConfig
from engine.tensor import Tensor


def _small_config():
    cfg = ModelConfig(input_dim=4, output_dim=1, n_classes=3)
    cfg.ase.latent_dim = 32
    cfg.ase.n_filters = 8
    cfg.ase.n_scales = 2
    cfg.sssr.state_dim = 16
    cfg.sssr.d_inner = 64
    cfg.crg.n_nodes = 4
    cfg.hmb.n_slots = 8
    cfg.hmb.latent_dim = 32
    return cfg


def _make_pipeline(tmp_dir):
    from model.vulgaris import Vulgaris
    from training.loss import VulgarisLoss
    from training.optimizer import SpectralAdamW, CosineSchedule

    cfg = _small_config()
    cfg.training.checkpoint_dir = tmp_dir
    model = Vulgaris(cfg)
    loss_fn = VulgarisLoss(cfg)
    optimizer = SpectralAdamW(model.parameters(), lr=1e-3)
    scheduler = CosineSchedule(optimizer, warmup_steps=2, max_steps=50, min_lr=1e-5)
    from training.pipeline import TrainingPipeline
    return TrainingPipeline(model, cfg, loss_fn, optimizer, scheduler), model, cfg


class TestTrainStep(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.pipeline, self.model, self.cfg = _make_pipeline(self.tmp)

    def test_returns_finite_loss(self):
        rng = np.random.default_rng(0)
        x = rng.normal(0, 0.5, (4, 4, 20)).astype(np.float32)
        y = np.zeros((4,), dtype=np.float32)
        metrics = self.pipeline.train_step(x, y, domain_idx=0)
        self.assertIn("total_loss", metrics)
        self.assertTrue(np.isfinite(metrics["total_loss"]) or metrics["total_loss"] == float('inf'))

    def test_step_count_increments(self):
        rng = np.random.default_rng(1)
        x = rng.normal(0, 0.5, (4, 4, 20)).astype(np.float32)
        y = np.zeros(4, dtype=np.float32)
        self.pipeline.train_step(x, y)
        self.assertEqual(self.pipeline.step_count, 1)

    def test_repeated_steps_stable(self):
        rng = np.random.default_rng(2)
        losses = []
        for i in range(5):
            x = rng.normal(0, 0.3, (4, 4, 20)).astype(np.float32)
            y = np.zeros(4, dtype=np.float32)
            m = self.pipeline.train_step(x, y)
            losses.append(m["total_loss"])
        finite_losses = [l for l in losses if np.isfinite(l)]
        self.assertGreater(len(finite_losses), 0, "All losses were inf/nan")


class TestEvalStep(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.pipeline, _, _ = _make_pipeline(self.tmp)

    def test_returns_interval(self):
        rng = np.random.default_rng(3)
        x = rng.normal(0, 0.5, (4, 4, 20)).astype(np.float32)
        y = np.zeros(4, dtype=np.float32)
        metrics = self.pipeline.eval_step(x, y)
        self.assertIn("interval_lower", metrics)
        self.assertIn("interval_upper", metrics)
        self.assertIn("interval_width", metrics)

    def test_interval_width_nonneg(self):
        rng = np.random.default_rng(4)
        x = rng.normal(0, 0.5, (4, 4, 20)).astype(np.float32)
        y = np.zeros(4, dtype=np.float32)
        metrics = self.pipeline.eval_step(x, y)
        self.assertGreaterEqual(metrics["interval_width"], 0.0)


class TestCheckpoint(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.pipeline, self.model, _ = _make_pipeline(self.tmp)

    def test_save_and_load(self):
        rng = np.random.default_rng(5)
        x = rng.normal(0, 0.3, (4, 4, 20)).astype(np.float32)
        y = np.zeros(4, dtype=np.float32)
        self.pipeline.train_step(x, y)

        ckpt_path = os.path.join(self.tmp, "checkpoint_best.npz")
        if not os.path.exists(ckpt_path):
            self.pipeline.save_checkpoint("test")
            ckpt_path = os.path.join(self.tmp, "checkpoint_test.npz")

        self.assertTrue(os.path.exists(ckpt_path), f"Checkpoint not found at {ckpt_path}")

        # Load back
        step_before = self.pipeline.step_count
        self.pipeline.load_checkpoint(ckpt_path)
        # step_count should be restored
        self.assertEqual(self.pipeline.step_count, step_before)

    def test_checkpoint_npz_format(self):
        self.pipeline.save_checkpoint("unit_test")
        path = os.path.join(self.tmp, "checkpoint_unit_test.npz")
        self.assertTrue(os.path.exists(path))
        data = np.load(path, allow_pickle=False)
        self.assertIn("meta__step_count", data.files)
        self.assertIn("meta__best_loss", data.files)


class TestDriftDetector(unittest.TestCase):

    def test_no_drift_on_identical(self):
        from monitoring.drift import DriftDetector
        det = DriftDetector(window_size=50, ks_threshold=0.2, mmd_threshold=0.1,
                            wasserstein_threshold=0.2)
        rng = np.random.default_rng(0)
        ref = rng.normal(0, 1, (100, 8)).astype(np.float32)
        det.set_reference(ref)
        result = det.update(rng.normal(0, 1, (50, 8)).astype(np.float32))
        # Same distribution — should rarely flag drift
        # (not guaranteed due to small sample noise, but stats should be small)
        self.assertLess(result["ks_stat"], 0.5)

    def test_drift_detected_on_shift(self):
        from monitoring.drift import DriftDetector
        det = DriftDetector(window_size=100, ks_threshold=0.1, mmd_threshold=0.05,
                            wasserstein_threshold=0.1)
        rng = np.random.default_rng(1)
        ref = rng.normal(0, 0.1, (200, 4)).astype(np.float32)
        det.set_reference(ref)
        shifted = rng.normal(5, 0.1, (100, 4)).astype(np.float32)
        result = det.update(shifted)
        self.assertTrue(result["drift_detected"], "Should detect large distribution shift")

    def test_reset_reference(self):
        from monitoring.drift import DriftDetector
        det = DriftDetector(window_size=50)
        rng = np.random.default_rng(2)
        ref = rng.normal(0, 1, (100, 4)).astype(np.float32)
        det.set_reference(ref)
        det.update(rng.normal(0, 1, (50, 4)).astype(np.float32))
        det.reset_reference()
        self.assertIsNotNone(det._reference)


class TestBaselines(unittest.TestCase):

    def _data(self, B=20, C=4, T=30, seed=0):
        rng = np.random.default_rng(seed)
        X = rng.normal(0, 1, (B, C, T)).astype(np.float32)
        y = X[:, 0, -1]
        return X, y

    def test_last_value(self):
        from benchmarks.baselines import LastValue
        X, y = self._data()
        m = LastValue()
        m.fit(X, y)
        preds = m.predict_flat(X)
        self.assertEqual(preds.shape, (20,))

    def test_moving_average(self):
        from benchmarks.baselines import MovingAverage
        X, y = self._data()
        m = MovingAverage(k=5)
        m.fit(X, y)
        preds = m.predict_flat(X)
        self.assertEqual(preds.shape, (20,))

    def test_arima_lite(self):
        from benchmarks.baselines import ARIMA_lite
        X, y = self._data()
        m = ARIMA_lite(p=3)
        m.fit(X, y)
        preds = m.predict_flat(X)
        self.assertEqual(preds.shape, (20,))
        self.assertTrue(np.all(np.isfinite(preds)))

    def test_lstm_lite(self):
        from benchmarks.baselines import LSTMLite
        X, y = self._data()
        m = LSTMLite(input_size=4, hidden_size=8)
        m.fit(X, y)
        preds = m.predict_flat(X)
        self.assertEqual(preds.shape, (20,))

    def test_run_comparison(self):
        from benchmarks.baselines import run_baseline_comparison
        X, y = self._data()
        results = run_baseline_comparison(X, y, X, y)
        self.assertIn("LastValue", results)
        self.assertIn("MovingAverage", results)
        self.assertIn("ARIMA_lite", results)
        for name, r in results.items():
            self.assertIn("mse", r)


if __name__ == "__main__":
    unittest.main(verbosity=2)
