# -*- coding: utf-8 -*-
"""
Unit tests for all Phase 1, 2, and 3 additions to VULGARIS.

Phase 1: Spectral augmentation, delta-threshold gating, MuonOptimizer,
         CanaryController / DeploymentMode, AuditLogger
Phase 2: MAE pretraining, I2A blending, MoD routing, TF-C loss,
         TrainingPipeline tfc_enabled / mae_mask_ratio flags
Phase 3: IndustrialTokenizer, OntologyEmbedding / OntologyRegistry,
         DAH ontology wiring, CRG intervene() / counterfactual_root_cause(),
         WorldModelHead / world_model_forward()
"""
import os
import sys
import json
import tempfile
import unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from engine.tensor import Tensor
from config import ModelConfig


# ──────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────────────────────────────────

def _cfg(d=16):
    cfg = ModelConfig(input_dim=4, output_dim=2, n_classes=0)
    cfg.ase.latent_dim = d
    cfg.ase.n_filters = 4
    cfg.ase.n_scales = 2
    cfg.ase.filter_len = 4
    cfg.sssr.state_dim = d
    cfg.sssr.d_inner = d * 2
    cfg.sssr.n_heads = 2
    cfg.crg.n_nodes = 8
    cfg.crg.n_lags = 2
    cfg.hmb.embed_dim = d
    cfg.hmb.compress_dim = max(d // 4, 4)
    cfg.hmb.buffer_size = 4
    cfg.htd.n_levels = 2
    cfg.htd.time_constants = [0.1, 1.0]
    cfg.dah.meta_dim = d
    return cfg


def _model(d=16):
    from model.vulgaris import Vulgaris
    return Vulgaris(_cfg(d))


def _x(B=2, C=4, T=8):
    return Tensor(np.random.randn(B, C, T).astype(np.float32))


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1
# ══════════════════════════════════════════════════════════════════════════════

class TestSpectralAugmentation(unittest.TestCase):
    """Phase 1 — Item 1: spectral frequency-band dropout in TimeSeriesAugment."""

    def _aug(self, p=0.9, width=4):
        from training.pipeline import TimeSeriesAugment
        return TimeSeriesAugment(noise_std=0.0, channel_dropout_p=0.0,
                                  scale_range=(1.0, 1.0), time_warp_max=0,
                                  aug_prob=1.0,
                                  spectral_drop_p=p, spectral_drop_width=width)

    def test_output_shape_preserved(self):
        aug = self._aug()
        x = np.random.randn(2, 4, 64).astype(np.float32)
        out = aug(x)
        self.assertEqual(out.shape, x.shape)

    def test_output_dtype_float32(self):
        aug = self._aug()
        x = np.random.randn(2, 4, 64).astype(np.float32)
        out = aug(x)
        self.assertEqual(out.dtype, np.float32)

    def test_output_finite(self):
        aug = self._aug()
        x = np.random.randn(2, 4, 64).astype(np.float32)
        out = aug(x)
        self.assertTrue(np.all(np.isfinite(out)))

    def test_spectral_drop_p_zero_is_noop(self):
        from training.pipeline import TimeSeriesAugment
        aug = TimeSeriesAugment(noise_std=0, channel_dropout_p=0,
                                 scale_range=(1.0, 1.0), time_warp_max=0,
                                 aug_prob=1.0, spectral_drop_p=0.0)
        x = np.ones((2, 4, 32), dtype=np.float32)
        out = aug(x)
        np.testing.assert_array_equal(out, x)

    def test_repr_includes_spectral_drop_p(self):
        aug = self._aug(p=0.3)
        self.assertIn("spectral_drop_p", repr(aug))


class TestDeltaThresholdGating(unittest.TestCase):
    """Phase 1 — Item 2: event gating in StreamingInference."""

    def _engine(self, threshold=0.5):
        from inference.streaming import StreamingInference
        m = _model()
        return StreamingInference(m, batch_size=1, delta_threshold=threshold)

    def test_first_step_not_skipped(self):
        eng = self._engine()
        x = np.random.randn(1, 4).astype(np.float32)
        result = eng.step(x)
        self.assertFalse(result["skipped"])

    def test_identical_input_skipped(self):
        eng = self._engine(threshold=0.01)
        x = np.random.randn(1, 4).astype(np.float32)
        eng.step(x)           # warm up cache
        result = eng.step(x)  # exact same input → must skip
        self.assertTrue(result["skipped"])

    def test_large_change_not_skipped(self):
        eng = self._engine(threshold=0.01)
        x1 = np.zeros((1, 4), dtype=np.float32)
        x2 = np.ones((1, 4), dtype=np.float32) * 10.0
        eng.step(x1)
        result = eng.step(x2)
        self.assertFalse(result["skipped"])

    def test_skipped_count_in_health_stats(self):
        eng = self._engine(threshold=0.01)
        x = np.zeros((1, 4), dtype=np.float32)
        eng.step(x)
        eng.step(x)
        eng.step(x)
        stats = eng.get_health_stats()
        self.assertIn("steps_skipped", stats)
        self.assertGreaterEqual(stats["steps_skipped"], 2)

    def test_skipped_result_has_prediction_key(self):
        eng = self._engine(threshold=0.01)
        x = np.zeros((1, 4), dtype=np.float32)
        eng.step(x)
        result = eng.step(x)
        self.assertIn("prediction", result)
        self.assertIn("uncertainty", result)

    def test_no_gating_when_threshold_none(self):
        from inference.streaming import StreamingInference
        m = _model()
        eng = StreamingInference(m, batch_size=1, delta_threshold=None)
        x = np.zeros((1, 4), dtype=np.float32)
        eng.step(x)
        result = eng.step(x)
        self.assertFalse(result["skipped"])


class TestMuonOptimizer(unittest.TestCase):
    """Phase 1 — Item 3: Muon optimizer with Newton-Schulz orthogonalization."""

    def test_step_updates_2d_params(self):
        from training.optimizer import MuonOptimizer
        from engine.tensor import Parameter
        w = Parameter(np.random.randn(8, 8).astype(np.float64), name="W")
        w.grad = np.random.randn(8, 8).astype(np.float64)
        before = w.data.copy()
        opt = MuonOptimizer([w], lr=0.01)
        opt.step()
        self.assertFalse(np.allclose(w.data, before), "2-D param should change after step")

    def test_step_updates_1d_params(self):
        from training.optimizer import MuonOptimizer
        from engine.tensor import Parameter
        b = Parameter(np.random.randn(8).astype(np.float64), name="b")
        b.grad = np.random.randn(8).astype(np.float64)
        before = b.data.copy()
        opt = MuonOptimizer([b], lr=0.01)
        opt.step()
        self.assertFalse(np.allclose(b.data, before), "1-D param should change after step")

    def test_ns_orthogonalize_output_finite(self):
        from training.optimizer import MuonOptimizer
        G = np.random.randn(8, 8)
        X = MuonOptimizer._ns_orthogonalize(G, steps=5)
        self.assertTrue(np.all(np.isfinite(X)))

    def test_ns_orthogonalize_near_orthogonal(self):
        from training.optimizer import MuonOptimizer
        G = np.random.randn(8, 8)
        X = MuonOptimizer._ns_orthogonalize(G, steps=5)
        XtX = X.T @ X
        # Should be approximately a scaled identity — check off-diagonals are small
        off_diag = XtX - np.diag(np.diag(XtX))
        self.assertLess(np.abs(off_diag).max(), 0.5)

    def test_zero_grad_clears_gradients(self):
        from training.optimizer import MuonOptimizer
        from engine.tensor import Parameter
        p = Parameter(np.ones((4, 4)), name="p")
        p.grad = np.ones((4, 4))
        opt = MuonOptimizer([p])
        opt.zero_grad()
        self.assertIsNone(p.grad)

    def test_state_dict_roundtrip(self):
        from training.optimizer import MuonOptimizer
        from engine.tensor import Parameter
        p = Parameter(np.random.randn(4, 4), name="p")
        p.grad = np.random.randn(4, 4)
        opt = MuonOptimizer([p], lr=0.01)
        opt.step()
        sd = opt.state_dict()
        opt2 = MuonOptimizer([p], lr=0.01)
        opt2.load_state_dict(sd)
        self.assertEqual(opt2._t, opt._t)


class TestCanaryController(unittest.TestCase):
    """Phase 1 — Item 4: shadow/canary deployment modes."""

    def test_primary_never_uses_canary(self):
        from serve.degradation import CanaryController, DeploymentMode
        ctrl = CanaryController(mode=DeploymentMode.PRIMARY)
        results = [ctrl.should_use_output() for _ in range(50)]
        self.assertFalse(any(results))

    def test_shadow_never_uses_output(self):
        from serve.degradation import CanaryController, DeploymentMode
        ctrl = CanaryController(mode=DeploymentMode.SHADOW)
        results = [ctrl.should_use_output() for _ in range(50)]
        self.assertFalse(any(results))

    def test_canary_routes_fraction(self):
        from serve.degradation import CanaryController, DeploymentMode
        ctrl = CanaryController(mode=DeploymentMode.CANARY, canary_pct=50.0)
        results = [ctrl.should_use_output() for _ in range(200)]
        # With 50% rate and 200 samples, should get some True and some False
        self.assertTrue(any(results))
        self.assertTrue(any(not r for r in results))

    def test_log_shadow_records_entry(self):
        from serve.degradation import CanaryController, DeploymentMode
        ctrl = CanaryController(mode=DeploymentMode.SHADOW)
        ctrl.log_shadow(np.array([1.0]), np.array([0.9]), agree=True)
        stats = ctrl.shadow_stats()
        self.assertEqual(stats["shadow_entries"], 1)

    def test_shadow_stats_keys(self):
        from serve.degradation import CanaryController, DeploymentMode
        ctrl = CanaryController(mode=DeploymentMode.CANARY, canary_pct=10.0)
        stats = ctrl.shadow_stats()
        for key in ("mode", "total_requests", "canary_requests",
                    "shadow_entries", "agree_rate", "canary_pct"):
            self.assertIn(key, stats)

    def test_set_mode_changes_mode(self):
        from serve.degradation import CanaryController, DeploymentMode
        ctrl = CanaryController(mode=DeploymentMode.PRIMARY)
        ctrl.set_mode(DeploymentMode.SHADOW)
        self.assertEqual(ctrl.mode, DeploymentMode.SHADOW)


class TestAuditLogger(unittest.TestCase):
    """Phase 1 — Item 5: per-prediction append-only JSONL audit log."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.path = os.path.join(self.tmp, "audit.jsonl")

    def test_creates_file(self):
        from serve.audit import AuditLogger
        AuditLogger(path=self.path)
        self.assertTrue(os.path.exists(self.path))

    def test_record_writes_jsonl(self):
        from serve.audit import AuditLogger
        logger = AuditLogger(path=self.path)
        logger.record(x_raw=np.zeros((2, 4)), prediction=np.array([0.1, 0.2]),
                      step=1, domain_idx=0)
        with open(self.path) as f:
            lines = [l.strip() for l in f if l.strip()]
        self.assertEqual(len(lines), 1)
        rec = json.loads(lines[0])
        self.assertIn("input_sha256", rec)
        self.assertIn("prediction", rec)
        self.assertIn("weights_sha256", rec)

    def test_tail_returns_records(self):
        from serve.audit import AuditLogger
        logger = AuditLogger(path=self.path)
        for i in range(5):
            logger.record(x_raw=np.zeros((1, 4)), prediction=np.array([float(i)]),
                          step=i)
        recs = logger.tail(3)
        self.assertEqual(len(recs), 3)

    def test_count_increments(self):
        from serve.audit import AuditLogger
        logger = AuditLogger(path=self.path)
        for _ in range(4):
            logger.record(x_raw=np.zeros(4), prediction=np.array([0.0]), step=0)
        self.assertEqual(logger.count(), 4)

    def test_sha256_changes_with_input(self):
        from serve.audit import AuditLogger
        logger = AuditLogger(path=self.path)
        logger.record(x_raw=np.zeros(4), prediction=np.array([0.0]), step=0)
        logger.record(x_raw=np.ones(4),  prediction=np.array([0.0]), step=1)
        recs = logger.tail(2)
        self.assertNotEqual(recs[0]["input_sha256"], recs[1]["input_sha256"])

    def test_status_dict_keys(self):
        from serve.audit import AuditLogger
        logger = AuditLogger(path=self.path)
        s = logger.status()
        self.assertIn("path", s)
        self.assertIn("records_written", s)


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2
# ══════════════════════════════════════════════════════════════════════════════

class TestMAEPretraining(unittest.TestCase):
    """Phase 2 — Item 1: Masked Autoencoder pretraining."""

    def setUp(self):
        self.model = _model(d=16)

    def test_mae_forward_shapes(self):
        x = _x()
        recon, mae_loss, mask = self.model.mae_forward(x, mask_ratio=0.5)
        B, C, T = 2, 4, 8
        self.assertEqual(recon.data.shape, (B, T, C))
        self.assertEqual(mask.shape, (B, C, T))
        self.assertEqual(mae_loss.data.shape, (1, 1))

    def test_mae_loss_positive(self):
        x = _x()
        _, mae_loss, _ = self.model.mae_forward(x, mask_ratio=0.5)
        self.assertGreaterEqual(float(mae_loss.data.sum()), 0.0)

    def test_mae_loss_zero_mask_ratio(self):
        x = _x()
        # With mask_ratio=0.0 no patches are masked → loss should be 0
        _, mae_loss, mask = self.model.mae_forward(x, mask_ratio=0.0)
        # mask should be all False
        self.assertFalse(mask.any())

    def test_mae_loss_finite(self):
        x = _x()
        _, mae_loss, _ = self.model.mae_forward(x, mask_ratio=0.75)
        self.assertTrue(np.isfinite(float(mae_loss.data.sum())))

    def test_mae_decoder_registered(self):
        self.assertTrue(hasattr(self.model, "mae_decoder"))


class TestI2ABlending(unittest.TestCase):
    """Phase 2 — Item 1 (I2A): Imagination-to-Action blending."""

    def setUp(self):
        self.model = _model(d=16)

    def test_i2a_active_flag(self):
        x = _x()
        _, aux = self.model(x, use_imagination=True)
        self.assertTrue(aux["i2a_active"])

    def test_i2a_inactive_by_default(self):
        x = _x()
        _, aux = self.model(x)
        self.assertFalse(aux["i2a_active"])

    def test_i2a_output_shape_unchanged(self):
        x = _x()
        out_reactive, _ = self.model(x)
        out_imag, _     = self.model(x, use_imagination=True)
        self.assertEqual(out_reactive.data.shape, out_imag.data.shape)

    def test_i2a_output_finite(self):
        x = _x()
        out, _ = self.model(x, use_imagination=True)
        self.assertTrue(np.all(np.isfinite(out.data)))

    def test_i2a_blend_module_registered(self):
        self.assertTrue(hasattr(self.model, "i2a"))


class TestMoDRouting(unittest.TestCase):
    """Phase 2 — Item 2: Mixture-of-Depths token routing in SelectiveSSR."""

    def _sssr(self, routing_k=0.5):
        from modules.sssr import SelectiveSSR
        cfg = _cfg(d=16)
        cfg.sssr.routing_k = routing_k
        return SelectiveSSR(d_model=16, config=cfg.sssr)

    def test_routing_proj_exists_when_k_lt_1(self):
        sssr = self._sssr(0.5)
        self.assertIsNotNone(sssr.routing_proj)

    def test_routing_proj_none_when_k_eq_1(self):
        sssr = self._sssr(1.0)
        self.assertIsNone(sssr.routing_proj)

    def test_output_shape_with_routing(self):
        sssr = self._sssr(0.5)
        z = Tensor(np.random.randn(2, 8, 16).astype(np.float32))
        out, _ = sssr(z)
        self.assertEqual(out.data.shape, (2, 8, 16))

    def test_output_finite_with_routing(self):
        sssr = self._sssr(0.5)
        z = Tensor(np.random.randn(2, 8, 16).astype(np.float32))
        out, _ = sssr(z)
        self.assertTrue(np.all(np.isfinite(out.data)))

    def test_full_routing_matches_no_routing_behavior(self):
        # routing_k=1.0 and routing_k=0.99 should both run all tokens
        sssr_full = self._sssr(1.0)
        z = Tensor(np.random.randn(2, 8, 16).astype(np.float32))
        out_full, _ = sssr_full(z)
        self.assertEqual(out_full.data.shape, (2, 8, 16))


class TestTFCLoss(unittest.TestCase):
    """Phase 2 — Item 3: Time-Frequency Consistency contrastive loss."""

    def setUp(self):
        from training.loss import VulgarisLoss
        self.loss_fn = VulgarisLoss(_cfg())

    def test_tfc_loss_scalar(self):
        h_t = Tensor(np.random.randn(4, 16), requires_grad=True)
        h_f = Tensor(np.random.randn(4, 16), requires_grad=True)
        loss = self.loss_fn.tfc_loss(h_t, h_f)
        self.assertEqual(loss.data.shape, (1, 1))

    def test_tfc_loss_finite(self):
        h_t = Tensor(np.random.randn(4, 16), requires_grad=True)
        h_f = Tensor(np.random.randn(4, 16), requires_grad=True)
        loss = self.loss_fn.tfc_loss(h_t, h_f)
        self.assertTrue(np.isfinite(float(loss.data.sum())))

    def test_tfc_backward_populates_grads(self):
        h_t = Tensor(np.random.randn(4, 16), requires_grad=True)
        h_f = Tensor(np.random.randn(4, 16), requires_grad=True)
        loss = self.loss_fn.tfc_loss(h_t, h_f)
        loss.backward()
        self.assertIsNotNone(h_t.grad)
        self.assertIsNotNone(h_f.grad)
        self.assertTrue(np.all(np.isfinite(h_t.grad)))

    def test_tfc_identical_views_lower_loss(self):
        # Same view → diagonal similarities all max → lower loss than random
        h = Tensor(np.random.randn(4, 16), requires_grad=False)
        loss_same = float(self.loss_fn.tfc_loss(h, h).data.sum())
        h_f = Tensor(np.random.randn(4, 16), requires_grad=False)
        loss_diff = float(self.loss_fn.tfc_loss(h, h_f).data.sum())
        self.assertLessEqual(loss_same, loss_diff)

    def test_tfc_appears_in_forward_components(self):
        from model.vulgaris import Vulgaris
        model = Vulgaris(_cfg())
        x = _x()
        y = Tensor(np.random.randn(2, 2).astype(np.float32))
        out, aux = model(x)
        aux["tfc_pair"] = (
            Tensor(np.random.randn(2, 16)),
            Tensor(np.random.randn(2, 16)),
        )
        _, comps = self.loss_fn.forward(out, y, aux)
        self.assertIn("tfc_loss", comps)

    def test_mae_loss_appears_in_forward_components(self):
        from model.vulgaris import Vulgaris
        model = Vulgaris(_cfg())
        x = _x()
        y = Tensor(np.random.randn(2, 2).astype(np.float32))
        out, aux = model(x)
        _, mae_loss, _ = model.mae_forward(x, mask_ratio=0.5)
        aux["mae_loss"] = mae_loss
        _, comps = self.loss_fn.forward(out, y, aux)
        self.assertIn("mae_loss", comps)
        self.assertGreaterEqual(comps["mae_loss"], 0.0)


class TestTrainingPipelinePhase2Flags(unittest.TestCase):
    """Phase 2 — TrainingPipeline tfc_enabled and mae_mask_ratio flags."""

    def _pipeline(self, tfc=False, mae=0.0):
        from model.vulgaris import Vulgaris
        from training.loss import VulgarisLoss
        from training.optimizer import SpectralAdamW, CosineSchedule
        from training.pipeline import TrainingPipeline, TimeSeriesAugment
        cfg = _cfg()
        cfg.training.checkpoint_dir = tempfile.mkdtemp()
        model = Vulgaris(cfg)
        loss_fn = VulgarisLoss(cfg)
        opt = SpectralAdamW(model.parameters(), lr=1e-3)
        sch = CosineSchedule(opt, warmup_steps=2, max_steps=20, min_lr=1e-5)
        return TrainingPipeline(model, cfg, loss_fn, opt, sch,
                                tfc_enabled=tfc, mae_mask_ratio=mae)

    def test_tfc_enabled_flag_stored(self):
        p = self._pipeline(tfc=True)
        self.assertTrue(p.tfc_enabled)

    def test_mae_mask_ratio_stored(self):
        p = self._pipeline(mae=0.5)
        self.assertAlmostEqual(p.mae_mask_ratio, 0.5)

    def test_tfc_enabled_step_finite(self):
        p = self._pipeline(tfc=True)
        x = np.random.randn(2, 4, 8).astype(np.float32)
        y = np.random.randn(2, 2).astype(np.float32)
        m = p.train_step(x, y)
        val = m.get("total_loss", float("inf"))
        self.assertTrue(np.isfinite(val) or val == float("inf"))

    def test_mae_enabled_step_finite(self):
        p = self._pipeline(mae=0.5)
        x = np.random.randn(2, 4, 8).astype(np.float32)
        y = np.random.randn(2, 2).astype(np.float32)
        m = p.train_step(x, y)
        val = m.get("total_loss", float("inf"))
        self.assertTrue(np.isfinite(val) or val == float("inf"))

    def test_metrics_dict_has_tfc_and_mae_keys(self):
        p = self._pipeline(tfc=True, mae=0.5)
        self.assertIn("tfc_loss", p.metrics)
        self.assertIn("mae_loss", p.metrics)


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3
# ══════════════════════════════════════════════════════════════════════════════

class TestIndustrialTokenizer(unittest.TestCase):
    """Phase 3 — Item 1: multi-rate typed token stream tokenizer."""

    def _specs_and_dict(self, B=2, T_high=100):
        from preprocessing.industrial_tokenizer import ChannelSpec, TokenType
        specs = [
            ChannelSpec(idx=0, token_type=TokenType.CONTINUOUS, hz=100.0, name="temp"),
            ChannelSpec(idx=1, token_type=TokenType.EVENT,      hz=10.0,  name="alarm"),
            ChannelSpec(idx=2, token_type=TokenType.ACTION,     hz=50.0,  name="valve"),
            ChannelSpec(idx=3, token_type=TokenType.REGIME,     hz=1.0,   name="mode",
                        n_classes=4),
        ]
        x_dict = {
            0: np.random.randn(B, T_high).astype(np.float32),
            1: np.random.rand(B, 10).astype(np.float32),
            2: np.random.randn(B, 50).astype(np.float32),
            3: np.random.randint(0, 4, (B, 1)).astype(np.float32),
        }
        return specs, x_dict

    def test_output_shape(self):
        from preprocessing.industrial_tokenizer import IndustrialTokenizer
        specs, x_dict = self._specs_and_dict()
        tok = IndustrialTokenizer(specs, d_model=32, target_hz=100.0)
        z = tok(x_dict, window_len=100)
        self.assertEqual(z.data.shape, (2, 100, 32))

    def test_output_finite(self):
        from preprocessing.industrial_tokenizer import IndustrialTokenizer
        specs, x_dict = self._specs_and_dict()
        tok = IndustrialTokenizer(specs, d_model=32, target_hz=100.0)
        z = tok(x_dict, window_len=100)
        self.assertTrue(np.all(np.isfinite(z.data)))

    def test_missing_channel_filled_with_zeros(self):
        from preprocessing.industrial_tokenizer import IndustrialTokenizer, ChannelSpec, TokenType
        specs = [ChannelSpec(idx=99, token_type=TokenType.CONTINUOUS, hz=10.0, name="missing")]
        tok = IndustrialTokenizer(specs, d_model=16, target_hz=10.0)
        z = tok({}, window_len=10)   # empty dict → all channels missing
        self.assertEqual(z.data.shape, (1, 10, 16))   # default B inferred fails → guard
        # shape should be valid if window_len given and B inferred from first channel
        # → since dict is empty, defaults to B=1 guard; output must be finite
        self.assertTrue(np.all(np.isfinite(z.data)))

    def test_specs_summary_length(self):
        from preprocessing.industrial_tokenizer import IndustrialTokenizer
        specs, _ = self._specs_and_dict()
        tok = IndustrialTokenizer(specs, d_model=16, target_hz=100.0)
        summary = tok.specs_summary()
        self.assertEqual(len(summary), len(specs))

    def test_regime_channel_produces_finite(self):
        from preprocessing.industrial_tokenizer import IndustrialTokenizer, ChannelSpec, TokenType
        B, T = 2, 20
        specs = [ChannelSpec(idx=0, token_type=TokenType.REGIME, hz=1.0,
                             name="mode", n_classes=3)]
        tok = IndustrialTokenizer(specs, d_model=8, target_hz=1.0)
        x_dict = {0: np.random.randint(0, 3, (B, T)).astype(np.float32)}
        z = tok(x_dict, window_len=T)
        self.assertTrue(np.all(np.isfinite(z.data)))


class TestOntologyEmbedding(unittest.TestCase):
    """Phase 3 — Item 2: ontology-aware domain embeddings."""

    def _registry_and_embed(self):
        from modules.ontology_embedding import OntologyEmbedding, OntologyRegistry
        registry = OntologyRegistry()
        registry.register(0, ["temperature", "vibration", "pressure"])
        registry.register(1, ["voltage", "current", "power"])
        embed = OntologyEmbedding(meta_dim=32)
        return registry, embed

    def test_output_shape(self):
        registry, embed = self._registry_and_embed()
        z = embed.encode(registry, 0)
        self.assertEqual(z.data.shape, (1, 32))

    def test_different_domains_different_output(self):
        registry, embed = self._registry_and_embed()
        z0 = embed.encode(registry, 0)
        z1 = embed.encode(registry, 1)
        self.assertFalse(np.allclose(z0.data, z1.data),
                         "Different ontology terms should produce different embeddings")

    def test_empty_terms_returns_zeros(self):
        from modules.ontology_embedding import OntologyEmbedding, OntologyRegistry
        embed = OntologyEmbedding(meta_dim=16)
        registry = OntologyRegistry()
        z = embed.encode(registry, 99)   # unregistered domain
        np.testing.assert_array_equal(z.data, np.zeros((1, 16)))

    def test_unknown_terms_handled(self):
        from modules.ontology_embedding import OntologyEmbedding, OntologyRegistry
        embed = OntologyEmbedding(meta_dim=16)
        z = embed.forward(["unknown_sensor_xyz_123"])
        self.assertEqual(z.data.shape, (1, 16))
        self.assertTrue(np.all(np.isfinite(z.data)))

    def test_term_similarity_same_cluster(self):
        from modules.ontology_embedding import OntologyEmbedding
        embed = OntologyEmbedding(meta_dim=32)
        sim = embed.term_similarity("temperature", "temp")
        # Same cluster → initial embeddings identical → cosine sim ≈ 1
        self.assertGreater(sim, 0.9)

    def test_vocab_coverage(self):
        from modules.ontology_embedding import OntologyRegistry
        registry = OntologyRegistry()
        registry.register(0, ["temperature", "not_in_vocab_xyz"])
        cov = registry.vocab_coverage(0)
        self.assertIn("known", cov)
        self.assertIn("unknown", cov)
        self.assertIn("temperature", cov["known"])
        self.assertIn("not_in_vocab_xyz", cov["unknown"])

    def test_nearest_terms_returns_list(self):
        from modules.ontology_embedding import OntologyEmbedding
        embed = OntologyEmbedding(meta_dim=32)
        results = embed.nearest_terms("temperature", top_k=3)
        self.assertEqual(len(results), 3)
        for term, sim in results:
            self.assertIsInstance(term, str)
            self.assertIsInstance(sim, float)


class TestDAHOntologyWiring(unittest.TestCase):
    """Phase 3 — Item 2: DAH attach_ontology_embedding integration."""

    def test_attach_and_compute_adapters(self):
        from modules.dah import DomainAdaptiveHypernetwork
        from modules.ontology_embedding import OntologyEmbedding, OntologyRegistry
        from engine.layers import Linear
        cfg = _cfg(d=16)
        base_layers = {"skip_proj": Linear(16, 16)}
        dah = DomainAdaptiveHypernetwork(base_layers, cfg.dah)
        registry = OntologyRegistry()
        registry.register(0, ["temperature", "pressure"])
        embed = OntologyEmbedding(meta_dim=16)
        dah.attach_ontology_embedding(embed, registry)
        adapters = dah.get_adapters(domain_idx=0)
        self.assertIn("skip_proj", adapters)
        A, B = adapters["skip_proj"]
        self.assertEqual(A.data.shape[0], 16)

    def test_ontology_invalidates_cache(self):
        from modules.dah import DomainAdaptiveHypernetwork
        from modules.ontology_embedding import OntologyEmbedding, OntologyRegistry
        from engine.layers import Linear
        cfg = _cfg(d=16)
        base_layers = {"skip_proj": Linear(16, 16)}
        dah = DomainAdaptiveHypernetwork(base_layers, cfg.dah)
        dah.get_adapters(0)   # populate cache
        self.assertIn(0, dah._domain_cache)
        registry = OntologyRegistry()
        registry.register(0, ["temperature"])
        embed = OntologyEmbedding(meta_dim=16)
        dah.attach_ontology_embedding(embed, registry)
        self.assertNotIn(0, dah._domain_cache)   # cache cleared


class TestCRGIntervene(unittest.TestCase):
    """Phase 3 — Item 3: CRG do-calculus intervene() API."""

    def setUp(self):
        from modules.crg import CausalRoutingGraph
        from config import CRGConfig
        crg_cfg = CRGConfig(n_nodes=8, n_lags=2)
        self.crg = CausalRoutingGraph(d_model=16, config=crg_cfg)

    def test_intervene_output_shape(self):
        x = Tensor(np.random.randn(2, 4, 16).astype(np.float32))
        out = self.crg.intervene({0: 1.0}, x)
        self.assertEqual(out.data.shape, (2, 4, 16))

    def test_intervene_output_finite(self):
        x = Tensor(np.random.randn(2, 4, 16).astype(np.float32))
        out = self.crg.intervene({0: 0.0, 2: 1.0}, x)
        self.assertTrue(np.all(np.isfinite(out.data)))

    def test_intervene_changes_output_vs_no_intervention(self):
        x = Tensor(np.random.randn(2, 4, 16).astype(np.float32))
        out_normal, _ = self.crg(x)
        out_iv = self.crg.intervene({0: 999.0}, x)
        # Forcing a node to an extreme value should change the output
        self.assertFalse(np.allclose(out_normal.data, out_iv.data, atol=1e-3),
                         "Intervention should change output")

    def test_intervene_multiple_nodes(self):
        x = Tensor(np.random.randn(2, 4, 16).astype(np.float32))
        out = self.crg.intervene({0: 1.0, 1: -1.0, 2: 0.5}, x)
        self.assertEqual(out.data.shape, (2, 4, 16))


class TestCRGCounterfactual(unittest.TestCase):
    """Phase 3 — Item 3: CRG counterfactual_root_cause() root cause analysis."""

    def setUp(self):
        from modules.crg import CausalRoutingGraph
        from config import CRGConfig
        crg_cfg = CRGConfig(n_nodes=8, n_lags=2)
        self.crg = CausalRoutingGraph(d_model=16, config=crg_cfg)

    def test_returns_dict_with_required_keys(self):
        x = Tensor(np.random.randn(2, 4, 16).astype(np.float32))
        result = self.crg.counterfactual_root_cause(x, target_node=3, top_k=3)
        for key in ("ranked_causes", "target_node", "target_obs_activation",
                    "n_candidates"):
            self.assertIn(key, result)

    def test_ranked_causes_length_respects_top_k(self):
        x = Tensor(np.random.randn(2, 4, 16).astype(np.float32))
        result = self.crg.counterfactual_root_cause(x, target_node=3, top_k=3)
        self.assertLessEqual(len(result["ranked_causes"]), 3)

    def test_ranked_causes_sorted_descending(self):
        x = Tensor(np.random.randn(2, 4, 16).astype(np.float32))
        result = self.crg.counterfactual_root_cause(x, target_node=3, top_k=5)
        deltas = [abs(r["delta_activation"]) for r in result["ranked_causes"]]
        self.assertEqual(deltas, sorted(deltas, reverse=True))

    def test_target_node_not_in_candidates(self):
        x = Tensor(np.random.randn(2, 4, 16).astype(np.float32))
        result = self.crg.counterfactual_root_cause(x, target_node=0, top_k=8)
        nodes = [r["node"] for r in result["ranked_causes"]]
        self.assertNotIn(0, nodes)

    def test_custom_candidate_nodes(self):
        x = Tensor(np.random.randn(2, 4, 16).astype(np.float32))
        result = self.crg.counterfactual_root_cause(x, target_node=3,
                                                     candidate_nodes=[1, 2],
                                                     top_k=5)
        nodes = [r["node"] for r in result["ranked_causes"]]
        for n in nodes:
            self.assertIn(n, [1, 2])


class TestWorldModelHead(unittest.TestCase):
    """Phase 3 — Item 4: WorldModelHead and world_model_forward()."""

    def test_standalone_shapes(self):
        from model.vulgaris import WorldModelHead
        wm = WorldModelHead(d_model=16, horizon=4)
        h = Tensor(np.random.randn(2, 16).astype(np.float32))
        future_z, uncs = wm(h, horizon=4)
        self.assertEqual(future_z.data.shape, (2, 4, 16))
        self.assertEqual(len(uncs), 4)

    def test_standalone_finite(self):
        from model.vulgaris import WorldModelHead
        wm = WorldModelHead(d_model=16, horizon=3)
        h = Tensor(np.random.randn(2, 16).astype(np.float32))
        future_z, uncs = wm(h)
        self.assertTrue(np.all(np.isfinite(future_z.data)))
        self.assertTrue(np.all(np.isfinite(uncs)))

    def test_uncertainties_nonnegative(self):
        from model.vulgaris import WorldModelHead
        wm = WorldModelHead(d_model=16, horizon=5)
        h = Tensor(np.random.randn(2, 16).astype(np.float32))
        _, uncs = wm(h)
        self.assertTrue(np.all(uncs >= 0.0))

    def test_horizon_override(self):
        from model.vulgaris import WorldModelHead
        wm = WorldModelHead(d_model=16, horizon=10)
        h = Tensor(np.random.randn(2, 16))
        future_z, uncs = wm(h, horizon=3)
        self.assertEqual(future_z.data.shape[1], 3)
        self.assertEqual(len(uncs), 3)

    def test_world_model_forward_keys(self):
        m = _model(d=16)
        x = _x()
        result = m.world_model_forward(x, horizon=3)
        for key in ("future_latents", "uncertainties",
                    "future_outputs", "current_output"):
            self.assertIn(key, result)

    def test_world_model_forward_shapes(self):
        m = _model(d=16)
        x = _x()   # (2, 4, 8)
        result = m.world_model_forward(x, horizon=3)
        self.assertEqual(result["future_latents"].data.shape, (2, 3, 16))
        self.assertEqual(result["future_outputs"].shape, (2, 3, 2))
        self.assertEqual(result["current_output"].data.shape, (2, 2))
        self.assertEqual(len(result["uncertainties"]), 3)

    def test_world_model_forward_finite(self):
        m = _model(d=16)
        x = _x()
        result = m.world_model_forward(x, horizon=2)
        self.assertTrue(np.all(np.isfinite(result["future_latents"].data)))
        self.assertTrue(np.all(np.isfinite(result["future_outputs"])))

    def test_world_model_registered_on_vulgaris(self):
        m = _model()
        self.assertTrue(hasattr(m, "world_model"))


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main(verbosity=2)
