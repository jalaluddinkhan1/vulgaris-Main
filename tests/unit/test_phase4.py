"""
Tests for Phase 4 additions:
  - Regime Mixture Core (RMC)
  - Active Learning
  - Distillation
  - Speculative Rollout
"""

import sys, os, unittest, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
from engine.tensor import Tensor
from config import ModelConfig
from model.vulgaris import Vulgaris


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _model(D=32, C=4, out=2):
    cfg = ModelConfig(input_dim=C, output_dim=out, n_classes=0)
    cfg.ase.latent_dim = D
    cfg.ase.n_filters  = 4
    cfg.ase.n_scales   = 2
    cfg.hmb.embed_dim  = D
    cfg.hmb.compress_dim = D // 4
    cfg.dah.meta_dim   = D
    return Vulgaris(cfg)


# ──────────────────────────────────────────────────────────────────────────────
# Phase 4 — Item 1: Regime Mixture Core
# ──────────────────────────────────────────────────────────────────────────────

class TestRegimeMixtureCore(unittest.TestCase):

    def setUp(self):
        from modules.rmc import RegimeMixtureCore
        self.RMC = RegimeMixtureCore
        np.random.seed(0)

    def test_output_shape(self):
        rmc = self.RMC(d_model=32, n_experts=4)
        z = Tensor(np.random.randn(2, 8, 32).astype(np.float64))
        z_out, balance = rmc(z)
        self.assertEqual(z_out.data.shape, (2, 8, 32))

    def test_balance_loss_scalar(self):
        rmc = self.RMC(d_model=32, n_experts=4)
        z = Tensor(np.random.randn(2, 8, 32).astype(np.float64))
        _, balance = rmc(z)
        self.assertEqual(balance.data.shape, (1, 1))

    def test_balance_loss_nonnegative(self):
        rmc = self.RMC(d_model=32, n_experts=4)
        z = Tensor(np.random.randn(2, 8, 32).astype(np.float64))
        _, balance = rmc(z)
        self.assertGreaterEqual(float(balance.data.sum()), 0.0)

    def test_balance_loss_finite(self):
        rmc = self.RMC(d_model=32, n_experts=4)
        z = Tensor(np.random.randn(2, 8, 32).astype(np.float64))
        _, balance = rmc(z)
        self.assertTrue(np.isfinite(balance.data).all())

    def test_backward_runs(self):
        rmc = self.RMC(d_model=16, n_experts=2)
        z = Tensor(np.random.randn(2, 4, 16).astype(np.float64), requires_grad=True)
        z_out, balance = rmc(z)
        # Propagate gradient through the mixture output (not just balance)
        z_out.backward(np.ones_like(z_out.data))
        # gate_proj should have gradients from the mixture computation
        self.assertIsNotNone(rmc.gate_proj.weight.grad)

    def test_n_experts_configurable(self):
        rmc = self.RMC(d_model=16, n_experts=6)
        z = Tensor(np.random.randn(1, 4, 16).astype(np.float64))
        z_out, _ = rmc(z)
        self.assertEqual(z_out.data.shape, (1, 4, 16))

    def test_regime_assignments_shape(self):
        rmc = self.RMC(d_model=16, n_experts=4)
        z = Tensor(np.random.randn(3, 5, 16).astype(np.float64))
        assignments = rmc.regime_assignments(z)
        self.assertEqual(assignments.shape, (3, 5))

    def test_regime_assignments_valid_range(self):
        rmc = self.RMC(d_model=16, n_experts=4)
        z = Tensor(np.random.randn(2, 6, 16).astype(np.float64))
        assignments = rmc.regime_assignments(z)
        self.assertTrue(np.all(assignments >= 0))
        self.assertTrue(np.all(assignments < 4))

    def test_rmc_registered_on_vulgaris(self):
        m = _model()
        self.assertTrue(hasattr(m, 'rmc'))

    def test_rmc_balance_in_aux(self):
        m = _model()
        x = Tensor(np.random.randn(2, 4, 16).astype(np.float32))
        _, aux = m(x)
        self.assertIn('rmc_balance_loss', aux)
        self.assertTrue(np.isfinite(aux['rmc_balance_loss']))

    def test_rmc_in_loss_components(self):
        from training.loss import VulgarisLoss
        cfg = ModelConfig(input_dim=4, output_dim=2, n_classes=0)
        loss_fn = VulgarisLoss(cfg)
        m = _model()
        x = Tensor(np.random.randn(2, 4, 16).astype(np.float32))
        y = Tensor(np.random.randn(2, 2).astype(np.float32))
        out, aux = m(x)
        total, comps = loss_fn(out, y, aux)
        self.assertIn('rmc_balance_loss', comps)


# ──────────────────────────────────────────────────────────────────────────────
# Phase 4 — Item 2: Active Learning
# ──────────────────────────────────────────────────────────────────────────────

class TestActiveLearner(unittest.TestCase):

    def setUp(self):
        from training.active_learning import ActiveLearner
        self.ActiveLearner = ActiveLearner
        np.random.seed(1)
        self.model = _model()

    def test_score_returns_correct_shape(self):
        learner = self.ActiveLearner(self.model, strategy='uncertainty')
        pool = np.random.randn(10, 4, 16).astype(np.float32)
        scores = learner.score(pool)
        self.assertEqual(scores.shape, (10,))

    def test_score_values_nonnegative(self):
        learner = self.ActiveLearner(self.model, strategy='uncertainty')
        pool = np.random.randn(10, 4, 16).astype(np.float32)
        scores = learner.score(pool)
        self.assertTrue(np.all(scores >= 0))

    def test_query_returns_n_query_indices(self):
        learner = self.ActiveLearner(self.model, strategy='uncertainty')
        pool = np.random.randn(20, 4, 16).astype(np.float32)
        idx = learner.query(pool, n_query=5)
        self.assertEqual(len(idx), 5)

    def test_query_indices_in_range(self):
        learner = self.ActiveLearner(self.model)
        pool = np.random.randn(20, 4, 16).astype(np.float32)
        idx = learner.query(pool, n_query=5)
        self.assertTrue(np.all(idx >= 0))
        self.assertTrue(np.all(idx < 20))

    def test_mark_labeled_excludes_from_next_query(self):
        learner = self.ActiveLearner(self.model, strategy='uncertainty')
        pool = np.random.randn(20, 4, 16).astype(np.float32)
        idx1 = learner.query(pool, n_query=5)
        learner.mark_labeled(idx1, pool_size=20)
        idx2 = learner.query(pool, n_query=5, exclude_labeled=True)
        self.assertEqual(len(set(idx1.tolist()) & set(idx2.tolist())), 0)

    def test_entropy_strategy(self):
        learner = self.ActiveLearner(self.model, strategy='entropy')
        pool = np.random.randn(8, 4, 16).astype(np.float32)
        scores = learner.score(pool)
        self.assertEqual(scores.shape, (8,))
        self.assertTrue(np.isfinite(scores).all())

    def test_margin_strategy(self):
        learner = self.ActiveLearner(self.model, strategy='margin')
        pool = np.random.randn(8, 4, 16).astype(np.float32)
        scores = learner.score(pool)
        self.assertTrue(np.isfinite(scores).all())

    def test_random_strategy(self):
        learner = self.ActiveLearner(self.model, strategy='random')
        pool = np.random.randn(8, 4, 16).astype(np.float32)
        scores = learner.score(pool)
        self.assertEqual(scores.shape, (8,))

    def test_labeled_count_after_mark(self):
        learner = self.ActiveLearner(self.model)
        learner.mark_labeled(np.array([0, 1, 2]), pool_size=10)
        self.assertEqual(learner.labeled_count(), 3)

    def test_reset_clears_state(self):
        learner = self.ActiveLearner(self.model)
        learner.mark_labeled(np.array([0, 1]), pool_size=10)
        learner.reset()
        self.assertEqual(learner.labeled_count(), 0)

    def test_stats_keys(self):
        learner = self.ActiveLearner(self.model, strategy='uncertainty')
        s = learner.stats()
        for key in ('strategy', 'n_mc', 'labeled_count', 'query_rounds'):
            self.assertIn(key, s)

    def test_pipeline_active_step(self):
        from training.pipeline import TrainingPipeline
        from training.loss import VulgarisLoss
        from training.optimizer import SpectralAdamW, CosineSchedule
        cfg = ModelConfig(input_dim=4, output_dim=2, n_classes=0)
        m = _model()
        opt = SpectralAdamW(m.parameters(), lr=1e-4)
        sch = CosineSchedule(opt, warmup_steps=10, max_steps=100, min_lr=1e-6)
        loss_fn = VulgarisLoss(cfg)
        pipeline = TrainingPipeline(m, cfg, loss_fn, opt, sch)
        pool = np.random.randn(15, 4, 16).astype(np.float32)
        x_lab = np.random.randn(2, 4, 16).astype(np.float32)
        y_lab = np.random.randn(2, 2).astype(np.float32)
        result = pipeline.active_step(pool, x_lab, y_lab, n_query=5)
        self.assertIn('query_indices', result)
        self.assertEqual(result['n_queried'], 5)


# ──────────────────────────────────────────────────────────────────────────────
# Phase 4 — Item 3: Distillation
# ──────────────────────────────────────────────────────────────────────────────

class TestDistillationLoss(unittest.TestCase):

    def setUp(self):
        from training.distillation import DistillationLoss
        self.KD = DistillationLoss
        np.random.seed(2)

    def test_soft_kl_positive(self):
        kd = self.KD(temperature=4.0)
        t_logits = np.random.randn(4, 8)
        s_logits = Tensor(np.random.randn(4, 8).astype(np.float64), requires_grad=True)
        loss = kd.soft_kl_tensor(s_logits, t_logits)
        self.assertGreater(float(loss.data.sum()), 0)

    def test_soft_kl_finite(self):
        kd = self.KD(temperature=4.0)
        t_logits = np.random.randn(4, 4)
        s_logits = Tensor(np.random.randn(4, 4).astype(np.float64), requires_grad=True)
        loss = kd.soft_kl_tensor(s_logits, t_logits)
        self.assertTrue(np.isfinite(loss.data).all())

    def test_soft_kl_backward_populates_grad(self):
        kd = self.KD(temperature=2.0)
        t_logits = np.random.randn(4, 4)
        s_logits = Tensor(np.random.randn(4, 4).astype(np.float64), requires_grad=True)
        loss = kd.soft_kl_tensor(s_logits, t_logits)
        loss.backward()
        self.assertIsNotNone(s_logits.grad)
        self.assertTrue(np.isfinite(s_logits.grad).all())

    def test_identical_logits_lower_loss(self):
        kd = self.KD(temperature=4.0)
        logits = np.random.randn(4, 4)
        same  = Tensor(logits.copy().astype(np.float64), requires_grad=True)
        diff  = Tensor((logits + np.random.randn(*logits.shape)).astype(np.float64),
                       requires_grad=True)
        l_same = kd.soft_kl_tensor(same, logits)
        l_diff = kd.soft_kl_tensor(diff, logits)
        self.assertLess(float(l_same.data.sum()), float(l_diff.data.sum()))

    def test_hint_loss_finite(self):
        kd = self.KD()
        s = Tensor(np.random.randn(4, 8).astype(np.float64), requires_grad=True)
        t = np.random.randn(4, 8)
        loss = kd.hint_loss_tensor(s, t)
        self.assertTrue(np.isfinite(loss.data).all())

    def test_hint_loss_zero_when_identical(self):
        kd = self.KD()
        arr = np.random.randn(4, 8)
        s = Tensor(arr.copy().astype(np.float64), requires_grad=True)
        loss = kd.hint_loss_tensor(s, arr)
        self.assertAlmostEqual(float(loss.data.sum()), 0.0, places=8)

    def test_hard_loss_finite(self):
        kd = self.KD()
        out = Tensor(np.random.randn(4, 2).astype(np.float64), requires_grad=True)
        y   = Tensor(np.random.randn(4, 2).astype(np.float32))
        loss = kd.hard_loss_tensor(out, y)
        self.assertTrue(np.isfinite(loss.data).all())

    def test_standalone_scalar_kl(self):
        kd = self.KD(temperature=3.0)
        t = np.random.randn(8, 6)
        s = np.random.randn(8, 6)
        val = kd.soft_kl(s, t)
        self.assertIsInstance(val, float)
        self.assertGreater(val, 0)


class TestDistillationTrainer(unittest.TestCase):

    def setUp(self):
        from training.distillation import DistillationTrainer
        from training.optimizer import SpectralAdamW
        self.DT = DistillationTrainer
        self.Opt = SpectralAdamW
        np.random.seed(3)
        self.teacher = _model(D=32)
        self.student = _model(D=32)

    def test_distil_step_returns_metrics(self):
        opt = self.Opt(self.student.parameters(), lr=1e-4)
        trainer = self.DT(self.teacher, self.student, opt)
        x = np.random.randn(2, 4, 16).astype(np.float32)
        y = np.random.randn(2, 2).astype(np.float32)
        _, metrics = trainer.distil_step(x, y)
        for key in ('total_loss', 'hard_loss', 'soft_loss', 'hint_loss', 'step'):
            self.assertIn(key, metrics)

    def test_distil_step_finite_loss(self):
        opt = self.Opt(self.student.parameters(), lr=1e-4)
        trainer = self.DT(self.teacher, self.student, opt)
        x = np.random.randn(2, 4, 16).astype(np.float32)
        y = np.random.randn(2, 2).astype(np.float32)
        _, metrics = trainer.distil_step(x, y)
        self.assertTrue(np.isfinite(metrics['total_loss']))

    def test_teacher_params_frozen(self):
        opt = self.Opt(self.student.parameters(), lr=1e-4)
        trainer = self.DT(self.teacher, self.student, opt)
        for p in self.teacher.parameters():
            self.assertFalse(p.requires_grad)

    def test_stats_compression_ratio(self):
        opt = self.Opt(self.student.parameters(), lr=1e-4)
        trainer = self.DT(self.teacher, self.student, opt)
        s = trainer.stats()
        self.assertIn('compression_ratio', s)
        self.assertGreater(s['compression_ratio'], 0)

    def test_step_count_increments(self):
        opt = self.Opt(self.student.parameters(), lr=1e-4)
        trainer = self.DT(self.teacher, self.student, opt)
        x = np.random.randn(2, 4, 16).astype(np.float32)
        y = np.random.randn(2, 2).astype(np.float32)
        trainer.distil_step(x, y)
        trainer.distil_step(x, y)
        self.assertEqual(trainer._step_count, 2)


# ──────────────────────────────────────────────────────────────────────────────
# Phase 4 — Item 4: Speculative Rollout
# ──────────────────────────────────────────────────────────────────────────────

class TestSpeculativeRollout(unittest.TestCase):

    def setUp(self):
        from inference.speculative import SpeculativeRollout
        self.SR = SpeculativeRollout
        np.random.seed(4)
        self.model = _model(D=32)

    def test_rollout_output_shape(self):
        sr = self.SR(self.model, gamma=3, accept_threshold=1e6)
        ctx = np.random.randn(1, 4, 16).astype(np.float32)
        result = sr.rollout(ctx, horizon=6)
        self.assertEqual(result['outputs'].shape[0], 6)

    def test_rollout_accepted_mask_shape(self):
        sr = self.SR(self.model, gamma=3, accept_threshold=1e6)
        ctx = np.random.randn(1, 4, 16).astype(np.float32)
        result = sr.rollout(ctx, horizon=5)
        self.assertEqual(result['accepted_mask'].shape, (5,))

    def test_rollout_all_accepted_with_high_threshold(self):
        sr = self.SR(self.model, gamma=4, accept_threshold=1e9)
        ctx = np.random.randn(1, 4, 16).astype(np.float32)
        result = sr.rollout(ctx, horizon=4)
        self.assertTrue(result['accepted_mask'].all())

    def test_rollout_outputs_finite(self):
        sr = self.SR(self.model, gamma=2, accept_threshold=1e6)
        ctx = np.random.randn(1, 4, 16).astype(np.float32)
        result = sr.rollout(ctx, horizon=4)
        self.assertTrue(np.isfinite(result['outputs']).all())

    def test_rollout_with_return_latents(self):
        sr = self.SR(self.model, gamma=2, accept_threshold=1e6)
        ctx = np.random.randn(1, 4, 16).astype(np.float32)
        result = sr.rollout(ctx, horizon=4, return_latents=True)
        self.assertIn('latents', result)
        self.assertEqual(result['latents'].shape[0], 4)

    def test_stats_keys(self):
        sr = self.SR(self.model, gamma=2)
        stats = sr.stats()
        for key in ('gamma', 'accept_threshold', 'total_steps',
                    'accepted_steps', 'full_model_calls', 'acceptance_rate',
                    'effective_speedup'):
            self.assertIn(key, stats)

    def test_reset_stats(self):
        sr = self.SR(self.model, gamma=2, accept_threshold=1e6)
        ctx = np.random.randn(1, 4, 16).astype(np.float32)
        sr.rollout(ctx, horizon=4)
        sr.reset_stats()
        self.assertEqual(sr.stats()['total_steps'], 0)
        self.assertEqual(sr.stats()['full_model_calls'], 0)

    def test_effective_speedup_positive(self):
        sr = self.SR(self.model, gamma=3, accept_threshold=1e6)
        ctx = np.random.randn(1, 4, 16).astype(np.float32)
        sr.rollout(ctx, horizon=6)
        self.assertGreater(sr.stats()['effective_speedup'], 0)

    def test_rollout_single_step(self):
        sr = self.SR(self.model, gamma=2, accept_threshold=1e6)
        x_last = np.random.randn(1, 4).astype(np.float32)
        state = self.model.init_state(batch_size=1)
        # rollout_single expects (B, C) — single timestep input
        pred, accepted = sr.rollout_single(x_last, state)
        self.assertIsNotNone(pred)
        self.assertIsInstance(accepted, bool)


if __name__ == '__main__':
    unittest.main()
