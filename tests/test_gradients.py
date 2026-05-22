# -*- coding: utf-8 -*-
"""End-to-end gradient flow verification for VULGARIS.

Tests that gradients propagate correctly from each loss signal back through
the full model using both numerical (finite-difference) and structural
(grad is not None) checks.
"""
import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Warm the module cache first to avoid circular import
import vulgaris  # noqa: F401
from engine.tensor import Tensor
from config import ModelConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tiny_config():
    """Smallest valid config for fast gradient checks."""
    cfg = ModelConfig(input_dim=4, output_dim=1, n_classes=2)
    cfg.ase.latent_dim = 16
    cfg.ase.n_filters = 4
    cfg.ase.n_scales = 2
    cfg.sssr.state_dim = 8
    cfg.sssr.d_inner = 32
    cfg.crg.n_nodes = 4
    cfg.hmb.n_slots = 4
    cfg.hmb.latent_dim = 16
    return cfg


def _model():
    from model.vulgaris import Vulgaris
    return Vulgaris(_tiny_config())


def _finite_diff_grad(fn, tensor: Tensor, eps: float = 1e-4) -> np.ndarray:
    """
    Compute numerical gradient of scalar fn(tensor) w.r.t. tensor.data
    via central differences.
    """
    grad = np.zeros_like(tensor.data)
    it = np.nditer(tensor.data, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        orig = float(tensor.data[idx])

        tensor.data[idx] = orig + eps
        fp = float(fn())

        tensor.data[idx] = orig - eps
        fm = float(fn())

        grad[idx] = (fp - fm) / (2 * eps)
        tensor.data[idx] = orig
        it.iternext()
    return grad


def _relative_error(analytical: np.ndarray, numerical: np.ndarray) -> float:
    diff = np.abs(analytical - numerical)
    denom = np.abs(analytical) + np.abs(numerical) + 1e-8
    return float((diff / denom).max())


# ---------------------------------------------------------------------------
# Structural gradient flow tests
# ---------------------------------------------------------------------------

class TestGradientFlow(unittest.TestCase):
    """Verify that loss.backward() populates gradients in model parameters."""

    def setUp(self):
        np.random.seed(0)
        self.model = _model()
        self.B, self.C, self.T = 2, 4, 24

    def _run_forward_backward(self, x_np):
        """Forward pass + MSE loss on output, return loss tensor."""
        x = Tensor(x_np.astype(np.float32), requires_grad=False)
        out, _ = self.model(x, domain_idx=0)    # (B, output_dim)
        target = np.zeros_like(out.data)
        diff = out.data - target
        loss_scalar = float((diff ** 2).mean())
        loss = Tensor(
            np.array([[loss_scalar]], dtype=np.float32),
            requires_grad=out.requires_grad,
            _children=(out,),
            _op="test_mse"
        )
        _out = out; _diff = diff; _denom = float(diff.size)

        def _back():
            if _out.requires_grad and loss.grad is not None:
                _out.grad = (
                    _out.grad + 2.0 * _diff / _denom
                    if _out.grad is not None
                    else (2.0 * _diff / _denom).astype(np.float32)
                )

        loss._backward = _back
        loss.backward()
        return loss

    def test_ase_params_receive_grad(self):
        """ASE parameters must have non-None gradients after backward."""
        x_np = np.random.randn(self.B, self.C, self.T).astype(np.float32)
        self._run_forward_backward(x_np)
        ase_params = list(self.model.ase.parameters())
        self.assertGreater(len(ase_params), 0)
        grads_populated = sum(1 for p in ase_params if p.grad is not None)
        self.assertGreater(grads_populated, 0,
            "No ASE parameter received a gradient — gradient is cut before ASE.")

    def test_sssr_params_receive_grad(self):
        """SSSR parameters must receive gradients."""
        x_np = np.random.randn(self.B, self.C, self.T).astype(np.float32)
        self._run_forward_backward(x_np)
        params = list(self.model.sssr.parameters())
        self.assertGreater(len(params), 0)
        grads = sum(1 for p in params if p.grad is not None)
        self.assertGreater(grads, 0,
            "No SSSR parameter received a gradient.")

    def test_crg_params_receive_grad(self):
        """CRG/DAG parameters must receive gradients."""
        x_np = np.random.randn(self.B, self.C, self.T).astype(np.float32)
        self._run_forward_backward(x_np)
        params = list(self.model.crg.parameters())
        self.assertGreater(len(params), 0)
        grads = sum(1 for p in params if p.grad is not None)
        self.assertGreater(grads, 0,
            "No CRG parameter received a gradient.")

    def test_no_nan_grad(self):
        """Gradients must be finite — no NaN or Inf after backward."""
        x_np = np.random.randn(self.B, self.C, self.T).astype(np.float32)
        self._run_forward_backward(x_np)
        for name, p in self.model.named_parameters():
            if p.grad is not None:
                self.assertFalse(
                    np.any(np.isnan(p.grad)) or np.any(np.isinf(p.grad)),
                    f"NaN/Inf gradient in parameter '{name}'"
                )

    def test_multitask_head_grad_flow(self):
        """MultitaskHead mean-pool fix: gradient must reach pre-pool latent."""
        from modules.multitask_head import MultitaskHead
        d_model = 16
        head = MultitaskHead(d_model=d_model, task_dims={"y": 1})
        B, T = 2, 8
        z_np = np.random.randn(B, T, d_model).astype(np.float32)
        z = Tensor(z_np, requires_grad=True)
        out = head(z)["y"]
        target = np.zeros_like(out.data)
        diff = out.data - target
        loss = Tensor(
            np.array([[float((diff**2).mean())]], dtype=np.float32),
            requires_grad=out.requires_grad,
            _children=(out,), _op="mth_test"
        )
        _o = out; _d = diff
        def _b():
            if _o.requires_grad and loss.grad is not None:
                _o.grad = (2.0 * _d / _d.size).astype(np.float32)
        loss._backward = _b
        loss.backward()
        self.assertIsNotNone(z.grad,
            "Gradient did not reach z through MultitaskHead mean-pool.")
        self.assertFalse(np.all(z.grad == 0),
            "z.grad is all zeros — mean-pool backward may be broken.")


# ---------------------------------------------------------------------------
# Numerical gradient checks (finite differences vs autograd)
# ---------------------------------------------------------------------------

class TestNumericalGradients(unittest.TestCase):
    """Compare autograd gradients against finite-difference estimates."""

    def _check(self, fn_build, param_tensor: Tensor, eps=1e-4, tol=5e-2):
        """
        fn_build: callable() -> scalar Tensor using param_tensor.
        Runs autograd backward once, then computes FD gradient.
        Asserts relative error < tol (lenient since float32).
        """
        # Analytical
        param_tensor.grad = None
        loss = fn_build()
        loss.backward()
        analytical = param_tensor.grad.copy() if param_tensor.grad is not None else np.zeros_like(param_tensor.data)

        # Numerical — rebuild graph each call to avoid stale backward state
        def _f():
            param_tensor.grad = None
            t = fn_build()
            return float(t.data.sum())

        numerical = _finite_diff_grad(_f, param_tensor, eps=eps)
        rel_err = _relative_error(analytical, numerical)
        self.assertLess(rel_err, tol,
            f"Gradient check failed: max relative error {rel_err:.4f} >= {tol}")

    def test_linear_layer_weight_grad(self):
        """Linear layer weight gradient vs finite differences."""
        from engine.layers import Linear
        layer = Linear(4, 3)
        x_np = np.random.randn(2, 4).astype(np.float32)

        def _fn():
            x = Tensor(x_np, requires_grad=False)
            out = layer(x)
            return out.reshape(1, -1)

        # Only check weight (bias check is identical structure)
        self._check(_fn, layer.weight)

    def test_rms_norm_grad(self):
        """RMSNorm gradient vs finite differences."""
        from engine.layers import RMSNorm
        norm = RMSNorm(8)
        x_np = np.random.randn(3, 8).astype(np.float32)
        x_t = Tensor(x_np, requires_grad=True)

        def _fn():
            out = norm(x_t)
            return out.reshape(1, -1)

        self._check(_fn, x_t)

    def test_causal_attention_query_proj_grad(self):
        """CausalAttention query projection weight gradient."""
        from engine.layers import CausalAttention
        d = 8
        attn = CausalAttention(d_model=d, n_heads=2)
        x_np = np.random.randn(1, 4, d).astype(np.float32) * 0.1

        def _fn():
            x = Tensor(x_np, requires_grad=False)
            out = attn(x)
            return out.reshape(1, -1)

        self._check(_fn, attn.W_q.weight, tol=0.1)

    def test_masked_reconstruction_mse_grad(self):
        """Masked MSE loss backward vs finite differences on recon output."""
        B, T, C = 2, 6, 3
        recon_np = np.random.randn(B, T, C).astype(np.float32)
        target_np = np.random.randn(B, T, C).astype(np.float32)
        mask = np.zeros((B, T), dtype=bool)
        mask[:, :3] = True
        mask_exp = mask[:, :, None].astype(np.float32)
        n_masked = int(mask.sum())
        denom = float(n_masked * C)

        recon_t = Tensor(recon_np.copy(), requires_grad=True)

        def _fn():
            diff = recon_t.data - target_np
            loss_s = float((diff**2 * mask_exp).sum()) / denom
            loss = Tensor(
                np.array([[loss_s]], dtype=np.float32),
                requires_grad=recon_t.requires_grad,
                _children=(recon_t,), _op="test_mmse"
            )
            _r = recon_t; _d = diff; _me = mask_exp

            def _b():
                if _r.requires_grad and loss.grad is not None:
                    g = float(loss.grad.sum())
                    grad = (2.0 * _d * _me / denom * g).astype(np.float32)
                    _r.grad = _r.grad + grad if _r.grad is not None else grad

            loss._backward = _b
            return loss

        self._check(_fn, recon_t)


# ---------------------------------------------------------------------------
# ICL gradient flow
# ---------------------------------------------------------------------------

class TestICLGradients(unittest.TestCase):
    def test_icl_gate_receives_grad(self):
        """ICL gate parameter must receive gradients when context is provided."""
        from modules.icl import InContextLearning
        d, out_dim, B, T, n_ctx = 16, 1, 2, 8, 3
        icl = InContextLearning(d_model=d, output_dim=out_dim, n_heads=2)

        # Build fake context latents and labels
        ctx_latents = [Tensor(np.random.randn(B, 4, d).astype(np.float32), requires_grad=False)
                       for _ in range(n_ctx)]
        ctx_labels  = [Tensor(np.random.randn(B, out_dim).astype(np.float32), requires_grad=False)
                       for _ in range(n_ctx)]
        z = Tensor(np.random.randn(B, T, d).astype(np.float32), requires_grad=True)

        ctx_stack = icl.encode_context(ctx_latents, ctx_labels)
        out = icl(z, ctx_stack)           # (B, T, d)

        loss_s = float((out.data ** 2).mean())
        loss = Tensor(np.array([[loss_s]], dtype=np.float32),
                      requires_grad=out.requires_grad,
                      _children=(out,), _op="icl_test")
        _o = out

        def _b():
            if _o.requires_grad and loss.grad is not None:
                _o.grad = (2.0 * _o.data / _o.data.size).astype(np.float32)

        loss._backward = _b
        loss.backward()

        gate = icl.adapter.gate
        self.assertIsNotNone(gate.grad, "ICL gate did not receive a gradient.")
        self.assertFalse(np.all(gate.grad == 0), "ICL gate gradient is all zeros.")


if __name__ == "__main__":
    unittest.main()
