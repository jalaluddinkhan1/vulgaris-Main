# -*- coding: utf-8 -*-
"""Unit tests for engine/tensor.py — numerical gradient verification."""
import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from engine.tensor import Tensor, zeros, ones, randn


def check_grad(fn, inputs, eps=1e-4, rtol=1e-2, atol=1e-4):
    """
    Finite-difference gradient check.
    fn: callable(*inputs) -> scalar Tensor
    inputs: list of Tensor with requires_grad=True
    Returns max relative error across all inputs.
    """
    # Compute analytical gradients
    out = fn(*inputs)
    out.backward()
    analytical = [inp.grad.copy() if inp.grad is not None else np.zeros_like(inp.data)
                  for inp in inputs]

    # Compute numerical gradients
    numerical = []
    for k, inp in enumerate(inputs):
        num_grad = np.zeros_like(inp.data)
        flat = inp.data.flat
        for idx in range(inp.data.size):
            orig = inp.data.flat[idx]
            inp.data.flat[idx] = orig + eps
            # Reset grads
            for i in inputs:
                i.grad = None
            f_plus = float(fn(*inputs).data.sum())
            inp.data.flat[idx] = orig - eps
            for i in inputs:
                i.grad = None
            f_minus = float(fn(*inputs).data.sum())
            inp.data.flat[idx] = orig
            num_grad.flat[idx] = (f_plus - f_minus) / (2 * eps)
        numerical.append(num_grad)

    errors = []
    for a, n in zip(analytical, numerical):
        if n.size > 0:
            denom = np.maximum(np.abs(n), np.abs(a)) + atol
            rel_err = np.max(np.abs(a - n) / denom)
            errors.append(float(rel_err))
    return max(errors) if errors else 0.0


class TestBasicOps(unittest.TestCase):

    def _make(self, shape, seed=0):
        rng = np.random.default_rng(seed)
        data = rng.normal(0, 0.5, shape).astype(np.float32)
        return Tensor(data, requires_grad=True)

    def test_add_grad(self):
        a = self._make((3, 4))
        b = self._make((3, 4), 1)
        err = check_grad(lambda x, y: x + y, [a, b])
        self.assertLess(err, 0.05, f"add grad error {err:.4f}")

    def test_mul_grad(self):
        a = self._make((3, 4))
        b = self._make((3, 4), 1)
        err = check_grad(lambda x, y: x * y, [a, b])
        self.assertLess(err, 0.05, f"mul grad error {err:.4f}")

    def test_matmul_grad(self):
        a = self._make((4, 6))
        b = self._make((6, 3), 1)
        err = check_grad(lambda x, y: x @ y, [a, b])
        self.assertLess(err, 0.05, f"matmul grad error {err:.4f}")

    def test_sub_grad(self):
        a = self._make((3, 3))
        b = self._make((3, 3), 1)
        err = check_grad(lambda x, y: x - y, [a, b])
        self.assertLess(err, 0.05, f"sub grad error {err:.4f}")

    def test_sum_grad(self):
        a = self._make((4, 5))
        err = check_grad(lambda x: x.sum(), [a])
        self.assertLess(err, 0.05)


class TestActivations(unittest.TestCase):

    def _make(self, shape, seed=0, scale=0.5):
        rng = np.random.default_rng(seed)
        return Tensor(rng.normal(0, scale, shape).astype(np.float32), requires_grad=True)

    def test_relu_grad(self):
        a = self._make((4, 6))
        err = check_grad(lambda x: x.relu(), [a])
        self.assertLess(err, 0.05)

    def test_tanh_grad(self):
        a = self._make((3, 4))
        err = check_grad(lambda x: x.tanh(), [a])
        self.assertLess(err, 0.05)

    def test_sigmoid_grad(self):
        a = self._make((3, 4))
        err = check_grad(lambda x: x.sigmoid(), [a], eps=1e-3)
        self.assertLess(err, 0.05)

    def test_softmax_grad(self):
        a = self._make((3, 5))
        # softmax + sum is constant=1, so grad check sum of softmax * weights
        w = np.random.default_rng(1).normal(0, 1, (3, 5)).astype(np.float32)
        W = Tensor(w, requires_grad=False)
        err = check_grad(lambda x: (x.softmax(axis=-1) * W).sum(), [a], eps=1e-3)
        self.assertLess(err, 0.05)


class TestShapeOps(unittest.TestCase):

    def test_reshape_grad(self):
        a = Tensor(np.random.randn(4, 6).astype(np.float32), requires_grad=True)
        err = check_grad(lambda x: x.reshape(24,), [a])
        self.assertLess(err, 0.05)

    def test_transpose_grad(self):
        a = Tensor(np.random.randn(3, 4).astype(np.float32), requires_grad=True)
        err = check_grad(lambda x: x.T, [a])
        self.assertLess(err, 0.05)


class TestNorm(unittest.TestCase):

    def test_output_dtype_float32(self):
        from engine.layers import LayerNorm, RMSNorm
        ln = LayerNorm(8)
        x = Tensor(np.random.randn(4, 8).astype(np.float32), requires_grad=False)
        out = ln(x)
        self.assertEqual(out.data.dtype, np.float32)

    def test_rmsnorm_dtype_float32(self):
        from engine.layers import RMSNorm
        rn = RMSNorm(8)
        x = Tensor(np.random.randn(4, 8).astype(np.float32), requires_grad=False)
        out = rn(x)
        self.assertEqual(out.data.dtype, np.float32)


class TestMemoryManagement(unittest.TestCase):

    def test_graph_freed_after_backward(self):
        """After backward(), _prev should be cleared on all nodes."""
        a = Tensor(np.array([[1.0, 2.0]], dtype=np.float32), requires_grad=True)
        b = Tensor(np.array([[3.0, 4.0]], dtype=np.float32), requires_grad=True)
        c = a * b
        d = c.sum()
        d.backward()
        # After backward, _prev should be empty (graph freed)
        self.assertEqual(len(d._prev), 0, "_prev not cleared after backward")
        self.assertEqual(len(c._prev), 0, "intermediate _prev not cleared")

    def test_dtype_float32(self):
        """Tensors created from float64 arrays should be stored as float32."""
        x = Tensor(np.array([1.0, 2.0], dtype=np.float64))
        self.assertEqual(x.data.dtype, np.float32)

    def test_dtype_preserved_float32(self):
        """Tensors created from float32 arrays should stay float32."""
        x = Tensor(np.array([1.0, 2.0], dtype=np.float32))
        self.assertEqual(x.data.dtype, np.float32)


if __name__ == "__main__":
    unittest.main(verbosity=2)
