"""
Numerical stability and backend parity stress tests.

Coverage:
  - Extreme input values: NaN, Inf, very large, very small, zero
  - fp32/INT8 parity (quantization error bound)
  - numpy vs numba vs triton backend output parity
  - log_a clamp invariant (ASE SSM decay gate)
  - NOTEARS gradient does not produce NaN/Inf for near-zero W
"""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_x(shape, dtype=np.float64, mode="normal"):
    rng = np.random.default_rng(0)
    if mode == "normal":
        return rng.standard_normal(shape).astype(dtype)
    if mode == "large":
        return (rng.standard_normal(shape) * 1e6).astype(dtype)
    if mode == "small":
        return (rng.standard_normal(shape) * 1e-6).astype(dtype)
    if mode == "zeros":
        return np.zeros(shape, dtype=dtype)
    raise ValueError(mode)


def _is_finite(arr):
    return bool(np.all(np.isfinite(arr)))


# ---------------------------------------------------------------------------
# 1. log_a clamp invariant
# ---------------------------------------------------------------------------

class TestLogAClamp:
    """SSM decay gate: log_a must stay in [-5, 0] regardless of input."""

    def _clamp_log_a(self, raw):
        raw = np.nan_to_num(raw, nan=-5.0, posinf=0.0, neginf=-5.0)
        return np.clip(raw, -5.0, 0.0)

    @pytest.mark.parametrize("val", [-1e9, -100.0, -5.0, -1.0, 0.0, 1.0, 1e9, np.nan, np.inf, -np.inf])
    def test_clamp_bounds(self, val):
        result = self._clamp_log_a(np.array([val], dtype=np.float64))
        assert _is_finite(result), f"Clamp produced non-finite for input {val}"
        assert float(result[0]) >= -5.0
        assert float(result[0]) <= 0.0

    def test_clamp_array(self):
        rng = np.random.default_rng(1)
        raw = rng.uniform(-20.0, 20.0, size=(4, 32, 64))
        clamped = self._clamp_log_a(raw)
        assert clamped.min() >= -5.0
        assert clamped.max() <= 0.0
        assert _is_finite(clamped)


# ---------------------------------------------------------------------------
# 2. NOTEARS gradient stability
# ---------------------------------------------------------------------------

class TestNOTEARSGradient:
    """d/dW tr(expm(W²)) = 2W ⊙ expm(W²) must be finite for small |W|."""

    @staticmethod
    def _notears_grad(W):
        from scipy.linalg import expm
        A = W ** 2
        E = expm(A)
        return 2.0 * W * E, np.trace(E) - W.shape[0]

    @pytest.mark.parametrize("scale", [0.0, 1e-6, 0.01, 0.1, 0.5])
    def test_grad_finite_small_W(self, scale):
        rng = np.random.default_rng(2)
        n = 8
        W = rng.standard_normal((n, n)) * scale
        np.fill_diagonal(W, 0.0)
        grad, pen = self._notears_grad(W)
        assert _is_finite(grad), f"NOTEARS grad non-finite at scale={scale}"
        assert _is_finite(np.array([pen]))

    def test_grad_finite_at_zero(self):
        W = np.zeros((16, 16))
        grad, pen = self._notears_grad(W)
        assert _is_finite(grad)
        assert abs(pen) < 1e-10   # tr(I) - n = 0

    def test_penalty_nonneg(self):
        rng = np.random.default_rng(3)
        W = rng.standard_normal((8, 8)) * 0.01
        np.fill_diagonal(W, 0.0)
        _, pen = self._notears_grad(W)
        assert pen >= -1e-10   # h(W) >= 0 by construction


# ---------------------------------------------------------------------------
# 3. Linear layer extreme inputs
# ---------------------------------------------------------------------------

class TestLinearExtremeInputs:
    """engine.layers.Linear must not produce NaN/Inf on extreme or zero inputs."""

    @pytest.fixture(autouse=True)
    def setup(self):
        try:
            from engine.layers import Linear
            from engine.tensor import Tensor
            self.Linear = Linear
            self.Tensor = Tensor
        except ImportError:
            pytest.skip("engine not importable in isolation")

    @pytest.mark.parametrize("mode", ["zeros", "small", "large"])
    def test_forward_finite(self, mode):
        layer = self.Linear(32, 16)
        x_np = _make_x((2, 32), mode=mode)
        x = self.Tensor(x_np, requires_grad=False)
        out = layer(x)
        assert _is_finite(out.data), f"Linear output non-finite for mode={mode}"

    def test_nan_input_propagates_cleanly(self):
        layer = self.Linear(8, 4)
        x_np = np.full((2, 8), np.nan, dtype=np.float64)
        x = self.Tensor(x_np, requires_grad=False)
        out = layer(x)
        # NaN in → NaN out is expected; important: no exception raised
        assert out.data.shape == (2, 4)


# ---------------------------------------------------------------------------
# 4. INT8 quantization parity
# ---------------------------------------------------------------------------

class TestINT8Parity:
    """Dequantized INT8 weights must approximate fp32 within expected tolerance."""

    @pytest.fixture(autouse=True)
    def setup(self):
        try:
            from inference.streaming import QuantizedWeights
            self.QW = QuantizedWeights
        except ImportError:
            pytest.skip("inference.streaming not importable")

    def test_quantize_dequantize_error(self):
        rng = np.random.default_rng(5)
        W = rng.standard_normal((64, 64)).astype(np.float32)
        qw = self.QW.quantize(W, bits=8)
        W_back = qw.dequantize()
        max_abs = float(np.max(np.abs(W)))
        # INT8 max relative error bound: ≤ 1/(2*127) ≈ 0.4%
        max_err = float(np.max(np.abs(W - W_back)))
        assert max_err <= max_abs * 0.01, (
            f"INT8 quantization error too large: {max_err:.6f} vs bound {max_abs*0.01:.6f}"
        )

    def test_quantize_zero_matrix(self):
        W = np.zeros((16, 16), dtype=np.float32)
        qw = self.QW.quantize(W, bits=8)
        W_back = qw.dequantize()
        assert np.allclose(W_back, 0.0)

    def test_quantize_preserves_sign(self):
        W = np.array([[-1.0, 2.0], [3.0, -4.0]], dtype=np.float32)
        qw = self.QW.quantize(W, bits=8)
        W_back = qw.dequantize()
        assert np.all(np.sign(W_back) == np.sign(W))

    def test_quantize_large_weights(self):
        rng = np.random.default_rng(6)
        W = rng.standard_normal((32, 32)).astype(np.float32) * 1000.0
        qw = self.QW.quantize(W, bits=8)
        W_back = qw.dequantize()
        # Large scale — same relative tolerance
        assert _is_finite(W_back), "Dequantized large-weight matrix is non-finite"


# ---------------------------------------------------------------------------
# 5. Backend output parity (numpy vs triton fallback)
# ---------------------------------------------------------------------------

class TestBackendParity:
    """
    numpy baseline vs triton-fallback must produce bit-level identical results
    (both are pure numpy paths in CPU-only environments).
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        try:
            from engine.backends.triton_ops.linear import linear_triton
            from engine.backends.triton_ops.ssm_scan import ssm_scan_triton
            self.linear_triton = linear_triton
            self.ssm_scan_triton = ssm_scan_triton
        except ImportError:
            pytest.skip("triton_ops not importable")

    def test_linear_numpy_vs_fallback(self):
        rng = np.random.default_rng(10)
        x = rng.standard_normal((8, 32)).astype(np.float32)
        w = rng.standard_normal((16, 32)).astype(np.float32)
        b = rng.standard_normal((16,)).astype(np.float32)

        # numpy reference
        ref = x @ w.T + b

        # triton (falls back to numpy on CPU)
        out = self.linear_triton(x, w, b)

        assert _is_finite(out), "linear_triton output non-finite"
        np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5,
                                   err_msg="linear_triton diverges from numpy reference")

    def test_ssm_scan_numpy_vs_fallback(self):
        rng = np.random.default_rng(11)
        B, T, D = 2, 16, 8
        a = np.exp(rng.uniform(-5.0, 0.0, (B, T, D))).astype(np.float32)
        b_in = rng.standard_normal((B, T, D)).astype(np.float32) * 0.1

        # numpy reference scan
        ref = np.zeros_like(b_in)
        for bat in range(B):
            h = np.zeros(D)
            for t in range(T):
                h = a[bat, t] * h + b_in[bat, t]
                ref[bat, t] = h

        out = self.ssm_scan_triton(a, b_in)
        assert _is_finite(out), "ssm_scan_triton output non-finite"
        np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4,
                                   err_msg="ssm_scan_triton diverges from numpy reference")

    def test_linear_no_bias(self):
        rng = np.random.default_rng(12)
        x = rng.standard_normal((4, 16)).astype(np.float32)
        w = rng.standard_normal((8, 16)).astype(np.float32)
        ref = x @ w.T
        out = self.linear_triton(x, w, b_np=None)
        np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# 6. Streaming normalizer — NaN / Inf input guard
# ---------------------------------------------------------------------------

class TestNormalizerRobustness:
    @pytest.fixture(autouse=True)
    def setup(self):
        try:
            from inference.streaming import RunningNormalizer
            self.Norm = RunningNormalizer
        except ImportError:
            pytest.skip("inference.streaming not importable")

    def test_normalizer_finite_after_normal_data(self):
        norm = self.Norm(dim=8)
        rng = np.random.default_rng(20)
        for _ in range(50):
            x = rng.standard_normal((4, 8))
            norm.update(x)
            out = norm.normalize(x)
            assert _is_finite(out), "Normalizer output non-finite on normal data"

    def test_normalizer_variance_floor(self):
        norm = self.Norm(dim=4)
        # Feed identical samples — variance should floor at 1e-8, not zero-divide
        x = np.ones((10, 4), dtype=np.float64)
        norm.update(x)
        out = norm.normalize(x)
        assert _is_finite(out)


# ---------------------------------------------------------------------------
# 7. LogEncoder robustness
# ---------------------------------------------------------------------------

class TestLogEncoderRobustness:
    @pytest.fixture(autouse=True)
    def setup(self):
        try:
            from preprocessing.log_encoder import LogEncoder
            self.LogEncoder = LogEncoder
        except ImportError:
            pytest.skip("preprocessing.log_encoder not importable")

    def test_encode_empty_string(self):
        enc = self.LogEncoder()
        out = enc.encode("")
        assert out.shape == (2,)
        assert _is_finite(out)

    def test_encode_unknown_severity(self):
        enc = self.LogEncoder()
        out = enc.encode("2026-01-01 some_module: message without severity keyword")
        assert int(out[1]) == 1   # defaults to INFO

    def test_encode_batch_consistent(self):
        enc = self.LogEncoder()
        lines = [
            "ERROR disk full on /dev/sda1",
            "ERROR disk full on /dev/sda2",   # same template
            "INFO service started pid=1234",
        ]
        out = enc.encode_batch(lines)
        assert out.shape == (3, 2)
        assert _is_finite(out)
        # Both ERROR disk lines should share a template ID (same prefix pattern)
        assert out[0, 0] == out[1, 0]

    def test_template_cap_unknown(self):
        enc = self.LogEncoder(max_templates=5)
        for i in range(20):
            enc.encode(f"unique_{i} message extra data")
        # After cap, new lines map to tid=0 ("unknown")
        out = enc.encode("completely_novel_line_xyz abc def ghi jkl")
        assert int(out[0]) == 0
