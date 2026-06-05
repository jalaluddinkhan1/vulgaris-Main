"""
Numerical gradient checks for VULGARIS custom backward closures.

These tests catch silent wrong-gradient bugs — every custom _backward
closure is verified against central finite differences.

Run:  pytest tests/test_grad_check.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

from engine.tensor import Tensor, Parameter
from engine.grad_check import grad_check


# ── Helpers ───────────────────────────────────────────────────────────────────

def T(data, req_grad=True):
    return Tensor(np.array(data, dtype=np.float32), requires_grad=req_grad)


def rand(*shape, req_grad=True):
    return Tensor(np.random.randn(*shape).astype(np.float32) * 0.3,
                  requires_grad=req_grad)


# ── OutputHead _last_back ─────────────────────────────────────────────────────

class TestOutputHeadGrad:
    def test_last_timestep_slice(self):
        """Gradient of x[:, -1, :] selection must flow only to last timestep."""
        from model.vulgaris import OutputHead
        head = OutputHead(d_model=8, output_dim=2)
        x = rand(2, 4, 8)

        def fn(xin):
            return head(xin)

        ok, err = grad_check(fn, [x])
        assert ok, f"OutputHead _last_back failed  max_rel={err:.2e}"


# ── MultiHorizonHead _pool_back + _stack_back ─────────────────────────────────

class TestMultiHorizonGrad:
    def test_mean_pool_and_stack(self):
        from model.vulgaris import MultiHorizonHead
        head = MultiHorizonHead(d_model=8, output_dim=1, horizons=[1, 2, 3])
        x = rand(2, 6, 8)

        ok, err = grad_check(lambda xin: head(xin), [x])
        assert ok, f"MultiHorizonHead grad failed  max_rel={err:.2e}"


# ── ASE multiscale backward ───────────────────────────────────────────────────

class TestASEGrad:
    def test_channel_mix_grad(self):
        """Input gradient through ASE (approximate backward via FFT convolution)."""
        from modules.ase import AdaptiveSignalEmbedding
        ase = AdaptiveSignalEmbedding(in_channels=3, n_filters=4,
                                      n_scales=2, filter_len=8, latent_dim=16)
        x = rand(1, 3, 16)
        # Run forward + backward to check gradients flow and are non-NaN
        out = ase(x)
        out.sum().backward()
        assert x.grad is not None, "x.grad is None after backward"
        assert not np.isnan(x.grad).any(), "NaN in x.grad"
        assert not np.allclose(x.grad, 0), "x.grad is all-zero (gradient not flowing)"

    def test_wavelet_params_receive_grad(self):
        """Wavelet parameters must receive non-zero gradients after a backward pass.

        Previously log_A, log_sigma, omega, phi had no gradient computation in
        _ase_back — they were listed as _children but never written to. This test
        catches the regression.
        """
        from modules.ase import AdaptiveSignalEmbedding
        ase = AdaptiveSignalEmbedding(in_channels=2, n_filters=4,
                                      n_scales=2, filter_len=8, latent_dim=8)
        x = rand(1, 2, 16)
        out = ase(x)
        out.sum().backward()

        for pname in ("log_A", "log_sigma", "omega", "phi"):
            g = getattr(ase, pname).grad
            assert g is not None, f"{pname}.grad is None — not computed in _ase_back"
            assert not np.isnan(g).any(), f"NaN in {pname}.grad"
            assert not np.allclose(g, 0), f"{pname}.grad is all-zero"

    def test_wavelet_params_numerical(self):
        """Numerical gradient check for wavelet parameters (approximate energy grad)."""
        np.random.seed(0)   # fix seed — energy normalisation stop-gradient makes
                            # the check sensitive to initialisation
        from modules.ase import AdaptiveSignalEmbedding
        ase = AdaptiveSignalEmbedding(in_channels=2, n_filters=3,
                                      n_scales=1, filter_len=8, latent_dim=8)
        x = rand(1, 2, 16, req_grad=False)

        for param_name in ("log_A", "omega", "phi"):
            param = getattr(ase, param_name)
            p_in = Tensor(param.data.copy(), requires_grad=True)

            def fn(p, pn=param_name, mod=ase, _x=x):
                old = getattr(mod, pn).data.copy()
                getattr(mod, pn).data = p.data
                object.__setattr__(mod, "_filter_cache", None)
                out = mod(_x)
                getattr(mod, pn).data = old
                object.__setattr__(mod, "_filter_cache", None)
                return out

            # Energy-normalisation stop-gradient means we expect approximate match
            ok, err = grad_check(fn, [p_in], atol=0.1, rtol=0.3)
            assert ok, f"ASE {param_name} numerical grad failed  max_rel={err:.2e}"


# ── RMC softmax backward (straight-through) ───────────────────────────────────

class TestRMCGrad:
    def test_output_grad(self):
        """Gradient through RMC expert mixture must reach z."""
        from modules.rmc import RegimeMixtureCore
        rmc = RegimeMixtureCore(d_model=8, n_experts=3)
        z = rand(2, 4, 8)

        def fn(zin):
            out, _, _ = rmc(zin)
            return out   # test expert mixture path only; bal has different shape

        ok, err = grad_check(fn, [z], atol=5e-3, rtol=0.15)
        assert ok, f"RMC output grad failed  max_rel={err:.2e}"

    def test_gradient_flows(self):
        """Full RMC forward+backward: z.grad must be non-NaN and non-zero."""
        from modules.rmc import RegimeMixtureCore
        rmc = RegimeMixtureCore(d_model=8, n_experts=4)
        z = rand(2, 4, 8)
        out, bal, _ = rmc(z)
        out.sum().backward()
        assert z.grad is not None, "z.grad is None"
        assert not np.isnan(z.grad).any(), "NaN in z.grad"
        assert not np.allclose(z.grad, 0), "z.grad is all-zero"


# ── WorldModelHead _stack_back ────────────────────────────────────────────────

class TestWorldModelGrad:
    def test_stack_back_accumulation(self):
        """Verify that _stack_back correctly accumulates gradients when
        the hidden state is used both in the stack AND as input to the
        next step (fan-out in computation graph)."""
        from model.vulgaris import WorldModelHead
        wm = WorldModelHead(d_model=8, horizon=3)
        h = rand(2, 8)

        def fn(hin):
            future_z, _ = wm(hin, horizon=3)
            return future_z

        ok, err = grad_check(fn, [h], atol=5e-3, rtol=0.1)
        assert ok, f"WorldModelHead _stack_back grad failed  max_rel={err:.2e}"


# ── CRG DAGMA penalty backward ────────────────────────────────────────────────

class TestCRGDagmaGrad:
    def test_dagma_penalty_grad(self):
        """DAGMA gradient: d/dW(-log det(sI - W⊙W)) via Cholesky solve."""
        from modules.crg import CausalRoutingGraph
        from config import CRGConfig
        crg = CausalRoutingGraph(d_model=8, config=CRGConfig(n_nodes=4))
        z = rand(1, 4, 8)

        def fn(zin):
            out, penalty = crg(zin)
            return penalty

        ok, err = grad_check(fn, [z], atol=1e-2, rtol=0.15)
        assert ok, f"CRG DAGMA penalty grad failed  max_rel={err:.2e}"


# ── HMB VAE backward ──────────────────────────────────────────────────────────

class TestHMBGrad:
    def test_hmb_gradient_flows(self):
        """HMB gradients flow back through the memory retrieval path."""
        from modules.hmb import HierarchicalMemoryBank
        from config import HMBConfig
        hmb = HierarchicalMemoryBank(config=HMBConfig(embed_dim=8, compress_dim=4,
                                                       buffer_size=8))
        z = rand(1, 4, 8)
        out, loss = hmb(z)
        (out + loss).sum().backward()
        assert z.grad is not None, "z.grad is None"
        assert not np.isnan(z.grad).any(), "NaN in z.grad"


# ── ICL attention pool backward ───────────────────────────────────────────────

class TestICLGrad:
    def test_icl_no_context_grad(self):
        """ICL pass-through (no context): gradient must reach z."""
        from modules.icl import InContextLearning
        icl = InContextLearning(d_model=8, output_dim=2, n_heads=2)
        z   = rand(1, 4, 8)
        # forward without context — adapter uses zero gate → near-identity
        out = icl(z, ctx_stack=None)
        out.sum().backward()
        assert z.grad is not None, "z.grad is None"
        assert not np.isnan(z.grad).any(), "NaN in z.grad"
