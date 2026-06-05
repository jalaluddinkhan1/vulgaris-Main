"""
Numerical gradient checker for VULGARIS custom backward closures.

For every custom _backward closure (ASE, MultiHorizonHead, RMC softmax,
OutputHead, etc.) the analytic gradient is verified against a central
finite-difference approximation:

    g_numerical[i] = (f(x + eps*e_i) - f(x - eps*e_i)) / (2*eps)

Usage
-----
    from engine.grad_check import grad_check, grad_check_module

    # Check a single function
    ok, max_err = grad_check(lambda x: x.sum() * x, [x_tensor])

    # Check every parameter of a module
    results = grad_check_module(ase_module, x_tensor)
"""
from __future__ import annotations

from typing import Callable, List, Tuple
import numpy as np

from engine.tensor import Tensor


def _flatten(tensors: list[Tensor]) -> Tuple[np.ndarray, list]:
    """Flatten a list of Tensors into a single 1-D numpy vector + metadata."""
    parts, meta = [], []
    for t in tensors:
        flat = t.data.ravel().astype(np.float64)
        meta.append((t, t.data.shape, len(flat)))
        parts.append(flat)
    return np.concatenate(parts), meta


def _unflatten(vec: np.ndarray, meta: list) -> list[Tensor]:
    """Reconstruct Tensors from the flat vector using saved metadata."""
    out, offset = [], 0
    for t, shape, n in meta:
        chunk = vec[offset : offset + n].reshape(shape).astype(np.float32)
        new_t = Tensor(chunk, requires_grad=t.requires_grad)
        out.append(new_t)
        offset += n
    return out


def grad_check(
    fn: Callable[..., Tensor],
    inputs: List[Tensor],
    eps: float = 1e-4,
    atol: float = 1e-3,
    rtol: float = 1e-2,
    verbose: bool = False,
) -> Tuple[bool, float]:
    """
    Check analytic gradients against central finite differences.

    Parameters
    ----------
    fn      : function that takes *inputs and returns a scalar Tensor
    inputs  : list of Tensors with requires_grad=True
    eps     : finite-difference step size
    atol    : absolute tolerance for pass/fail
    rtol    : relative tolerance for pass/fail
    verbose : print per-element comparison

    Returns
    -------
    (passed: bool, max_relative_error: float)
    """
    # ── Analytic gradients ─────────────────────────────────────────────
    for inp in inputs:
        inp.grad = None

    out = fn(*inputs)
    if out.data.size != 1:
        # Sum to scalar so backward works
        scalar = Tensor(
            np.array([[out.data.sum()]], dtype=np.float32),
            requires_grad=True,
            _children=(out,),
            _op="sum_scalar",
        )
        _out = out

        def _sum_back():
            if _out.requires_grad and scalar.grad is not None:
                _out.grad = (_out.grad + np.ones_like(_out.data) * scalar.grad.sum()
                             if _out.grad is not None
                             else np.ones_like(_out.data) * scalar.grad.sum())

        scalar._backward = _sum_back
        scalar.backward()
    else:
        out.backward()

    analytic = {}
    for inp in inputs:
        if inp.requires_grad and inp.grad is not None:
            analytic[id(inp)] = inp.grad.ravel().copy()

    # ── Numerical gradients ────────────────────────────────────────────
    flat_vec, meta = _flatten(inputs)
    numerical = np.zeros_like(flat_vec)

    for i in range(len(flat_vec)):
        # f(x + eps)
        vec_p = flat_vec.copy(); vec_p[i] += eps
        inp_p = _unflatten(vec_p, meta)
        out_p = fn(*inp_p)
        f_plus = float(out_p.data.sum())

        # f(x - eps)
        vec_m = flat_vec.copy(); vec_m[i] -= eps
        inp_m = _unflatten(vec_m, meta)
        out_m = fn(*inp_m)
        f_minus = float(out_m.data.sum())

        numerical[i] = (f_plus - f_minus) / (2.0 * eps)

    # ── Compare ────────────────────────────────────────────────────────
    analytic_flat = np.concatenate(
        [analytic.get(id(inp), np.zeros(n))
         for inp, _, n in meta]
    )

    abs_err = np.abs(analytic_flat - numerical)
    denom   = np.maximum(np.abs(analytic_flat) + np.abs(numerical), 1e-8)
    rel_err = abs_err / denom
    max_rel = float(rel_err.max()) if len(rel_err) > 0 else 0.0

    passed = bool((abs_err <= atol + rtol * denom).all())

    if verbose:
        for i, (a, n, ae, re) in enumerate(
            zip(analytic_flat, numerical, abs_err, rel_err)
        ):
            status = "✓" if ae <= atol + rtol * abs(a + n) / 2 else "✗"
            print(f"  [{i:4d}] analytic={a:+.6f}  numerical={n:+.6f}"
                  f"  abs={ae:.2e}  rel={re:.2e}  {status}")

    return passed, max_rel


def grad_check_module(
    module,
    *inputs,
    eps: float = 1e-4,
    atol: float = 1e-3,
    rtol: float = 1e-2,
    verbose: bool = False,
) -> dict[str, Tuple[bool, float]]:
    """
    Check gradients for every trainable parameter of a Module.

    Runs grad_check independently for each parameter, holding all others fixed.

    Returns
    -------
    dict mapping parameter name → (passed, max_rel_error)
    """
    results = {}

    def _run(*inp):
        return module(*inp)

    for name, param in _named_params(module):
        if not param.requires_grad or param.data.size == 0:
            continue
        param_as_input = [Tensor(param.data.copy(), requires_grad=True)]

        def _fn_for_param(p, name=name, param=param):
            old = param.data.copy()
            param.data = p.data
            out = module(*inputs)
            if hasattr(out, "sum"):
                result = out
            else:
                # Module returned (out, aux) tuple
                result = out[0]
            param.data = old
            return result

        ok, err = grad_check(_fn_for_param, param_as_input,
                             eps=eps, atol=atol, rtol=rtol, verbose=verbose)
        results[name] = (ok, err)
        status = "PASS" if ok else "FAIL"
        print(f"  {name:40s}: {status}  max_rel={err:.2e}")

    return results


def _named_params(module, prefix=""):
    from engine.module import Module
    from engine.tensor import Parameter
    for attr, val in vars(module).items():
        full = f"{prefix}.{attr}" if prefix else attr
        if isinstance(val, Parameter):
            yield full, val
        elif isinstance(val, Module):
            yield from _named_params(val, full)
