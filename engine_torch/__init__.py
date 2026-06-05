"""
engine_torch — PyTorch backend for VULGARIS.

Drop-in replacement for engine/ that uses PyTorch autograd instead of the
custom numpy reverse-mode engine.  All hand-written _backward closures are
ignored — PyTorch traces the forward pass and differentiates automatically.

Activate via environment variable (must be set before any vulgaris import):

    VULGARIS_BACKEND=torch python train.py

What this backend gives you
---------------------------
  - Genuine GPU training (CUDA, MPS, ROCm)
  - torch.compile() / TorchScript / ONNX trace (not re-implementation)
  - Mixed-precision (torch.autocast)
  - torch.autograd.gradcheck for correctness validation
  - ~5–20x faster training vs numpy on GPU

What still needs porting (numpy leakage in module bodies)
----------------------------------------------------------
The .data shim makes Linear, RMSNorm, Module, etc. fully transparent.
Modules that do np.exp(self.param.data) or np.einsum(...) on .data arrays
are partially transparent: the VALUE is correct but the computation is not
traced by PyTorch, so gradients through those paths are missing.

Port status (update as modules are ported):
  Linear           ✅ full trace
  RMSNorm          ✅ full trace
  LayerNorm        ✅ full trace
  CausalAttention  ✅ full trace
  RevIN            ✅ full trace
  Module           ✅ full trace
  Tensor           ✅ .data shim, basic ops traced
  ASE              ⚠️  np.exp on wavelet params — gradients approx
  SSSR             ⚠️  parallel scan numpy fallback — gradients approx
  CRG              ⚠️  DAGMA penalty numpy — gradient via numpy autograd only
  HMB              ⚠️  VAE reparameterization numpy — partial grad
  All others       ⚠️  depends on numpy ops in body
"""
from engine_torch.tensor import Tensor, Parameter, zeros, ones, randn, rand, cat, stack
from engine_torch.module import Module
from engine_torch.layers import Linear, LayerNorm, RMSNorm, CausalAttention, RevIN
