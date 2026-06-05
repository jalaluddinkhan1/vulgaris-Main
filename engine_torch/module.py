"""engine_torch/module.py — Module backed by torch.nn.Module."""
from __future__ import annotations
import torch.nn as nn
from engine_torch.tensor import Tensor, Parameter


class Module(nn.Module):
    """
    Drop-in for engine.module.Module.

    Subclasses torch.nn.Module so:
    - All torch.nn layers registered as attributes are automatically found by
      parameters() and moved to device with .to()
    - .train() and .eval() work correctly
    - torch.compile() can trace the full forward graph
    """

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def parameters(self):
        """Yield Parameter objects (compat with VULGARIS optimizer interface)."""
        for p in super().parameters():
            yield Parameter(p.data.cpu().numpy())

    def state_dict(self) -> dict:
        return {k: v.cpu().numpy() for k, v in super().state_dict().items()}

    def load_state_dict(self, d: dict, strict: bool = True):
        import torch
        torch_d = {k: torch.from_numpy(v) for k, v in d.items()}
        super().load_state_dict(torch_d, strict=strict)
