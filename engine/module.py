from typing import Dict, Iterator, Optional, Tuple, Any
import numpy as np
from .tensor import Tensor, Parameter


class Module:
    """Base class for all neural network modules."""

    def __init__(self):
        # Use object.__setattr__ to bypass our custom __setattr__ during init
        object.__setattr__(self, "_parameters", {})
        object.__setattr__(self, "_modules", {})
        object.__setattr__(self, "_buffers", {})
        object.__setattr__(self, "training", True)

    def __setattr__(self, name: str, value: Any):
        # Remove from other registries if previously stored
        params = object.__getattribute__(self, "_parameters")
        modules = object.__getattribute__(self, "_modules")
        buffers = object.__getattribute__(self, "_buffers")

        if isinstance(value, Parameter):
            params[name] = value
            # Also store as normal attribute for direct access
            object.__setattr__(self, name, value)
        elif isinstance(value, Module):
            modules[name] = value
            object.__setattr__(self, name, value)
        elif isinstance(value, np.ndarray) and name in buffers:
            buffers[name] = value
            object.__setattr__(self, name, value)
        else:
            # Remove from param/module dicts if overwriting
            params.pop(name, None)
            modules.pop(name, None)
            object.__setattr__(self, name, value)

    def register_buffer(self, name: str, data: np.ndarray):
        """Register a non-parameter numpy array accessible as attribute."""
        buffers = object.__getattribute__(self, "_buffers")
        buffers[name] = data
        object.__setattr__(self, name, data)

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        """Yield all parameters, optionally recursing into submodules."""
        seen = set()
        for param in self._parameters.values():
            if id(param) not in seen:
                seen.add(id(param))
                yield param
        if recurse:
            for mod in self._modules.values():
                for param in mod.parameters(recurse=True):
                    if id(param) not in seen:
                        seen.add(id(param))
                        yield param

    def named_parameters(self, prefix: str = "", recurse: bool = True) -> Iterator[Tuple[str, Parameter]]:
        """Yield (name, param) tuples with dotted names."""
        seen = set()
        for name, param in self._parameters.items():
            full_name = f"{prefix}.{name}" if prefix else name
            if id(param) not in seen:
                seen.add(id(param))
                yield full_name, param
        if recurse:
            for mod_name, mod in self._modules.items():
                sub_prefix = f"{prefix}.{mod_name}" if prefix else mod_name
                for name, param in mod.named_parameters(prefix=sub_prefix, recurse=True):
                    if id(param) not in seen:
                        seen.add(id(param))
                        yield name, param

    def zero_grad(self):
        """Set all parameter gradients to None."""
        for p in self.parameters():
            p.grad = None

    def train(self, mode: bool = True) -> "Module":
        """Set training mode for this module and all submodules."""
        object.__setattr__(self, "training", mode)
        for mod in self._modules.values():
            mod.train(mode)
        return self

    def eval(self) -> "Module":
        """Set evaluation mode."""
        return self.train(False)

    def __call__(self, *args, **kwargs) -> Any:
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError(f"{self.__class__.__name__} must implement forward()")

    def state_dict(self) -> Dict[str, np.ndarray]:
        """Return dict of parameter name -> data copy."""
        return {name: param.data.copy() for name, param in self.named_parameters()}

    def load_state_dict(self, d: Dict[str, np.ndarray], strict: bool = True):
        """Load parameter data from dict."""
        own_params = dict(self.named_parameters())
        if strict:
            own_keys = set(own_params.keys())
            dict_keys = set(d.keys())
            missing = own_keys - dict_keys
            unexpected = dict_keys - own_keys
            if missing:
                raise KeyError(f"Missing keys in state_dict: {missing}")
            if unexpected:
                raise KeyError(f"Unexpected keys in state_dict: {unexpected}")
        for name, data in d.items():
            if name in own_params:
                own_params[name].data = np.asarray(data, dtype=np.float64)

    def n_params(self) -> int:
        """Total number of scalar parameters."""
        return sum(p.data.size for p in self.parameters())

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        n = self.n_params()
        lines = [f"{cls}(n_params={n:,})"]
        for name, mod in self._modules.items():
            mod_repr = repr(mod).replace("\n", "\n  ")
            lines.append(f"  ({name}): {mod_repr}")
        return "\n".join(lines)
