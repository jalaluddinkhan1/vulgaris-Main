"""
Utilities for downloading and loading pre-trained VULGARIS weights.
"""
from __future__ import annotations

import os
import json
import tempfile
import urllib.request
from typing import Optional

import numpy as np


# Registry of known hosted weights
_REGISTRY = {
    "vulgaris-base-v1": {
        "hf_repo": "keysparktech/vulgaris",
        "filename": "vulgaris-base-v1.npz",
        "config":   "vulgaris-base-v1-config.json",
        "description": "Base model pre-trained on SKAB + NAB + SMD telemetry (256-dim)",
    },
}


def _download_url(url: str, dest: str):
    """Download url to dest with a progress indicator."""
    def _hook(count, block, total):
        if total > 0:
            pct = min(100, int(count * block * 100 / total))
            print(f"\r  {pct}%", end="", flush=True)
    urllib.request.urlretrieve(url, dest, reporthook=_hook)
    print()


def from_pretrained(
    name_or_path: str,
    config=None,
    cache_dir: Optional[str] = None,
) -> "Vulgaris":
    """
    Load a pre-trained VULGARIS model.

    Parameters
    ----------
    name_or_path : str
        Either a registered name ("vulgaris-base-v1"), a Hugging Face repo
        ("keysparktech/vulgaris"), or a local path to a .npz checkpoint file.
    config : ModelConfig, optional
        If provided, use this config instead of the saved one.
    cache_dir : str, optional
        Directory to cache downloaded weights (default: ~/.cache/vulgaris).

    Returns
    -------
    Vulgaris model with loaded weights.

    Examples
    --------
    >>> import vulgaris
    >>> model = vulgaris.from_pretrained("vulgaris-base-v1")
    >>> out, aux = model(x)
    """
    from model.vulgaris import Vulgaris
    from config import ModelConfig

    cache_dir = cache_dir or os.path.join(os.path.expanduser("~"), ".cache", "vulgaris")
    os.makedirs(cache_dir, exist_ok=True)

    # ── 1. Resolve the path to the .npz file ─────────────────────────────
    if os.path.isfile(name_or_path):
        weights_path = name_or_path
        cfg_path = None
    else:
        # Try registry first
        entry = _REGISTRY.get(name_or_path)

        if entry is not None:
            # Try Hugging Face Hub
            weights_path = _hf_download(entry["hf_repo"], entry["filename"], cache_dir)
            cfg_path     = _hf_download(entry["hf_repo"], entry["config"],   cache_dir,
                                         required=False)
        else:
            # Treat as "owner/repo" HF path with default filename
            repo = name_or_path
            filename = "vulgaris-weights.npz"
            weights_path = _hf_download(repo, filename, cache_dir)
            cfg_path     = _hf_download(repo, "config.json", cache_dir, required=False)

    # ── 2. Load config ────────────────────────────────────────────────────
    if config is None:
        if cfg_path and os.path.isfile(cfg_path):
            config = _load_config_json(cfg_path)
        else:
            config = ModelConfig()

    # ── 3. Build model and load weights ──────────────────────────────────
    model = Vulgaris(config)

    data = np.load(weights_path, allow_pickle=False)
    state = {k[len("model__"):]: data[k]
             for k in data.files if k.startswith("model__")}
    if not state:
        # Support plain weight dict (no model__ prefix)
        state = {k: data[k] for k in data.files
                 if not k.startswith(("opt__", "meta__"))}

    model.load_state_dict(state)
    print(f"Loaded weights from {weights_path}")
    return model


# ── helpers ───────────────────────────────────────────────────────────────

def _hf_download(repo: str, filename: str, cache_dir: str,
                 required: bool = True) -> Optional[str]:
    """Download a file from Hugging Face Hub, return local path."""
    local = os.path.join(cache_dir, repo.replace("/", "__"), filename)
    if os.path.isfile(local):
        return local

    os.makedirs(os.path.dirname(local), exist_ok=True)

    # Try huggingface_hub package first (faster, handles auth)
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id=repo, filename=filename,
                               cache_dir=cache_dir)
        return path
    except Exception:
        pass

    # Fallback: direct HTTPS
    url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
    try:
        print(f"Downloading {url}")
        _download_url(url, local)
        return local
    except Exception as e:
        if required:
            raise RuntimeError(
                f"Could not download {filename} from {repo}.\n"
                f"Error: {e}\n"
                f"Install huggingface_hub: pip install huggingface_hub"
            ) from e
        return None


def _load_config_json(path: str):
    """Load ModelConfig from a JSON file."""
    from config import (ModelConfig, ASEConfig, SSSRConfig, CRGConfig,
                        HMBConfig, HTDConfig, DAHConfig)
    with open(path) as f:
        d = json.load(f)

    cfg = ModelConfig(
        input_dim=d.get("input_dim", 64),
        output_dim=d.get("output_dim", 1),
        n_classes=d.get("n_classes", 0),
    )
    if "ase" in d:
        for k, v in d["ase"].items():
            setattr(cfg.ase, k, v)
    if "sssr" in d:
        for k, v in d["sssr"].items():
            setattr(cfg.sssr, k, v)
    if "hmb" in d:
        for k, v in d["hmb"].items():
            setattr(cfg.hmb, k, v)
    if "htd" in d:
        for k, v in d["htd"].items():
            setattr(cfg.htd, k, v)
    if "crg" in d:
        for k, v in d["crg"].items():
            setattr(cfg.crg, k, v)
    return cfg


def save_pretrained(model, config, output_dir: str, name: str = "vulgaris-base-v1"):
    """
    Save model weights + config in from_pretrained-compatible format.

    Creates:
        output_dir/{name}.npz          — weights
        output_dir/{name}-config.json  — config
    """
    import dataclasses
    os.makedirs(output_dir, exist_ok=True)

    # Save weights
    state = model.state_dict()
    save_dict = {f"model__{k}": v for k, v in state.items()}
    weights_path = os.path.join(output_dir, f"{name}.npz")
    np.savez_compressed(weights_path, **save_dict)
    print(f"Saved weights → {weights_path}")

    # Save config as JSON
    cfg_dict = dataclasses.asdict(config)
    cfg_path = os.path.join(output_dir, f"{name}-config.json")
    with open(cfg_path, "w") as f:
        json.dump(cfg_dict, f, indent=2)
    print(f"Saved config  → {cfg_path}")

    return weights_path, cfg_path
