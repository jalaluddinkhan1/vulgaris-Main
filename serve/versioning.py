"""Model version registry with promotion, rollback, and SHA256 checksums."""
import hashlib
import json
import os
import time
from typing import Dict, List, Optional


class ModelVersionRegistry:
    """
    Tracks model checkpoint versions in a JSON index file.
    Supports promote (set active), rollback (revert to previous), SHA256 verification.
    """

    def __init__(self, index_path: str = "checkpoints/version_index.json"):
        self.index_path = index_path
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        self._index: dict = self._load()

    def _load(self) -> dict:
        if os.path.exists(self.index_path):
            with open(self.index_path, "r") as f:
                return json.load(f)
        return {"versions": [], "active": None, "previous": None}

    def _save(self):
        with open(self.index_path, "w") as f:
            json.dump(self._index, f, indent=2)

    @staticmethod
    def _sha256(path: str) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    def register(self, path: str, tag: str, metadata: Optional[dict] = None) -> dict:
        """Register a checkpoint file and return version entry."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        checksum = self._sha256(path)
        entry = {
            "tag": tag,
            "path": path,
            "checksum": checksum,
            "size_bytes": os.path.getsize(path),
            "registered_at": time.time(),
            "metadata": metadata or {},
        }
        self._index["versions"].append(entry)
        self._save()
        return entry

    def promote(self, tag: str) -> dict:
        """Set a version as active."""
        entry = self._get_by_tag(tag)
        self._index["previous"] = self._index["active"]
        self._index["active"] = tag
        self._save()
        return entry

    def rollback(self) -> Optional[dict]:
        """Revert to previous active version."""
        prev = self._index.get("previous")
        if prev is None:
            return None
        entry = self._get_by_tag(prev)
        self._index["active"] = prev
        self._index["previous"] = None
        self._save()
        return entry

    def verify(self, tag: str) -> bool:
        """Verify checksum of a registered version."""
        entry = self._get_by_tag(tag)
        current = self._sha256(entry["path"])
        return current == entry["checksum"]

    def list_versions(self) -> List[dict]:
        return self._index.get("versions", [])

    def get_active(self) -> Optional[dict]:
        tag = self._index.get("active")
        if tag is None:
            return None
        try:
            return self._get_by_tag(tag)
        except KeyError:
            return None

    def _get_by_tag(self, tag: str) -> dict:
        for v in self._index["versions"]:
            if v["tag"] == tag:
                return v
        raise KeyError(f"Version not found: {tag}")
