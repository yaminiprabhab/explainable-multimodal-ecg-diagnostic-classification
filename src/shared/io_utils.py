"""Config I/O, JSON dumps, path resolution, and seed 42 helpers.

Paths in YAML are resolved relative to the repository root. set_seed also
fixes PyTorch and cuDNN when torch is installed.
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def repo_root() -> Path:
    """Return the repository root (contains artifacts/, requirements.txt, src/)."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "artifacts" / "label_mapping.yaml").exists() and (parent / "requirements.txt").exists():
            return parent
    return here.parents[2]


def default_config(stem: str) -> Path:
    """Path to configs/<stem>.yaml."""
    name = stem if stem.endswith(".yaml") else f"{stem}.yaml"
    return repo_root() / "configs" / name


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def save_yaml(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, default=_json_default)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Cannot serialize {type(obj)}")


def resolve_path(root: Path, maybe_relative: str | Path) -> Path:
    path = Path(maybe_relative)
    if path.is_absolute():
        return path
    return (root / path).resolve()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def package_versions() -> dict[str, str]:
    versions = {"python": sys.version.split()[0]}
    for name in ("pandas", "numpy", "yaml", "torch", "sklearn", "wfdb", "matplotlib"):
        try:
            mod = __import__(name)
            versions[name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[name] = "not_installed"
    return versions


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
