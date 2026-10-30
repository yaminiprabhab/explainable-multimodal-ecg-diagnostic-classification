"""Helpers for Model 2B. Reuses Model 2 serialization and seed utilities."""

from __future__ import annotations

from typing import Any

from model_2_text_only.utils import (
    LABEL_ORDER,
    ensure_dir,
    load_yaml,
    package_versions as _package_versions,
    resolve_path,
    save_json,
    save_yaml,
    set_seed,
)

FORBIDDEN_FEATURE_PREFIXES = ("report_",)
FORBIDDEN_FEATURE_NAMES = {
    "subject_id",
    "study_id",
    "cart_id",
    "ecg_time",
    "bandwidth",
    "filtering",
    "split",
    "unmapped",
    "ambiguous",
    "poor_quality",
    "uncertain_labels",
    "raw_report_text",
    "trigger_statements",
    "clinical_text",
    "exclude_reason",
    "n_valid_primary_features",
    *LABEL_ORDER,
    *[f"evidence_{name}" for name in LABEL_ORDER],
}


def package_versions() -> dict[str, str]:
    versions = _package_versions()
    for name in ("xgboost", "matplotlib", "joblib"):
        try:
            mod = __import__(name)
            versions[name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[name] = "not_installed"
    return versions


def assert_feature_allowlist(columns: list[str], allowed: list[str]) -> None:
    extra = [c for c in columns if c not in set(allowed)]
    missing = [c for c in allowed if c not in set(columns)]
    if extra:
        raise ValueError(f"Feature matrix contains columns outside the allowlist: {extra}")
    if missing:
        raise ValueError(f"Feature matrix is missing allowlisted columns: {missing}")
    leaked = forbidden_columns(columns)
    if leaked:
        raise ValueError(f"Leakage: forbidden columns present in the feature matrix: {leaked}")


def forbidden_columns(columns: list[str]) -> list[str]:
    leaked = []
    for col in columns:
        if col in FORBIDDEN_FEATURE_NAMES:
            leaked.append(col)
        elif col.startswith(FORBIDDEN_FEATURE_PREFIXES):
            leaked.append(col)
    return leaked


def json_ready(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_ready(v) for v in obj]
    return obj
