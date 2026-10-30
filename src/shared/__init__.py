"""Shared experimental protocol for Models 1–4.

Freeze Model 2's label mapping (v1.0.0) and patient-level split. All later models
must reuse those artifacts rather than resampling patients or retuning rules after
seeing test metrics. Identifiers are never model features.

Scores measure agreement with weak, report-derived labels, not clinical accuracy.
"""

from .constants import LABEL_ORDER, LEAD_ORDER, TARGET_LENGTH_SAMPLES

__all__ = ["LABEL_ORDER", "LEAD_ORDER", "TARGET_LENGTH_SAMPLES"]
