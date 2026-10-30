"""Construct leakage-controlled clinical text from report fields."""

from __future__ import annotations

import re
from typing import Iterable

import pandas as pd

from .label_generation import LabelMapper


def combine_statements(statements: Iterable[str], separator: str = " | ") -> str:
    cleaned = []
    for item in statements:
        text = re.sub(r"\s+", " ", str(item).strip())
        if text:
            cleaned.append(text)
    return separator.join(cleaned)


def _contains_phrase(text: str, phrases: list[str]) -> bool:
    lowered = text.lower()
    return any(p.lower() in lowered for p in phrases)


def mask_statements(
    statements: list[str],
    trigger_statements: list[str],
    mapper: LabelMapper,
    mask_token: str = "[MASKED]",
    also_mask_summary: bool = True,
    summary_phrases: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Drop or replace statements that generated labels.

    Remaining statements are eligible model input. This does not remove all
    synonymous leakage.
    """
    trigger_set = {re.sub(r"\s+", " ", t.strip()) for t in trigger_statements}
    summary_phrases = summary_phrases or []
    kept: list[str] = []
    masked: list[str] = []
    for stmt in statements:
        normalized = re.sub(r"\s+", " ", stmt.strip())
        drop = normalized in trigger_set
        if also_mask_summary and _contains_phrase(normalized, summary_phrases):
            drop = True
        if not drop:
            # Mask any statement that still matches a label rule (synonym residual).
            relabeled = mapper.label_statements([normalized])
            if any(relabeled.labels[name] == 1 for name in mapper.label_order):
                drop = True
        if drop:
            masked.append(normalized)
        else:
            kept.append(normalized)
    return kept, masked


def build_clinical_text(
    labels_frame: pd.DataFrame,
    mapper: LabelMapper,
    separator: str = " | ",
    mask_token: str = "[MASKED]",
    also_mask_summary: bool = True,
    summary_phrases: list[str] | None = None,
    leakage_mode: str = "mask_trigger_statements",
) -> pd.DataFrame:
    out = labels_frame.copy()
    clinical = []
    n_remaining = []
    n_masked = []
    for _, row in out.iterrows():
        raw = row.get("raw_report_text") or ""
        statements = [s.strip() for s in str(raw).split(" | ") if s.strip()]
        triggers = [s.strip() for s in str(row.get("trigger_statements") or "").split(" || ") if s.strip()]
        if leakage_mode == "none":
            kept, masked = statements, []
            text = combine_statements(kept, separator=separator)
        else:
            kept, masked = mask_statements(
                statements,
                triggers,
                mapper,
                mask_token=mask_token,
                also_mask_summary=also_mask_summary,
                summary_phrases=summary_phrases,
            )
            text = combine_statements(kept, separator=separator)
        clinical.append(text)
        n_remaining.append(len(kept))
        n_masked.append(len(masked))
    out["clinical_text"] = clinical
    out["n_remaining_statements"] = n_remaining
    out["n_masked_statements"] = n_masked
    out["has_usable_text"] = out["clinical_text"].str.len() > 0
    return out
