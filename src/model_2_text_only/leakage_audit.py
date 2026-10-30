"""Leakage audit for text-derived labels used as both targets and inputs."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

from .label_generation import LabelMapper
from .utils import LABEL_ORDER


def audit_leakage(
    frame: pd.DataFrame,
    mapper: LabelMapper,
    label_order: list[str] | None = None,
) -> dict[str, Any]:
    order = label_order or LABEL_ORDER
    n = len(frame)
    residual_hits: dict[str, int] = {k: 0 for k in order}
    empty_text = int((~frame["has_usable_text"]).sum()) if "has_usable_text" in frame.columns else None
    remaining_lens = (
        frame["clinical_text"].fillna("").astype(str).str.split().map(len)
        if "clinical_text" in frame.columns
        else pd.Series(dtype=int)
    )
    reconstructable = 0
    examples_remaining: list[str] = []

    residual_cfg = mapper.mapping.get("residual_leakage_clues", {})

    for _, row in frame.iterrows():
        text = str(row.get("clinical_text") or "")
        if text and len(examples_remaining) < 8:
            examples_remaining.append(text[:300])
        recovered = []
        relabeled = mapper.label_statements(
            [s.strip() for s in text.split(" | ") if s.strip()]
        )
        for label in order:
            if relabeled.labels[label] == 1:
                residual_hits[label] += 1
                recovered.append(label)
            clues = residual_cfg.get(label, [])
            lowered = text.lower()
            if any(c.lower() in lowered for c in clues) and int(row.get(label, 0)) == 1:
                residual_hits[label] += 0  # already counted via rules if matched
        if recovered:
            reconstructable += 1

    residual_clue_rates = {}
    for label in order:
        clues = residual_cfg.get(label, [])
        if not clues or "clinical_text" not in frame.columns:
            residual_clue_rates[label] = None
            continue
        pattern = "|".join(re.escape(c) for c in clues)
        hit = frame["clinical_text"].fillna("").str.lower().str.contains(pattern, regex=True)
        pos = frame[label] == 1
        residual_clue_rates[label] = {
            "clue_hits": int(hit.sum()),
            "clue_hits_among_positives": int((hit & pos).sum()),
            "positives": int(pos.sum()),
        }

    mean_remaining = float(remaining_lens.mean()) if len(remaining_lens) else 0.0
    median_remaining = float(remaining_lens.median()) if len(remaining_lens) else 0.0
    frac_with_text = float(frame["has_usable_text"].mean()) if n and "has_usable_text" in frame.columns else 0.0

    experiment_type = "residual_text_prediction"
    stop_training = False
    if frac_with_text < 0.05 or median_remaining < 1:
        experiment_type = "insufficient_independent_text"
        stop_training = True
    elif reconstructable / n > 0.2 if n else False:
        experiment_type = "partially_leaky_residual_text"
    notes = [
        "Targets are generated from the same report fields used to build text features.",
        "Trigger statements are masked, but synonyms, rhythm context, and report structure can still correlate with labels.",
        "This experiment measures whether leftover non-trigger text predicts weak labels, not cardiologist-verified diagnosis.",
        "Masking does not eliminate leakage.",
    ]
    return {
        "n_studies": n,
        "empty_text_studies": empty_text,
        "fraction_with_remaining_text": frac_with_text,
        "mean_remaining_tokens": mean_remaining,
        "median_remaining_tokens": median_remaining,
        "studies_where_rules_still_fire_on_masked_text": reconstructable,
        "fraction_reconstructable_from_masked_text": float(reconstructable / n) if n else None,
        "rule_hits_on_masked_text": residual_hits,
        "residual_clue_rates": residual_clue_rates,
        "example_remaining_text": examples_remaining,
        "experiment_type": experiment_type,
        "stop_training": stop_training,
        "notes": notes,
        "source_fields_for_labels": "report_0 ... report_17",
        "fields_supplied_to_model": "clinical_text after statement-level masking",
    }
