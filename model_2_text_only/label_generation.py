"""Rule-based weak-label generation for five diagnostic categories."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import yaml

from .data_loader import report_columns_of
from .utils import LABEL_ORDER

DEFAULT_MAPPING_PATH = Path(__file__).resolve().parent.parent / "artifacts" / "label_mapping.yaml"


@dataclass
class LabelResult:
    labels: dict[str, int]
    evidence: dict[str, list[str]]
    uncertain_labels: list[str]
    ambiguous: bool
    poor_quality: bool
    unmapped: bool
    source_statements: list[str] = field(default_factory=list)
    trigger_statements: list[str] = field(default_factory=list)


class LabelMapper:
    def __init__(self, mapping: dict[str, Any]):
        self.mapping = mapping
        self.label_order = list(mapping.get("label_order") or LABEL_ORDER)
        self.uncertain_policy = mapping.get("uncertain_policy", "positive_weak")
        self.missing_tokens = {
            str(x).strip().lower() for x in mapping.get("missing_value_tokens", [])
        }
        self.negation_re = _compile_union(mapping.get("negation_patterns", []))
        self.uncertainty_re = _compile_union(mapping.get("uncertainty_patterns", []))
        self.poor_quality_re = _compile_union(mapping.get("poor_quality_patterns", []))
        self.rules: dict[str, list[re.Pattern[str]]] = {}
        for label in self.label_order:
            patterns = mapping.get("rules", {}).get(label, {}).get("patterns", [])
            self.rules[label] = [re.compile(p, flags=re.IGNORECASE) for p in patterns]
        self._pacing_false_positive = re.compile(
            r"electronic simulator|without pacing", flags=re.IGNORECASE
        )

    @classmethod
    def from_path(cls, path: Path | None = None) -> "LabelMapper":
        mapping_path = path or DEFAULT_MAPPING_PATH
        with mapping_path.open("r", encoding="utf-8") as handle:
            mapping = yaml.safe_load(handle)
        return cls(mapping)

    def is_missing(self, value: Any) -> bool:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return True
        text = str(value).strip().lower()
        return text in self.missing_tokens

    def statements(self, row: pd.Series, report_cols: Iterable[str]) -> list[str]:
        lines: list[str] = []
        for col in report_cols:
            if col not in row.index:
                continue
            value = row[col]
            if self.is_missing(value):
                continue
            text = re.sub(r"\s+", " ", str(value).strip())
            if text:
                lines.append(text)
        return lines

    def label_statements(self, statements: list[str]) -> LabelResult:
        labels = {name: 0 for name in self.label_order}
        evidence: dict[str, list[str]] = {name: [] for name in self.label_order}
        uncertain_labels: set[str] = set()
        trigger_statements: list[str] = []
        poor_quality = False

        for raw in statements:
            line = raw.strip()
            if not line:
                continue
            if self.poor_quality_re.search(line):
                poor_quality = True
            negated = bool(self.negation_re.search(line))
            uncertain = bool(self.uncertainty_re.search(line))
            line_triggered = False
            for label in self.label_order:
                if label == "NORM":
                    continue
                if not _any_match(self.rules[label], line):
                    continue
                if label == "CD" and self._pacing_false_positive.search(line):
                    continue
                if label == "MI" and not _mi_allowed(line):
                    continue
                if negated:
                    evidence[label].append(f"NEGATED:{line}")
                    continue
                if uncertain and self.uncertain_policy == "skip":
                    evidence[label].append(f"UNCERTAIN_SKIP:{line}")
                    continue
                if uncertain and self.uncertain_policy == "negative":
                    evidence[label].append(f"UNCERTAIN_AS_NEG:{line}")
                    continue
                labels[label] = 1
                tag = f"UNCERTAIN:{line}" if uncertain else line
                evidence[label].append(tag)
                if uncertain:
                    uncertain_labels.add(label)
                line_triggered = True
            if line_triggered:
                trigger_statements.append(line)

        explicit_norm = False
        for line in statements:
            if not _any_match(self.rules["NORM"], line):
                continue
            if re.search(r"normal variant", line, flags=re.IGNORECASE):
                continue
            if self.negation_re.search(line):
                evidence["NORM"].append(f"NEGATED:{line}")
                continue
            explicit_norm = True
            evidence["NORM"].append(line)
            trigger_statements.append(line)

        abnormality = any(labels[name] == 1 for name in self.label_order if name != "NORM")
        if explicit_norm and not abnormality:
            labels["NORM"] = 1
        else:
            labels["NORM"] = 0
            if explicit_norm and abnormality:
                evidence["NORM"].append("SUPPRESSED_DUE_TO_ABNORMALITY")

        unmapped = all(v == 0 for v in labels.values())
        ambiguous = bool(uncertain_labels) or (
            explicit_norm and abnormality
        )
        unique_triggers = list(dict.fromkeys(trigger_statements))
        return LabelResult(
            labels=labels,
            evidence=evidence,
            uncertain_labels=sorted(uncertain_labels),
            ambiguous=ambiguous,
            poor_quality=poor_quality,
            unmapped=unmapped,
            source_statements=statements,
            trigger_statements=unique_triggers,
        )

    def label_row(self, row: pd.Series, report_cols: Iterable[str]) -> LabelResult:
        return self.label_statements(self.statements(row, report_cols))


def _compile_union(patterns: list[str]) -> re.Pattern[str]:
    if not patterns:
        return re.compile(r"(?!x)x")
    return re.compile("|".join(f"(?:{p})" for p in patterns), flags=re.IGNORECASE)


def _any_match(patterns: list[re.Pattern[str]], text: str) -> bool:
    return any(p.search(text) for p in patterns)


def _mi_allowed(line: str) -> bool:
    """Ischemia without infarct/STEMI/MI is not myocardial infarction."""
    if re.search(r"\binfarct(ion)?\b|\bstemi\b|\bacute mi\b|\bst elevation mi\b", line, flags=re.IGNORECASE):
        return True
    if re.search(r"myocardial infarction", line, flags=re.IGNORECASE):
        return True
    if re.search(r"\bq waves?\b", line, flags=re.IGNORECASE) and re.search(
        r"infarct", line, flags=re.IGNORECASE
    ):
        return True
    return False


def generate_labels(
    frame: pd.DataFrame,
    mapper: LabelMapper,
    report_prefix: str = "report_",
) -> pd.DataFrame:
    report_cols = report_columns_of(frame, prefix=report_prefix)
    subject_ids = frame["subject_id"].to_numpy()
    study_ids = frame["study_id"].to_numpy()
    report_values = frame[report_cols].to_numpy()
    records = []
    for i in range(len(frame)):
        if i and i % 100000 == 0:
            print(f"  labeled {i}/{len(frame)} studies")
        statements: list[str] = []
        for value in report_values[i]:
            if mapper.is_missing(value):
                continue
            text = re.sub(r"\s+", " ", str(value).strip())
            if text:
                statements.append(text)
        result = mapper.label_statements(statements)
        rec = {
            "subject_id": subject_ids[i],
            "study_id": study_ids[i],
            "poor_quality": result.poor_quality,
            "unmapped": result.unmapped,
            "ambiguous": result.ambiguous,
            "uncertain_labels": "|".join(result.uncertain_labels),
            "n_statements": len(result.source_statements),
            "raw_report_text": " | ".join(result.source_statements),
            "trigger_statements": " || ".join(result.trigger_statements),
        }
        for label in mapper.label_order:
            rec[label] = int(result.labels[label])
            rec[f"evidence_{label}"] = " || ".join(result.evidence[label])
        records.append(rec)
    out = pd.DataFrame.from_records(records)
    for label in mapper.label_order:
        if out[label].isna().any():
            raise ValueError(f"Missing values generated for {label}")
        if set(out[label].unique()) - {0, 1}:
            raise ValueError(f"Non-binary values generated for {label}")
    return out


def label_prevalence(labels: pd.DataFrame, label_order: list[str] | None = None) -> pd.DataFrame:
    order = label_order or LABEL_ORDER
    rows = []
    n = len(labels)
    for name in order:
        pos = int(labels[name].sum())
        rows.append(
            {
                "label": name,
                "positives": pos,
                "negatives": int(n - pos),
                "prevalence": float(pos / n) if n else None,
                "no_positives": pos == 0,
                "no_negatives": pos == n and n > 0,
            }
        )
    return pd.DataFrame(rows)
