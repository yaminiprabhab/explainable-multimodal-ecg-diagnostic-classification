"""Build outputs/comparison from each model’s test_predictions.csv.

Run after the individual pipelines. Inner-joins study_id so Models 1/3/4
are only compared on studies that also have residual text (and 2B if present).
Does not retune thresholds.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from shared.compare import compare_output_dirs, default_named_dirs
from shared.io_utils import ensure_dir, repo_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a common-cohort comparison report")
    parser.add_argument("--root", default=None)
    parser.add_argument("--dest", default="outputs/comparison")
    args = parser.parse_args()
    root = Path(args.root).resolve() if args.root else repo_root()
    dest = ensure_dir(root / args.dest) if not Path(args.dest).is_absolute() else ensure_dir(Path(args.dest))
    report = compare_output_dirs(default_named_dirs(root), dest)
    print("models compared", report.get("n_models_compared"), "common studies", report.get("n_common_test_studies"))
    print("wrote", dest / "comparison.md")


if __name__ == "__main__":
    main()
