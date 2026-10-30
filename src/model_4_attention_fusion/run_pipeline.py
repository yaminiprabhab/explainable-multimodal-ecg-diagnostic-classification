"""CLI wrapper: fusion_type='gated' via shared.fusion_pipeline.

Writes attention_gates.json (mean softmax gates) and per-study gate columns
on test_predictions.csv in addition to the usual metrics and occlusions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from shared.fusion_pipeline import run_fusion_pipeline
from shared.io_utils import default_config


def run_pipeline(
    config_path: Path,
    artifacts_dir: Path | None = None,
    output_dir: Path | None = None,
    waveform_root: Path | None = None,
    smoke: bool = False,
):
    return run_fusion_pipeline(
        config_path,
        fusion_type="gated",
        artifacts_dir=artifacts_dir,
        output_dir=output_dir,
        waveform_root=waveform_root,
        smoke=smoke,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Model 4 gated attention fusion")
    parser.add_argument("--config", default=str(default_config("model_4_attention_fusion")))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    summary = run_pipeline(Path(args.config), smoke=args.smoke)
    metrics = summary.get("test_metrics") or {}
    print("macro_f1", metrics.get("macro_f1"), "micro_f1", metrics.get("micro_f1"))


if __name__ == "__main__":
    main()
