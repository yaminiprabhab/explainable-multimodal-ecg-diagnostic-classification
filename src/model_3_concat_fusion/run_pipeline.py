"""CLI wrapper: fusion_type='concat' via shared.fusion_pipeline.

Expects frozen Model 2 artifacts plus usable waveforms. Optional Model 1/2
checkpoints are loaded when present (skipped during --smoke).
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
        fusion_type="concat",
        artifacts_dir=artifacts_dir,
        output_dir=output_dir,
        waveform_root=waveform_root,
        smoke=smoke,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Model 3 concat fusion")
    parser.add_argument("--config", default=str(default_config("model_3_concat_fusion")))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    summary = run_pipeline(Path(args.config), smoke=args.smoke)
    metrics = summary.get("test_metrics") or {}
    print("macro_f1", metrics.get("macro_f1"), "micro_f1", metrics.get("micro_f1"))


if __name__ == "__main__":
    main()
