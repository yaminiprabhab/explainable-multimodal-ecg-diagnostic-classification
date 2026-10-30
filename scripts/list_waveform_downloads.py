"""List PhysioNet paths for the frozen split only — not the full ~800k dump.

Reads artifacts/patient_split_manifest.csv (created by Model 2). Writes
relative folders plus .hea/.dat URLs you can feed to wget. Models 1/3/4
still skip any study whose file is missing; downloading a subset is valid.
"""

from __future__ import annotations

import argparse

from shared.io_utils import repo_root, resolve_path
from shared.waveforms import PHYSIONET_MIMIC_IV_ECG, write_download_lists


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write WFDB download paths for the frozen Model 2 split only"
    )
    parser.add_argument("--manifest", default="artifacts/patient_split_manifest.csv")
    parser.add_argument("--dest", default="artifacts")
    parser.add_argument("--base-url", default=PHYSIONET_MIMIC_IV_ECG)
    args = parser.parse_args()
    root = repo_root()
    manifest = resolve_path(root, args.manifest)
    dest = resolve_path(root, args.dest)
    if not manifest.is_file():
        raise SystemExit(
            f"Missing {manifest}. Run Model 2 first so the frozen split exists."
        )
    stats = write_download_lists(manifest, dest, base_url=args.base_url)
    print(
        "studies",
        stats["n_studies"],
        "urls",
        stats["n_urls"],
        "wrote",
        stats["relative_list"],
        "and",
        stats["url_list"],
    )


if __name__ == "__main__":
    main()
