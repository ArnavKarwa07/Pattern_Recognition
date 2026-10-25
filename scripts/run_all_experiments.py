"""Full pipeline: prepare -> validate -> train all -> evaluate all -> figures -> report.

Usage:  python scripts/run_all_experiments.py --config configs/default.yaml [--resume]

Stops at the first failing stage. Validation is skipped when the manifest is
unchanged; data preparation re-reads the protocols/parquet but does not
rewrite CD-ADD audio files that already exist; cached features are reused.
"""

import argparse
import sys

from speechdeepfake.cli import main


def run() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--resume", action="store_true", help="resume CNN training from last.pt")
    args = ap.parse_args()
    stages = [
        ["prepare-data", "--config", args.config],
        ["validate-data", "--config", args.config],
        ["run-all", "--config", args.config, *(["--resume"] if args.resume else [])],
    ]
    for argv in stages:
        print(f"\n=== speechdeepfake {' '.join(argv)}")
        code = main(argv)
        if code != 0:
            print(f"Stage {argv[0]} failed with exit code {code}; stopping.")
            return code
    return 0


if __name__ == "__main__":
    sys.exit(run())
