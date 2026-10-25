"""Validate manifests and audio files (equivalent to `speechdeepfake validate-data`).

Usage:  python scripts/validate_data.py --config configs/default.yaml [--workers 8] [--force]
"""

import sys

from speechdeepfake.cli import main

if __name__ == "__main__":
    sys.exit(main(["validate-data", *sys.argv[1:]]))
