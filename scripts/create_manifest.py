"""Build normalised manifests (equivalent to `speechdeepfake prepare-data`).

Usage:  python scripts/create_manifest.py --config configs/default.yaml [--corpus cdadd]
"""

import sys

from speechdeepfake.cli import main

if __name__ == "__main__":
    sys.exit(main(["prepare-data", *sys.argv[1:]]))
