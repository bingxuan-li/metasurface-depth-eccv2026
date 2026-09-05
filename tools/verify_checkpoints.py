"""Verify locally supplied release weights against the published manifest."""

import argparse
import json
from pathlib import Path

from metasurface_depth.io import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("checkpoints"))
    args = parser.parse_args()
    manifest = json.loads((args.directory / "manifest.json").read_text())
    for variant, entry in manifest.items():
        path = args.directory / entry["filename"]
        if path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
            raise ValueError(f"Checkpoint verification failed: {variant}")
        print(f"{variant}: verified")


if __name__ == "__main__":
    main()
