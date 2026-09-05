"""Strip training state from a new-format checkpoint for distribution."""

import argparse
from pathlib import Path

import torch

from .io import sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".json").exists():
        parser.error("Output or checksum sidecar already exists")
    source = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if source.get("format_version") != 1:
        raise ValueError("Expected a format_version=1 project checkpoint")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        torch.save({key: source[key] for key in ("format_version", "config", "state_dict")}, stream)
    write_json(
        args.output.with_suffix(".json"),
        dict(filename=args.output.name, sha256=sha256(args.output)),
    )
    print(args.output)


if __name__ == "__main__":
    main()
