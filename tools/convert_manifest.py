"""Convert legacy image1,image2,depth[,rgb] text lists without guessing path migrations."""

import argparse
import csv
import os
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    entries = []
    for line in args.source.read_text().splitlines():
        if not line.strip():
            continue
        values = next(csv.reader([line], skipinitialspace=True))
        if len(values) < 3:
            raise ValueError("Expected at least image1,image2,depth")
        paths = [(args.source.resolve().parent / value.strip()).resolve() for value in values[:3]]
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(path)
        entries.append([os.path.relpath(path, args.output.resolve().parent) for path in paths])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["image1", "image2", "depth"])
        writer.writerows(entries)


if __name__ == "__main__":
    main()
