"""Batch RGB-D simulation; preserve full image extent before network cropping."""

import argparse
import csv
import os
import re
from pathlib import Path

from PIL import Image

from .io import new_directory, read_manifest, write_json
from .simulator.api import PSF_SHA256, Simulator


def run(manifest, output, device="cuda", max_memory=24):
    rows = read_manifest(manifest, ("rgb", "depth"))
    ids = [row.get("id") or f"{i:06d}" for i, row in enumerate(rows)]
    if len(set(ids)) != len(ids) or any(not re.fullmatch(r"[A-Za-z0-9_-]+", x) for x in ids):
        raise ValueError("IDs must be unique and contain only letters, digits, '_' or '-'")
    simulator = Simulator(device=device, max_memory=max_memory)
    output = new_directory(output).resolve()
    with (output / "encoded.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["id", "image1", "image2", "depth"])
        writer.writeheader()
        for id_, row in zip(ids, rows):
            pair = simulator.simulate(row["rgb"], row["depth"])
            names = [f"{id_}-{i + 1}.png" for i in range(2)]
            for name, image in zip(names, pair):
                Image.fromarray(image).save(output / name)
            writer.writerow(
                dict(
                    id=id_,
                    image1=names[0],
                    image2=names[1],
                    depth=os.path.relpath(row["depth"], output),
                )
            )
    write_json(
        output / "simulation.json",
        dict(
            version="v5-ext15",
            psf_sha256=PSF_SHA256,
            samples=len(rows),
            depth_units="meters",
            crop="none; simulate full RGB-D first",
        ),
    )
    return output / "encoded.csv"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-memory", type=int, default=24)
    args = parser.parse_args()
    print(run(args.manifest, args.output, args.device, args.max_memory))


if __name__ == "__main__":
    main()
