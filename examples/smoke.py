"""Exercise simulation, training, resume, export, evaluation and inference on one GPU."""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

from metasurface_depth.demo import create_demo
from metasurface_depth.io import new_directory
from metasurface_depth.simulate import run


def command(*arguments):
    subprocess.run([sys.executable, "-m", "metasurface_depth", *map(str, arguments)], check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--weights",
        type=Path,
        help="Optional project Small weights; otherwise random smoke initialization",
    )
    args = p.parse_args()
    root = new_directory(args.output).resolve()
    raw = create_demo(root / "rgbd")
    encoded = run(raw, root / "encoded")
    with encoded.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        rows = list(reader)
    for name, subset in (("train", rows[:2]), ("val", rows[2:])):
        with (encoded.parent / f"{name}.csv").open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(subset)
    config = dict(
        model="small",
        steps=2,
        warmup_steps=2,
        batch_size=1,
        crop_size=56,
        validate_every=1,
        save_every=1,
        log_every=1,
        augmentation_start=2,
        train=[dict(manifest="encoded/train.csv", weight=1, augment=True)],
        validation=[dict(manifest="encoded/val.csv")],
    )
    cfg = root / "smoke.json"
    cfg.write_text(json.dumps(config, indent=2))
    initial = ["--weights", args.weights.resolve()] if args.weights else ["--random-init"]
    command("train", "--config", cfg, "--output", root / "train", "--stop-after", 1, *initial)
    command(
        "train", "--config", cfg, "--output", root / "resume", "--resume", root / "train/last.pth"
    )
    command("export", "--checkpoint", root / "resume/last.pth", "--output", root / "export.pth")
    command(
        "evaluate",
        "--manifest",
        encoded.parent / "val.csv",
        "--weights",
        root / "export.pth",
        "--output",
        root / "evaluation.json",
    )
    command(
        "infer",
        "--model",
        "small",
        "--weights",
        root / "export.pth",
        "--image1",
        encoded.parent / rows[2]["image1"],
        "--image2",
        encoded.parent / rows[2]["image2"],
        "--output",
        root / "inference.npy",
    )
    command(
        "pipeline",
        "--rgb",
        root / "rgbd/scene_02.png",
        "--depth",
        root / "rgbd/scene_02.npy",
        "--weights",
        root / "export.pth",
        "--output",
        root / "pipeline",
    )
    import numpy as np

    np.testing.assert_array_equal(
        np.load(root / "inference.npy"), np.load(root / "pipeline/prediction.npy")
    )
    import torch

    trained = torch.load(root / "resume/last.pth", map_location="cpu", weights_only=True)
    exported = torch.load(root / "export.pth", map_location="cpu", weights_only=True)
    assert trained["training"]["step"] == 4
    assert "training" not in exported
    summary = dict(
        passed=True,
        steps=4,
        resumed=True,
        pipeline_matches_separate_inference=True,
        note="Functional smoke test, not a paper-quality training run",
    )
    (root / "SUCCESS.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
