"""Evaluate exported weights with native-resolution, center-cropped inputs."""

import argparse
from pathlib import Path

import numpy as np
import torch

from .data import EncodedDataset
from .io import sha256, write_json
from .metrics import metrics
from .model import load_model


@torch.inference_mode()
def evaluate_model(model, dataset, device, foreground=False):
    model.eval()
    rows = []
    for index in range(len(dataset)):
        image, target = dataset[index]
        pred = model(image.unsqueeze(0).to(device)).clamp(0.2, 1.2).cpu()
        row = dict(index=index, full=metrics(pred, target.unsqueeze(0)))
        if foreground:
            row["foreground"] = metrics(pred, target.unsqueeze(0), foreground=True)
        rows.append(row)
    regions = ["full", "foreground"] if foreground else ["full"]
    summary = {
        region: {
            key: dict(
                mean=float(np.mean([r[region][key] for r in rows])),
                std=float(np.std([r[region][key] for r in rows])),
            )
            for key in rows[0][region]
        }
        for region in regions
    }
    return dict(samples=len(rows), summary=summary, rows=rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--weights", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--foreground", action="store_true")
    args = p.parse_args()
    if args.output.exists():
        p.error("Output already exists")
    model, config = load_model(args.weights, args.device)
    result = evaluate_model(model, EncodedDataset(args.manifest), args.device, args.foreground)
    result.update(
        model=config,
        weights_sha256=sha256(args.weights),
        manifest_sha256=sha256(args.manifest),
        torch=str(torch.__version__),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(result["summary"])


if __name__ == "__main__":
    main()
