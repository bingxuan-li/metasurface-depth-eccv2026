"""Pair of encoded sensor images -> float32 metric depth and crop metadata."""

import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from .model import load_model


def preprocess(image1, image2):
    arrays = []
    for path in (image1, image2):
        raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise ValueError(f"Cannot read image: {path}")
        if raw.dtype != np.uint8:
            raise ValueError(
                "Inputs must be uint8 images; calibrated raw/16-bit conversion is not defined"
            )
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        arrays.append(image.astype(np.float32) / 255.0)
    a, b = arrays
    if a.shape != b.shape:
        raise ValueError("The two input images must have identical dimensions")
    h, w = a.shape
    ch, cw = h // 14 * 14, w // 14 * 14
    if min(ch, cw) < 14:
        raise ValueError("Input must be at least 14x14")
    y, x = (h - ch) // 2, (w - cw) // 2
    # Torch arithmetic matches the original SimulatedDataset exactly.
    a = torch.from_numpy(a[y : y + ch, x : x + cw]).unsqueeze(0)
    b = torch.from_numpy(b[y : y + ch, x : x + cw]).unsqueeze(0)
    tensor = torch.cat((a, b, (a + b) / 2), dim=0).unsqueeze(0)
    return tensor, dict(original_shape=[h, w], crop_xywh=[x, y, cw, ch], output_shape=[ch, cw])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["small", "base", "large"], required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--image1", type=Path, required=True)
    parser.add_argument("--image2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New .npy output, in meters")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.output.suffix.lower() != ".npy":
        parser.error("--output must end with .npy")
    metadata = args.output.with_suffix(".json")
    if args.output.exists() or metadata.exists():
        parser.error("Output or sidecar already exists")
    tensor, crop = preprocess(args.image1, args.image2)
    model, config = load_model(args.weights, args.device, args.model)
    with torch.inference_mode():
        depth = (
            model(tensor.to(args.device))
            .clamp(*config["depth_range_m"])
            .squeeze(0)
            .squeeze(0)
            .cpu()
            .numpy()
        )
    if not np.isfinite(depth).all():
        raise RuntimeError("Nonfinite model prediction")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as f:
        np.save(f, depth.astype(np.float32))
    metadata.write_text(
        json.dumps(dict(model=args.model, units="meters", **crop, config=config), indent=2)
    )
    print(f"Saved {args.output}, shape={depth.shape}, range={depth.min():.6f}..{depth.max():.6f} m")


if __name__ == "__main__":
    main()
