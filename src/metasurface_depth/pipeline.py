"""End-to-end RGB-D -> optical images -> neural metric-depth prediction."""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .infer import preprocess
from .io import new_directory, write_json
from .model import load_model
from .simulator.api import PSF_SHA256, Simulator


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rgb", required=True, type=Path)
    p.add_argument("--depth", required=True, type=Path)
    p.add_argument("--weights", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    simulator = Simulator(device=args.device)
    pair = simulator.simulate(args.rgb, args.depth)
    output = new_directory(args.output)
    paths = [output / f"image{i + 1}.png" for i in range(2)]
    for image, path in zip(pair, paths):
        Image.fromarray(image).save(path)
    # Release simulator buffers before loading a large neural model.
    del simulator
    torch.cuda.empty_cache()
    tensor, crop = preprocess(*paths)
    model, config = load_model(args.weights, args.device)
    with torch.inference_mode():
        prediction = model(tensor.to(args.device)).clamp(0.2, 1.2).squeeze().cpu().numpy()
    if not np.isfinite(prediction).all():
        raise RuntimeError("Nonfinite prediction")
    np.save(output / "prediction.npy", prediction.astype(np.float32))
    write_json(
        output / "prediction.json",
        dict(
            **crop,
            config=config,
            units="meters",
            psf_sha256=PSF_SHA256,
            ground_truth_used_by_network=False,
        ),
    )
    print(output / "prediction.npy")


if __name__ == "__main__":
    main()
