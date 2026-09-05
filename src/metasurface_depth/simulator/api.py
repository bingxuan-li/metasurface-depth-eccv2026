"""Validated, cached-PSF API around the frozen experimental image-formation kernel."""

from importlib.resources import files
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ..io import sha256

PSF_SHA256 = "6ed9694f916c3c7bc17b709a79a28a002fa4be9395bab0560955928bcfb5a256"


class Simulator:
    """Generate two uint8 encoded images from uncropped RGB and metric depth.

    The cached PSF and V5 extension=15 path are a single calibrated baseline.
    No network or ground-truth depth estimator is involved in simulation.
    """

    def __init__(self, psf_path=None, device="cuda", max_memory=24):
        if not str(device).startswith("cuda") or not torch.cuda.is_available():
            raise RuntimeError("The qualified V5 simulator requires CUDA")
        if not 1 <= max_memory <= 48:
            raise ValueError("max_memory must be between 1 and 48 GB")
        from ._kernel import DepthEncoder

        self.device = torch.device(device)
        path = (
            Path(psf_path)
            if psf_path
            else Path(str(files("metasurface_depth") / "assets/psf_v5.pt"))
        )
        if sha256(path) != PSF_SHA256:
            raise ValueError("PSF checksum differs from the locked V5 calibration")
        self.encoder = DepthEncoder(
            depth_sigma=2500,
            max_memory=max_memory,
            extension=15,
            shotnoise=False,
            requires_grad=False,
        )
        self.encoder.read_psflist(str(path))

    @torch.inference_mode()
    def simulate(self, rgb_path, depth_path):
        from ._kernel import ReadImageDepth

        rgb_path, depth_path = Path(rgb_path), Path(depth_path)
        if depth_path.suffix.lower() != ".npy":
            raise ValueError("Depth must be a floating-point .npy array in meters")
        depth = np.load(depth_path, allow_pickle=False)
        if depth.ndim != 2 or depth.dtype.kind != "f" or not np.isfinite(depth).all():
            raise ValueError("Depth must be a finite floating-point H-by-W array")
        if depth.min() < 0.2 - 0.0002 or depth.max() > 1.2 + 0.0012:
            raise ValueError("This release supports RGB-D depths in [0.2, 1.2] meters")
        with Image.open(rgb_path) as image:
            if image.mode not in ("L", "RGB", "RGBA"):
                raise ValueError("RGB input must be an 8-bit L/RGB/RGBA image")
            if image.size != (depth.shape[1], depth.shape[0]):
                raise ValueError("RGB and depth must have identical dimensions")
        sample = ReadImageDepth(str(rgb_path), str(depth_path), depth_magnification=1e6)
        output = self.encoder(sample, device=self.device, ifhard=False)
        array = output.cpu().numpy()
        if not np.isfinite(array).all():
            raise RuntimeError("Nonfinite simulator output")
        return array.clip(0, 255).astype(np.uint8)
