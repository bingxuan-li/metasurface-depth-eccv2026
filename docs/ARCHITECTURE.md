# Architecture and maintenance boundaries

The public API is under one `metasurface_depth` namespace. It does not install
top-level `promptda`, `dinov2` or `vision_transformer` modules, avoiding collisions
with other projects. Commands import optional dependencies only when required.

## Optical model

`simulator/api.py` validates inputs, verifies the fixed PSF SHA-256 and calls the
V5 kernel with extension=15, depth sigma=2500 µm and no additional shot noise.
The cached bank has 521 depth planes over 0.2–1.5 m and 35×35 kernels. Input RGB-D
for this release is limited to the model's 0.2–1.2 m range. Do not trim the bank
to the input range: that changes behavior near the endpoints.

`simulator/_kernel.py` and `_occlusion.py` preserve the audited scientific
implementation. Only package imports and safe checkpoint loading were adjusted.
They intentionally retain PSF-construction research functions for provenance,
but the qualified API uses the bundled calibration. Regenerating a new PSF or
changing physical parameters is a research change requiring separate validation.

## Neural model

`model.py` creates the exact Small/Base/Large DAV2-derived architecture. It loads
all parameter and buffer keys strictly. The DINOv2 defaults are fixed locally;
there are no Torch Hub downloads. Inputs are `[I1, I2, (I1+I2)/2]`; the auxiliary
prompt is zero. The prompt branch remains present because learned biases matter.

`models/` and `_vendor/` are compatibility code. Prefer wrappers and tests over
casual cleanup inside numerical kernels. No network architecture pruning was done.

## Learning and outputs

`data.py` uses the same crop on both encoded images and the target depth. `metrics.py`
owns the loss and metric definitions used by both training and evaluation.
`train.py` writes immutable serialized best snapshots, not live state references.
Local JSON/JSONL replaces mandatory telemetry and machine-specific job scripts.

The simulator is offline data generation, not a differentiable layer in the
training graph. Ground-truth depth must never reach the neural forward function.
The complete pipeline calls only the two simulated images into inference.

When changing code, distinguish exact kernel parity, quantized-image parity,
functional smoke tests, and full training reproduction. These are different claims.
