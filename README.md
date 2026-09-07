# Physically Grounded Monocular Depth

**Nanophotonic wavefront encoding · RGB-D simulation · Neural depth estimation**

Research code for [Physically Grounded Monocular Depth via Nanophotonic Wavefront Encoding](https://arxiv.org/abs/2503.15770), ECCV 2026.

This repository brings the optical simulator and the learning pipeline together:

```text
RGB image + metric depth → V5 metasurface simulator → two encoded images
                                                       ├─ training + depth labels
                                                       └─ inference → metric depth
Real sensor image pair ────────────────────────────────────────┘
```

The model sees the encoded image pair, **not ground-truth depth**. Depth is required
by the simulator and as a training/evaluation label. Real sensor pairs bypass simulation.

[Paper](https://arxiv.org/abs/2503.15770) ·
[Video](https://www.youtube.com/watch?v=Qkb4nXjKlwU) ·
[Models](https://huggingface.co/Bingxuan111/metasurface-depth-eccv2026)

The released checkpoints use mixed synthetic/real training. See
[validation and limitations](docs/VALIDATION.md) for paper-metric and
test-boundary differences before comparing results.

## What's included

| Component | Entry point | Output |
| --- | --- | --- |
| RGB-D simulation | `metasurface simulate` | Encoded image pairs and a dataset manifest |
| Training / resume | `metasurface train` | Checkpoints, configuration and JSONL logs |
| Evaluation | `metasurface evaluate` | Per-image metrics and mean/std summaries |
| Inference | `metasurface infer` | Float32 metric depth and crop metadata |
| Complete pipeline | `metasurface pipeline` | Simulated pair and neural depth prediction |
| Weight export | `metasurface export` | Inference-only model state |

## Installation

Use a dedicated Python 3.10+ environment and install a compatible PyTorch build
for your CUDA driver first. Then, from this repository:

```bash
python -m pip install -e '.[simulator,train]'
python -m metasurface_depth --help
```

The qualified simulator requires an NVIDIA GPU. Inference also accepts `--device cpu`.
The V5 PSF is bundled and checksum-verified; neither simulation nor inference makes
automatic network downloads. xFormers is optional, but memory use and numerical
behavior can differ without it. See [installation and runtimes](docs/INSTALLATION.md).

## Try the complete pipeline

Place the selected Small weight file at `checkpoints/small.pth` (see
[model files](checkpoints/README.md)). Generate a procedural RGB-D scene and run:

```bash
metasurface demo-data --output runs/demo-rgbd
metasurface pipeline \
  --rgb runs/demo-rgbd/scene_00.png \
  --depth runs/demo-rgbd/scene_00.npy \
  --weights checkpoints/small.pth \
  --output runs/demo-prediction
```

The result contains `image1.png`, `image2.png`, `prediction.npy` in meters, and
`prediction.json` with crop coordinates. The procedural scene tests the software;
it is not a scientific benchmark or a promised accuracy example.

To exercise **simulation, training, checkpoint resume, export, evaluation and
inference** without downloading any model:

```bash
python examples/smoke.py --output runs/smoke
```

This uses random initialization and four training updates. It verifies plumbing,
not convergence or paper-quality accuracy. A successful run writes `SUCCESS.json`.
Commands refuse to reuse existing output directories; choose a new run name.

## Use your own data

The real captured release is prepared separately: **5 training scenes and 42
evaluation scenes**, each with two encoded inputs and a depth label. See
[real dataset contents and protocol](docs/REAL_DATASET.md), including the important
validation/model-selection reuse disclosure. Public hosting is pending.

Create a CSV with `id,rgb,depth` columns; paths are relative to that CSV:

```csv
id,rgb,depth
scene001,images/scene001.png,depth/scene001.npy
```

Use uncropped 8-bit RGB and aligned floating-point depth in meters, within 0.2–1.2 m.

```bash
metasurface simulate --manifest data/rgbd.csv --output runs/simulated
```

**Simulate before cropping.** Cropping RGB-D first changes the physical boundary
conditions. The model subsequently center-crops to multiples of 14; training uses
aligned random crops. See the [data contract](docs/DATA.md).

## Train, evaluate, infer

Prepare disjoint train/validation CSVs and adapt the path-only entries in
`configs/small.json`, `base.json` or `large.json`. Initialize from the matching
DAV2 metric-Hypersim checkpoint:

```bash
metasurface train --config configs/small.json \
  --pretrained checkpoints/dav2_metric_hypersim_small.pth --output runs/train-small
metasurface evaluate --manifest data/test.csv \
  --weights runs/train-small/best.pth --output runs/test-metrics.json
metasurface infer --model small --weights checkpoints/small.pth \
  --image1 path/to/image1.png --image2 path/to/image2.png --output runs/depth.npy
```

Training also accepts `--weights` for fine-tuning a project checkpoint and `--resume`
for restoring training state. See [training and reproducibility](docs/TRAINING.md)
before comparing a new run with the released models.

## Repository guide

```text
src/metasurface_depth/
  simulator/     validated API + frozen scientific kernel
  models/        experiment-specific DPT components
  _vendor/       isolated DINOv2 implementation
  assets/        fixed PSF calibration
  data.py        aligned dataset loading and split checks
  train.py       explicit training loop and checkpoints
  metrics.py     shared loss and evaluation definitions
configs/         Small / Base / Large training recipes
examples/        executable end-to-end smoke workflow
tests/           CPU contract tests
docs/            architecture, data, training and validation
```

See [architecture](docs/ARCHITECTURE.md), [validation and limitations](docs/VALIDATION.md),
[contributing](CONTRIBUTING.md), and [third-party notices](THIRD_PARTY_NOTICES.md).

## Citation

Please cite the paper when using this research. Machine-readable metadata is in
[`CITATION.cff`](CITATION.cff).

```bibtex
@article{li2026physically,
  title={Physically Grounded Monocular Depth via Nanophotonic Wavefront Encoding},
  author={Li, Bingxuan and Wu, Jiahao and Xu, Yuan and Zhu, Zezheng and Zhang, Yunxiang and Chen, Kenneth and Liang, Yanqi and Yu, Nanfang and Sun, Qi},
  journal={arXiv preprint arXiv:2503.15770},
  year={2026}
}
```

## Licensing

Code and pretrained weights have different provenance and terms. Do not interpret
this repository as granting unrestricted commercial rights to all three models.
Original project code is released under [MIT](LICENSE). Third-party code retains
its original licenses. See [licensing](docs/LICENSING.md) for component and weight terms.
