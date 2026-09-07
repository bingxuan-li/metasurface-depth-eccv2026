# Real captured data

The prepared `metasurface-real-v1.zip` contains the exact real-data lists named
by the selected mixed-training Small, Base and Large checkpoints:

| Split | Scenes | PNG inputs | NPY depth labels | Height x width |
| --- | ---: | ---: | ---: | --- |
| train | 5 | 10 | 5 | 1200 x 1600 |
| test | 42 | 84 | 42 | 1190 x 1596 |

**The historical test split was also used for validation/model selection. It is
not an untouched holdout.** The splits have no shared source paths or identical
ordered input-pair hashes; this does not rule out shared objects or categories.

All 141 data files preserve the original bytes. The archive additionally contains
`train.csv`, `test.csv` and `manifest.json` with per-file SHA-256 checksums. Paths
are relative, without internal HPC paths. The archive SHA-256 is
`8a239a732a524c77c3b226741baca21fd46af7de9b1d773a3a63dd0facf21f5f`.

After extracting the archive and obtaining the selected weights:

```bash
metasurface evaluate --manifest metasurface-real-v1/test.csv \
  --weights checkpoints/small.pth --output runs/real-small.json
```

Each CSV row has `id,image1,image2,depth`. PNG input order is significant. Labels
are aligned float32 arrays in meters, with observed values approximately
0.25--1.10 m. Use the arrays, not distance tokens in filenames. The fourth field
in the old lists duplicated image1 and was not an additional RGB photograph.

These are the processed image pairs and supplied supervision arrays used by the
experiments, not all raw sensor streams. Annotation and measurement details still
need author confirmation. Real pairs bypass simulation; the five real training
scenes do not replace the synthetic portion of mixed training.

Point the real-data training manifest in a selected configuration to `train.csv`.
Retain the synthetic data source and mixture described in [TRAINING.md](TRAINING.md).
The shared loader handles aligned crops; evaluation images already have dimensions
divisible by 14. Ground truth is a label, never a neural input.

Publication status: publicly available on
[Hugging Face](https://huggingface.co/datasets/Bingxuan111/metasurface-real-eccv2026)
under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/), with
author-approved public noncommercial distribution as of 2026-09-07. Academic
research is permitted; commercial use is not permitted under this license.
Provide attribution, link to the license, and identify changes when reusing data.
The license and dataset card are recorded at commit `da94b86`; the original
archive bytes and SHA-256 above are unchanged. Code and model weights retain
their separate licenses.
