# Data contract

## RGB-D manifest

CSV header: `id,rgb,depth`. IDs must be unique letters/digits/underscores/hyphens.
Paths are relative to the manifest (absolute paths work for private runs).
RGB must be an 8-bit L/RGB/RGBA image. Depth must be a finite floating-point H×W
`.npy` file in meters, aligned to RGB and within 0.2–1.2 m. Raw 16-bit sensor
conversion is intentionally not guessed.

Simulation retains the original image extent. It writes two grayscale uint8
images, an `encoded.csv`, and a calibration/protocol record. The output manifest
references the original depth label; keep that label available when moving data.

## Encoded manifest

CSV header: `image1,image2,depth` (optional `id`). All three arrays must be spatially
aligned and have identical dimensions. The neural loader decodes each image as
OpenCV grayscale, divides by 255, and constructs `[I1,I2,(I1+I2)/2]` using Torch
arithmetic. Evaluation center-crops to multiples of 14; training random-crops to
the configured size (518 by default). Labels receive the identical crop.

Do not resize images independently, swap image1/image2, normalize each image by
its maximum, or silently rescale metric depth. Raw sensor pairs may bypass the
simulator but still require the same acquisition/calibration conventions.

`tools/convert_manifest.py old.txt new.csv` converts existing four-column text
lists after their paths are valid. It never guesses how private filesystem paths
should be migrated and never rewrites source data.

## Splits and scientific data

Keep train/validation/test disjoint. The trainer rejects shared input or label
paths across training and validation; this is a guardrail, not a proof against
duplicate content under different filenames. Check dataset-level identities too.
The released November model selection used a 95% synthetic / 5% real mixture.
The later February list must not silently replace the November split.

Scientific RGB-D and sensor datasets are not bundled. Obtain them under their
own terms and preserve the exact selected splits. The procedural demo is supplied
for software tests only; it is not a substitute benchmark.

## Why crop order matters

The established flow is **full RGB-D → optical simulation → aligned crop**.
Convolution and occlusion near image boundaries depend on pixels outside a crop.
Applying the simulator to already-cropped RGB-D does not generally reproduce a
crop of the original simulation. The MIT audit verified this distinction. A
Hypersim test-input boundary/provenance discrepancy remains unresolved; see validation.
