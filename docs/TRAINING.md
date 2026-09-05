# Training and reproducibility

The recipes cover Small/Base/Large, 518-pixel aligned random crops, batch sizes
8/8/2, 80,000 main steps plus 1,000 warm-up steps, and synthetic/real sampling
probabilities 0.95/0.05. Supply your own matching data manifests and DAV2
metric-Hypersim initialization file; there is no automatic download.

## Preserved numerical ingredients

- Unclipped prediction loss: L1 + 0.5 × gradient-magnitude L1. Horizontal and
  vertical absolute gradients are compared separately, matching the original.
- AdamW: backbone LR 3e-6, head LR 1e-5, weight decay 0.01. During warm-up only
  the backbone is trainable. At the warm-up boundary the optimizer is recreated,
  matching the original transition. StepLR decays every 10,000 updates by 0.8.
- The original sensor augmentation class is retained: blur, global/local
  imbalance, shot noise and Gaussian noise, applied independently to the two
  encoded images of synthetic samples. Defaults start at step 2,000, p=0.4.
  Real samples disable augmentation in the supplied recipes.
- Best-checkpoint selection defaults to the original combined metric on the
  configured validation dataset: sqrt(MAE × AbsRel × RMSE) × (1 − delta0.5).
  Recipes select the real validation dataset, not an average of unlike sets.
- Only evaluation clips predictions to 0.2–1.2 m. Statistics are per-image means
  and population standard deviations, not pixel-pooled dataset aggregates.

## Intentional engineering changes

This is a maintainable single-device training loop, not a promise of bitwise
historical training reproduction. It replaces DataParallel and multiworker
loading with a single device and deterministic worker-free sampling, uses an
explicit seed, drops incomplete training batches, and applies augmentation at
an explicit step boundary rather than relying on worker restart behavior.
An initial pre-training validation pass is not part of the loop.

Best checkpoints are serialized immediately from cloned tensors. The original
code retained references to live state tensors, so historical `step` metadata is
not reliable proof of an immutable best-weight snapshot. Selected released
weights are preserved exactly and are not silently replaced by retraining.

`--resume` restores model, optimizer, scheduler and absolute step with the same
config, into a **new** output directory. Data-loader cursor and RNG stream are
restarted from the configured seed; this is restartable training, not exact
continuation of a historical sample sequence. `--stop-after N` creates a bounded
run useful for testing resume without changing the experiment config.

## Output and export

Each run includes resolved settings, input-manifest hashes, initialization
provenance, local `metrics.jsonl`, `last.pth`, and `best.pth` when validation improves
the best score. A resumed run retains the previous best score; if it never improves,
its new directory has no `best.pth`. Keep the earlier run's best checkpoint or use
the resumed `last.pth` explicitly. The smoke test exports the final checkpoint.
Training checkpoints contain optimizer state and may include private paths in
their run settings. Do not publish them directly:

```bash
metasurface export --checkpoint runs/train-small/best.pth --output runs/small-export.pth
```

The export keeps model tensors and inference metadata only. Loading DAV2 weights
allows missing new head parameters but rejects a missing backbone or unexpected
keys. Loading a project checkpoint is strict. Full paper-quality training runs
and all-device qualification remain separate from the functional smoke test.
