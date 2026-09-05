# Validation and known limitations

## Frozen references

The initial inference extraction compared all three selected November checkpoints
against the original experimental network on 342 images per model: 42 real, 100
MIT-CGH, 100 NYU and 100 Hypersim. All 1,026 raw predictions were bitwise identical.
Every exported tensor was also exactly preserved. These results apply to that
reference extraction; current integrated-repository tests are recorded below.

The locked simulator is V5 with extension=15 and the bundled 521-plane PSF.
Eight training samples and four NYU samples regenerated exactly. Four MIT samples,
using full 384×384 RGB-D followed by cropping to 378×378, differed at only three
pixels total, each by one uint8 level. Twelve other already-cropped test inputs
were not treated as successful full-image reconstructions.

## Unresolved scientific differences

- **Base real-data metrics:** re-evaluation MAE/RMSE/AbsRel/delta1 =
  0.034382/0.080957/0.059448/0.860636; the inspected paper table reports
  0.032/0.089/0.055/0.895. Small and Large matched at reported precision. Do not
  claim that every paper entry is reproduced.
- **Hypersim test boundaries:** current RGB-D inputs are already cropped.
  Full-image differences remain (up to 115 uint8 levels in the audit); excluding
  35-pixel borders leaves sparse one-level differences. Crop-order provenance is
  a supported hypothesis, not a completed proof. Original data was not overwritten.
- **Training:** bounded smoke tests do not establish convergence, paper accuracy,
  exact sample ordering, historical optimizer trajectories or multi-GPU parity.
- **Optics:** the qualified API uses the fixed cached PSF. Regenerating PSFs or
  changing calibration/geometry is not covered by cached-kernel regression tests.

## Integrated repository qualification

The integrated package was installed as a wheel and checked outside the source
directory on 2026-09-05. CPU unit tests: **13 passed**. Ruff source checks and
wheel/source-distribution builds passed. The GitHub-hosted workflow has not run.

All three integrated models were independently compared with the saved outputs
of the original experimental source, including input tensor hashes. This is a
new regression of the namespaced integrated package, not only the earlier extraction:

| Model | Images | Raw output comparison | Maximum difference |
| --- | ---: | --- | ---: |
| Small | 342 | All bitwise identical | 0 |
| Base | 342 | All bitwise identical | 0 |
| Large | 342 | All bitwise identical | 0 |

Runtime: NVIDIA H100 80 GB, PyTorch 2.8.0+cu128, NumPy 2.2.6, OpenCV 4.12.0,
xFormers 0.0.32.post2.

### Complete workflow: passed

On H100, the installed package generated three procedural RGB-D scenes and their
encoded pairs, fine-tuned the selected Small export for one update, restored the
model/optimizer/scheduler into a new run, and completed updates 2–4. This crosses
the configured warm-up and augmentation boundaries. Export removed training state;
evaluation and inference loaded the export successfully. The one-command RGB-D
pipeline produced a depth array **bitwise identical** to separate simulation and
inference. The run wrote `SUCCESS.json`. This is functional qualification, not an
80,000-step scientific training reproduction. The executed smoke used `--weights`;
the no-weights random-initialization example is not a paper result.

### Integrated simulator versus frozen V5: passed

The public API and independently imported frozen experimental V5 were run in the
same PyTorch 2.8.0/H100 environment with identical cached PSF and settings. Three
112×140 procedural scenes, one 768×1024 Hypersim RGB-D pair, and one full 384×384
MIT RGB-D pair produced **bitwise-identical uint8 outputs: 5 pairs / 10 images**.
This verifies the packaging/refactor; it does not resolve the historical
already-cropped test-input discrepancies described above.

The simulation extras in this environment included SciPy 1.16.3, scikit-learn
1.9.0, h5py 3.15.1 and Matplotlib 3.10.7. An initial qualification attempt stopped
at a missing transitive dependency in the manually prepared environment. After
installing the normal dependency closure, both GPU gates passed; no scientific
kernel change was needed.

### Additional release gates

- The original DAV2 metric-Hypersim Small checkpoint loads through the new
  initialization contract: no unexpected keys or missing backbone keys; 44 new
  project parameters are intentionally absent. This was a CPU load/compatibility
  check, not an additional training-convergence experiment.
- All three local distribution files match the recorded sizes and SHA-256 hashes.
- Source syntax, local Markdown links and common credential/internal-path scans
  passed. Final whitespace cleanup preserved the AST of all 21 vendored/scientific
  modules compared with the GPU-qualified source snapshot.
- No original experimental repository, training dataset or selected model tensor
  was changed. No public upload has been performed.
