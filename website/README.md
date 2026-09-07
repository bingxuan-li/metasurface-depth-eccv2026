# Project website

Static project page for **Physically Grounded Monocular Depth via Nanophotonic
Wavefront Encoding**, ECCV 2026.

Published at https://bingxuan-li.github.io/metasurface-depth-eccv2026/ using GitHub
Pages. The `Project website` workflow builds and deploys changes to this folder.
It does not modify the author's personal homepage repository.

## Development

Use Node.js 22.13+ and pnpm 11.19.0:

```sh
pnpm install --frozen-lockfile
pnpm dev
pnpm test
PAGES_BASE_PATH=/metasurface-depth-eccv2026/ pnpm build
```

The production output is `dist/`. Vite builds the interactive gallery and a
build-time React render supplies the paper text as static HTML. No server,
database, account, API credentials or external image service is required.
Video playback uses the paper's YouTube embed.

## Content and maintenance

- `src/page.tsx`: paper text, links and figures.
- `src/supplementary-results.tsx`: polarization, model transfer and PSF analysis.
- `src/results-explorer.tsx`: read-only gallery and perspective point-cloud viewer.
- `src/lib/`: capture-anchored camera and depth-tested grid/ruler math.
- `public/results/manifest.json`: the five author-approved examples, with
  descriptive names, checkpoint hashes and projection assumptions.
- `scripts/check-public.mjs`: checks asset availability and excludes review/API links.

Only Camera, Cat, Dragon, Three objects and Four objects are included. Their
point arrays and images preserve the approved preview's bytes; only filenames
are changed. The private 42-scene review tool, decisions, authentication and
database are not copied into this repository or deployment. Simulation point
clouds are not included. Do not copy a complete review export into `public/`.

## Figure provenance

Figures come from the authors' paper materials, not generated illustrations.
No Overleaf archive, unpublished comments or revision history is distributed.

| Website figure | Paper source |
| --- | --- |
| `teaser.png`, `pipeline.png` | Figures referenced by `teaser.tex`, `pipeline.tex` |
| `simulation-more.png` | `qual_sim.pdf`, supplementary simulated comparisons |
| `simulation-ablation.png` | `supp_qualitative_ablation.png`, depth-prior ablation |
| `physical-comparison.png` | `qual_phy_2.pdf`, eleven real comparisons |
| `physical-distance-results.png` (not displayed) | `qual_phy_1.pdf`, six additional real comparisons |
| `depth-consistency.png` | `supp_video.png`, simulation and real motion sequences |
| `simulator-ablation.png` | `sim_real_gap_new2.pdf`, optical forward-model ablation |
| `polarization-robustness.png` | `images/rebuttal/heatmap_mae_3d.pdf`, included in the active supplement |
| `point-source-precision.png` | `images/supp_theory/point_source.pdf` |
| `edge-orientation.png` | `images/supp_theory/edge_phi.pdf` |
| `edge-precision.png` | `images/supp_theory/mean_edge.pdf` |
| `depth-correlation.png` | `images/supp_theory/correlation.pdf` |

The simulator ablation compares optical rendering, not neural depth predictions or new
measurements. Original comparison figures retain paper results; the interactive
gallery uses the released Large mixed-training checkpoint. Known evaluation
and nominal-camera limitations remain visible on the page.

The UniDepth V2 table reproduces only MAE from `tables/supp_generalization.tex`.
The source's real-data fine-tuned RMSE (0.0096) is below its MAE (0.0348), an
unresolved inconsistency; no replacement value is inferred or published here.
These are UniDepth experiments, not benchmarks of the released DAV2 checkpoints.
PSF analysis uses the supplement's theoretical 1–5 m configuration (50 mm focal
length, 5 mm aperture), not measured performance of the near-range prototype.

Website code follows the root license. The select primitive was retained from
shadcn/ui (MIT), using Base UI; package dependencies retain their own licenses.
Research figures and example data are not relicensed by the code license.
Attribute the paper when reusing its research content.
