# Publication checklist

This directory is the source-only release candidate. Experimental repositories,
raw training checkpoints, private manifests and audit inputs are not part of it.

## Before making a public repository

- Confirm contributor approval, names and the license for the new integration code.
- Confirm redistribution permission for the bundled PSF calibration and all source
  modifications; preserve the third-party notices.
- Review upstream weight terms separately for Small, Base and Large. The latter
  two inherit non-commercial upstream restrictions; do not apply a blanket MIT
  label to every artifact.
- Rotate any credential formerly embedded in experimental code. Do not publish
  experimental Git history, configuration secrets or telemetry credentials.
- Resolve or clearly retain the scientific caveats in `VALIDATION.md`.
- Select GitHub organization/repository and model hosting. Upload only exported
  weights after approval, then add real download URLs to `checkpoints/README.md`.
  Check every download against `checkpoints/manifest.json`.
- Run the documented installation and smoke workflow from a fresh checkout,
  review generated documentation, and enable the included CI workflow.
- Tag the reviewed release and attach its source archive and checksum manifest.

No repository creation, upload, public sharing or publication is performed by
the local preparation workflow. Existing datasets remain external and must be
obtained under their own licenses.
