# Model files

Use `small.pth`, `base.pth`, and `large.pth` with the matching model variant.
`manifest.json` records exact byte sizes, SHA-256 hashes and input configuration.
The selected exports preserve model tensors but omit optimizer/scheduler state.

After installing the package, verify all three files from the repository root:

```bash
python tools/verify_checkpoints.py
```

The weights are intentionally ignored by Git. Local release preparation may
place them here, but public hosting URLs are not yet configured. There are no
placeholder download commands or implicit substitutions with upstream models.

All three originate from DAV2 metric-Hypersim initialization. Source-code and
weight licenses are distinct; read `../docs/LICENSING.md` before redistribution.
The Base paper-metric discrepancy is recorded in `../docs/VALIDATION.md`.
