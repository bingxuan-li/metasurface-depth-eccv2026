# Model files

Use `small.pth`, `base.pth`, and `large.pth` with the matching model variant.
`manifest.json` records exact byte sizes, SHA-256 hashes and input configuration.
The selected exports preserve model tensors but omit optimizer/scheduler state.

After installing the package, verify all three files from the repository root:

```bash
python tools/verify_checkpoints.py
```

The weights are intentionally ignored by Git. All three exports are uploaded to
[Hugging Face](https://huggingface.co/Bingxuan111/metasurface-depth-eccv2026).
The repository is public; downloads do not require authentication. Model bytes
are unchanged from the verified exports. There are no implicit substitutions
with upstream models.

Download Small (replace `small` with `base` or `large` as needed):

```bash
curl -L --fail https://huggingface.co/Bingxuan111/metasurface-depth-eccv2026/resolve/main/small.pth -o checkpoints/small.pth
```

All three originate from DAV2 metric-Hypersim initialization. Source-code and
weight licenses are distinct; read `../docs/LICENSING.md` before redistribution.
The Base paper-metric discrepancy is recorded in `../docs/VALIDATION.md`.
