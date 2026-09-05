# Installation and runtime qualification

Use an isolated environment. Install a CUDA-compatible PyTorch build first, then:

```bash
python -m pip install -e '.[simulator,train,dev]'
python -m unittest discover -s tests -v
ruff check src/metasurface_depth
python -m build
```

For inference only, `pip install .` avoids the scientific simulation extras.
For simulation add `[simulator]`; for training add `[train]`. xFormers is optional
and must match your PyTorch/CUDA installation; it is not installed automatically.
Only the project namespace is installed, so it does not shadow upstream packages.

Reference audits used H100 GPUs: the neural reference used PyTorch 2.8.0+cu128,
NumPy 2.2.6, OpenCV 4.12.0 and xFormers 0.0.32.post2; the original simulator audit
used PyTorch 2.7.1+cu126. Integrated qualification is recorded separately in
`VALIDATION.md`. Broad package dependency ranges do not mean every combination
has been tested.

The simulator's `--max-memory` tunes internal batch/chunk sizes; it is not a hard
CUDA memory cap. Default is 24. Lower it on smaller GPUs, and recheck numerical
parity if changing batching or precision. Full-resolution sensor images can use
substantially more memory than the procedural smoke example.

CPU contract tests do not exercise the GPU optical pipeline. CI runs CPU tests,
lint and wheel building; the GPU smoke command is run separately on an available
CUDA machine. Do not use a CPU CI pass to claim GPU regression coverage.
