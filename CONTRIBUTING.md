# Contributing

Create an isolated environment and install `.[simulator,train,dev]`. Run CPU
tests, lint and a wheel build before proposing changes. If you change numerical
code, also run the GPU smoke workflow and compare against a frozen reference.

Keep image order, metric units, crop order, PSF checksum and checkpoint tensor keys
stable unless deliberately introducing a separately documented experiment.
Do not silently edit benchmark data to make a result match. Add a focused test for
each bug fix and document any changed predictions or training semantics.

Generated data, large checkpoints, logs and local configuration belong under
Git-ignored `data/`, `checkpoints/`, or `runs/`. Never commit credentials, account
names, private dataset paths, or scheduler-specific launch scripts.

The vendored and frozen scientific kernels are compatibility boundaries. Put
validation and API improvements in wrappers first; numerical refactors need
explicit regression evidence. See `docs/ARCHITECTURE.md`.
