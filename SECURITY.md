# Security

Load only checkpoints from trusted sources, verify their published hashes, and
retain `weights_only=True` loading. Do not put API keys into source files or model
metadata. This project does not require telemetry credentials.

Run `python tools/check_release_secrets.py` before publication. It scans common
credential patterns in available Git history and nonignored working files without
printing secret values. CI performs the same check. This does not inspect ignored
private archives or prove the absence of all possible secrets.

Removing a credential from code does not revoke it. If an old experimental key
was exposed, its owner must revoke it in W&B account settings and update any
dependent private jobs. This release works without W&B and needs no replacement
key. Never paste old or replacement credentials into issues or chat.

Do not include secrets or private data in public issue reports. Report sensitive
findings through the authors' existing
private communication channel. No unverified security-contact address is listed.
