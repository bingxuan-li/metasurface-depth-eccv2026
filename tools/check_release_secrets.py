"""Fail on common credential patterns in Git history and nonignored working files.

Reports paths and pattern names only, never matching values. This is a targeted
guardrail, not proof that every possible secret or binary artifact is safe.
"""

import re
import subprocess
from pathlib import Path


def git(*args):
    return subprocess.check_output(["git", *args])


PATTERNS = {
    "W&B token": rb"wandb_v[0-9]+_[A-Za-z0-9_\-]{20,}",
    "literal W&B login key": rb"wandb\.login\s*\([^)]*\bkey\s*=\s*['\"][^'\"]{20,}['\"]",
    "literal W&B environment key": rb"WANDB_API_KEY['\"]\s*\]\s*=\s*['\"][^'\"]{20,}['\"]",
    "Hugging Face token": rb"\bhf_[A-Za-z0-9]{25,}\b",
    "GitHub token": rb"\bgh[pousr]_[A-Za-z0-9]{30,}\b",
    "private key": rb"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----",
}
TEXT_SUFFIXES = {".py", ".md", ".json", ".toml", ".yaml", ".yml", ".txt", ".sh", ".cff", ".in"}


def is_text(path):
    return Path(path).suffix in TEXT_SUFFIXES or Path(path).name.startswith(".")


def findings(data):
    return [name for name, pattern in PATTERNS.items() if re.search(pattern, data, re.DOTALL)]


def main():
    root = Path(git("rev-parse", "--show-toplevel").decode().strip())
    seen, failures = set(), []
    commits = git("rev-list", "--all").decode().splitlines()
    for commit in commits:
        for entry in git("ls-tree", "-rz", commit).split(b"\0"):
            if not entry:
                continue
            meta, raw_path = entry.split(b"\t", 1)
            _, kind, oid = meta.split()
            path = raw_path.decode("utf-8")
            if kind != b"blob" or oid in seen or not is_text(path):
                continue
            seen.add(oid)
            for name in findings(git("cat-file", "blob", oid.decode())):
                failures.append(f"history:{commit[:8]}:{path}: {name}")
    for raw_path in set(git("ls-files", "-z", "--cached", "--others", "--exclude-standard").split(b"\0")):
        if not raw_path:
            continue
        path = raw_path.decode("utf-8")
        full_path = root / path
        if is_text(path) and full_path.is_file():
            for name in findings(full_path.read_bytes()):
                failures.append(f"working-tree:{path}: {name}")
    if failures:
        raise SystemExit("Credential-pattern findings (values redacted):\n" + "\n".join(failures))
    print(f"PASS: {len(commits)} available commits; {len(seen)} unique text blobs; working tree checked")


if __name__ == "__main__":
    main()
