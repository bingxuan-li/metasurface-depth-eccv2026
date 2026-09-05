"""Portable manifests and non-overwriting output helpers."""

import csv
import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_manifest(path, fields):
    path = Path(path).resolve()
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not set(fields).issubset(reader.fieldnames or []):
            raise ValueError(f"{path.name}: required CSV columns: {fields}")
        rows = []
        for row in reader:
            resolved = dict(row)
            for key in fields:
                if not row[key] or not row[key].strip():
                    raise ValueError(f"Empty {key} in {path.name}")
                value = Path(row[key])
                resolved[key] = (path.parent / value).resolve()
                if not resolved[key].is_file():
                    raise FileNotFoundError(resolved[key])
            rows.append(resolved)
    if not rows:
        raise ValueError(f"Empty manifest: {path}")
    return rows


def write_json(path, payload):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)


def new_directory(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path
