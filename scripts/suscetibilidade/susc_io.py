"""
SUSC shared I/O utilities (pure Python, deterministic).

CSV/JSON/Markdown read/write, safe directory creation, SHA256, deterministic
column ordering, and repo-path containment validation. No third-party deps.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def repo_root() -> Path:
    return ROOT


def is_within_repo(path) -> bool:
    """True if path is inside the repository tree."""
    try:
        Path(path).resolve().relative_to(ROOT)
        return True
    except (ValueError, OSError):
        return False


def ensure_dir(path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_csv(path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8", errors="ignore", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames=None) -> Path:
    """Deterministic CSV write (LF line terminator). If fieldnames is None, the
    union of keys is used in first-seen order."""
    p = Path(path)
    ensure_dir(p.parent)
    if not rows:
        if fieldnames:
            with p.open("w", encoding="utf-8", newline="") as f:
                csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n").writeheader()
        else:
            p.write_text("", encoding="utf-8")
        return p
    if fieldnames is None:
        fieldnames = []
        for r in rows:
            for k in r:
                if k not in fieldnames:
                    fieldnames.append(k)
    with p.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})
    return p


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, obj) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return p


def write_markdown(path, text) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    p.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
    return p


def rel(path) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")
