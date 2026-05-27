"""Tests for v1ni official negative request targets."""

from __future__ import annotations

import csv
import hashlib
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
SCRIPT = ROOT / "scripts/protocolo_c/revp_v1ni_official_negative_request_target_builder.py"
TARGETS = ROOT / "datasets/official_negative_evidence_request_target_registry.csv"
FIELDS = ROOT / "datasets/official_negative_evidence_required_fields_matrix.csv"
SCHEMA = ROOT / "datasets/schemas/official_negative_evidence_request_target_schema.csv"
PUBLIC = [TARGETS, FIELDS, SCHEMA, ROOT / "docs/metodologia_cientifica/protocolo_c_alvos_pedido_oficial_negativo_v1ni.md"]
ABS_PATH = re.compile(r"[A-Za-z]:[\\/]|\\\\")


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v1ni_generates_required_files_and_schema() -> None:
    result = subprocess.run([sys.executable, str(SCRIPT), "--force", "--emit-evidence"], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr + result.stdout
    for path in PUBLIC:
        assert path.exists()
    schema_fields = {row["field"] for row in rows(SCHEMA)}
    for field in ["request_target_id", "c4_blocker", "required_fields", "forbidden_substitutes", "can_unlock_c4_alone"]:
        assert field in schema_fields


def test_v1ni_keeps_targets_non_unlocking_and_no_abs_paths() -> None:
    target_rows = rows(TARGETS)
    assert target_rows
    assert {row["can_unlock_c4_alone"] for row in target_rows} == {"false"}
    assert all("pseudo-absence" in row["forbidden_substitutes"] for row in target_rows)
    for path in PUBLIC:
        assert not ABS_PATH.search(path.read_text(encoding="utf-8", errors="replace"))


def test_v1ni_is_deterministic() -> None:
    subprocess.run([sys.executable, str(SCRIPT), "--force", "--emit-evidence"], cwd=ROOT, check=True, timeout=120)
    before = {path: digest(path) for path in PUBLIC}
    subprocess.run([sys.executable, str(SCRIPT), "--force", "--emit-evidence"], cwd=ROOT, check=True, timeout=120)
    assert before == {path: digest(path) for path in PUBLIC}
