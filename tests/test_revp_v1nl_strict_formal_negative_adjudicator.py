"""Tests for v1nl strict formal negative adjudicator."""

from __future__ import annotations

import csv
import hashlib
import os
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
DEP = ROOT / "scripts/protocolo_c/revp_v1nk_official_negative_response_intake_validator.py"
SCRIPT = ROOT / "scripts/protocolo_c/revp_v1nl_strict_formal_negative_adjudicator.py"
ADJ = ROOT / "datasets/strict_formal_negative_adjudication_registry.csv"
GATES = ROOT / "datasets/strict_formal_negative_gate_matrix.csv"
SCHEMA = ROOT / "datasets/schemas/strict_formal_negative_adjudication_schema.csv"
PUBLIC = [ADJ, GATES, SCHEMA, ROOT / "docs/metodologia_cientifica/protocolo_c_adjudicacao_negativo_formal_v1nl.md"]
ABS_PATH = re.compile(r"[A-Za-z]:[\\/]|\\\\")


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v1nl_blocks_without_official_response(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["REVP_OFFICIAL_NEGATIVE_INBOX"] = str(tmp_path / "official_negative_response_inbox")
    subprocess.run([sys.executable, str(DEP), "--force", "--emit-evidence"], cwd=ROOT, env=env, check=True, timeout=120)
    result = subprocess.run([sys.executable, str(SCRIPT), "--force", "--emit-evidence"], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr + result.stdout
    row = rows(ADJ)[0]
    assert row["can_be_formal_negative"] == "false"
    assert row["can_be_used_for_c4"] == "false"
    assert row["can_create_training_negative_label"] == "false"
    assert rows(GATES)[0]["all_gates_pass"] == "FAIL"


def test_v1nl_schema_and_no_abs_paths(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["REVP_OFFICIAL_NEGATIVE_INBOX"] = str(tmp_path / "official_negative_response_inbox")
    subprocess.run([sys.executable, str(DEP), "--force", "--emit-evidence"], cwd=ROOT, env=env, check=True, timeout=120)
    subprocess.run([sys.executable, str(SCRIPT), "--force", "--emit-evidence"], cwd=ROOT, check=True, timeout=120)
    schema_fields = {row["field"] for row in rows(SCHEMA)}
    for field in ["candidate_negative_id", "failed_gates", "can_be_formal_negative", "can_be_used_for_c4", "can_create_training_negative_label"]:
        assert field in schema_fields
    for path in PUBLIC:
        assert not ABS_PATH.search(path.read_text(encoding="utf-8", errors="replace"))


def test_v1nl_is_deterministic(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["REVP_OFFICIAL_NEGATIVE_INBOX"] = str(tmp_path / "official_negative_response_inbox")
    subprocess.run([sys.executable, str(DEP), "--force", "--emit-evidence"], cwd=ROOT, env=env, check=True, timeout=120)
    subprocess.run([sys.executable, str(SCRIPT), "--force", "--emit-evidence"], cwd=ROOT, check=True, timeout=120)
    before = {path: digest(path) for path in PUBLIC}
    subprocess.run([sys.executable, str(SCRIPT), "--force", "--emit-evidence"], cwd=ROOT, check=True, timeout=120)
    assert before == {path: digest(path) for path in PUBLIC}
