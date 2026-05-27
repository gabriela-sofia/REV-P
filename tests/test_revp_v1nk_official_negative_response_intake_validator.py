"""Tests for v1nk official response intake validator."""

from __future__ import annotations

import csv
import hashlib
import os
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
SCRIPT = ROOT / "scripts/protocolo_c/revp_v1nk_official_negative_response_intake_validator.py"
INTAKE = ROOT / "datasets/official_negative_response_intake_registry.csv"
PREVALIDATION = ROOT / "datasets/official_negative_response_prevalidation_matrix.csv"
SCHEMA = ROOT / "datasets/schemas/official_negative_response_intake_schema.csv"
PUBLIC = [INTAKE, PREVALIDATION, SCHEMA, ROOT / "docs/metodologia_cientifica/protocolo_c_intake_respostas_oficiais_negativas_v1nk.md"]
ABS_PATH = re.compile(r"[A-Za-z]:[\\/]|\\\\")


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v1nk_empty_inbox_generates_no_intake_status(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["REVP_OFFICIAL_NEGATIVE_INBOX"] = str(tmp_path / "official_negative_response_inbox")
    result = subprocess.run([sys.executable, str(SCRIPT), "--force", "--emit-evidence"], cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr + result.stdout
    assert rows(INTAKE)[0]["intake_status"] == "NO_OFFICIAL_RESPONSE_INTAKE"
    assert rows(PREVALIDATION)[0]["prevalidation_status"] == "NO_OFFICIAL_RESPONSE_INTAKE"
    for path in PUBLIC:
        assert path.exists()


def test_v1nk_outputs_no_abs_paths_and_no_promotion() -> None:
    for path in PUBLIC:
        text = path.read_text(encoding="utf-8", errors="replace")
        assert not ABS_PATH.search(text)
        assert "can_promote_without_adjudication,true" not in text


def test_v1nk_is_deterministic_for_empty_inbox(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["REVP_OFFICIAL_NEGATIVE_INBOX"] = str(tmp_path / "official_negative_response_inbox")
    subprocess.run([sys.executable, str(SCRIPT), "--force", "--emit-evidence"], cwd=ROOT, env=env, check=True, timeout=120)
    before = {path: digest(path) for path in PUBLIC}
    subprocess.run([sys.executable, str(SCRIPT), "--force", "--emit-evidence"], cwd=ROOT, env=env, check=True, timeout=120)
    assert before == {path: digest(path) for path in PUBLIC}
