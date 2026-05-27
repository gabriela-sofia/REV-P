"""Tests for v1nm C4 recheck after official intake."""

from __future__ import annotations

import csv
import hashlib
import os
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
DEPS = [
    ROOT / "scripts/protocolo_c/revp_v1ni_official_negative_request_target_builder.py",
    ROOT / "scripts/protocolo_c/revp_v1nj_official_negative_lai_request_packet_generator.py",
    ROOT / "scripts/protocolo_c/revp_v1nk_official_negative_response_intake_validator.py",
    ROOT / "scripts/protocolo_c/revp_v1nl_strict_formal_negative_adjudicator.py",
]
SCRIPT = ROOT / "scripts/protocolo_c/revp_v1nm_c4_recheck_after_official_negative_intake.py"
C4 = ROOT / "datasets/c4_recheck_after_official_negative_intake.csv"
READY = ROOT / "datasets/c4_label_readiness_after_official_negative_intake.csv"
SUMMARY = ROOT / "datasets/protocol_c_official_negative_intake_summary.csv"
SCHEMA = ROOT / "datasets/schemas/c4_recheck_after_official_negative_intake_schema.csv"
PUBLIC = [C4, READY, SUMMARY, SCHEMA, ROOT / "docs/metodologia_cientifica/protocolo_c_recheck_c4_pos_intake_oficial_v1nm.md"]
ABS_PATH = re.compile(r"[A-Za-z]:[\\/]|\\\\")


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_deps(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["REVP_OFFICIAL_NEGATIVE_INBOX"] = str(tmp_path / "official_negative_response_inbox")
    subprocess.run([sys.executable, str(DEPS[0]), "--force", "--emit-evidence"], cwd=ROOT, check=True, timeout=120)
    subprocess.run([sys.executable, str(DEPS[1]), "--force", "--emit-packet"], cwd=ROOT, check=True, timeout=120)
    subprocess.run([sys.executable, str(DEPS[2]), "--force", "--emit-evidence"], cwd=ROOT, env=env, check=True, timeout=120)
    subprocess.run([sys.executable, str(DEPS[3]), "--force", "--emit-evidence"], cwd=ROOT, check=True, timeout=120)


def test_v1nm_keeps_c4_blocked_without_formal_negative(tmp_path: Path) -> None:
    run_deps(tmp_path)
    result = subprocess.run([sys.executable, str(SCRIPT), "--force", "--emit-evidence"], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr + result.stdout
    row = rows(C4)[0]
    assert row["summary_decision"] == "C4_BLOCKED_NO_FORMAL_NEGATIVES"
    assert row["c3_event_count"] == "9"
    assert row["formal_negative_count"] == "0"
    assert row["can_create_operational_label"] == "false"
    assert row["can_train_model"] == "false"
    assert row["dino_role"] == "REVIEW_ONLY_REPRESENTATION"


def test_v1nm_schema_no_abs_paths_and_no_training_claim() -> None:
    assert "summary_decision" in {row["field"] for row in rows(SCHEMA)}
    for path in PUBLIC:
        text = path.read_text(encoding="utf-8", errors="replace")
        assert not ABS_PATH.search(text)
        assert "can_train_model,true" not in text
        assert "can_create_operational_label,true" not in text


def test_v1nm_is_deterministic() -> None:
    subprocess.run([sys.executable, str(SCRIPT), "--force", "--emit-evidence"], cwd=ROOT, check=True, timeout=120)
    before = {path: digest(path) for path in PUBLIC}
    subprocess.run([sys.executable, str(SCRIPT), "--force", "--emit-evidence"], cwd=ROOT, check=True, timeout=120)
    assert before == {path: digest(path) for path in PUBLIC}
