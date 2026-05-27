"""Tests for v1nj official negative LAI packet."""

from __future__ import annotations

import csv
import hashlib
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
DEP = ROOT / "scripts/protocolo_c/revp_v1ni_official_negative_request_target_builder.py"
SCRIPT = ROOT / "scripts/protocolo_c/revp_v1nj_official_negative_lai_request_packet_generator.py"
MANIFEST = ROOT / "datasets/official_negative_request_packet_manifest.csv"
QUESTIONS = ROOT / "datasets/official_negative_request_question_bank.csv"
SCHEMA = ROOT / "datasets/schemas/official_negative_request_question_bank_schema.csv"
PUBLIC = [
    MANIFEST,
    QUESTIONS,
    SCHEMA,
    ROOT / "docs/metodologia_cientifica/protocolo_c_pacote_lai_negativos_v1nj.md",
    ROOT / "docs/metodologia_cientifica/modelo_pedido_lai_defesa_civil_petropolis_2022_v1nj.md",
]
ABS_PATH = re.compile(r"[A-Za-z]:[\\/]|\\\\")


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v1nj_generates_packet_and_question_schema() -> None:
    subprocess.run([sys.executable, str(DEP), "--force", "--emit-evidence"], cwd=ROOT, check=True, timeout=120)
    result = subprocess.run([sys.executable, str(SCRIPT), "--force", "--emit-packet"], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr + result.stdout
    for path in PUBLIC:
        assert path.exists()
    schema_fields = {row["field"] for row in rows(SCHEMA)}
    for field in ["question_id", "official_channel", "question_text", "forbidden_inference_if_absent"]:
        assert field in schema_fields


def test_v1nj_does_not_infer_absence_or_send_request() -> None:
    assert {row["send_status"] for row in rows(MANIFEST)} == {"NOT_SENT_BY_SCRIPT"}
    assert all("Do not infer absence" in row["forbidden_inference_if_absent"] for row in rows(QUESTIONS))
    for path in PUBLIC:
        text = path.read_text(encoding="utf-8", errors="replace")
        assert not ABS_PATH.search(text)
        assert "can_train_model,true" not in text
        assert "can_create_operational_label,true" not in text


def test_v1nj_is_deterministic() -> None:
    subprocess.run([sys.executable, str(SCRIPT), "--force", "--emit-packet"], cwd=ROOT, check=True, timeout=120)
    before = {path: digest(path) for path in PUBLIC}
    subprocess.run([sys.executable, str(SCRIPT), "--force", "--emit-packet"], cwd=ROOT, check=True, timeout=120)
    assert before == {path: digest(path) for path in PUBLIC}
