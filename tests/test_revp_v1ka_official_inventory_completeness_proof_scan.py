"""Tests for REV-P v1ka inventory completeness proof scan."""

from __future__ import annotations

import csv
import subprocess
import sys
from functools import lru_cache
from pathlib import Path


REVP_ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
DATASETS = REVP_ROOT / "datasets"
SCRIPT = REVP_ROOT / "scripts/protocolo_c/revp_v1ka_official_inventory_completeness_proof_scan.py"
COMMAND = [sys.executable, str(SCRIPT), "--scan-completeness-proof", "--emit-phrase-audit"]
sys.path.insert(0, str(REVP_ROOT / "scripts" / "protocolo_c"))

from revp_v1ka_official_inventory_completeness_proof_scan import classify_text


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


@lru_cache(maxsize=1)
def run_once() -> None:
    result = subprocess.run(COMMAND, cwd=str(REVP_ROOT), capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stderr + result.stdout


def test_script_runs_and_outputs_exist() -> None:
    assert SCRIPT.exists()
    run_once()
    for path in [
        DATASETS / "inventory_completeness_proof_scan_registry.csv",
        DATASETS / "inventory_completeness_phrase_audit.csv",
        DATASETS / "schemas/inventory_completeness_proof_scan_schema.csv",
        DATASETS / "schemas/inventory_completeness_phrase_audit_schema.csv",
    ]:
        assert path.exists(), path


def test_metadata_only_does_not_support_complete_inventory_gate() -> None:
    classification, _, complete, _, _, _ = classify_text("FONTE=Fotointerpretacao; TIPO=Deslizamento; escala 1:25000")
    assert classification in {"EXPLICIT_MAPPING_METHOD_ONLY", "IMPLICIT_LAYER_METADATA_ONLY"}
    assert complete is False


def test_synthetic_explicit_complete_statement_is_detected() -> None:
    text = "Inventario completo de todas as feicoes de cicatrizes de deslizamento na area de estudo de Petropolis."
    classification, _, complete, coverage, _, _ = classify_text(text)
    assert classification == "EXPLICIT_COMPLETE_INVENTORY_STATEMENT"
    assert complete is True
    assert coverage is True


def test_public_outputs_have_no_private_paths() -> None:
    run_once()
    for path in [
        DATASETS / "inventory_completeness_proof_scan_registry.csv",
        DATASETS / "inventory_completeness_phrase_audit.csv",
    ]:
        text = path.read_text(encoding="utf-8")
        assert "C:\\" not in text and "C:/" not in text
        assert "gabriela" not in text.lower()
        assert "PROJETO" not in text


def test_supporting_complete_proof_must_be_explicit_statement() -> None:
    run_once()
    rows = read_csv(DATASETS / "inventory_completeness_proof_scan_registry.csv")
    for row in rows:
        if row["can_support_complete_inventory_gate"] == "true":
            assert row["evidence_classification"] == "EXPLICIT_COMPLETE_INVENTORY_STATEMENT"
