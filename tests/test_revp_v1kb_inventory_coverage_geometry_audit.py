"""Tests for REV-P v1kb inventory coverage geometry audit."""

from __future__ import annotations

import csv
import subprocess
import sys
from functools import lru_cache
from pathlib import Path


REVP_ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
DATASETS = REVP_ROOT / "datasets"
SCRIPT = REVP_ROOT / "scripts/protocolo_c/revp_v1kb_inventory_coverage_geometry_audit.py"
COMMAND = [sys.executable, str(SCRIPT), "--audit-coverage-geometry", "--emit-sampling-area-decision"]
sys.path.insert(0, str(REVP_ROOT / "scripts" / "protocolo_c"))

from revp_v1kb_inventory_coverage_geometry_audit import classify_layer


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
        DATASETS / "inventory_coverage_geometry_audit_registry.csv",
        DATASETS / "inventory_negative_sampling_area_decision.csv",
        DATASETS / "schemas/inventory_coverage_geometry_audit_schema.csv",
        DATASETS / "schemas/inventory_negative_sampling_area_decision_schema.csv",
    ]:
        assert path.exists(), path


def test_municipal_boundary_is_context_not_completeness() -> None:
    decision, _, _ = classify_layer(Path("Limites/Petropolis_Sirgas2000_UTM23.shp"))
    assert decision == "MUNICIPAL_BOUNDARY_ONLY_CONTEXT"


def test_convex_or_inventory_extent_does_not_pass_completeness() -> None:
    decision, relation, _ = classify_layer(Path("Feicoes/Cicatriz_Area_A.shp"))
    assert decision == "DERIVED_COVERAGE_ONLY_REVIEW"
    assert relation == "INVENTORY_FEATURE_EXTENT_ONLY"


def test_sampling_area_decision_keeps_training_blocked() -> None:
    run_once()
    row = read_csv(DATASETS / "inventory_negative_sampling_area_decision.csv")[0]
    assert row["complete_inventory_gate"] == "FAIL"
    assert row["can_create_training_label"] == "false"
    assert row["can_train_model"] == "false"
    if row["coverage_geometry_decision"] == "MUNICIPAL_BOUNDARY_ONLY_CONTEXT":
        assert row["study_area_coverage_gate"] == "FAIL"


def test_public_outputs_have_no_private_paths() -> None:
    run_once()
    for path in [
        DATASETS / "inventory_coverage_geometry_audit_registry.csv",
        DATASETS / "inventory_negative_sampling_area_decision.csv",
    ]:
        text = path.read_text(encoding="utf-8")
        assert "C:\\" not in text and "C:/" not in text
        assert "gabriela" not in text.lower()
