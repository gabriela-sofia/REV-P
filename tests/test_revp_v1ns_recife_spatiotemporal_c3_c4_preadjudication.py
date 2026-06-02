"""Tests for v1ns Recife C3/C4 pre-adjudication."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
MINER = ROOT / "scripts/protocolo_c/revp_v1nr_recife_official_event_negative_candidate_miner.py"
SCRIPT = ROOT / "scripts/protocolo_c/revp_v1ns_recife_spatiotemporal_c3_c4_preadjudication.py"
PREADJ = ROOT / "datasets/recife_c3_c4_preadjudication_registry.csv"
LINK = ROOT / "datasets/recife_official_candidate_patch_linkage_registry.csv"
TEMP = ROOT / "datasets/recife_official_candidate_temporal_compatibility_matrix.csv"
SPATIAL = ROOT / "datasets/recife_official_candidate_spatial_compatibility_matrix.csv"
BLOCK = ROOT / "datasets/recife_c4_preflight_blocker_matrix.csv"
SCHEMA = ROOT / "datasets/schemas/recife_c3_c4_preadjudication_schema.csv"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_v1ns_creates_preadjudication_without_labels(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "recife_fixture.csv").write_text("data;endereco;bairro;descricao\n2022-05-25;Rua A;Ibura;alagamento em canal\n2022-05-26;Rua B;Ibura;vistoria de barreira sem risco\n", encoding="utf-8")
    env = __import__("os").environ.copy()
    env["REVP_RECIFE_RAW_DIR"] = str(raw)
    subprocess.run([sys.executable, str(MINER), "--force", "--emit-evidence"], cwd=ROOT, env=env, check=True, timeout=120)
    result = subprocess.run([sys.executable, str(SCRIPT), "--force", "--emit-evidence"], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr + result.stdout
    assert rows(PREADJ)
    assert {row["can_create_operational_label"] for row in rows(PREADJ)} == {"false"}
    assert {row["can_train_model"] for row in rows(PREADJ)} == {"false"}
    assert all(row["overlap"] == "NOT_COMPUTED_NO_VALIDATED_GEOMETRY" for row in rows(LINK))
    assert rows(TEMP) and rows(SPATIAL)


def test_v1ns_schema_and_c4_not_opened() -> None:
    assert "c3_c4_classification" in {row["field"] for row in rows(SCHEMA)}
    text = PREADJ.read_text(encoding="utf-8", errors="replace") + BLOCK.read_text(encoding="utf-8", errors="replace")
    assert "can_create_operational_label,true" not in text
    assert "can_train_model,true" not in text
