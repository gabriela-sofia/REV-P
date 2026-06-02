"""Tests for v1nu Recife official event date normalization."""

from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
SCRIPT = ROOT / "scripts/protocolo_c/revp_v1nu_recife_official_event_date_normalizer.py"
OUT = ROOT / "datasets/recife_positive_candidate_date_normalized_registry.csv"
QUALITY = ROOT / "datasets/recife_positive_candidate_date_quality_matrix.csv"
SCHEMA = ROOT / "datasets/schemas/recife_positive_candidate_date_normalized_registry_schema.csv"


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_v1nu_normalizes_valid_and_rejects_invalid_dates(tmp_path: Path) -> None:
    fixture = tmp_path / "positive.csv"
    write_rows(
        fixture,
        [
            {"candidate_id": "C1", "source_id": "RECIFE_EMLURB_156", "source_file": "fixture.csv", "source_row_id": "1", "region": "RECIFE", "phenomenon": "alagamento", "date_raw": "24/05/2022", "date_parsed": "", "bairro": "Boa Viagem"},
            {"candidate_id": "C2", "source_id": "RECIFE_DADOS_VIVOS_SEDEC", "source_file": "fixture.csv", "source_row_id": "2", "region": "RECIFE", "phenomenon": "barreira", "date_raw": "impossivel", "date_parsed": "", "bairro": "Casa Forte"},
        ],
    )
    env = os.environ.copy()
    env["REVP_RECIFE_POSITIVE_CANDIDATE_REGISTRY"] = str(fixture)
    result = subprocess.run([sys.executable, str(SCRIPT), "--force", "--emit-evidence"], cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr + result.stdout
    rows = read_rows(OUT)
    assert rows[0]["date_parsed"] == "2022-05-24"
    assert rows[0]["date_quality"] == "VALID_DATE"
    assert rows[1]["date_quality"] == "INVALID_DATE"
    assert "candidate_id" in {row["field"] for row in read_rows(SCHEMA)}
    assert "can_train_model,true" not in OUT.read_text(encoding="utf-8")
    assert "can_create_operational_label,true" not in QUALITY.read_text(encoding="utf-8")
