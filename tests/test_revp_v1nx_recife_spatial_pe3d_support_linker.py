"""Tests for v1nx Recife spatial and PE3D support linker."""

from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
SCRIPT = ROOT / "scripts/protocolo_c/revp_v1nx_recife_spatial_pe3d_support_linker.py"
OUT = ROOT / "datasets/recife_candidate_spatial_support_registry.csv"
PE3D = ROOT / "datasets/recife_candidate_pe3d_support_matrix.csv"


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_v1nx_uses_point_or_textual_spatial_support_without_fake_overlap(tmp_path: Path) -> None:
    positive = tmp_path / "positive.csv"
    normalized = tmp_path / "normalized.csv"
    write_rows(
        positive,
        [
            {"candidate_id": "C1", "bairro": "", "rpa": "", "microrregiao": "", "latitude": "-8.05", "longitude": "-34.9", "address_raw": ""},
            {"candidate_id": "C2", "bairro": "Boa Viagem", "rpa": "6", "microrregiao": "", "latitude": "", "longitude": "", "address_raw": ""},
        ],
    )
    write_rows(normalized, [{"candidate_id": "C1"}, {"candidate_id": "C2"}])
    env = os.environ.copy()
    env["REVP_RECIFE_POSITIVE_CANDIDATE_REGISTRY"] = str(positive)
    env["REVP_RECIFE_DATE_NORMALIZED_REGISTRY"] = str(normalized)
    result = subprocess.run([sys.executable, str(SCRIPT), "--force", "--emit-evidence"], cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr + result.stdout
    rows = read_rows(OUT)
    assert rows[0]["spatial_support_status"] == "SPATIAL_POINT_AVAILABLE"
    assert rows[1]["spatial_support_status"] == "SPATIAL_APPROXIMATION_REVIEW"
    assert read_rows(PE3D)[0]["support_role"] == "TERRITORIAL_CONTEXT_REVIEW_ONLY"
    assert "can_create_operational_label,true" not in OUT.read_text(encoding="utf-8")
