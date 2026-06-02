"""Tests for v1ny Recife C3+ adjudication."""

from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
SCRIPT = ROOT / "scripts/protocolo_c/revp_v1ny_recife_c3_plus_candidate_adjudication.py"
OUT = ROOT / "datasets/recife_c3_plus_candidate_adjudication_registry.csv"
SUMMARY = ROOT / "datasets/recife_c3_plus_summary_matrix.csv"
C4 = ROOT / "datasets/recife_c4_positive_only_blocker_matrix.csv"


def write_rows(path: Path, rows: list[dict[str, str]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = fields or list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_v1ny_blocks_c4_when_formal_negative_count_is_zero(tmp_path: Path) -> None:
    positive = tmp_path / "positive.csv"
    dates = tmp_path / "dates.csv"
    temporal = tmp_path / "temporal.csv"
    spatial = tmp_path / "spatial.csv"
    pe3d = tmp_path / "pe3d.csv"
    negative = tmp_path / "negative.csv"
    write_rows(positive, [{"candidate_id": "C1", "source_id": "RECIFE_EMLURB_156", "phenomenon": "alagamento"}])
    write_rows(dates, [{"candidate_id": "C1", "date_parsed": "2022-05-24", "date_quality": "VALID_DATE"}])
    write_rows(temporal, [{"candidate_id": "C1", "patch_id": "REC_PATCH", "temporal_class": "TEMPORAL_STRONG"}])
    write_rows(spatial, [{"candidate_id": "C1", "spatial_support_status": "SPATIAL_APPROXIMATION_REVIEW"}])
    write_rows(pe3d, [{"candidate_id": "C1", "pe3d_mde_support_status": "PE3D_MDE_CONTEXT_REGISTERED_REVIEW_ONLY"}])
    write_rows(negative, [], ["candidate_id", "event_or_negative"])
    env = os.environ.copy()
    env["REVP_RECIFE_POSITIVE_CANDIDATE_REGISTRY"] = str(positive)
    env["REVP_RECIFE_DATE_NORMALIZED_REGISTRY"] = str(dates)
    env["REVP_RECIFE_EVENT_PATCH_TEMPORAL_MATCH"] = str(temporal)
    env["REVP_RECIFE_SPATIAL_SUPPORT_REGISTRY"] = str(spatial)
    env["REVP_RECIFE_PE3D_SUPPORT_MATRIX"] = str(pe3d)
    env["REVP_RECIFE_NEGATIVE_CANDIDATE_REGISTRY"] = str(negative)
    result = subprocess.run([sys.executable, str(SCRIPT), "--force", "--emit-evidence"], cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr + result.stdout
    assert read_rows(OUT)[0]["adjudication_status"] == "C4_PREFLIGHT_POSITIVE_ONLY_BLOCKED_NO_NEGATIVE"
    assert read_rows(SUMMARY)[0]["formal_negative_count"] == "0"
    assert read_rows(C4)[0]["c4_open"] == "false"
    assert "can_train_model,true" not in OUT.read_text(encoding="utf-8")
    assert "ground_truth,true" not in OUT.read_text(encoding="utf-8")
