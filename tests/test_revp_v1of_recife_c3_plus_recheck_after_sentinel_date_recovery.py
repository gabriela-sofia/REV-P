"""Tests for v1of Recife C3+ recheck after Sentinel date recovery."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
SCRIPT = ROOT / "scripts/protocolo_c/revp_v1of_recife_c3_plus_recheck_after_sentinel_date_recovery.py"
POSITIVE = ROOT / "datasets/recife_official_positive_candidate_registry.csv"
DATES = ROOT / "datasets/recife_positive_candidate_date_normalized_registry.csv"
REMATCH = ROOT / "datasets/recife_event_patch_temporal_rematch_registry.csv"
SPATIAL = ROOT / "datasets/recife_candidate_spatial_support_registry.csv"
PE3D = ROOT / "datasets/recife_candidate_pe3d_support_matrix.csv"
QUALITY = ROOT / "datasets/recife_sentinel_scene_date_quality_matrix.csv"
DINO = ROOT / "datasets/recife_dino_c3_plus_training_boundary_matrix.csv"
NEGATIVE = ROOT / "datasets/recife_official_negative_candidate_registry.csv"
OUT = ROOT / "datasets/recife_c3_plus_recheck_after_sentinel_date_recovery.csv"
SUMMARY = ROOT / "datasets/recife_c3_plus_unlocked_summary.csv"
C4 = ROOT / "datasets/recife_c4_status_after_sentinel_date_recovery.csv"


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


def test_v1of_can_unlock_review_candidate_but_not_c4_without_formal_negative() -> None:
    write_rows(POSITIVE, [{"candidate_id": "C1", "source_id": "RECIFE_EMLURB_156", "phenomenon": "alagamento"}])
    write_rows(DATES, [{"candidate_id": "C1", "date_parsed": "2022-05-24", "date_quality": "VALID_DATE"}])
    write_rows(REMATCH, [{"candidate_id": "C1", "patch_id": "REC_PATCH_A", "scene_date": "2022-05-29", "temporal_class": "TEMPORAL_MODERATE"}])
    write_rows(SPATIAL, [{"candidate_id": "C1", "spatial_support_status": "SPATIAL_APPROXIMATION_REVIEW"}])
    write_rows(PE3D, [{"candidate_id": "C1", "pe3d_mde_support_status": "PE3D_MDE_CONTEXT_REGISTERED_REVIEW_ONLY"}])
    write_rows(QUALITY, [{"scene_date_confirmed_count": "1"}])
    write_rows(DINO, [{"dino_status": "REVIEW_ONLY_REPRESENTATION", "dino_can_create_label": "false", "dino_can_train_model": "false"}])
    write_rows(NEGATIVE, [], ["candidate_id", "event_or_negative"])
    result = subprocess.run([sys.executable, str(SCRIPT), "--force", "--emit-evidence"], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr + result.stdout
    assert read_rows(OUT)[0]["recheck_status"] == "REC_C3_PLUS_POSITIVE_REVIEW_CANDIDATE"
    assert read_rows(SUMMARY)[0]["rec_c3_plus_positive_review_candidate_count"] == "1"
    assert read_rows(C4)[0]["c4_open"] == "false"
    assert "can_train_model,true" not in OUT.read_text(encoding="utf-8")
