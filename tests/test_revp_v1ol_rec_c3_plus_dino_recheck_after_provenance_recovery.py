"""Tests for v1ol REC C3+ and DINO recheck.

All I/O is redirected to tmp_path — datasets/ is never touched.
"""

from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
SCRIPT = ROOT / "scripts/protocolo_c/revp_v1ol_rec_c3_plus_dino_recheck_after_provenance_recovery.py"
DATASETS = ROOT / "datasets"


def _write(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fnames = fields or (list(rows[0].keys()) if rows else [])
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fnames)
        w.writeheader()
        w.writerows(rows)


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def test_v1ol_creates_review_queue_but_keeps_c4_closed_without_negatives(tmp_path: Path) -> None:
    in_pos = tmp_path / "pos.csv"
    in_date = tmp_path / "date.csv"
    in_spatial = tmp_path / "spatial.csv"
    in_support = tmp_path / "support.csv"
    in_rematch = tmp_path / "rematch.csv"
    in_dino_boundary = tmp_path / "dino_boundary.csv"
    in_neg = tmp_path / "neg.csv"
    out_c3 = tmp_path / "c3.csv"
    out_dino = tmp_path / "dino.csv"
    out_c4 = tmp_path / "c4.csv"

    # Use realistic IDs — not REC_00001/C1 fixture patterns
    _write(in_pos, [{"candidate_id": "RECIFE_POSITIVE_CANDIDATE_V1NR_00263",
                     "source_id": "RECIFE_EMLURB_156", "phenomenon": "alagamento"}])
    _write(in_date, [{"candidate_id": "RECIFE_POSITIVE_CANDIDATE_V1NR_00263",
                      "date_parsed": "2022-05-24", "date_quality": "VALID_DATE"}])
    _write(in_spatial, [{"candidate_id": "RECIFE_POSITIVE_CANDIDATE_V1NR_00263",
                         "spatial_support_status": "SPATIAL_POINT_AVAILABLE"}])
    _write(in_support, [{"candidate_id": "RECIFE_POSITIVE_CANDIDATE_V1NR_00263",
                         "pe3d_mde_support_status": "PE3D_MDE_CONTEXT_REGISTERED_REVIEW_ONLY"}])
    _write(in_rematch, [{"candidate_id": "RECIFE_POSITIVE_CANDIDATE_V1NR_00263",
                         "patch_id": "RECIFE_SENTINEL_PATCH_S2A_001",
                         "scene_date": "2022-05-25",
                         "temporal_class": "TEMPORAL_STRONG"}])
    _write(in_dino_boundary, [{"dino_status": "REVIEW_ONLY_REPRESENTATION"}])
    _write(in_neg, [], ["candidate_id", "event_or_negative"])

    env = {**os.environ,
           "REVP_V1OL_IN_POS": str(in_pos),
           "REVP_V1OL_IN_DATE": str(in_date),
           "REVP_V1OL_IN_SPATIAL": str(in_spatial),
           "REVP_V1OL_IN_SUPPORT": str(in_support),
           "REVP_V1OL_IN_REMATCH": str(in_rematch),
           "REVP_V1OL_OUT_C3": str(out_c3),
           "REVP_V1OL_OUT_DINO": str(out_dino),
           "REVP_V1OL_OUT_C4": str(out_c4),
           "REVP_V1OL_SCHEMA_C3": str(tmp_path / "s_c3.csv"),
           "REVP_V1OL_SCHEMA_DINO": str(tmp_path / "s_dino.csv"),
           "REVP_V1OL_SCHEMA_C4": str(tmp_path / "s_c4.csv"),
           "REVP_V1OL_DOC": str(tmp_path / "doc.md")}

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--force", "--emit-evidence"],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    assert _read(out_c3)[0]["recheck_status"] == "REC_C3_PLUS_POSITIVE_REVIEW_CANDIDATE"
    assert _read(out_dino)[0]["dino_can_create_label"] == "false"
    assert _read(out_c4)[0]["c4_open"] == "false"

    # Verify the test outputs are in tmp_path, not in real datasets/
    assert out_c3.exists(), "Output must be in tmp_path"
    assert not (DATASETS / "recife_c3_plus_recheck_after_provenance_recovery.csv").samefile(out_c3), \
        "Test must not write to real datasets/"
