from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "multimodal"))

import revp_v2cx_to_v2dd_common as common  # noqa: E402
from revp_v2db_patch_boundary_readiness_audit import main  # noqa: E402


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def prepared(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    common.write_csv(root / "outputs_public/tables/revp_external_patch_pairing_v2cr.csv", [
        {"pairing_id": "PAIR1", "patch_id": "REC_00019", "evidence_id": "E1", "region": "Recife", "patch_boundary_available": "false"},
        {"pairing_id": "PAIR2", "patch_id": "PET_001", "evidence_id": "E2", "region": "Petropolis", "patch_boundary_available": "true", "crs": ""},
        {"pairing_id": "PAIR3", "patch_id": "CTB_001", "evidence_id": "E3", "region": "Curitiba", "patch_boundary_available": "false"},
    ], ["pairing_id", "patch_id", "evidence_id", "region", "patch_boundary_available", "crs"])
    return root


def test_patch_without_boundary_blocks_pairing(tmp_path: Path) -> None:
    rows = common.build_boundary_audit(prepared(tmp_path))
    ctb = next(row for row in rows if row["patch_id"] == "CTB_001")
    assert ctb["patch_boundary_status"] == "NO_BOUNDARY_AVAILABLE"
    assert ctb["pairing_ready"] == "false"


def test_boundary_without_crs_blocks(tmp_path: Path) -> None:
    rows = common.build_boundary_audit(prepared(tmp_path))
    pet = next(row for row in rows if row["patch_id"] == "PET_001")
    assert pet["patch_boundary_status"] == "BOUNDARY_BLOCKED_MISSING_CRS"


def test_rec_00019_stays_candidate_not_final(tmp_path: Path) -> None:
    rows = common.build_boundary_audit(prepared(tmp_path))
    rec = next(row for row in rows if row["patch_id"] == "REC_00019")
    assert rec["patch_boundary_status"] == "BOUNDARY_CANDIDATE_FROM_BOUNDS"
    assert rec["pairing_ready"] == "false"


def test_human_review_required_for_all(tmp_path: Path) -> None:
    assert all(row["human_review_required"] == "true" for row in common.build_boundary_audit(prepared(tmp_path)))


def test_cli_writes_boundary_table(tmp_path: Path) -> None:
    root = prepared(tmp_path)
    assert main(["--repo-root", str(root), "--offline", "--force"]) == 0
    assert common.boundary_path(root).exists()


def test_boundary_guardrail_generated(tmp_path: Path) -> None:
    root = prepared(tmp_path)
    main(["--repo-root", str(root), "--offline", "--force"])
    rows = read_csv(root / "outputs_public/logs_summary/revp_patch_boundary_guardrails_v2db.csv")
    assert any(row["guardrail"] == "rec_00019_not_final" for row in rows)


@pytest.mark.parametrize("field", common.BOUNDARY_FIELDS)
def test_boundary_has_required_fields(tmp_path: Path, field: str) -> None:
    row = common.build_boundary_audit(prepared(tmp_path))[0]
    assert field in row


def test_no_boundary_ground_truth_status(tmp_path: Path) -> None:
    assert all("PATCH_GROUND_TRUTH_READY" not in row["patch_boundary_status"] for row in common.build_boundary_audit(prepared(tmp_path)))
