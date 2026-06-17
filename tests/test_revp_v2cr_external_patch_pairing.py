from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "multimodal"))

import revp_v2cn_to_v2cr_common as common  # noqa: E402


FORBIDDEN = ["GROUND_TRUTH_READY", "LABEL_READY", "TRAINING_READY", "MODEL_VALIDATED", "DETECTION_CONFIRMED", "PREDICTION_VALIDATED"]


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def qa_row(**overrides: str) -> dict[str, str]:
    row = {
        "qa_id": "QA_A",
        "evidence_id": "EVID_A",
        "source_family": "COPERNICUS_EMS",
        "region": "Recife",
        "local_path": "a.geojson",
        "file_type": "GEOJSON",
        "is_geospatial": "true",
        "is_vector": "true",
        "is_raster": "false",
        "crs": "EPSG:4326",
        "crs_known": "true",
        "bounds_available": "true",
        "geometry_valid": "true",
        "geometry_count": "1",
        "area_available": "true",
        "qa_status": "VALIDATED_EXTERNAL_GEOMETRY_CANDIDATE",
        "tp2_candidate_allowed": "true",
        "replay_candidate_allowed": "true",
        "blocking_reason": "",
        "allowed_claim": common.ALLOWED_CLAIM,
        "forbidden_claim": common.FORBIDDEN_CLAIM,
    }
    row.update(overrides)
    return row


def patch_row(**overrides: str) -> dict[str, str]:
    row = {"candidate_id": "CAND_A", "region": "Recife", "patch_id": "PATCH_A", "patch_boundary_available": "true", "intersection_test_possible": "true"}
    row.update(overrides)
    return row


def prepare(root: Path, qa: dict[str, str], patch: dict[str, str]) -> None:
    write_csv(root / "outputs_public/tables/revp_external_geospatial_qa_v2cq.csv", common.QA_FIELDS, [qa])
    write_csv(root / "outputs_public/tables/revp_tp2_patch_pair_candidates_v2ci.csv", ["candidate_id", "region", "patch_id", "patch_boundary_available", "intersection_test_possible"], [patch])


def test_pairing_blocks_without_patch_boundary(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    prepare(root, qa_row(), patch_row(patch_boundary_available="false"))
    assert common.build_patch_pairing(root)[0]["pairing_status"] == "PAIRING_BLOCKED_NO_PATCH_BOUNDARY"


def test_pairing_blocks_without_validated_external_geometry(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    prepare(root, qa_row(qa_status="BLOCKED_MISSING_CRS"), patch_row())
    assert common.build_patch_pairing(root)[0]["pairing_status"] == "PAIRING_BLOCKED_NO_VALIDATED_EXTERNAL_GEOMETRY"


def test_pairing_blocks_without_crs(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    prepare(root, qa_row(crs="", crs_known="false"), patch_row())
    assert common.build_patch_pairing(root)[0]["pairing_status"] == "PAIRING_BLOCKED_MISSING_CRS"


def test_intersection_not_executed_with_invalid_input(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    prepare(root, qa_row(qa_status="BLOCKED_NOT_GEOSPATIAL"), patch_row())
    row = common.build_patch_pairing(root)[0]
    assert row["intersection_executed"] == "false"


def test_area_fields_empty_when_blocked(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    prepare(root, qa_row(), patch_row(patch_boundary_available="false"))
    row = common.build_patch_pairing(root)[0]
    assert row["candidate_intersection_area"] == ""
    assert row["candidate_intersection_ratio_patch"] == ""
    assert row["candidate_intersection_ratio_evidence"] == ""


def test_ready_pairing_remains_not_executed_candidate_only(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    prepare(root, qa_row(), patch_row())
    row = common.build_patch_pairing(root)[0]
    assert row["pairing_status"] == "PAIRING_READY_NOT_EXECUTED"
    assert row["tp2_status"] == "TP2_EXTERNAL_CANDIDATE_ONLY"


def test_no_qa_rows_create_region_blockers(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    rows = common.build_patch_pairing(root)
    assert len(rows) == 3
    assert all(row["pairing_status"] == "PAIRING_BLOCKED_NO_PATCH_BOUNDARY" for row in rows)


def test_pairing_report_and_guardrails_generated(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    prepare(root, qa_row(), patch_row(patch_boundary_available="false"))
    common.run_patch_pairing(root, force=True)
    assert (root / "outputs_public/execution_reports/revp_external_patch_pairing_report_v2cr.md").exists()
    assert (root / "outputs_public/logs_summary/revp_external_patch_pairing_guardrails_v2cr.csv").exists()


def test_forbidden_status_tokens_absent_from_pairing_output(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    prepare(root, qa_row(), patch_row())
    common.run_patch_pairing(root, force=True)
    text = (root / "outputs_public/tables/revp_external_patch_pairing_v2cr.csv").read_text(encoding="utf-8")
    assert all(token not in text for token in FORBIDDEN)


def test_pairing_allowed_claim_is_explicit(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    prepare(root, qa_row(), patch_row())
    row = common.build_patch_pairing(root)[0]
    assert "candidata" in row["allowed_claim"]
