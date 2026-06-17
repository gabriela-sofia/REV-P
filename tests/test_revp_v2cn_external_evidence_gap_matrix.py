from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "multimodal"))

from revp_v2cn_to_v2cr_common import GAP_FIELDS, build_gap_matrix, run_gap_matrix  # noqa: E402


FORBIDDEN = ["GROUND_TRUTH_READY", "LABEL_READY", "TRAINING_READY", "MODEL_VALIDATED", "DETECTION_CONFIRMED", "PREDICTION_VALIDATED"]


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    write_csv(
        root / "outputs_public/tables/revp_tp2_candidate_inventory_v2ci.csv",
        ["candidate_id", "region", "candidate_status", "has_observed_geometry", "crs_known", "provenance_available", "hash_available", "license_status"],
        [
            {
                "candidate_id": "RECIFE_A",
                "region": "Recife",
                "candidate_status": "TP2_CANDIDATE_ONLY",
                "has_observed_geometry": "false",
                "crs_known": "false",
                "provenance_available": "true",
                "hash_available": "false",
                "license_status": "UNKNOWN",
            }
        ],
    )
    write_csv(
        root / "outputs_public/tables/revp_tp2_patch_pair_candidates_v2ci.csv",
        ["candidate_id", "region", "patch_id", "patch_boundary_available", "intersection_test_possible"],
        [{"candidate_id": "RECIFE_A", "region": "Recife", "patch_id": "PATCH_A", "patch_boundary_available": "false", "intersection_test_possible": "false"}],
    )
    return root


def test_gap_matrix_output_exists(tmp_path: Path) -> None:
    root = repo(tmp_path)
    assert run_gap_matrix(root, force=True) == 0
    assert (root / "outputs_public/tables/revp_external_evidence_gap_matrix_v2cn.csv").exists()


def test_gap_matrix_required_columns_exist(tmp_path: Path) -> None:
    root = repo(tmp_path)
    run_gap_matrix(root, force=True)
    rows = read_csv(root / "outputs_public/tables/revp_external_evidence_gap_matrix_v2cn.csv")
    assert set(GAP_FIELDS).issubset(rows[0].keys())


def test_missing_geometry_becomes_critical_gap(tmp_path: Path) -> None:
    rows = build_gap_matrix(repo(tmp_path))
    recife = next(row for row in rows if row["event_or_candidate_id"] == "RECIFE_A")
    assert recife["missing_observed_geometry"] == "true"
    assert recife["priority"] == "CRITICAL_GEOSPATIAL_GAP"


def test_recife_petropolis_curitiba_appear(tmp_path: Path) -> None:
    rows = build_gap_matrix(repo(tmp_path))
    assert {"Recife", "Petropolis", "Curitiba"}.issubset({row["region"] for row in rows})


def test_allowed_and_forbidden_claims_are_explicit(tmp_path: Path) -> None:
    rows = build_gap_matrix(repo(tmp_path))
    assert all(row["allowed_claim"] for row in rows)
    assert all("ground_truth_operacional" in row["forbidden_claim"] for row in rows)


def test_patch_boundary_gap_is_recorded(tmp_path: Path) -> None:
    rows = build_gap_matrix(repo(tmp_path))
    recife = next(row for row in rows if row["event_or_candidate_id"] == "RECIFE_A")
    assert recife["missing_patch_boundary"] == "true"
    assert "MISSING_PATCH_BOUNDARY" in recife["blocking_reason"]


def test_high_gap_when_only_metadata_missing(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    write_csv(
        root / "outputs_public/tables/revp_tp2_candidate_inventory_v2ci.csv",
        ["candidate_id", "region", "candidate_status", "has_observed_geometry", "crs_known", "provenance_available", "hash_available", "license_status"],
        [{"candidate_id": "A", "region": "Curitiba", "candidate_status": "CANDIDATE", "has_observed_geometry": "true", "crs_known": "true", "provenance_available": "false", "hash_available": "false", "license_status": "UNKNOWN"}],
    )
    write_csv(root / "outputs_public/tables/revp_tp2_patch_pair_candidates_v2ci.csv", ["candidate_id", "region", "patch_boundary_available", "intersection_test_possible"], [{"candidate_id": "A", "region": "Curitiba", "patch_boundary_available": "true", "intersection_test_possible": "true"}])
    row = next(row for row in build_gap_matrix(root) if row["event_or_candidate_id"] == "A")
    assert row["priority"] == "HIGH_EVIDENCE_GAP"


def test_gap_report_is_generated(tmp_path: Path) -> None:
    root = repo(tmp_path)
    run_gap_matrix(root, force=True)
    text = (root / "outputs_public/execution_reports/revp_external_evidence_gap_matrix_report_v2cn.md").read_text(encoding="utf-8")
    assert "lacunas geoespaciais" in text


def test_forbidden_status_tokens_absent_from_outputs(tmp_path: Path) -> None:
    root = repo(tmp_path)
    run_gap_matrix(root, force=True)
    text = (root / "outputs_public/tables/revp_external_evidence_gap_matrix_v2cn.csv").read_text(encoding="utf-8")
    assert all(token not in text for token in FORBIDDEN)


def test_gap_matrix_is_deterministic(tmp_path: Path) -> None:
    root = repo(tmp_path)
    run_gap_matrix(root, force=True)
    first = (root / "outputs_public/tables/revp_external_evidence_gap_matrix_v2cn.csv").read_text(encoding="utf-8")
    run_gap_matrix(root, force=True)
    second = (root / "outputs_public/tables/revp_external_evidence_gap_matrix_v2cn.csv").read_text(encoding="utf-8")
    assert first == second
