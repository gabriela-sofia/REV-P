from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "multimodal"))

from revp_v2cj_tp2_candidate_prioritization import FIELDS, main  # noqa: E402


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    inv_fields = [
        "candidate_id", "region", "event_name", "event_date", "source_name", "source_reference",
        "evidence_type", "has_observed_geometry", "geometry_format", "crs_known",
        "provenance_available", "hash_available", "human_review_required", "can_be_digitized",
        "can_be_replayed_against_patch", "candidate_status", "blocking_reason", "allowed_claim",
        "forbidden_claim",
    ]
    write_csv(root / "outputs_public/tables/revp_tp2_candidate_inventory_v2ci.csv", inv_fields, [
        {"candidate_id": "A", "region": "Recife", "event_name": "A", "event_date": "2022-01-01",
         "source_name": "Charter", "source_reference": "local", "evidence_type": "GEOMETRIA_CANDIDATA",
         "has_observed_geometry": "false", "geometry_format": "RASTER", "crs_known": "false",
         "provenance_available": "true", "hash_available": "false", "human_review_required": "true",
         "can_be_digitized": "true", "can_be_replayed_against_patch": "false",
         "candidate_status": "TP2_CANDIDATE_ONLY", "blocking_reason": "CRS_AUSENTE",
         "allowed_claim": "review", "forbidden_claim": "ground_truth"},
        {"candidate_id": "B", "region": "Curitiba", "event_name": "B", "event_date": "",
         "source_name": "UNKNOWN", "source_reference": "", "evidence_type": "EVIDENCIA_TEXTUAL",
         "has_observed_geometry": "false", "geometry_format": "UNKNOWN", "crs_known": "false",
         "provenance_available": "false", "hash_available": "false", "human_review_required": "true",
         "can_be_digitized": "false", "can_be_replayed_against_patch": "false",
         "candidate_status": "TP2_BLOCKED", "blocking_reason": "SEM_GEOMETRIA",
         "allowed_claim": "review", "forbidden_claim": "ground_truth"},
    ])
    write_csv(root / "outputs_public/tables/revp_tp2_patch_pair_candidates_v2ci.csv",
              ["pair_id", "candidate_id", "patch_id", "region", "patch_boundary_available",
               "event_geometry_available", "intersection_test_possible", "intersection_confirmed",
               "tp2_status", "tp3_ready", "blocking_reason"],
              [{"pair_id": "P1", "candidate_id": "A", "patch_id": "P", "region": "Recife",
                "patch_boundary_available": "true", "event_geometry_available": "false",
                "intersection_test_possible": "false", "intersection_confirmed": "false",
                "tp2_status": "TP2_CANDIDATE_ONLY", "tp3_ready": "false", "blocking_reason": "blocked"}])
    return root


def run_rows(tmp_path: Path) -> list[dict[str, str]]:
    root = repo(tmp_path)
    assert main(["--repo-root", str(root), "--force"]) == 0
    return read_csv(root / "outputs_public/tables/revp_tp2_candidate_priority_v2cj.csv")


def test_csv_exists(tmp_path: Path) -> None:
    root = repo(tmp_path)
    main(["--repo-root", str(root), "--force"])
    assert (root / "outputs_public/tables/revp_tp2_candidate_priority_v2cj.csv").exists()


def test_required_columns(tmp_path: Path) -> None:
    rows = run_rows(tmp_path)
    assert set(FIELDS).issubset(rows[0].keys())


def test_ranking_is_deterministic(tmp_path: Path) -> None:
    rows1 = run_rows(tmp_path)
    rows2 = run_rows(tmp_path)
    assert [r["candidate_id"] for r in rows1] == [r["candidate_id"] for r in rows2]


def test_score_name_is_review_priority(tmp_path: Path) -> None:
    rows = run_rows(tmp_path)
    assert "review_priority_score" in rows[0]
    assert "truth_score" not in rows[0]


def test_no_candidate_becomes_ground_truth(tmp_path: Path) -> None:
    rows = run_rows(tmp_path)
    assert all("GROUND_TRUTH_READY" not in row.values() for row in rows)


def test_high_priority_does_not_imply_tp2_ready(tmp_path: Path) -> None:
    rows = run_rows(tmp_path)
    high = [r for r in rows if r["priority_class"] == "HIGH_REVIEW_PRIORITY"]
    assert all(r["tp2_status"] != "TP2_READY_FOR_REPLAY" for r in high)


def test_missing_geometry_penalized(tmp_path: Path) -> None:
    rows = {r["candidate_id"]: r for r in run_rows(tmp_path)}
    assert int(rows["A"]["review_priority_score"]) > int(rows["B"]["review_priority_score"])


def test_report_exists_and_is_conservative(tmp_path: Path) -> None:
    root = repo(tmp_path)
    main(["--repo-root", str(root), "--force"])
    text = (root / "outputs_public/execution_reports/revp_tp2_candidate_priority_report_v2cj.md").read_text()
    assert "nao fecha TP2" in text
    assert "nao cria ground truth operacional" in text

