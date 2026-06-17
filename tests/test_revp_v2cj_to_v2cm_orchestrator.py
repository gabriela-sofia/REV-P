from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "multimodal"))

from revp_v2cj_to_v2cm_orchestrator import main  # noqa: E402


FORBIDDEN = "GROUND_TRUTH_READY|LABEL_READY|TRAINING_READY|MODEL_VALIDATED|DETECTION_CONFIRMED|PREDICTION_VALIDATED"


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
    write_csv(root / "outputs_public/tables/revp_tp2_candidate_inventory_v2ci.csv",
              ["candidate_id", "region", "event_name", "event_date", "source_name", "source_reference",
               "evidence_type", "has_observed_geometry", "geometry_format", "crs_known",
               "provenance_available", "hash_available", "human_review_required", "can_be_digitized",
               "can_be_replayed_against_patch", "candidate_status", "blocking_reason", "allowed_claim",
               "forbidden_claim"],
              [{"candidate_id": "A", "region": "Recife", "event_name": "A", "event_date": "2022",
                "source_name": "src", "source_reference": "ref", "evidence_type": "EVIDENCIA_VISUAL",
                "has_observed_geometry": "false", "geometry_format": "PNG", "crs_known": "false",
                "provenance_available": "true", "hash_available": "false", "human_review_required": "true",
                "can_be_digitized": "true", "can_be_replayed_against_patch": "false",
                "candidate_status": "TP2_CANDIDATE_ONLY", "blocking_reason": "blocked",
                "allowed_claim": "review", "forbidden_claim": "ground_truth"}])
    write_csv(root / "outputs_public/tables/revp_tp2_patch_pair_candidates_v2ci.csv",
              ["pair_id", "candidate_id", "patch_id", "region", "patch_boundary_available",
               "event_geometry_available", "intersection_test_possible", "intersection_confirmed",
               "tp2_status", "tp3_ready", "blocking_reason"],
              [{"pair_id": "P", "candidate_id": "A", "patch_id": "PATCH", "region": "Recife",
                "patch_boundary_available": "false", "event_geometry_available": "false",
                "intersection_test_possible": "false", "intersection_confirmed": "false",
                "tp2_status": "TP2_CANDIDATE_ONLY", "tp3_ready": "false", "blocking_reason": "blocked"}])
    return root


def test_pipeline_runs_complete(tmp_path: Path) -> None:
    assert main(["--repo-root", str(repo(tmp_path)), "--force"]) == 0


def test_integrated_report_generated(tmp_path: Path) -> None:
    root = repo(tmp_path)
    main(["--repo-root", str(root), "--force"])
    assert (root / "outputs_public/execution_reports/revp_v2cj_to_v2cm_integrated_report.md").exists()


def test_commit_checklist_generated(tmp_path: Path) -> None:
    root = repo(tmp_path)
    main(["--repo-root", str(root), "--force"])
    assert (root / "outputs_public/execution_reports/revp_v2cj_to_v2cm_commit_checklist.md").exists()


def test_test_rollup_generated(tmp_path: Path) -> None:
    root = repo(tmp_path)
    main(["--repo-root", str(root), "--force"])
    rows = read_csv(root / "outputs_public/logs_summary/revp_v2cj_to_v2cm_test_rollup.csv")
    assert len(rows) == 4
    assert all(row["status"] == "PASS" for row in rows)


def test_guardrail_rollup_generated(tmp_path: Path) -> None:
    root = repo(tmp_path)
    main(["--repo-root", str(root), "--force"])
    rows = read_csv(root / "outputs_public/logs_summary/revp_v2cj_to_v2cm_guardrail_rollup.csv")
    assert any(row["guardrail"] == "training_ready" and row["observed_value"] == "BLOCKED" for row in rows)


def test_no_forbidden_status_in_outputs(tmp_path: Path) -> None:
    root = repo(tmp_path)
    main(["--repo-root", str(root), "--force"])
    text = "\n".join(p.read_text(encoding="utf-8", errors="ignore") for p in (root / "outputs_public").rglob("*") if p.is_file())
    for token in FORBIDDEN.split("|"):
        assert token not in text


def test_replay_remains_blocked(tmp_path: Path) -> None:
    root = repo(tmp_path)
    main(["--repo-root", str(root), "--force"])
    rows = read_csv(root / "outputs_public/tables/revp_patch_event_replay_v2cm.csv")
    assert rows[0]["replay_status"].startswith("REPLAY_BLOCKED")


def test_training_ready_remains_blocked(tmp_path: Path) -> None:
    root = repo(tmp_path)
    main(["--repo-root", str(root), "--force"])
    rows = read_csv(root / "outputs_public/logs_summary/revp_v2cj_to_v2cm_guardrail_rollup.csv")
    assert next(row for row in rows if row["guardrail"] == "training_ready")["observed_value"] == "BLOCKED"

