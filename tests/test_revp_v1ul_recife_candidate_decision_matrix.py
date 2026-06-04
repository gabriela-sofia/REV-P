import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.protocolo_c.revp_v1ul_recife_common import run_candidate_decision_matrix


def _write(path, columns, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def test_decision_matrix_never_creates_label_or_ground_reference(tmp_path):
    router = tmp_path / "router.csv"
    preflight = tmp_path / "preflight.csv"
    _write(router, [
        "event_id", "candidate_row_id", "review_route", "review_priority",
        "hazard_signal", "coordinate_status", "locality_status",
        "can_enter_supervisor_review", "can_enter_overlay_preflight", "asset_id",
    ], [{
        "event_id": "REC_2022_05_24_30",
        "candidate_row_id": "c1",
        "review_route": "ROUTE_COORDINATE_OCCURRENCE_REVIEW",
        "review_priority": "1",
        "hazard_signal": "HAS_HAZARD_SIGNAL",
        "coordinate_status": "OCCURRENCE_COORDINATES_CANDIDATE",
        "locality_status": "NO_LOCALITY",
        "can_enter_supervisor_review": "true",
        "can_enter_overlay_preflight": "true",
        "asset_id": "a1",
    }])
    _write(preflight, [
        "candidate_row_id", "overlay_preflight_status", "sensitive_status"
    ], [{
        "candidate_row_id": "c1",
        "overlay_preflight_status": "OVERLAY_PREFLIGHT_ELIGIBLE_AFTER_SUPERVISOR_REVIEW",
        "sensitive_status": "NO_SENSITIVE_FIELDS_DETECTED",
    }])
    rows = run_candidate_decision_matrix(
        str(tmp_path / "decision.csv"),
        str(tmp_path / "queue.csv"),
        str(router),
        str(preflight),
    )
    assert rows[0]["recommended_decision"] == "READY_FOR_SUPERVISOR_REVIEW_COORDINATE_CANDIDATE"
    assert rows[0]["can_execute_overlay_now"] == "false"
    assert rows[0]["can_create_ground_reference"] == "false"
    assert rows[0]["can_create_training_label"] == "false"
    assert "LABEL" not in rows[0]["recommended_decision"]
