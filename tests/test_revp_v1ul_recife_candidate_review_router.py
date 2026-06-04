import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.protocolo_c.revp_v1ul_recife_common import run_candidate_review_router


def _candidate(**overrides):
    row = {
        "candidate_row_id": "c1",
        "event_id": "REC_2022_05_24_30",
        "asset_id": "a1",
        "row_hash": "rh",
        "candidate_class": "ROW_LEVEL_OCCURRENCE_WITH_COORDINATES_FOR_REVIEW",
        "event_window_match": "event_core_window",
        "hazard_term_status": "HAS_HAZARD_SIGNAL",
        "coordinate_status": "OCCURRENCE_COORDINATES_CANDIDATE",
        "locality_status": "NO_LOCALITY",
        "review_priority": "1",
    }
    row.update(overrides)
    return row


def _write(path, rows):
    columns = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def test_router_classifies_coordinate_candidate_without_promotion(tmp_path):
    candidates = tmp_path / "candidates.csv"
    _write(candidates, [_candidate()])
    rows = run_candidate_review_router(str(tmp_path / "router.csv"), str(candidates))
    assert rows[0]["review_route"] == "ROUTE_COORDINATE_OCCURRENCE_REVIEW"
    assert rows[0]["can_enter_overlay_preflight"] == "true"
    assert rows[0]["can_create_ground_reference"] == "false"
    assert rows[0]["can_create_training_label"] == "false"


def test_router_classifies_locality_only_candidate(tmp_path):
    candidates = tmp_path / "candidates.csv"
    _write(candidates, [_candidate(
        candidate_class="ROW_LEVEL_OCCURRENCE_WITH_LOCALITY_ONLY_FOR_REVIEW",
        coordinate_status="NO_COORDINATES",
        locality_status="ADDRESS_TEXT_AVAILABLE",
        review_priority="2",
    )])
    rows = run_candidate_review_router(str(tmp_path / "router.csv"), str(candidates))
    assert rows[0]["review_route"] == "ROUTE_LOCALITY_ONLY_REVIEW"
    assert rows[0]["can_enter_supervisor_review"] == "true"
    assert rows[0]["can_enter_overlay_preflight"] == "false"
    assert rows[0]["sensitive_review_required"] == "true"
