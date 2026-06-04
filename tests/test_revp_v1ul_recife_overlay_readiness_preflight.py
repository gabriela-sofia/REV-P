import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.protocolo_c.revp_v1ul_recife_common import run_overlay_readiness_preflight


def _write(path, rows):
    columns = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _route(route, coord="NO_COORDINATES"):
    return {
        "event_id": "REC_2022_05_24_30",
        "candidate_row_id": route,
        "asset_id": "a1",
        "event_window_match": "event_core_window",
        "hazard_signal": "HAS_HAZARD_SIGNAL",
        "coordinate_status": coord,
        "sensitive_review_required": "false",
        "review_route": route,
    }


def test_locality_only_never_enters_overlay(tmp_path):
    router = tmp_path / "router.csv"
    _write(router, [_route("ROUTE_LOCALITY_ONLY_REVIEW")])
    rows = run_overlay_readiness_preflight(str(tmp_path / "preflight.csv"), str(router))
    assert rows[0]["overlay_preflight_status"] == "LOCALITY_ONLY_NOT_OVERLAY_ELIGIBLE"
    assert rows[0]["can_execute_overlay_now"] == "false"


def test_contextual_layer_never_enters_overlay(tmp_path):
    router = tmp_path / "router.csv"
    _write(router, [_route("ROUTE_CONTEXT_ONLY")])
    rows = run_overlay_readiness_preflight(str(tmp_path / "preflight.csv"), str(router))
    assert rows[0]["overlay_preflight_status"] == "CONTEXTUAL_LAYER_NOT_ELIGIBLE"
    assert rows[0]["can_create_ground_reference"] == "false"


def test_coordinate_candidate_is_preflight_only_not_overlay_execution(tmp_path):
    router = tmp_path / "router.csv"
    _write(router, [_route("ROUTE_COORDINATE_OCCURRENCE_REVIEW", "OCCURRENCE_COORDINATES_CANDIDATE")])
    rows = run_overlay_readiness_preflight(str(tmp_path / "preflight.csv"), str(router))
    assert rows[0]["overlay_preflight_status"] == "OVERLAY_PREFLIGHT_ELIGIBLE_AFTER_SUPERVISOR_REVIEW"
    assert rows[0]["can_execute_overlay_now"] == "false"
    assert rows[0]["can_create_training_label"] == "false"
