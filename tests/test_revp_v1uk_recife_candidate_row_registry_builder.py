import csv

from scripts.protocolo_c.revp_v1uk_recife_common import run_candidate_builder


def test_locality_only_candidate_never_ground_reference(tmp_path):
    matches = tmp_path / "matches.csv"
    coords = tmp_path / "coords.csv"
    locs = tmp_path / "locs.csv"
    with open(matches, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["asset_id", "row_hash", "window_type", "has_hazard_term", "coordinate_status"])
        w.writeheader()
        w.writerow({"asset_id": "asset1", "row_hash": "rh", "window_type": "event_core_window",
                    "has_hazard_term": "true", "coordinate_status": "NO_COORDINATES"})
    with open(coords, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["asset_id", "coordinate_classification"])
        w.writeheader()
        w.writerow({"asset_id": "asset1", "coordinate_classification": "NO_COORDINATES"})
    with open(locs, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["asset_id", "locality_classification"])
        w.writeheader()
        w.writerow({"asset_id": "asset1", "locality_classification": "ADDRESS_TEXT_AVAILABLE"})
    rows = run_candidate_builder(str(tmp_path / "candidates.csv"), str(matches), str(coords), str(locs))
    assert rows[0]["candidate_class"] == "ROW_LEVEL_OCCURRENCE_WITH_LOCALITY_ONLY_FOR_REVIEW"
    assert rows[0]["can_create_ground_reference"] == "false"
    assert rows[0]["can_create_training_label"] == "false"
