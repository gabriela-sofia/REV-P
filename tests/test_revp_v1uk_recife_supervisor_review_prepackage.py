import csv

from scripts.protocolo_c.revp_v1uk_recife_common import run_supervisor_prepackage


def test_supervisor_prepackage_counts_candidates(tmp_path):
    candidates = tmp_path / "candidates.csv"
    with open(candidates, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["candidate_class"])
        w.writeheader()
        w.writerow({"candidate_class": "ROW_LEVEL_OCCURRENCE_WITH_COORDINATES_FOR_REVIEW"})
        w.writerow({"candidate_class": "ROW_LEVEL_OCCURRENCE_WITH_LOCALITY_ONLY_FOR_REVIEW"})
    rows = run_supervisor_prepackage(str(tmp_path / "pre.csv"), str(candidates))
    assert rows[0]["coordinate_candidates_count"] == "1"
    assert rows[0]["locality_only_candidates_count"] == "1"
    assert rows[0]["can_advance_to_v1ul"] == "true"
