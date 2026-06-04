from tests.test_revp_v1uz_curitiba_context_only_hold_builder import (
    install_base_inputs, set_env,
)
import scripts.protocolo_c.revp_v1uz_common as common


def test_ranker_selects_target_by_score(tmp_path, monkeypatch):
    data = set_env(tmp_path, monkeypatch)
    install_base_inputs(data)
    rows = common.run_next_programming_target_ranker(common.parse_args([]))
    # All candidate targets are ranked and ranks are contiguous from 1.
    assert [r["rank"] for r in rows] == [str(i) for i in range(1, len(rows) + 1)]
    # With every event-patch candidate missing a Sentinel date, recovery wins.
    assert rows[0]["next_target"] == "SENTINEL_DATE_RECOVERY_FOR_EVENT_PATCH_PACKAGES"
    assert rows[0]["recommended_action"] == "SELECTED_NEXT_TARGET"
    # No target claims ground-truth value.
    assert all(r["ground_truth_value"] == "0" for r in rows)

    # Score really drives the order: scores are monotonically non-increasing.
    scores = [
        0.5 * int(r["programming_value"]) + 0.5 * int(r["blocker_reduction_value"])
        - common.EFFORT_PENALTY[r["expected_effort"]]
        - common.OVERCLAIM_PENALTY[r["overclaim_risk"]]
        for r in rows
    ]
    assert scores == sorted(scores, reverse=True)


def test_ranker_changes_with_inputs(tmp_path, monkeypatch):
    data = set_env(tmp_path, monkeypatch)
    install_base_inputs(data)
    # Remove the sentinel-missing blocker so sentinel recovery loses value.
    import csv as _csv
    path = data / "v1us_event_patch_candidate_registry.csv"
    with open(path, newline="", encoding="utf-8") as f:
        cand = list(_csv.DictReader(f))
    for r in cand:
        if r["blocker"] == "SENTINEL_DATE_AND_GEOMETRY_MISSING":
            r["blocker"] = "GEOMETRY_MISSING_ONLY"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=cand[0].keys())
        w.writeheader()
        w.writerows(cand)
    rows = common.run_next_programming_target_ranker(common.parse_args([]))
    # Without sentinel-missing pressure, sentinel recovery is no longer rank 1.
    assert rows[0]["next_target"] != "SENTINEL_DATE_RECOVERY_FOR_EVENT_PATCH_PACKAGES"
