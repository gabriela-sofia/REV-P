from tests.test_revp_v2aa_patch_source_registry_scanner import (
    install_base_inputs, set_env,
)
import scripts.protocolo_c.revp_v2aa_common as common


def test_ranker_orders_targets_by_score(tmp_path, monkeypatch):
    data, scan = set_env(tmp_path, monkeypatch)
    install_base_inputs(data, scan)
    common.run_all(common.parse_args([]))
    rows = common.run_next_programming_target_ranker(common.parse_args([]))
    assert [r["rank"] for r in rows] == [str(i) for i in range(1, len(rows) + 1)]
    assert rows[0]["recommended_action"] == "SELECTED_NEXT_TARGET"
    assert all(r["ground_truth_value"] == "0" for r in rows)
    # Scores must be monotonically non-increasing with rank.
    scores = [
        0.5 * int(r["programming_value"]) + 0.5 * int(r["blocker_reduction_value"])
        - common.EFFORT_PENALTY[r["expected_effort"]]
        - common.OVERCLAIM_PENALTY[r["overclaim_risk"]]
        for r in rows
    ]
    assert scores == sorted(scores, reverse=True)
    # With the candidate namespace mostly unrecoverable, "continue" is not rank 1.
    assert rows[0]["next_target"] != "SENTINEL_DATE_RECOVERY_CONTINUE"
    assert rows[0]["recommended_version"].startswith("v2ab")
