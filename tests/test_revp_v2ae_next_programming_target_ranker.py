from tests.test_revp_v2ae_canonical_region_registry_builder import install_base_inputs, set_env
import scripts.protocolo_c.revp_v2ae_common as common


def test_ranker_orders_targets_by_score(tmp_path, monkeypatch):
    data = set_env(tmp_path, monkeypatch)
    install_base_inputs(data)
    common.run_all(common.parse_args([]))
    rows = common.run_next_programming_target_ranker(common.parse_args([]))
    assert [r["rank"] for r in rows] == [str(i) for i in range(1, len(rows) + 1)]
    assert rows[0]["recommended_action"] == "SELECTED_NEXT_TARGET"
    assert all(r["ground_truth_value"] == "0" for r in rows)
    scores = [
        0.5 * int(r["programming_value"]) + 0.5 * int(r["blocker_reduction_value"])
        - common.EFFORT_PENALTY[r["expected_effort"]]
        - common.OVERCLAIM_PENALTY[r["overclaim_risk"]]
        for r in rows
    ]
    assert scores == sorted(scores, reverse=True)
    # Registries consistent and QA green -> automation is the top target.
    assert rows[0]["next_target"] == "EVENT_PATCH_PACKAGE_V2_QA_AUTOMATION"
    assert rows[0]["recommended_version"].startswith("v2af")
