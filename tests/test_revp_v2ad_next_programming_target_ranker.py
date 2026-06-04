from tests.test_revp_v2ad_package_contract_qa import install_base_inputs, set_env
import scripts.protocolo_c.revp_v2ad_common as common


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
    # With the QA gate green, consolidating the registries is the top target.
    assert rows[0]["next_target"] == "MULTI_REGION_REGISTRY_HARDENING"
    assert rows[0]["recommended_version"].startswith("v2ae")
