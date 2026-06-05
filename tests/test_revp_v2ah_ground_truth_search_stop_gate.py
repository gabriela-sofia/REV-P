import scripts.protocolo_c.revp_v2ah_common as common
from tests.test_revp_v2ah_common import install_inputs, set_env


def test_stop_gate_is_fail_closed(tmp_path, monkeypatch):
    data = set_env(tmp_path, monkeypatch)
    install_inputs(data)
    rows = common.run_ground_truth_search_stop_gate(common.parse_args([]))
    row = rows[0]
    assert row["ground_truth_search_status"] == "GROUND_TRUTH_SEARCH_STOPPED_UNTIL_NEW_QUALIFIED_SOURCE"
    assert row["required_action"] == "maintain_candidate_review_only_layer"
    assert row["can_create_ground_reference"] == "false"
    assert row["can_create_training_label"] == "false"
    assert row["can_reopen_protocol_b"] == "false"
    assert row["can_apply_overlay"] == "false"
    assert row["can_infer_sentinel_date"] == "false"
