from tests.test_revp_v2ab_patch_namespace_inventory import install_base_inputs, set_env
import scripts.protocolo_c.revp_v2ab_common as common


def _run_to_completeness(data, scan):
    install_base_inputs(data, scan)
    common.run_patch_namespace_inventory(common.parse_args([]))
    common.run_patch_identity_crosswalk_audit(common.parse_args([]))
    common.run_event_patch_schema_contract_builder(common.parse_args([]))
    common.run_temporal_field_contract_enforcer(common.parse_args([]))
    common.run_event_patch_package_validator(common.parse_args([]))
    return common.run_package_completeness_scorer(common.parse_args([]))


def test_completeness_is_structural_not_ground_truth(tmp_path, monkeypatch):
    data, scan = set_env(tmp_path, monkeypatch)
    rows = _run_to_completeness(data, scan)
    by_epc = {r["event_patch_candidate_id"]: r for r in rows}
    # Score is a bounded structural percentage, never a performance/truth metric.
    for r in rows:
        assert 0 <= int(r["completeness_score"]) <= 100
        assert r["can_create_ground_reference"] == "false"
        assert r["can_create_training_label"] == "false"
        assert r["safe_for_review_only_use"] == "true"
    # Candidate missing patch id is blocked, not scored as complete.
    assert by_epc["EPC2"]["completeness_class"] == "PACKAGE_BLOCKED_MISSING_PATCH_ID"
    # A structurally complete package still carries a temporal blocker.
    assert by_epc["EPC0"]["completeness_class"] in {
        "PACKAGE_COMPLETE_WITH_TEMPORAL_BLOCKER", "PACKAGE_INCOMPLETE_SCHEMA",
    }
    # No completeness class is ever an operational/ground-truth status.
    forbidden = {"OPERATIONAL_VALIDATED", "GROUND_TRUTH", "FLOOD_DETECTED"}
    assert all(r["completeness_class"] not in forbidden for r in rows)
