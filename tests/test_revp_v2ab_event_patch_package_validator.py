from tests.test_revp_v2ab_patch_namespace_inventory import install_base_inputs, set_env
import scripts.protocolo_c.revp_v2ab_common as common


def _run_to_validation(data, scan):
    install_base_inputs(data, scan)
    common.run_patch_namespace_inventory(common.parse_args([]))
    common.run_patch_identity_crosswalk_audit(common.parse_args([]))
    common.run_event_patch_schema_contract_builder(common.parse_args([]))
    common.run_temporal_field_contract_enforcer(common.parse_args([]))
    return common.run_event_patch_package_validator(common.parse_args([]))


def test_validator_detects_missing_namespace_and_crosswalk(tmp_path, monkeypatch):
    data, scan = set_env(tmp_path, monkeypatch)
    rows = _run_to_validation(data, scan)
    by_epc = {r["event_patch_candidate_id"]: r for r in rows}
    # Candidate with empty patch_id -> blocked missing patch id / namespace missing.
    cur = by_epc["EPC2"]
    assert cur["namespace_status"].startswith("NAMESPACE_MISSING")
    assert cur["validation_status"] in {"PACKAGE_BLOCKED_MISSING_PATCH_ID", "PACKAGE_INCOMPLETE_SCHEMA"}
    # Normal candidate resolves a namespace and carries the NO_EXPLICIT_CROSSWALK
    # status toward the anchor namespace.
    rec = by_epc["EPC0"]
    assert rec["namespace_status"] == "NAMESPACE_RESOLVED"
    assert rec["crosswalk_status"] == "NO_EXPLICIT_CROSSWALK"
    assert rec["temporal_field_status"] == "SENTINEL_DATE_RECOVERED_UNLINKABLE_NAMESPACE"
    assert all(r["can_create_ground_reference"] == "false" for r in rows)
    assert all(r["can_create_training_label"] == "false" for r in rows)
