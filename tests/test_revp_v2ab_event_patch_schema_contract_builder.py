from tests.test_revp_v2ab_patch_namespace_inventory import install_base_inputs, set_env
import scripts.protocolo_c.revp_v2ab_common as common


def test_schema_contract_has_required_fields_and_blockers(tmp_path, monkeypatch):
    data, scan = set_env(tmp_path, monkeypatch)
    install_base_inputs(data, scan)
    rows = common.run_event_patch_schema_contract_builder(common.parse_args([]))
    by_name = {r["field_name"]: r for r in rows}
    for required in ("event_patch_candidate_id", "event_id", "patch_id", "patch_namespace",
                     "sentinel_date_status", "blocker", "safe_use", "prohibited_use"):
        assert by_name[required]["required"] == "true"
        assert by_name[required]["requires_blocker_if_null"] == "true"
    # Optional fields are nullable but must require an explicit blocker if null.
    for optional in ("sentinel_scene_date", "refpatch_id", "explicit_crosswalk_id"):
        assert by_name[optional]["required"] == "false"
        assert by_name[optional]["nullable"] == "true"
        assert by_name[optional]["requires_blocker_if_null"] == "true"
    # Guardrail status fields are pinned to BLOCKED.
    assert by_name["overlay_status"]["allowed_values"] == "BLOCKED"
    assert by_name["ground_reference_status"]["allowed_values"] == "BLOCKED"
