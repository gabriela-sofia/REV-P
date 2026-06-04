from tests.test_revp_v2ac_event_patch_v2_package_builder import install_base_inputs, set_env
import scripts.protocolo_c.revp_v2ac_common as common


def test_readiness_matrix_keeps_overlay_ground_reference_training_blocked(tmp_path, monkeypatch):
    data = set_env(tmp_path, monkeypatch)
    install_base_inputs(data)
    common.run_event_patch_v2_package_builder(common.parse_args([]))
    common.run_schema_contract_validator(common.parse_args([]))
    rows = common.run_v2_readiness_matrix_builder(common.parse_args([]))
    blocked = {"overlay_readiness", "ground_reference_readiness", "training_readiness"}
    for r in rows:
        if r["dimension"] in blocked:
            assert r["classification"] == "BLOCKED"
    # Occurrence coordinate / geometry support always ABSENT.
    for r in rows:
        if r["dimension"] in {"coordinate_support", "geometry_support"}:
            assert r["classification"] == "ABSENT"
    assert all(r["can_create_training_label"] == "false" for r in rows)
    assert all(r["schema_migration_only"] == "true" for r in rows)
