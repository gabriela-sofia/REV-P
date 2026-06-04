from tests.test_revp_v2ae_canonical_region_registry_builder import install_base_inputs, set_env
import scripts.protocolo_c.revp_v2ae_common as common


def test_readiness_consolidation_keeps_blocked_dimensions(tmp_path, monkeypatch):
    data = set_env(tmp_path, monkeypatch)
    install_base_inputs(data)
    common.run_canonical_region_registry_builder(common.parse_args([]))
    common.run_canonical_event_registry_builder(common.parse_args([]))
    rows = common.run_multiregion_readiness_consolidator(common.parse_args([]))
    scopes = {r["readiness_scope"] for r in rows}
    assert {"REGION", "EVENT", "PACKAGE"} <= scopes
    blocked = {"overlay_readiness", "ground_reference_readiness", "training_readiness"}
    for r in rows:
        if r["dimension"] in blocked:
            assert r["status"] == "BLOCKED"
    assert all(r["multiregion_registry_hardening_only"] == "true" for r in rows)
    assert all(r["can_create_ground_reference"] == "false" for r in rows)
