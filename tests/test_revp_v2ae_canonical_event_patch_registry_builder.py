from tests.test_revp_v2ae_canonical_region_registry_builder import install_base_inputs, set_env
import scripts.protocolo_c.revp_v2ae_common as common


def test_event_patch_registry_preserves_packages(tmp_path, monkeypatch):
    data = set_env(tmp_path, monkeypatch)
    install_base_inputs(data)
    src = common.load_csv(str(data / "v2ac_event_patch_v2_package_registry.csv"))
    rows = common.run_canonical_event_patch_registry_builder(common.parse_args([]))
    assert len(rows) == len(src)
    assert {r["event_patch_candidate_id"] for r in rows} == {r["event_patch_candidate_id"] for r in src}
    for r in rows:
        assert r["overlay_status"] == "BLOCKED"
        assert r["ground_reference_status"] == "BLOCKED"
        assert r["training_label_status"] == "BLOCKED"
        assert r["can_create_ground_reference"] == "false"
        assert r["can_create_training_label"] == "false"
