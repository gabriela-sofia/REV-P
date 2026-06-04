from tests.test_revp_v1ux_curitiba_download_target_builder import seed_fixture_downloads, set_env
import scripts.protocolo_c.revp_v1ux_curitiba_common as common


def test_geodata_metadata_audit_blocks_overlay_and_ground_reference(tmp_path, monkeypatch):
    data, raw = set_env(tmp_path, monkeypatch)
    seed_fixture_downloads(data, raw)
    common.run_artifact_inventory(common.parse_args([]))
    common.run_schema_audit(common.parse_args([]))
    rows = common.run_geodata_metadata_audit(common.parse_args([]))
    classes = {r["geodata_class"] for r in rows}
    assert "context layer" in classes
    assert "possible occurrence" in classes
    assert all(r["can_support_overlay_preflight"] == "false" for r in rows)
    assert all(r["can_create_ground_reference"] == "false" for r in rows)
