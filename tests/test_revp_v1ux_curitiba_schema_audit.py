from tests.test_revp_v1ux_curitiba_download_target_builder import seed_fixture_downloads, set_env
import scripts.protocolo_c.revp_v1ux_curitiba_common as common


def test_schema_audit_detects_date_hazard_locality_without_values(tmp_path, monkeypatch):
    data, raw = set_env(tmp_path, monkeypatch)
    seed_fixture_downloads(data, raw)
    common.run_artifact_inventory(common.parse_args([]))
    rows = common.run_schema_audit(common.parse_args([]))
    assert any(r["schema_class"] == "EVENT_OR_SERVICE_TABLE_SCHEMA" for r in rows)
    assert any(r["schema_class"] == "DATE_ONLY_SCHEMA" for r in rows)
    assert any(r["has_coordinate_fields"] == "true" for r in rows)
    assert all("Boqueirao" not in str(r) for r in rows)
