from tests.test_revp_v1ux_curitiba_download_target_builder import seed_fixture_downloads, set_env
import scripts.protocolo_c.revp_v1ux_curitiba_common as common


def test_field_mapper_hashes_address_fields_and_does_not_geocode(tmp_path, monkeypatch):
    data, raw = set_env(tmp_path, monkeypatch)
    seed_fixture_downloads(data, raw)
    common.run_artifact_inventory(common.parse_args([]))
    common.run_schema_audit(common.parse_args([]))
    rows = common.run_hazard_date_locality_field_mapper(common.parse_args([]))
    assert any(r["address_hash_fields"] for r in rows)
    assert all(r["raw_address_versioned"] == "false" for r in rows)
    assert all(r["geocoding_executed"] == "false" for r in rows)
