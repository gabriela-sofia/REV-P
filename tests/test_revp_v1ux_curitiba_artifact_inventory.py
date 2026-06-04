from tests.test_revp_v1ux_curitiba_download_target_builder import seed_fixture_downloads, set_env
import scripts.protocolo_c.revp_v1ux_curitiba_common as common


def test_artifact_inventory_classifies_fixture_types(tmp_path, monkeypatch):
    data, raw = set_env(tmp_path, monkeypatch)
    seed_fixture_downloads(data, raw)
    rows = common.run_artifact_inventory(common.parse_args([]))
    types = {r["artifact_type"] for r in rows}
    assert {"CSV", "GeoJSON", "ZIP", "PDF", "JSON"}.issubset(types)
    assert any(r["row_count"] == "1" and r["artifact_type"] == "CSV" for r in rows)
    assert any(r["zip_member_count"] == "1" for r in rows)
    assert all("local_only" not in str(r) for r in rows)
