from tests.test_revp_v1uv_curitiba_source_target_builder import set_env
import scripts.protocolo_c.revp_v1uv_curitiba_common as common


def test_open_data_resolver_records_formats_without_event_date(tmp_path, monkeypatch):
    set_env(tmp_path, monkeypatch)
    common.run_source_target_builder(common.parse_args([]))
    rows = common.run_open_data_resolver(common.parse_args(["--dry-run"]))
    assert {r["resource_format"] for r in rows} >= {"CSV", "GeoJSON", "PDF"}
    assert all(r["can_create_event_without_date"] == "false" for r in rows)
