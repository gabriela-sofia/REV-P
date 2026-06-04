import scripts.protocolo_c.revp_v1uo_multiregion_common as common
from tests.test_revp_v1uo_multiregion_event_registry_builder import make_base


def test_schema_audit_reuses_recife_and_blocks_curitiba_without_assets(tmp_path, monkeypatch):
    data = make_base(tmp_path, monkeypatch)
    common.run_multiregion_event_registry_builder(str(data / "v1uo_multiregion_event_registry.csv"))
    rows = common.run_multiregion_schema_audit_runner(str(data / "schema.csv"))
    by_region = {r["region"]: r for r in rows}
    assert by_region["REC"]["schema_status"] == "REUSED_V1UK_SCHEMA_AUDIT"
    assert by_region["CUR"]["schema_status"] == "NO_DOWNLOADED_ASSETS_YET"
