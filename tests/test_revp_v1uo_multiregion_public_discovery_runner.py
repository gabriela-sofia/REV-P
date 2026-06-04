import scripts.protocolo_c.revp_v1uo_multiregion_common as common
from tests.test_revp_v1uo_multiregion_event_registry_builder import make_base


def test_discovery_runner_is_dry_run_without_download(tmp_path, monkeypatch):
    data = make_base(tmp_path, monkeypatch)
    common.run_multiregion_event_registry_builder(str(data / "v1uo_multiregion_event_registry.csv"))
    common.run_public_source_registry_builder(str(data / "v1uo_public_source_registry.csv"))
    rows = common.run_multiregion_public_discovery_runner(str(data / "discovery.csv"))
    assert rows
    assert all(r["http_status"] == "NOT_REQUESTED" for r in rows)
    assert all(r["discovery_status"] == "DRY_RUN_NOT_DOWNLOADED" for r in rows)
