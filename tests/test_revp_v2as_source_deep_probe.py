import scripts.protocolo_c.revp_v2as_common as common
from tests.test_revp_v2as_common import install_all


def test_offline_deterministic_no_versioning(tmp_path, monkeypatch):
    install_all(tmp_path, monkeypatch)
    common.run_deep_probe_priority_builder(common.parse_args([]))
    rows = common.run_source_deep_probe(common.parse_args([]))
    assert rows
    assert all(r["network_enabled"] == "false" for r in rows)
    assert all(r["probe_status"] == "NETWORK_DISABLED_DETERMINISTIC_RUN" for r in rows)
    assert all(r["raw_data_versioned"] == "false" for r in rows)
    assert all(r["cached_temporarily"] == "false" for r in rows)
