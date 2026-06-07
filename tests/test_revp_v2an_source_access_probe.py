import scripts.protocolo_c.revp_v2an_common as common
from tests.test_revp_v2an_common import install_all


def test_probe_offline_is_fail_closed(tmp_path, monkeypatch):
    data, protocol, docs, _ = install_all(tmp_path, monkeypatch)
    rows = common.run_source_access_probe(common.parse_args([]))
    assert rows
    assert all(r["raw_data_versioned"] == "false" for r in rows)
    assert all(r["raw_data_downloaded"] == "false" for r in rows)
    assert all(r["access_status"] == "NETWORK_UNAVAILABLE_OR_SKIPPED" for r in rows)
    # urls in content must be repo-relative-safe (no absolute drive paths)
    for r in rows:
        assert ":\\" not in r["url"]
