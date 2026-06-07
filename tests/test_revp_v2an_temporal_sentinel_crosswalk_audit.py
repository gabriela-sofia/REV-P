import scripts.protocolo_c.revp_v2an_common as common
from tests.test_revp_v2an_common import install_all


def test_no_sentinel_date_inferred(tmp_path, monkeypatch):
    data, protocol, docs, _ = install_all(tmp_path, monkeypatch)
    rows = common.run_temporal_sentinel_crosswalk_audit(common.parse_args([]))
    assert len(rows) == 9
    assert all(r["explicit_crosswalk_found"] == "false" for r in rows)
    assert all(r["sentinel_asset_date_found"] == "" for r in rows)
    assert all(r["temporal_gate_status"].startswith("BLOCKED") for r in rows)
