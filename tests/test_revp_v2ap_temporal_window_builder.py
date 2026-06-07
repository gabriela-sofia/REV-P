import scripts.protocolo_c.revp_v2ap_common as common
from tests.test_revp_v2ap_common import install_all


def test_temporal_windows(tmp_path, monkeypatch):
    datasets, protocol, docs = install_all(tmp_path, monkeypatch)
    rows = common.run_temporal_window_builder(common.parse_args([]))
    assert len(rows) == 9
    by = {r["candidate_id"]: r for r in rows}
    rec = by["REC_2022_05_24_30"]
    assert rec["acceptable_sentinel_window_start"] == "2022-05-16"
    assert rec["acceptable_sentinel_window_end"] == "2022-06-07"
    assert "policy" in rec["window_policy"].lower()
