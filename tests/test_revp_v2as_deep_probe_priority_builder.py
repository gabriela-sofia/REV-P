import scripts.protocolo_c.revp_v2as_common as common
from tests.test_revp_v2as_common import install_all


def test_priority_high_for_can_digitize(tmp_path, monkeypatch):
    install_all(tmp_path, monkeypatch)
    rows = common.run_deep_probe_priority_builder(common.parse_args([]))
    by = {r["candidate_id"]: r for r in rows}
    assert by["PET_2022_02_15"]["deep_probe_priority"] == "DEEP_PROBE_HIGH"
    assert "can_digitize_now" in by["PET_2022_02_15"]["deep_probe_reason"]
    assert by["PET_2024_03_21_28"]["deep_probe_priority"] == "DEEP_PROBE_LOW"
    assert by["REC_2022_05_24_30"]["deep_probe_priority"] == "DEEP_PROBE_MEDIUM"
