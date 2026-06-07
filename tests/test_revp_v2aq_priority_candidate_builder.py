import scripts.protocolo_c.revp_v2aq_common as common
from tests.test_revp_v2aq_common import install_all


def test_priority_includes_all_and_marks_high(tmp_path, monkeypatch):
    datasets, protocol, docs, geojson = install_all(tmp_path, monkeypatch)
    rows = common.run_priority_candidate_builder(common.parse_args([]))
    assert len(rows) == 9
    assert all(r["included_in_v2aq"] == "true" for r in rows)
    by = {r["candidate_id"]: r for r in rows}
    assert "C4" in by["PET_2022_02_15"]["priority_reason"]
    assert by["PET_2022_02_15"]["priority_rank"] == "1"
