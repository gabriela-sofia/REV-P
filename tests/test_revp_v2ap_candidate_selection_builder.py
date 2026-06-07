import scripts.protocolo_c.revp_v2ap_common as common
from tests.test_revp_v2ap_common import install_all


def test_selection_priorities(tmp_path, monkeypatch):
    datasets, protocol, docs = install_all(tmp_path, monkeypatch)
    rows = common.run_candidate_selection_builder(common.parse_args([]))
    assert len(rows) == 9
    assert all(r["included_in_v2ap"] == "true" for r in rows)
    by = {r["candidate_id"]: r for r in rows}
    assert by["PET_2022_02_15"]["selection_reason"] == "HIGH_PRIORITY_GEOMETRY_AND_CROSSWALK"
    assert by["PET_2024_03_21_28"]["selection_reason"] == "LOW_PRIORITY_GEOMETRY_COLLECTION"
    assert all("ground_truth" in r["forbidden_use"] for r in rows)
