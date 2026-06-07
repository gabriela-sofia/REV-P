import scripts.protocolo_c.revp_v2ap_common as common
from tests.test_revp_v2ap_common import install_all
from tests.test_revp_v2ap_patch_event_link_readiness_builder import _prep


def test_boundary_update_patch_truth_blocked(tmp_path, monkeypatch):
    datasets, protocol, docs = install_all(tmp_path, monkeypatch)
    _prep(common)
    common.run_patch_event_link_readiness_builder(common.parse_args([]))
    common.run_patch_reference_readiness_scorer(common.parse_args([]))
    rows = common.run_patch_truth_boundary_update_builder(common.parse_args([]))
    assert len(rows) == 9
    assert all(r["patch_truth_allowed"] == "false" for r in rows)
    assert all("ground truth" in r["why_still_blocked"].lower() for r in rows)
    assert all("ground_truth" in r["forbidden_use"] for r in rows)
