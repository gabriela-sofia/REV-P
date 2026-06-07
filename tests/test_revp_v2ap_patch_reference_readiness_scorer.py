import scripts.protocolo_c.revp_v2ap_common as common
from tests.test_revp_v2ap_common import install_all
from tests.test_revp_v2ap_patch_event_link_readiness_builder import _prep


def test_scorer_blocks_everything(tmp_path, monkeypatch):
    datasets, protocol, docs = install_all(tmp_path, monkeypatch)
    _prep(common)
    common.run_patch_event_link_readiness_builder(common.parse_args([]))
    rows = common.run_patch_reference_readiness_scorer(common.parse_args([]))
    assert len(rows) == 9
    assert all(r["can_create_ground_truth"] == "false" for r in rows)
    assert all(r["can_create_label"] == "false" for r in rows)
    assert all(r["protocol_b_status"] == "BLOCKED" for r in rows)
    for r in rows:
        assert int(r["overall_patch_reference_score"]) == (
            int(r["geometry_score"]) + int(r["crosswalk_score"]) + int(r["patch_registry_score"]))
