import scripts.protocolo_c.revp_v2ap_common as common
from tests.test_revp_v2ap_common import install_all
from tests.test_revp_v2ap_patch_event_link_readiness_builder import _prep


def test_crosswalk_collection_packet(tmp_path, monkeypatch):
    datasets, protocol, docs = install_all(tmp_path, monkeypatch)
    _prep(common)
    common.run_patch_event_link_readiness_builder(common.parse_args([]))
    rows = common.run_crosswalk_collection_packet_builder(common.parse_args([]))
    assert rows
    assert all("nao inferir" in r["do_not_infer"].lower() for r in rows)
    for r in rows:
        assert ":\\" not in r["recommended_manifest_to_check"]
