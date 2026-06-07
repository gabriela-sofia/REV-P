import scripts.protocolo_c.revp_v2ap_common as common
from tests.test_revp_v2ap_common import install_all


def test_geometry_collection_packet(tmp_path, monkeypatch):
    datasets, protocol, docs = install_all(tmp_path, monkeypatch)
    common.run_patch_registry_inventory_builder(common.parse_args([]))
    common.run_spatial_geometry_readiness_builder(common.parse_args([]))
    rows = common.run_geometry_collection_packet_builder(common.parse_args([]))
    assert rows
    assert all("nao inventar" in r["do_not_infer"].lower() for r in rows)
    assert all(r["collection_priority"] in ("HIGH", "MEDIUM", "LOW") for r in rows)
