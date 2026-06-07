import scripts.protocolo_c.revp_v2ap_common as common
from tests.test_revp_v2ap_common import install_all


def test_patch_registry_inventory(tmp_path, monkeypatch):
    datasets, protocol, docs = install_all(tmp_path, monkeypatch)
    rows = common.run_patch_registry_inventory_builder(common.parse_args([]))
    assert rows
    for r in rows:
        assert r["geometry_status"] in ("HAS_GEOMETRY_COLUMNS", "NO_GEOMETRY_COLUMNS_BLOCKER")
        assert ":\\" not in r["source_file"]
