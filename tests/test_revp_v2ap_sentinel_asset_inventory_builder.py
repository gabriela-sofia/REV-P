import scripts.protocolo_c.revp_v2ap_common as common
from tests.test_revp_v2ap_common import install_all


def test_asset_inventory_explicit_dates_only(tmp_path, monkeypatch):
    datasets, protocol, docs = install_all(tmp_path, monkeypatch)
    rows = common.run_sentinel_asset_inventory_builder(common.parse_args([]))
    assert rows
    methods = {r["date_detection_method"] for r in rows}
    assert "visual_similarity" not in methods
    assert "dino_similarity" not in methods
    dino = [r for r in rows if "dino" in r["path"].lower()]
    assert all(r["safe_to_use_as_crosswalk_evidence"] == "false" for r in dino)
    for r in rows:
        assert ":\\" not in r["path"]
