import scripts.protocolo_c.revp_v2an_common as common
from tests.test_revp_v2an_common import install_all


def test_anchors_have_no_invented_coordinates(tmp_path, monkeypatch):
    data, protocol, docs, _ = install_all(tmp_path, monkeypatch)
    rows = common.run_spatial_anchor_extractor(common.parse_args([]))
    assert rows
    assert all(r["coordinate_available"] == "false" for r in rows)
    assert all(r["geometry_available"] == "false" for r in rows)
    assert all(r["manual_geometry_review_required"] == "true" for r in rows)
    assert all(r["anchor_type"] for r in rows)
