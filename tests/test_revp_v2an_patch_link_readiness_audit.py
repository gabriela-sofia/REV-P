import scripts.protocolo_c.revp_v2an_common as common
from tests.test_revp_v2an_common import install_all


def test_overlay_never_ready(tmp_path, monkeypatch):
    data, protocol, docs, _ = install_all(tmp_path, monkeypatch)
    common.run_spatial_anchor_extractor(common.parse_args([]))
    rows = common.run_patch_link_readiness_audit(common.parse_args([]))
    assert len(rows) == 9
    assert all(r["overlay_ready"] == "false" for r in rows)
    assert all(r["has_event_geometry_available"] == "false" for r in rows)
    assert all(r["manual_patch_review_required"] == "true" for r in rows)
