import scripts.protocolo_c.revp_v2am_common as common
from tests.test_revp_v2am_common import install_all


def test_review_queue_all_pending_blocked(tmp_path, monkeypatch):
    data, docs, atlas, _ = install_all(tmp_path, monkeypatch)
    rows = common.run_review_queue_appendix_builder(common.parse_args([]))
    assert rows
    assert all(r["review_status"] == "PENDING_HUMAN_REVIEW" for r in rows)
    assert all(r["adjudication_status"] == "PENDING_ADJUDICATION" for r in rows)
    assert all(r["promotion_status"] == "PROMOTION_BLOCKED" for r in rows)
    common.assert_no_fake_review(rows)
    md = (atlas / "v2am_review_queue_appendix.md").read_text(encoding="utf-8")
    assert "pendente" in md.lower()
    assert "bloqueada" in md.lower()
    common.assert_safe_text(md)
