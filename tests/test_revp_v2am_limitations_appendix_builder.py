import scripts.protocolo_c.revp_v2am_common as common
from tests.test_revp_v2am_common import install_all


def test_limitations_framed_as_control(tmp_path, monkeypatch):
    data, docs, atlas, _ = install_all(tmp_path, monkeypatch)
    rows = common.run_limitations_appendix_builder(common.parse_args([]))
    assert rows
    for r in rows:
        assert r["what_it_does_not_imply"]
        assert r["mitigation"]
        assert r["future_work"]
    md = (atlas / "v2am_limitations_appendix.md").read_text(encoding="utf-8")
    assert "controle metodologico" in md.lower()
    common.assert_safe_text(md)
