import scripts.protocolo_c.revp_v2am_common as common
from tests.test_revp_v2am_common import install_all


def test_catalog_has_required_items(tmp_path, monkeypatch):
    data, docs, atlas, _ = install_all(tmp_path, monkeypatch)
    rows = common.run_tables_figures_catalog_builder(common.parse_args([]))
    titles = " ".join(r["suggested_title"].lower() for r in rows)
    assert "candidatos" in titles
    assert "blockers" in titles
    assert "claims" in titles
    assert "revisao humana" in titles
    assert "limitacoes" in titles
    assert "dag" in titles
    assert "atlas" in titles
    assert all(r["manual_review_required"] == "true" for r in rows)
    # no accuracy/validation/training as a positive suggested item
    for r in rows:
        assert "acuracia" not in r["suggested_title"].lower()
    md = (atlas / "v2am_tables_and_figures_catalog.md").read_text(encoding="utf-8")
    common.assert_safe_text(md)
