import scripts.protocolo_c.revp_v2am_common as common
from tests.test_revp_v2am_common import install_all


def test_appendix_index_order(tmp_path, monkeypatch):
    data, docs, atlas, _ = install_all(tmp_path, monkeypatch)
    rows = common.run_appendix_index_builder(common.parse_args([]))
    titles = [r["title"] for r in rows]
    assert titles[0] == "Appendix index"
    assert titles[1] == "Evidence atlas"
    assert titles[2] == "Traceability DAG"
    assert titles[-1] == "Final claim consistency audit"
    md = (atlas / "v2am_appendix_index.md").read_text(encoding="utf-8")
    assert "ordem sugerida" in md.lower()
    for r in rows:
        assert r["appendix_file"].startswith("docs/tcc_exports/v2am_appendix_evidence_atlas/")
