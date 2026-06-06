import scripts.protocolo_c.revp_v2al_common as common
from tests.test_revp_v2al_common import install_all, read_csv


def test_patch_plan_is_manual_only(tmp_path, monkeypatch):
    data, docs, integration = install_all(tmp_path, monkeypatch)
    (tmp_path / "tcc").mkdir()
    (tmp_path / "tcc" / "main.tex").write_text(
        "\\section{Metodologia}\n\\section{Resultados}\n", encoding="utf-8")
    common.run_manuscript_candidate_scanner(common.parse_args([]))
    rows = common.run_manuscript_patch_plan_builder(common.parse_args([]))
    assert rows
    assert all("manual" in r["manual_action"].lower() for r in rows)
    assert all(r["insert_before_or_after"] in ("before", "after") for r in rows)
    assert all(r["anchor_to_find"].startswith("% v2al-anchor:") for r in rows)
    md = (integration / "v2al_manual_patch_plan.md").read_text(encoding="utf-8")
    assert "Nenhuma escrita automatica" in md


def test_patch_plan_handles_no_candidates(tmp_path, monkeypatch):
    data, docs, integration = install_all(tmp_path, monkeypatch)
    common.run_manuscript_candidate_scanner(common.parse_args([]))
    rows = common.run_manuscript_patch_plan_builder(common.parse_args([]))
    assert rows
    # still produced, still manual, never autowrite
    assert all("manual" in r["manual_action"].lower() for r in rows)
