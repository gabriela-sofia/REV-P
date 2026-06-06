import scripts.protocolo_c.revp_v2al_common as common
from tests.test_revp_v2al_common import install_all


def test_latex_bundles_headers_and_subsections(tmp_path, monkeypatch):
    data, docs, integration = install_all(tmp_path, monkeypatch)
    written = common.run_latex_section_bundle_builder(common.parse_args([]))
    assert len(written) == 4
    text = (integration / "v2al_metodologia_section_candidate.tex").read_text(encoding="utf-8")
    assert text.startswith("% v2al candidate section -- manual review required")
    assert "% no operational ground truth, no labels, no prediction" in text
    assert "\\subsection{" in text
    assert "\\cite{" not in text
    assert "review-only" in text
    # underscore from [CITAR_FONTE] must be escaped, command not destroyed
    assert "[CITAR\\_FONTE]" in text


def test_latex_bundles_are_safe(tmp_path, monkeypatch):
    data, docs, integration = install_all(tmp_path, monkeypatch)
    common.run_latex_section_bundle_builder(common.parse_args([]))
    for name in ("v2al_metodologia_section_candidate.tex",
                 "v2al_resultados_section_candidate.tex",
                 "v2al_discussao_section_candidate.tex",
                 "v2al_limitacoes_trabalhos_futuros_section_candidate.tex"):
        common.assert_safe_manuscript_language(
            (integration / name).read_text(encoding="utf-8"))
