import scripts.protocolo_c.revp_v2al_common as common
from tests.test_revp_v2al_common import install_all


def test_markdown_bundles_have_front_matter(tmp_path, monkeypatch):
    data, docs, integration = install_all(tmp_path, monkeypatch)
    written = common.run_markdown_section_bundle_builder(common.parse_args([]))
    assert len(written) == 4
    text = (integration / "v2al_metodologia_section_candidate.md").read_text(encoding="utf-8")
    assert "stage: v2al" in text
    assert "source_stage: v2ak" in text
    assert "integration_status: manual_review_required" in text
    assert "operational_claims: false" in text
    assert "ground_truth_created: false" in text
    assert "auto_insert: false" in text
    assert "[CITAR_FONTE]" in text
    assert "review-only" in text


def test_markdown_bundles_are_safe(tmp_path, monkeypatch):
    data, docs, integration = install_all(tmp_path, monkeypatch)
    common.run_markdown_section_bundle_builder(common.parse_args([]))
    for name in ("v2al_metodologia_section_candidate.md",
                 "v2al_resultados_section_candidate.md",
                 "v2al_discussao_section_candidate.md",
                 "v2al_limitacoes_trabalhos_futuros_section_candidate.md"):
        common.assert_safe_manuscript_language(
            (integration / name).read_text(encoding="utf-8"))
