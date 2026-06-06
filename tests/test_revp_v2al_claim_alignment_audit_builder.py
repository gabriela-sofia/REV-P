import pytest

import scripts.protocolo_c.revp_v2al_common as common
from tests.test_revp_v2al_common import install_all, read_csv


def test_claim_alignment_passes_on_safe_bundles(tmp_path, monkeypatch):
    data, docs, integration = install_all(tmp_path, monkeypatch)
    common.run_markdown_section_bundle_builder(common.parse_args([]))
    common.run_latex_section_bundle_builder(common.parse_args([]))
    rows = common.run_claim_alignment_audit_builder(common.parse_args([]))
    assert all(r["violation"] == "false" for r in rows)
    written = read_csv(data / "v2al_claim_alignment_audit.csv")
    assert written


def test_claim_alignment_fails_on_overclaim_bundle(tmp_path, monkeypatch):
    data, docs, integration = install_all(tmp_path, monkeypatch)
    common.run_markdown_section_bundle_builder(common.parse_args([]))
    # inject an overclaiming bundle as a positive assertion
    (integration / "v2al_discussao_section_candidate.md").write_text(
        "# Discussao\n\nO sistema entrega ground truth validado.\n", encoding="utf-8")
    with pytest.raises(ValueError):
        common.run_claim_alignment_audit_builder(common.parse_args([]))


def test_forbidden_claim_as_negative_example_is_allowed(tmp_path, monkeypatch):
    data, docs, integration = install_all(tmp_path, monkeypatch)
    common.run_markdown_section_bundle_builder(common.parse_args([]))
    (integration / "v2al_limitacoes_trabalhos_futuros_section_candidate.md").write_text(
        "# Limitacoes\n\nNao pode dizer ground truth validado nesta etapa.\n",
        encoding="utf-8")
    rows = common.run_claim_alignment_audit_builder(common.parse_args([]))
    assert all(r["violation"] == "false" for r in rows)
