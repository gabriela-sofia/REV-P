import pytest

import scripts.protocolo_c.revp_v2am_common as common
from tests.test_revp_v2am_common import install_all, read_csv


def _build_docs(monkeypatch_funcs=None):
    pass


def test_final_audit_passes_on_safe_docs(tmp_path, monkeypatch):
    data, docs, atlas, integration = install_all(tmp_path, monkeypatch)
    common.run_evidence_atlas_registry_builder(common.parse_args([]))
    common.run_claims_guardrails_appendix_builder(common.parse_args([]))
    common.run_defense_question_bank_builder(common.parse_args([]))
    rows = common.run_final_claim_consistency_audit(common.parse_args([]))
    assert all(r["violation"] == "false" for r in rows)


def test_final_audit_fails_on_positive_forbidden(tmp_path, monkeypatch):
    data, docs, atlas, integration = install_all(tmp_path, monkeypatch)
    (atlas / "v2am_protocol_c_evidence_atlas.md").write_text(
        "# Atlas\n\nO sistema entrega ground truth validado e deteccao de enchente.\n",
        encoding="utf-8")
    with pytest.raises(ValueError):
        common.run_final_claim_consistency_audit(common.parse_args([]))


def test_final_audit_allows_negated_forbidden(tmp_path, monkeypatch):
    data, docs, atlas, integration = install_all(tmp_path, monkeypatch)
    (atlas / "v2am_protocol_c_evidence_atlas.md").write_text(
        "# Atlas\n\nNao pode dizer ground truth validado; nao ha deteccao de enchente.\n",
        encoding="utf-8")
    rows = common.run_final_claim_consistency_audit(common.parse_args([]))
    assert all(r["violation"] == "false" for r in rows)
