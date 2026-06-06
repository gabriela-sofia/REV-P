import csv

import pytest

import scripts.protocolo_c.revp_v2al_common as common


def write_csv(path, cols, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def set_env(tmp_path, monkeypatch):
    data = tmp_path / "datasets" / "protocolo_c"
    docs = tmp_path / "docs" / "tcc_exports"
    integration = docs / "v2al_manuscript_integration"
    cfg = tmp_path / "configs" / "protocolo_c"
    for path in (data, docs, integration, cfg):
        path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(common, "DATASET_DIR", str(data))
    monkeypatch.setattr(common, "DOCS_DIR", str(docs))
    monkeypatch.setattr(common, "INTEGRATION_DIR", str(integration))
    monkeypatch.setattr(common, "CONFIG_DIR", str(cfg))
    monkeypatch.setattr(common, "REPO_ROOT", str(tmp_path))
    return data, docs, integration


V2AK_DRAFT_BODIES = {
    "protocolo_c_v2ak_metodologia_draft.md": [
        "# Protocolo C - metodologia draft seguro",
        "",
        "## Papel metodologico",
        "O Protocolo C foi tratado como camada de evidencia contextual e candidatos revisaveis, review-only.",
        "",
        "## Citacoes",
        "Manter marcador [CITAR_FONTE] ate verificacao manual.",
    ],
    "protocolo_c_v2ak_resultados_draft.md": [
        "# Protocolo C - resultados draft seguro",
        "",
        "## Estado quantitativo",
        "Foram preservados 172 candidatos revisaveis com blockers de promocao.",
    ],
    "protocolo_c_v2ak_discussao_draft.md": [
        "# Protocolo C - discussao draft seguro",
        "",
        "## Maturidade review-only",
        "O resultado interpretativo e a maturidade review-only sem promocao operacional.",
    ],
    "protocolo_c_v2ak_limitacoes_trabalhos_futuros_draft.md": [
        "# Protocolo C - limitacoes e trabalhos futuros draft seguro",
        "",
        "## Limitacoes",
        "Nao ha referencia operacional patch-level; revisao humana permanece pendente.",
    ],
    "protocolo_c_v2ak_orientador_briefing.md": [
        "# Protocolo C - briefing para orientador",
        "",
        "## Estado atual",
        "- 172 candidatos revisaveis preservados.",
    ],
}


def install_v2ak(data, docs):
    for name, body in V2AK_DRAFT_BODIES.items():
        front = ["---", "stage: v2ak", "operational_claims: false",
                 "ground_truth_created: false", "---", ""]
        (docs / name).write_text("\n".join(front + body) + "\n", encoding="utf-8")
    write_csv(
        data / "v2ak_writeup_traceability_matrix.csv",
        ["trace_id", "draft_file", "section_heading", "trace_status"],
        [{"trace_id": "TR1", "draft_file": "docs/tcc_exports/protocolo_c_v2ak_metodologia_draft.md",
          "section_heading": "Papel metodologico", "trace_status": "TRACE_OK"}],
    )
    write_csv(
        data / "v2ak_claim_usage_audit.csv",
        ["audit_id", "draft_file", "claim_text", "claim_status", "violation"],
        [{"audit_id": "A1", "draft_file": "d", "claim_text": "evidencia contextual",
          "claim_status": "allowed", "violation": "false"}],
    )
    write_csv(
        data / "v2ak_safe_language_glossary.csv",
        ["term_id", "term", "status"],
        [{"term_id": "T1", "term": "review-only", "status": "safe"},
         {"term_id": "T2", "term": "ground truth validado", "status": "prohibited"}],
    )


def install_v2aj(data):
    write_csv(
        data / "v2aj_tcc_protocol_c_claims_matrix.csv",
        ["claim_id", "section_target", "claim_type", "claim_allowed", "safe_wording",
         "unsafe_wording", "reason", "source_artifact", "required_disclaimer",
         "guardrail_category"],
        [
            {"claim_id": "CLM1", "section_target": "metodologia", "claim_type": "allowed",
             "claim_allowed": "true", "safe_wording": "evidencia contextual",
             "unsafe_wording": "deteccao de enchente", "reason": "safe",
             "source_artifact": "datasets/protocolo_c/v2aj.csv",
             "required_disclaimer": "review-only", "guardrail_category": "no_detection"},
            {"claim_id": "CLM2", "section_target": "results", "claim_type": "forbidden",
             "claim_allowed": "false", "safe_wording": "nao afirmar resultado",
             "unsafe_wording": "ground truth validado", "reason": "unsafe",
             "source_artifact": "datasets/protocolo_c/v2aj.csv",
             "required_disclaimer": "review-only", "guardrail_category": "no_gt"},
        ],
    )
    write_csv(
        data / "v2aj_results_tables_export_registry.csv",
        ["table_id", "suggested_title", "source_artifacts", "tcc_section", "safe_caption",
         "unsafe_caption", "allowed_interpretation", "forbidden_interpretation",
         "include_in_main_text", "include_in_appendix"],
        [
            {"table_id": "TAB_v2aj_000", "suggested_title": "Review-only candidate state",
             "source_artifacts": "datasets/protocolo_c/v2ah.csv", "tcc_section": "results",
             "safe_caption": "Candidates retained as review-only packages.",
             "unsafe_caption": "operational results table",
             "allowed_interpretation": "Describe review queue.",
             "forbidden_interpretation": "Do not infer accuracy.",
             "include_in_main_text": "true", "include_in_appendix": "true"},
        ],
    )


def install_all(tmp_path, monkeypatch):
    data, docs, integration = set_env(tmp_path, monkeypatch)
    install_v2ak(data, docs)
    install_v2aj(data)
    return data, docs, integration


# --- common tests ----------------------------------------------------------
def test_assert_v2ak_ready_fails_when_absent(tmp_path, monkeypatch):
    set_env(tmp_path, monkeypatch)
    with pytest.raises(FileNotFoundError):
        common.assert_v2ak_ready()


def test_assert_v2ak_ready_passes(tmp_path, monkeypatch):
    data, docs, _ = set_env(tmp_path, monkeypatch)
    install_v2ak(data, docs)
    assert common.assert_v2ak_ready()


def test_is_true_is_fail_closed():
    assert common.is_true("true")
    assert common.is_true("TRUE")
    assert common.is_true("  true  ")
    for bad in ("", "false", "1", "yes", "True!", None):
        assert not common.is_true(bad)


def test_assert_no_operational_claim_blocks_forbidden_true():
    with pytest.raises(ValueError):
        common.assert_no_operational_claim([{"ground_truth_created": "true"}])
    with pytest.raises(ValueError):
        common.assert_no_operational_claim([{"safe_to_autowrite": "true"}])
    common.assert_no_operational_claim([{"safe_to_autowrite": "false"}])


def test_assert_no_operational_claim_allows_unsafe_in_example_field():
    common.assert_no_operational_claim([{"unsafe_wording": "ground truth validado"}])
    with pytest.raises(ValueError):
        common.assert_no_operational_claim([{"body": "ground truth validado"}])


def test_assert_no_operational_claim_blocks_forbidden_kv_and_paths():
    with pytest.raises(ValueError):
        common.assert_no_operational_claim([{"note": "ground_truth=true"}])
    with pytest.raises(ValueError):
        common.assert_no_operational_claim([{"note": "C:\\Users\\x\\file.tex"}])
    with pytest.raises(ValueError):
        common.assert_no_operational_claim([{"note": "local" + "_only path"}])


def test_assert_safe_manuscript_language():
    common.assert_safe_manuscript_language("Camada review-only sem promocao.")
    common.assert_safe_manuscript_language(
        "Nao pode dizer ground truth validado nesta etapa.")
    with pytest.raises(ValueError):
        common.assert_safe_manuscript_language(
            "O sistema entrega ground truth validado e deteccao de enchente.")


def test_assert_no_auto_manuscript_overwrite():
    common.assert_no_auto_manuscript_overwrite(
        "docs/tcc_exports/v2al_manuscript_integration/v2al_metodologia_section_candidate.md")
    with pytest.raises(ValueError):
        common.assert_no_auto_manuscript_overwrite("docs/manuscrito_principal.tex")
    with pytest.raises(ValueError):
        common.assert_no_auto_manuscript_overwrite(
            "docs/v2al_x.csv", manuscript_paths=["docs/v2al_x.csv"])
    # absolute output dir with a v2al_ basename is allowed (controlled output location)
    common.assert_no_auto_manuscript_overwrite("C:\\tmp\\out\\v2al_x.csv")


def test_convert_markdown_to_latex_safe():
    latex = common.convert_markdown_to_latex_safe(
        "# Titulo\n\n## Sub\nTexto com _underscore_ e review-only e DINOv2.")
    assert "\\subsection{Titulo}" in latex
    assert "\\cite{" not in latex
    assert "review-only" in latex
    assert "DINOv2" in latex
    assert "\\_underscore\\_" in latex


def test_build_section_anchor_and_trace_row():
    anchor = common.build_section_anchor("Limitacoes e Trabalhos Futuros")
    assert anchor == "% v2al-anchor: limitacoes-e-trabalhos-futuros"
    row = common.build_integration_trace_row(
        0, "docs/tcc_exports/x.md", "Metodologia", "subsecao", "resumo", "nao promover", "risco")
    assert row["insertion_mode"] == "manual_review_required"
    assert row["required_review"] == "true"
    assert row["insertion_id"] == "INS_v2al_000"


def test_write_csv_rejects_non_v2al_name(tmp_path, monkeypatch):
    set_env(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        common.write_csv(str(tmp_path / "manuscrito.csv"), ["a"], [{"a": "1"}])
