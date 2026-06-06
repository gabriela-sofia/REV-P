import csv

import pytest

import scripts.protocolo_c.revp_v2am_common as common


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
    atlas = docs / "v2am_appendix_evidence_atlas"
    integration = docs / "v2al_manuscript_integration"
    cfg = tmp_path / "configs" / "protocolo_c"
    for path in (data, docs, atlas, integration, cfg):
        path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(common, "DATASET_DIR", str(data))
    monkeypatch.setattr(common, "DOCS_DIR", str(docs))
    monkeypatch.setattr(common, "ATLAS_DIR", str(atlas))
    monkeypatch.setattr(common, "V2AL_INTEGRATION_DIR", str(integration))
    monkeypatch.setattr(common, "CONFIG_DIR", str(cfg))
    return data, docs, atlas, integration


def install_stack(data, docs, integration):
    # v2ah
    write_csv(
        data / "v2ah_candidate_reference_review_queue.csv",
        ["review_queue_id", "package_id", "event_id", "region", "candidate_status"],
        [{"review_queue_id": f"RQ_{i}", "package_id": f"EPC_{i}", "event_id": "E1",
          "region": ("PET" if i % 2 == 0 else "REC"), "candidate_status": "BLOCKED_REFERENCE_CANDIDATE"}
         for i in range(6)],
    )
    write_csv(
        data / "v2ah_ground_truth_search_stop_gate.csv",
        ["stop_gate_id", "ground_truth_search_status", "guardrail_status"],
        [{"stop_gate_id": "STOP_0",
          "ground_truth_search_status": "GROUND_TRUTH_SEARCH_STOPPED_UNTIL_NEW_QUALIFIED_SOURCE",
          "guardrail_status": "FAIL_CLOSED_REVIEW_ONLY"}],
    )
    # v2ai
    write_csv(
        data / "v2ai_review_assignment_registry.csv",
        ["assignment_id", "package_id", "region", "assignment_status"],
        [{"assignment_id": f"ASN_{i}", "package_id": f"EPC_{i // 2}",
          "region": ("PET" if i % 2 == 0 else "REC"),
          "assignment_status": "ASSIGNED_SLOT_PENDING_HUMAN_REVIEW"} for i in range(12)],
    )
    write_csv(
        data / "v2ai_adjudication_queue.csv",
        ["adjudication_id", "package_id", "adjudication_status", "can_promote_after_adjudication"],
        [{"adjudication_id": f"ADJ_{i}", "package_id": f"EPC_{i}",
          "adjudication_status": "WAITING_FOR_HUMAN_REVIEW",
          "can_promote_after_adjudication": "false"} for i in range(6)],
    )
    write_csv(
        data / "v2ai_safe_promotion_blockers.csv",
        ["package_id", "promotion_status", "promotion_allowed", "promotion_reason"],
        [{"package_id": f"EPC_{i}", "promotion_status": "PROMOTION_BLOCKED_PENDING_REAL_REVIEW",
          "promotion_allowed": "false", "promotion_reason": "human review pending"} for i in range(6)],
    )
    # v2aj
    write_csv(
        data / "v2aj_tcc_protocol_c_claims_matrix.csv",
        ["claim_id", "section_target", "claim_type", "claim_allowed", "safe_wording",
         "unsafe_wording", "reason", "source_artifact", "required_disclaimer", "guardrail_category"],
        [
            {"claim_id": "CLM1", "section_target": "metodologia", "claim_type": "allowed",
             "claim_allowed": "true", "safe_wording": "camada revisavel de candidatos",
             "unsafe_wording": "ground truth validado", "reason": "safe",
             "source_artifact": "datasets/protocolo_c/v2ah.csv",
             "required_disclaimer": "review-only", "guardrail_category": "no_gt"},
            {"claim_id": "CLM2", "section_target": "results", "claim_type": "forbidden",
             "claim_allowed": "false", "safe_wording": "nao afirmar deteccao",
             "unsafe_wording": "deteccao de enchente", "reason": "unsafe",
             "source_artifact": "datasets/protocolo_c/v2ah.csv",
             "required_disclaimer": "review-only", "guardrail_category": "no_detection"},
        ],
    )
    write_csv(
        data / "v2aj_tcc_evidence_summary_table.csv",
        ["summary_id", "metric_name", "metric_value"],
        [{"summary_id": "SUM_0", "metric_name": "total_candidates", "metric_value": "6"}],
    )
    write_csv(
        data / "v2aj_methodological_limitations_export.csv",
        ["limitation_id", "limitation_name", "what_it_means", "what_it_does_not_mean",
         "safe_tcc_wording", "unsafe_tcc_wording", "mitigation_already_done", "future_work"],
        [{"limitation_id": "LIM_0", "limitation_name": "no_operational_patch_ground_truth",
          "what_it_means": "No operational patch-level reference exists.",
          "what_it_does_not_mean": "Nao implica que o pipeline falhou.",
          "safe_tcc_wording": "Limitacao controlada: sem referencia operacional patch-level.",
          "unsafe_tcc_wording": "ground truth validado",
          "mitigation_already_done": "Stop gate e blockers documentados.",
          "future_work": "Usar nova fonte qualificada e revisao."}],
    )
    write_csv(
        data / "v2aj_results_tables_export_registry.csv",
        ["table_id", "suggested_title", "tcc_section", "safe_caption", "unsafe_caption",
         "allowed_interpretation", "forbidden_interpretation"],
        [{"table_id": "TAB_v2aj_000", "suggested_title": "Review-only state",
          "tcc_section": "results", "safe_caption": "Candidates review-only.",
          "unsafe_caption": "operational results table",
          "allowed_interpretation": "Describe queue.", "forbidden_interpretation": "No accuracy."}],
    )
    # v2ak
    write_csv(
        data / "v2ak_safe_language_glossary.csv",
        ["term_id", "term", "status"],
        [{"term_id": "T1", "term": "review-only", "status": "safe"},
         {"term_id": "T2", "term": "ground truth validado", "status": "prohibited"}],
    )
    write_csv(
        data / "v2ak_claim_usage_audit.csv",
        ["audit_id", "draft_file", "claim_text", "claim_status", "violation"],
        [{"audit_id": "A1", "draft_file": "d", "claim_text": "evidencia contextual",
          "claim_status": "allowed", "violation": "false"}],
    )
    (docs / "protocolo_c_v2ak_metodologia_draft.md").write_text(
        "---\nstage: v2ak\n---\n\n# Metodologia\n\nCamada review-only de candidatos.\n",
        encoding="utf-8")
    (docs / "protocolo_c_v2ak_resultados_draft.md").write_text(
        "# Resultados\n\nCandidatos revisaveis com blockers; sem promocao.\n", encoding="utf-8")
    # v2al
    write_csv(
        data / "v2al_table_caption_export.csv",
        ["caption_id", "source_table_id", "target_section", "safe_title", "safe_caption",
         "forbidden_caption", "allowed_interpretation", "manual_review_required"],
        [{"caption_id": "CAP_0", "source_table_id": "TAB_v2aj_000", "target_section": "results",
          "safe_title": "Review-only state", "safe_caption": "Candidates review-only, sem validacao para uso operacional.",
          "forbidden_caption": "operational results table", "allowed_interpretation": "Describe queue.",
          "manual_review_required": "true"}],
    )
    write_csv(
        data / "v2al_section_insertion_matrix.csv",
        ["insertion_id", "v2ak_source_draft", "target_tcc_section", "insertion_mode"],
        [{"insertion_id": "INS_0", "v2ak_source_draft": "docs/tcc_exports/protocolo_c_v2ak_metodologia_draft.md",
          "target_tcc_section": "Metodologia", "insertion_mode": "manual_review_required"}],
    )
    (integration / "v2al_metodologia_section_candidate.md").write_text(
        "# Metodologia\n\nCamada review-only.\n", encoding="utf-8")


def install_all(tmp_path, monkeypatch):
    data, docs, atlas, integration = set_env(tmp_path, monkeypatch)
    install_stack(data, docs, integration)
    return data, docs, atlas, integration


# --- common tests ----------------------------------------------------------
def test_stage_ready_fails_when_absent(tmp_path, monkeypatch):
    set_env(tmp_path, monkeypatch)
    with pytest.raises(FileNotFoundError):
        common.assert_stage_artifacts_ready()


def test_stage_ready_passes(tmp_path, monkeypatch):
    install_all(tmp_path, monkeypatch)
    assert common.assert_stage_artifacts_ready()


def test_is_true_fail_closed():
    assert common.is_true("true") and common.is_true("TRUE")
    for bad in ("", "false", "1", "yes", None):
        assert not common.is_true(bad)


def test_repo_relative_path_rejects_absolute():
    assert common.repo_relative_path("datasets/x.csv") == "datasets/x.csv"
    assert common.repo_relative_path("a\\b\\c.csv") == "a/b/c.csv"
    with pytest.raises(ValueError):
        common.repo_relative_path("C:\\Users\\x\\y.csv")


def test_assert_no_absolute_paths_in_content():
    common.assert_no_absolute_paths_in_content([{"path": "datasets/x.csv"}])
    with pytest.raises(ValueError):
        common.assert_no_absolute_paths_in_content([{"path": "C:\\Users\\x.csv"}])


def test_assert_no_local_only():
    common.assert_no_local_only([{"note": "safe"}])
    with pytest.raises(ValueError):
        common.assert_no_local_only([{"note": "local" + "_only here"}])


def test_assert_no_operational_claim():
    common.assert_no_operational_claim([{"unsafe_wording": "ground truth validado"}])
    with pytest.raises(ValueError):
        common.assert_no_operational_claim([{"body": "ground truth validado"}])
    with pytest.raises(ValueError):
        common.assert_no_operational_claim([{"promotion_created": "true"}])
    with pytest.raises(ValueError):
        common.assert_no_operational_claim([{"note": "promotion_allowed=true"}])


def test_assert_no_fake_review():
    common.assert_no_fake_review([{"review_status": "PENDING_HUMAN_REVIEW"}])
    with pytest.raises(ValueError):
        common.assert_no_fake_review([{"human_review_completed": "true"}])


def test_assert_no_manuscript_overwrite():
    common.assert_no_manuscript_overwrite("datasets/protocolo_c/v2am_x.csv")
    with pytest.raises(ValueError):
        common.assert_no_manuscript_overwrite("docs/tcc_exports/manuscrito.tex")


def test_build_dag_edge_promotion_false():
    edge = common.build_dag_edge(0, "A", "B", "rel")
    assert edge["promotion_created"] == "false"
    assert edge["guardrail_preserved"] == "true"


def test_safe_slug_and_markdown_table():
    assert common.safe_slug("Evidence Atlas v2") == "evidence-atlas-v2"
    table = common.write_markdown_table(["a", "b"], [("1", "2")])
    assert table[0] == "| a | b |"
    assert table[1] == "| --- | --- |"
    assert table[2] == "| 1 | 2 |"


def test_status_enum_not_substring_flagged():
    # "promotion_allowed=false" must NOT be a forbidden status violation
    counts = common.scan_text_violations("o campo promotion_allowed=false e seguro")
    assert counts["forbidden_status"] == 0
    assert counts["forbidden_kv"] == 0
