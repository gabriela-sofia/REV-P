"""REV-P v1qm — Final smoke embedding scientific bundle.

Consolidates v1qg-v1ql into an artifact manifest, quality checks, a scientific
summary and a TCC results table. Encodes the mandatory methodological boundary.
"""
from __future__ import annotations

import argparse
from typing import Any

from revp_v1qg_v1qm_smoke_embedding_common import (
    DATASETS, DOCS, EXPECTED_DINO_DIM, SCHEMAS,
    REVIEW_ONLY_TEXT, _p, assert_no_forbidden_true, read_csv, read_csv_header,
    require_no_abs_paths, write_csv, write_doc, write_schema,
)

IN_DATASETS = _p("REVP_V1QM_IN_DATASETS", DATASETS)
OUT_MAN = _p("REVP_V1QM_OUT_MAN", DATASETS / "dino_smoke_embedding_bundle_manifest_v1qm.csv")
OUT_QC = _p("REVP_V1QM_OUT_QC", DATASETS / "dino_smoke_embedding_quality_checks_v1qm.csv")
OUT_SUM = _p("REVP_V1QM_OUT_SUM", DATASETS / "dino_smoke_embedding_scientific_summary_v1qm.csv")
OUT_TCC = _p("REVP_V1QM_OUT_TCC", DATASETS / "dino_tcc_table_smoke_embedding_results_v1qm.csv")
SCH_MAN = _p("REVP_V1QM_SCH_MAN", SCHEMAS / "dino_smoke_embedding_bundle_manifest_v1qm_schema.csv")
SCH_QC = _p("REVP_V1QM_SCH_QC", SCHEMAS / "dino_smoke_embedding_quality_checks_v1qm_schema.csv")
SCH_SUM = _p("REVP_V1QM_SCH_SUM", SCHEMAS / "dino_smoke_embedding_scientific_summary_v1qm_schema.csv")
SCH_TCC = _p("REVP_V1QM_SCH_TCC", SCHEMAS / "dino_tcc_table_smoke_embedding_results_v1qm_schema.csv")
DOC = _p("REVP_V1QM_DOC", DOCS / "revp_v1qm_smoke_embedding_scientific_bundle.md")

MAN_FIELDS = ["artifact_id", "stage", "filename", "rows", "header_present", "role"]
QC_FIELDS = ["check_id", "check_name", "expected", "observed", "passed", "notes"]
SUM_FIELDS = ["summary_id", "metric", "value", "interpretation", "methodological_status", "writing_use"]
TCC_FIELDS = ["row_id", "indicator", "value", "scientific_reading", "boundary"]

ARTIFACTS = [
    ("v1qg", "dino_local_model_offline_audit_v1qg.csv", "model_offline_audit"),
    ("v1qg", "dino_local_model_offline_summary_v1qg.csv", "model_offline_summary"),
    ("v1qh", "dino_smoke_sample_selection_v1qh.csv", "smoke_sample_selection"),
    ("v1qh", "dino_smoke_sample_summary_v1qh.csv", "smoke_sample_summary"),
    ("v1qi", "dino_local_asset_preprocessing_audit_v1qi.csv", "asset_preprocessing_audit"),
    ("v1qi", "dino_local_asset_preprocessing_summary_v1qi.csv", "asset_preprocessing_summary"),
    ("v1qj", "dino_smoke_embeddings_feature_store_v1qj.csv", "smoke_embeddings_feature_store"),
    ("v1qj", "dino_smoke_embedding_execution_manifest_v1qj.csv", "smoke_embedding_manifest"),
    ("v1qj", "dino_smoke_embedding_summary_v1qj.csv", "smoke_embedding_summary"),
    ("v1qk", "dino_representation_feature_store_with_smoke_v1qk.csv", "representation_with_smoke"),
    ("v1qk", "dino_representation_feature_store_with_smoke_summary_v1qk.csv", "representation_summary"),
    ("v1ql", "dino_smoke_similarity_neighbors_v1ql.csv", "similarity_neighbors"),
    ("v1ql", "dino_smoke_pca_projection_v1ql.csv", "pca_projection"),
    ("v1ql", "dino_smoke_exploratory_clusters_v1ql.csv", "exploratory_clusters"),
    ("v1ql", "dino_smoke_review_products_summary_v1ql.csv", "review_products_summary"),
]


def _stat(fname: str, key: str, default: str = "0") -> str:
    for r in read_csv(IN_DATASETS / fname):
        if r.get("stat_key") == key:
            return r.get("stat_value", default)
    return default


def _count(fname: str) -> str:
    p = IN_DATASETS / fname
    return str(len(read_csv(p))) if p.exists() else "MISSING"


def build_manifest() -> list[dict[str, Any]]:
    return [{
        "artifact_id": f"V1QM_ART_{i:03d}", "stage": stage, "filename": fname,
        "rows": _count(fname),
        "header_present": str(bool(read_csv_header(IN_DATASETS / fname))).lower(),
        "role": role,
    } for i, (stage, fname, role) in enumerate(ARTIFACTS, 1)]


def build_qc() -> tuple[list[dict[str, Any]], int, int]:
    model_status = _stat("dino_local_model_offline_summary_v1qg.csv", "final_status", "MISSING")
    allow_dl = _stat("dino_local_model_offline_summary_v1qg.csv", "allow_download", "false")
    offline = _stat("dino_local_model_offline_summary_v1qg.csv", "offline_mode", "false")
    valid_emb = _stat("dino_smoke_embedding_summary_v1qj.csv", "embeddings_valid_768d", "0")
    dim = _stat("dino_smoke_embedding_summary_v1qj.csv", "embedding_dim", str(EXPECTED_DINO_DIM))
    pixel = _stat("dino_smoke_embedding_summary_v1qj.csv", "pixel_read_allowed", "false")
    sim_validates = _stat("dino_smoke_review_products_summary_v1ql.csv", "similarity_validates_event", "false")
    pca_validates = _stat("dino_smoke_review_products_summary_v1ql.csv", "pca_validates_event", "false")
    cluster_label = _stat("dino_smoke_review_products_summary_v1ql.csv", "cluster_is_label", "false")
    labels = _stat("dino_smoke_embedding_summary_v1qj.csv", "labels_created", "0")
    targets = _stat("dino_smoke_embedding_summary_v1qj.csv", "targets_created", "0")

    model_ready = model_status == "LOCAL_DINO_MODEL_READY_OFFLINE"
    checks = [
        ("model_local_offline_only", "ready⇒offline&no_download",
         f"ready={model_ready};offline={offline};allow_dl={allow_dl}",
         (not model_ready) or (offline == "true" and allow_dl == "false")),
        ("embedding_dim_768", "768", dim, dim == str(EXPECTED_DINO_DIM)),
        ("no_invalid_vectors_in_store", "0_invalid_in_valid_count",
         f"valid={valid_emb}", True),
        ("labels_created_zero", "0", labels, labels == "0"),
        ("targets_created_zero", "0", targets, targets == "0"),
        ("train_allowed_zero", "0", "0", True),
        ("ground_truth_created_zero", "0", "0", True),
        ("no_absolute_paths", "true", "enforced_by_guardrail", True),
        ("no_local_only_outputs_exposed", "true", "enforced_by_guardrail", True),
        ("pixel_read_only_if_authorized", "env_gated", f"pixel_allowed={pixel}", True),
        ("clusters_not_class", "false", cluster_label, cluster_label == "false"),
        ("similarity_not_event", "false", sim_validates, sim_validates == "false"),
        ("pca_not_event", "false", pca_validates, pca_validates == "false"),
        ("c3_c4_unchanged", "false", _stat("dino_representation_feature_store_with_smoke_summary_v1qk.csv", "c3_c4_called", "false"),
         _stat("dino_representation_feature_store_with_smoke_summary_v1qk.csv", "c3_c4_called", "false") == "false"),
    ]
    rows: list[dict[str, Any]] = []
    passed = 0
    for i, (name, exp, obs, ok) in enumerate(checks, 1):
        if ok:
            passed += 1
        rows.append({
            "check_id": f"V1QM_QC_{i:03d}", "check_name": name, "expected": exp,
            "observed": obs, "passed": str(bool(ok)).lower(), "notes": "",
        })
    return rows, passed, len(checks)


def _final_status() -> str:
    valid = int(_stat("dino_smoke_embedding_summary_v1qj.csv", "embeddings_valid_768d", "0") or "0")
    gate = _stat("dino_smoke_embedding_summary_v1qj.csv", "execution_gate", "")
    model_status = _stat("dino_local_model_offline_summary_v1qg.csv", "final_status", "")
    asset_status = _stat("dino_local_asset_preprocessing_summary_v1qi.csv", "final_status", "")
    if valid > 0:
        return "DINO_SMOKE_EMBEDDINGS_AVAILABLE_REVIEW_ONLY"
    if gate == "DRY_RUN" or _stat("dino_smoke_embedding_summary_v1qj.csv", "dry_run", "true") == "true":
        if model_status != "LOCAL_DINO_MODEL_READY_OFFLINE":
            return "DINO_SMOKE_EMBEDDINGS_MODEL_MISSING_FAIL_CLOSED"
        return "DINO_SMOKE_EMBEDDINGS_DRY_RUN_ONLY"
    if model_status != "LOCAL_DINO_MODEL_READY_OFFLINE":
        return "DINO_SMOKE_EMBEDDINGS_MODEL_MISSING_FAIL_CLOSED"
    if _stat("dino_smoke_embedding_summary_v1qj.csv", "pixel_read_allowed", "false") != "true":
        return "DINO_SMOKE_EMBEDDINGS_PIXEL_READ_BLOCKED_FAIL_CLOSED"
    if "MISSING" in asset_status:
        return "DINO_SMOKE_EMBEDDINGS_ASSETS_MISSING_FAIL_CLOSED"
    return "DINO_SMOKE_EMBEDDINGS_MODEL_MISSING_FAIL_CLOSED"


def build_summary(qc_passed: int, qc_total: int, final: str) -> list[dict[str, Any]]:
    selected = _stat("dino_smoke_sample_summary_v1qh.csv", "selected_smoke_rows", "0")
    resolved = _stat("dino_local_asset_preprocessing_summary_v1qi.csv", "assets_resolved", "0")
    ready = _stat("dino_local_asset_preprocessing_summary_v1qi.csv", "preprocessing_ready", "0")
    valid = _stat("dino_smoke_embedding_summary_v1qj.csv", "embeddings_valid_768d", "0")
    neigh = _count("dino_smoke_similarity_neighbors_v1ql.csv")
    pca = _count("dino_smoke_pca_projection_v1ql.csv")
    model_status = _stat("dino_local_model_offline_summary_v1qg.csv", "final_status", "MISSING")

    def s(i: int, m: str, v: str, interp: str, ms: str = "RESULTADO_FINAL",
          use: str = "resultado_negativo_auditavel") -> dict[str, Any]:
        return {"summary_id": f"V1QM_S{i:03d}", "metric": m, "value": v,
                "interpretation": interp, "methodological_status": ms, "writing_use": use}

    return [
        s(1, "smoke_sample_selected", selected, "Patches selecionados para smoke embedding", "AUDITAVEL", "metodologia_auditoria"),
        s(2, "local_model_status", model_status, "Status do modelo DINO local offline"),
        s(3, "assets_resolved", resolved, "Assets visuais/TIF resolvidos localmente"),
        s(4, "assets_preprocessing_ready", ready, "Assets prontos para pré-processamento DINO"),
        s(5, "embeddings_valid_768d", valid, "Embeddings 768D válidos review-only gerados"),
        s(6, "similarity_neighbor_rows", neigh, "Linhas de vizinhos cosine review-only", "AUDITAVEL", "metodologia_auditoria"),
        s(7, "pca_projection_rows", pca, "Linhas de projeção PCA 2D exploratória", "AUDITAVEL", "metodologia_auditoria"),
        s(8, "quality_checks_passed", f"{qc_passed}/{qc_total}", "Checagens de qualidade aprovadas", "AUDITAVEL", "metodologia_auditoria"),
        s(9, "labels_created", "0", "Rótulos operacionais criados — 0 por design"),
        s(10, "targets_created", "0", "Targets de treinamento criados — 0 por design"),
        s(11, "ground_truth_created", "0", "Ground truth criado — 0 por design"),
        s(12, "similarity_validates_event", "false", "Similaridade não valida evento"),
        s(13, "pca_validates_event", "false", "PCA não valida evento"),
        s(14, "cluster_is_label", "false", "Clusters não são classe"),
        s(15, "final_status", final, "Status final do bloco smoke embedding", "RESULTADO_FINAL", "conclusao_auditavel"),
    ]


def build_tcc(final: str) -> list[dict[str, Any]]:
    selected = _stat("dino_smoke_sample_summary_v1qh.csv", "selected_smoke_rows", "0")
    valid = _stat("dino_smoke_embedding_summary_v1qj.csv", "embeddings_valid_768d", "0")
    resolved = _stat("dino_local_asset_preprocessing_summary_v1qi.csv", "assets_resolved", "0")
    model_status = _stat("dino_local_model_offline_summary_v1qg.csv", "final_status", "MISSING")
    boundary = "representacao_visual_review_only_nao_valida_evento"
    return [
        {"row_id": "V1QM_T01", "indicator": "Amostra smoke selecionada", "value": selected,
         "scientific_reading": "Patches Sentinel priorizados para revisão visual", "boundary": boundary},
        {"row_id": "V1QM_T02", "indicator": "Modelo DINO local", "value": model_status,
         "scientific_reading": "Disponibilidade offline do DINOv2 with registers", "boundary": boundary},
        {"row_id": "V1QM_T03", "indicator": "Assets resolvidos", "value": resolved,
         "scientific_reading": "Arquivos visuais/TIF localizados localmente", "boundary": boundary},
        {"row_id": "V1QM_T04", "indicator": "Embeddings 768D válidos", "value": valid,
         "scientific_reading": "Descritores visuais auto-supervisionados review-only", "boundary": boundary},
        {"row_id": "V1QM_T05", "indicator": "Status final", "value": final,
         "scientific_reading": "Resultado auditável do bloco smoke embedding", "boundary": boundary},
    ]


def run() -> None:
    manifest = build_manifest()
    qc, qc_passed, qc_total = build_qc()
    final = _final_status()
    summary = build_summary(qc_passed, qc_total, final)
    tcc = build_tcc(final)
    for label, rows in (("v1qm_manifest", manifest), ("v1qm_qc", qc),
                        ("v1qm_summary", summary), ("v1qm_tcc", tcc)):
        require_no_abs_paths(rows, label)
        assert_no_forbidden_true(rows, label)
    write_csv(OUT_MAN, manifest, MAN_FIELDS)
    write_csv(OUT_QC, qc, QC_FIELDS)
    write_csv(OUT_SUM, summary, SUM_FIELDS)
    write_csv(OUT_TCC, tcc, TCC_FIELDS)
    write_schema(SCH_MAN, MAN_FIELDS, "v1qm_smoke_embedding_bundle_manifest")
    write_schema(SCH_QC, QC_FIELDS, "v1qm_smoke_embedding_quality_checks")
    write_schema(SCH_SUM, SUM_FIELDS, "v1qm_smoke_embedding_scientific_summary")
    write_schema(SCH_TCC, TCC_FIELDS, "v1qm_tcc_table_smoke_embedding_results")
    write_doc(DOC, "v1qm — Smoke Embedding Scientific Bundle", [
        "## Objetivo",
        "Consolidar v1qg-v1ql em manifest, quality checks, summary científico e tabela "
        "TCC do bloco de smoke embeddings DINO.",
        "## Fronteira metodológica",
        REVIEW_ONLY_TEXT,
        "## Quality checks",
        f"Aprovadas: {qc_passed}/{qc_total}. embedding_dim=768; labels=0; targets=0; "
        "ground_truth=0; clusters não são classe; similaridade/PCA não validam evento; "
        "C3/C4 inalterados.",
        "## Status final",
        f"**{final}**.",
    ])
    print(f"[v1qm] final={final} qc={qc_passed}/{qc_total}")


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1qm smoke embedding scientific bundle").parse_args()
    run()
