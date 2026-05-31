"""REV-P v1pt — DINO execution bundle.

Consolidates v1pn-v1ps into manifest, QC, scientific summary and final doc.
Does not recompute — reads outputs of prior stages.
"""
from __future__ import annotations

import argparse
import os
from typing import Any

from revp_v1pn_v1pt_dino_execution_common import (
    DATASETS, DOCS, SCHEMAS,
    _p, assert_no_forbidden_true, require_no_abs_paths, write_csv, write_doc, write_schema,
)
from revp_v1pg_v1pm_dino_representation_common import read_csv

OUT_MANIFEST = _p("REVP_V1PT_OUT_MANIFEST", DATASETS / "dino_execution_manifest_v1pt.csv")
OUT_QC = _p("REVP_V1PT_OUT_QC", DATASETS / "dino_execution_quality_checks_v1pt.csv")
OUT_SUM = _p("REVP_V1PT_OUT_SUM", DATASETS / "dino_execution_scientific_summary_v1pt.csv")
SCH_MAN = _p("REVP_V1PT_SCH_MAN", SCHEMAS / "dino_execution_manifest_v1pt_schema.csv")
SCH_QC = _p("REVP_V1PT_SCH_QC", SCHEMAS / "dino_execution_quality_checks_v1pt_schema.csv")
SCH_SUM = _p("REVP_V1PT_SCH_SUM", SCHEMAS / "dino_execution_scientific_summary_v1pt_schema.csv")
DOC = _p("REVP_V1PT_DOC", DOCS / "revp_v1pt_dino_execution_bundle.md")

DRY_RUN = os.environ.get("REVP_DINO_DRY_RUN", "true").lower() == "true"

MANIFEST_FIELDS = ["artifact_id", "stage", "filename", "rows", "header_present", "role"]
QC_FIELDS = [
    "check_id", "stage", "filename", "check_name",
    "status", "severity", "observed", "expected",
]
SUM_FIELDS = ["summary_id", "metric", "value", "interpretation", "methodological_status", "writing_use"]

TCC_TEXT = (
    "A etapa de execução controlada de embeddings DINO foi estruturada como harness "
    "reprodutível e fail-closed. A geração de vetores depende da disponibilidade local "
    "do backend/modelo e, quando executada, produz representações 768D review-only, sem "
    "criação de rótulos, sem targets supervisionados e sem validação operacional de eventos."
)

ARTIFACTS: list[tuple[str, str, str, list[str]]] = [
    # (stage, filename, role, required_cols)
    ("v1pn", "dino_patch_visual_asset_inventory_v1pn.csv", "visual_asset_inventory",
     ["visual_asset_id", "eligible_for_embedding_queue"]),
    ("v1pn", "dino_patch_visual_asset_inventory_summary_v1pn.csv", "visual_asset_summary",
     ["stat_key", "stat_value"]),
    ("v1po", "dino_embedding_execution_queue_v1po.csv", "execution_queue",
     ["queue_id", "can_create_label", "can_train_model", "target_created"]),
    ("v1po", "dino_embedding_execution_queue_summary_v1po.csv", "execution_queue_summary",
     ["stat_key", "stat_value"]),
    ("v1pp", "dino_backend_model_probe_v1pp.csv", "backend_probe",
     ["probe_id", "available"]),
    ("v1pp", "dino_backend_model_probe_summary_v1pp.csv", "backend_probe_summary",
     ["stat_key", "stat_value"]),
    ("v1pq", "dino_controlled_smoke_embedding_results_v1pq.csv", "smoke_results",
     ["embedding_run_id", "can_create_label", "can_train_model", "target_created"]),
    ("v1pq", "dino_controlled_smoke_embedding_failures_v1pq.csv", "smoke_failures",
     ["failure_id"]),
    ("v1pq", "dino_controlled_smoke_embedding_summary_v1pq.csv", "smoke_summary",
     ["stat_key", "stat_value"]),
    ("v1pr", "dino_smoke_embedding_feature_store_v1pr.csv", "smoke_feature_store",
     ["embedding_id", "dino_can_create_label", "dino_can_train_model"]),
    ("v1pr", "dino_smoke_embedding_feature_store_summary_v1pr.csv", "smoke_store_summary",
     ["stat_key", "stat_value"]),
    ("v1ps", "dino_smoke_similarity_neighbors_v1ps.csv", "smoke_neighbors",
     ["query_patch_id", "can_create_label"]),
    ("v1ps", "dino_smoke_pca_projection_v1ps.csv", "smoke_pca",
     ["patch_id", "can_create_label"]),
    ("v1ps", "dino_smoke_cluster_exploratory_v1ps.csv", "smoke_cluster",
     ["patch_id", "can_be_used_as_class"]),
    ("v1ps", "dino_smoke_protocol_c_crosswalk_v1ps.csv", "smoke_crosswalk",
     ["crosswalk_id", "dino_can_validate_event"]),
    ("v1ps", "dino_smoke_review_products_summary_v1ps.csv", "smoke_review_summary",
     ["stat_key", "stat_value"]),
]


def _stat(path_or_fname: str, key: str) -> str:
    from pathlib import Path
    p = DATASETS / path_or_fname
    for r in read_csv(p):
        if r.get("stat_key") == key:
            return r.get("stat_value", "0")
    return "0"


def _count_rows(fname: str) -> int:
    from pathlib import Path
    p = DATASETS / fname
    if not p.exists():
        return -1
    rows = read_csv(p)
    return len(rows)


def _header(fname: str) -> list[str]:
    from pathlib import Path
    from revp_v1pg_v1pm_dino_representation_common import read_csv_header
    return read_csv_header(DATASETS / fname)


def build_manifest() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, (stage, fname, role, _) in enumerate(ARTIFACTS, 1):
        n = _count_rows(fname)
        h = _header(fname)
        rows.append({
            "artifact_id": f"V1PT_ART_{i:03d}",
            "stage": stage,
            "filename": fname,
            "rows": str(n) if n >= 0 else "MISSING",
            "header_present": str(bool(h)).lower(),
            "role": role,
        })
    return rows


def build_qc() -> list[dict[str, Any]]:
    qc: list[dict[str, Any]] = []
    qid = 0
    for stage, fname, role, req_cols in ARTIFACTS:
        p = DATASETS / fname
        h = _header(fname)
        exists = p.exists()
        qid += 1
        qc.append({"check_id": f"V1PT_QC_{qid:04d}", "stage": stage, "filename": fname,
                    "check_name": "file_exists", "status": "PASS" if exists else "FAIL",
                    "severity": "HIGH" if not exists else "INFO",
                    "observed": str(exists), "expected": "true"})
        qid += 1
        qc.append({"check_id": f"V1PT_QC_{qid:04d}", "stage": stage, "filename": fname,
                    "check_name": "header_present", "status": "PASS" if h else "FAIL",
                    "severity": "HIGH" if not h else "INFO",
                    "observed": str(bool(h)), "expected": "true"})
        for col in req_cols:
            qid += 1
            present = col in h
            qc.append({"check_id": f"V1PT_QC_{qid:04d}", "stage": stage, "filename": fname,
                        "check_name": f"col_{col}", "status": "PASS" if present else "FAIL",
                        "severity": "HIGH" if not present else "INFO",
                        "observed": str(present), "expected": "true"})
    return qc


def build_summary(dry_run: bool) -> tuple[list[dict[str, Any]], str]:
    visual = _stat("dino_patch_visual_asset_inventory_summary_v1pn.csv", "visual_assets_found")
    eligible = _stat("dino_patch_visual_asset_inventory_summary_v1pn.csv", "eligible_for_queue")
    queue = _stat("dino_embedding_execution_queue_summary_v1po.csv", "queue_total")
    backend = _stat("dino_backend_model_probe_summary_v1pp.csv", "final_status")
    can_exec = _stat("dino_backend_model_probe_summary_v1pp.csv", "can_execute_embeddings")
    attempted = _stat("dino_controlled_smoke_embedding_summary_v1pq.csv", "embeddings_attempted")
    executed = _stat("dino_controlled_smoke_embedding_summary_v1pq.csv", "embeddings_executed_review_only")
    valid = _stat("dino_smoke_embedding_feature_store_summary_v1pr.csv", "valid_768d")
    invalid = _stat("dino_smoke_embedding_feature_store_summary_v1pr.csv", "invalid_blocked")
    nb = _stat("dino_smoke_review_products_summary_v1ps.csv", "smoke_neighbors")
    pca = _stat("dino_smoke_review_products_summary_v1ps.csv", "smoke_pca_rows")
    cl = _stat("dino_smoke_review_products_summary_v1ps.csv", "smoke_clusters")

    if dry_run:
        final = "DINO_EXECUTION_PLAN_READY_DRY_RUN"
    elif can_exec != "true":
        final = "DINO_EXECUTION_FAIL_CLOSED_MODEL_UNAVAILABLE"
    elif int(valid or "0") > 0:
        final = "DINO_SMOKE_EMBEDDINGS_GENERATED_REVIEW_ONLY"
    else:
        final = "DINO_EXECUTION_NO_EMBEDDINGS_GENERATED_FAIL_CLOSED"

    def s(i: int, m: str, v: str, interp: str, ms: str = "RESULTADO_FINAL", use: str = "resultado_negativo_auditavel") -> dict[str, Any]:
        return {"summary_id": f"V1PT_S{i:03d}", "metric": m, "value": v,
                "interpretation": interp, "methodological_status": ms, "writing_use": use}

    rows = [
        s(1, "visual_assets_found", visual, "Imagens elegíveis encontradas no repositório", "AUDITAVEL", "metodologia_auditoria"),
        s(2, "eligible_queue_items", eligible, "Itens elegíveis para fila de embedding", "AUDITAVEL", "metodologia_auditoria"),
        s(3, "queue_total", queue, "Total na fila de execução", "AUDITAVEL", "metodologia_auditoria"),
        s(4, "backend_status", backend, "Status do backend de execução", "AUDITAVEL", "metodologia_auditoria"),
        s(5, "dry_run", str(dry_run).lower(), "Modo dry-run ativo (padrão=true)", "AUDITAVEL", "metodologia_auditoria"),
        s(6, "embeddings_attempted", attempted, "Embeddings tentados (0 se dry-run)"),
        s(7, "embeddings_generated", executed, "Embeddings executados em modo real"),
        s(8, "valid_768d_embeddings", valid, "Embeddings válidos 768D importados"),
        s(9, "invalid_embeddings", invalid, "Embeddings inválidos bloqueados"),
        s(10, "smoke_neighbors", nb, "Pares de vizinhança exploratória smoke"),
        s(11, "smoke_pca_rows", pca, "Linhas de projeção PCA smoke"),
        s(12, "smoke_clusters", cl, "Clusters exploratórios smoke (não são classe)"),
        s(13, "labels_created", "0", "Labels operacionais criadas — 0 por design"),
        s(14, "targets_created", "0", "Targets de treinamento criados — 0 por design"),
        s(15, "final_execution_status", final, "Status final da execução de embeddings DINO", "RESULTADO_FINAL", "conclusao_auditavel"),
    ]
    return rows, final


def run() -> None:
    manifest = build_manifest()
    qc = build_qc()
    summary, final = build_summary(DRY_RUN)
    for label, rows in [("v1pt_manifest", manifest), ("v1pt_qc", qc), ("v1pt_summary", summary)]:
        require_no_abs_paths(rows, label)
        assert_no_forbidden_true(rows, label)
    write_csv(OUT_MANIFEST, manifest, MANIFEST_FIELDS)
    write_csv(OUT_QC, qc, QC_FIELDS)
    write_csv(OUT_SUM, summary, SUM_FIELDS)
    write_schema(SCH_MAN, MANIFEST_FIELDS, "v1pt_dino_execution_manifest")
    write_schema(SCH_QC, QC_FIELDS, "v1pt_dino_execution_quality_checks")
    write_schema(SCH_SUM, SUM_FIELDS, "v1pt_dino_execution_scientific_summary")
    fails = sum(1 for r in qc if r["status"] == "FAIL")
    write_doc(DOC, "v1pt — DINO Execution Bundle", [
        "## Objetivo",
        "Consolidar v1pn-v1ps em manifest, QC, summary científico e doc final.",
        "## Texto metodológico TCC",
        TCC_TEXT,
        "## Guardrails",
        "DINO é representação visual review-only. Nenhum label, target ou treino criado.",
        f"## Resultado",
        f"Status final: **{final}**. QC: {len(qc)} checks, {fails} falhas.",
    ])
    print(f"[v1pt] final_status={final} qc={len(qc)} fails={fails}")


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1pt dino execution bundle").parse_args()
    run()
