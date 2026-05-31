"""REV-P v1qa — Expanded queue import bridge.

Converts v1pw expanded visual queue to executor-compatible format.
Preserves priority/reason/linkage_confidence. All items remain review-only.
Never executes embedding. Never creates labels/targets/ground truth.
"""
from __future__ import annotations

import argparse
from typing import Any

from revp_v1qa_v1qf_execution_bridge_common import (
    DATASETS, DOCS, SCHEMAS,
    _p, assert_no_forbidden_true, dedup_queue, mask_local, normalize_region,
    path_hash, read_backend_probe, read_expanded_queue,
    require_no_abs_paths, validate_queue_item, write_csv, write_doc, write_schema,
)

IN_QUEUE = _p("REVP_V1QA_IN_QUEUE",
              DATASETS / "dino_review_only_execution_queue_expanded_v1pw.csv")
IN_BACKEND = _p("REVP_V1QA_IN_BACKEND",
                DATASETS / "dino_backend_model_probe_summary_v1pp.csv")
OUT_QUEUE = _p("REVP_V1QA_OUT_QUEUE",
               DATASETS / "dino_execution_queue_from_visual_expansion_v1qa.csv")
OUT_SUM = _p("REVP_V1QA_OUT_SUM",
             DATASETS / "dino_execution_queue_from_visual_expansion_summary_v1qa.csv")
SCH_QUEUE = _p("REVP_V1QA_SCH_QUEUE",
               SCHEMAS / "dino_execution_queue_from_visual_expansion_v1qa_schema.csv")
SCH_SUM = _p("REVP_V1QA_SCH_SUM",
             SCHEMAS / "dino_execution_queue_from_visual_expansion_summary_v1qa_schema.csv")
DOC = _p("REVP_V1QA_DOC", DOCS / "revp_v1qa_expanded_queue_import_bridge.md")

QUEUE_FIELDS = [
    "execution_queue_id", "source_queue_id", "visual_asset_id", "patch_id",
    "alias", "region", "relative_path", "path_hash", "visual_type",
    "queue_priority", "queue_reason", "linkage_confidence", "dino_allowed_use",
    "can_create_label", "can_train_model", "target_created",
    "ready_for_dry_run", "ready_for_real_execution", "blocked_reason", "notes",
]
SUM_FIELDS = ["stat_key", "stat_value"]


def build_queue() -> list[dict[str, Any]]:
    raw = read_expanded_queue(IN_QUEUE)
    raw = dedup_queue(raw)
    backend = read_backend_probe(IN_BACKEND)
    model_ok = backend.get("can_execute_embeddings", "false") == "true"

    rows: list[dict[str, Any]] = []
    for i, r in enumerate(raw, 1):
        valid, reason = validate_queue_item(r)
        rel = mask_local(r.get("relative_path", ""))
        ph = r.get("path_hash") or path_hash(rel)
        ready_real = "true" if (valid and model_ok) else "false"
        rows.append({
            "execution_queue_id": f"V1QA_EQ_{i:05d}",
            "source_queue_id": r.get("queue_id", ""),
            "visual_asset_id": r.get("visual_asset_id", ""),
            "patch_id": r.get("patch_id", ""),
            "alias": r.get("alias", ""),
            "region": normalize_region(r.get("region", "")),
            "relative_path": rel,
            "path_hash": ph,
            "visual_type": r.get("visual_type", ""),
            "queue_priority": r.get("queue_priority", ""),
            "queue_reason": r.get("queue_reason", ""),
            "linkage_confidence": r.get("linkage_confidence", ""),
            "dino_allowed_use": r.get("dino_allowed_use", "REVIEW_ONLY_REPRESENTATION"),
            "can_create_label": "false",
            "can_train_model": "false",
            "target_created": "false",
            "ready_for_dry_run": "true" if valid else "false",
            "ready_for_real_execution": ready_real,
            "blocked_reason": reason if not valid else "",
            "notes": "",
        })
    return rows


def run() -> None:
    rows = build_queue()
    require_no_abs_paths(rows, "v1qa")
    assert_no_forbidden_true(rows, "v1qa")
    dry_ready = sum(1 for r in rows if r["ready_for_dry_run"] == "true")
    real_ready = sum(1 for r in rows if r["ready_for_real_execution"] == "true")
    summary = [
        {"stat_key": "queue_rows_imported", "stat_value": str(len(rows))},
        {"stat_key": "ready_for_dry_run", "stat_value": str(dry_ready)},
        {"stat_key": "ready_for_real_execution", "stat_value": str(real_ready)},
        {"stat_key": "blocked_rows", "stat_value": str(len(rows) - dry_ready)},
        {"stat_key": "labels_created", "stat_value": "0"},
        {"stat_key": "targets_created", "stat_value": "0"},
    ]
    write_csv(OUT_QUEUE, rows, QUEUE_FIELDS)
    write_csv(OUT_SUM, summary, SUM_FIELDS)
    write_schema(SCH_QUEUE, QUEUE_FIELDS, "v1qa_execution_queue_from_visual_expansion")
    write_schema(SCH_SUM, SUM_FIELDS, "v1qa_execution_queue_summary")
    write_doc(DOC, "v1qa — Expanded Queue Import Bridge", [
        "## Objetivo",
        "Converter fila visual expandida v1pw para formato compatível com executor "
        "v1pq. Preserva priority/reason/linkage_confidence. Tudo review-only.",
        "## Guardrails",
        "can_create_label, can_train_model e target_created sempre false. "
        "ready_for_real_execution=false enquanto modelo indisponível.",
        f"## Resultado",
        f"Itens importados: {len(rows)}. Prontos dry-run: {dry_ready}. "
        f"Prontos execução real: {real_ready}.",
    ])
    print(f"[v1qa] imported={len(rows)} dry_ready={dry_ready} real_ready={real_ready}")


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1qa expanded queue import bridge").parse_args()
    run()
