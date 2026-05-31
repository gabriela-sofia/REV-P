"""REV-P v1po — DINO embedding execution queue builder.

Builds a prioritized queue from v1pn visual assets and Protocol C v1oz review queue.
Does NOT execute any embedding. All rows are review-only with explicit anti-label guards.
"""
from __future__ import annotations

import argparse
import os
from typing import Any

from revp_v1pn_v1pt_dino_execution_common import (
    DATASETS, DOCS, ROOT, SCHEMAS,
    _p, assert_no_forbidden_true, normalize_region, path_hash,
    require_no_abs_paths, write_csv, write_doc, write_schema,
)
from revp_v1pg_v1pm_dino_representation_common import read_csv

OUT_QUEUE = _p("REVP_V1PO_OUT_QUEUE", DATASETS / "dino_embedding_execution_queue_v1po.csv")
OUT_SUM = _p("REVP_V1PO_OUT_SUM", DATASETS / "dino_embedding_execution_queue_summary_v1po.csv")
SCH_QUEUE = _p("REVP_V1PO_SCH_QUEUE", SCHEMAS / "dino_embedding_execution_queue_v1po_schema.csv")
SCH_SUM = _p("REVP_V1PO_SCH_SUM", SCHEMAS / "dino_embedding_execution_queue_summary_v1po_schema.csv")
DOC = _p("REVP_V1PO_DOC", DOCS / "revp_v1po_dino_embedding_execution_queue.md")
IN_INV = _p("REVP_V1PO_IN_INV", DATASETS / "dino_patch_visual_asset_inventory_v1pn.csv")
IN_V1OZ = _p("REVP_V1PO_IN_V1OZ", DATASETS / "recife_dino_review_only_representation_queue_v1oz.csv")

MAX_QUEUE = int(os.environ.get("REVP_DINO_MAX_QUEUE", "50"))

QUEUE_FIELDS = [
    "queue_id", "visual_asset_id", "patch_id", "alias", "region",
    "relative_path", "path_hash", "queue_priority", "queue_reason",
    "protocol_c_context", "dino_allowed_use", "can_create_label",
    "can_train_model", "target_created", "execution_status", "blocked_reason", "notes",
]
SUM_FIELDS = ["stat_key", "stat_value"]


def build_queue() -> list[dict[str, Any]]:
    inventory = read_csv(IN_INV)
    v1oz = read_csv(IN_V1OZ)

    # Index v1oz by patch_id
    oz_patches: set[str] = {r.get("patch_id", "").strip().upper() for r in v1oz if r.get("patch_id")}

    eligible = [r for r in inventory if r.get("eligible_for_embedding_queue") == "true"]
    seen: set[str] = set()
    queue: list[dict[str, Any]] = []

    def _add(row: dict[str, str], priority: int, reason: str, protocol_ctx: str) -> None:
        if len(queue) >= MAX_QUEUE:
            return
        k = row.get("path_hash", "") or path_hash(row.get("relative_path", ""))
        if k in seen:
            return
        seen.add(k)
        patch = row.get("patch_id", "UNKNOWN_PATCH").strip().upper()
        queue.append({
            "queue_id": f"V1PO_Q_{len(queue)+1:05d}",
            "visual_asset_id": row.get("visual_asset_id", ""),
            "patch_id": patch,
            "alias": row.get("alias", ""),
            "region": normalize_region(row.get("region", "")),
            "relative_path": row.get("relative_path", ""),
            "path_hash": row.get("path_hash", ""),
            "queue_priority": str(priority),
            "queue_reason": reason,
            "protocol_c_context": protocol_ctx,
            "dino_allowed_use": "REVIEW_ONLY_REPRESENTATION",
            "can_create_label": "false",
            "can_train_model": "false",
            "target_created": "false",
            "execution_status": "PENDING",
            "blocked_reason": row.get("blocked_reason", ""),
            "notes": "",
        })

    # Priority 1: protocol C DINO review queue patches with visual assets
    for row in eligible:
        pid = row.get("patch_id", "").strip().upper()
        if pid in oz_patches:
            _add(row, 1, "protocol_c_dino_review_queue", "v1oz_candidate")

    # Priority 2: remaining eligible
    for row in eligible:
        _add(row, 2, "visual_asset_eligible", "none")

    return queue


def run() -> None:
    rows = build_queue()
    require_no_abs_paths(rows, "v1po")
    assert_no_forbidden_true(rows, "v1po")
    p1 = sum(1 for r in rows if r["queue_priority"] == "1")
    summary = [
        {"stat_key": "queue_total", "stat_value": str(len(rows))},
        {"stat_key": "protocol_c_priority_items", "stat_value": str(p1)},
        {"stat_key": "max_queue_limit", "stat_value": str(MAX_QUEUE)},
        {"stat_key": "labels_queued", "stat_value": "0"},
    ]
    write_csv(OUT_QUEUE, rows, QUEUE_FIELDS)
    write_csv(OUT_SUM, summary, SUM_FIELDS)
    write_schema(SCH_QUEUE, QUEUE_FIELDS, "v1po_dino_embedding_execution_queue")
    write_schema(SCH_SUM, SUM_FIELDS, "v1po_dino_embedding_execution_queue_summary")
    write_doc(DOC, "v1po — DINO Embedding Execution Queue", [
        "## Objetivo",
        "Construir fila priorizada de assets visuais para geração de embedding. "
        "Não executa embedding. Tudo review-only.",
        "## Prioridade",
        "1 = patch em fila DINO review v1oz. 2 = asset elegível restante.",
        "## Guardrails",
        "`can_create_label`, `can_train_model` e `target_created` sempre false.",
        f"## Resultado",
        f"Itens na fila: {len(rows)}. Prioridade 1 (Protocol C): {p1}.",
    ])
    print(f"[v1po] queue={len(rows)} priority1={p1}")


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1po dino embedding execution queue").parse_args()
    run()
