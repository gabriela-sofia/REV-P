"""REV-P v1pl — DINO ↔ Protocol C crosswalk.

Cross-references the DINO embedding registry with Protocol C observational
outputs (v1oy decision audit, v1oz DINO review queue, v1ox event-patch linkage,
v1pf final bundle if present) by patch_id. DINO representation NEVER validates an
event, creates a label, or trains a model.
"""

from __future__ import annotations

import argparse
from typing import Any

from revp_v1pg_v1pm_dino_representation_common import (
    DATASETS, DOCS, SCHEMAS,
    _p, assert_no_forbidden_true, normalize_region, read_csv, require_no_abs_paths,
    write_csv, write_doc, write_schema,
)

IN_REGISTRY = _p("REVP_V1PL_IN_REGISTRY", DATASETS / "dino_embedding_feature_store_registry_v1ph.csv")
IN_V1OY = _p("REVP_V1PL_IN_V1OY", DATASETS / "recife_ground_truth_candidate_decision_audit_v1oy.csv")
IN_V1OZ = _p("REVP_V1PL_IN_V1OZ", DATASETS / "recife_dino_review_only_representation_queue_v1oz.csv")
IN_V1OX = _p("REVP_V1PL_IN_V1OX", DATASETS / "recife_event_patch_linkage_registry_v1ox.csv")
IN_V1PF = _p("REVP_V1PL_IN_V1PF", DATASETS / "recife_protocol_c_final_bundle_manifest_v1pf.csv")
OUT_CROSSWALK = _p("REVP_V1PL_OUT_CROSSWALK", DATASETS / "dino_protocol_c_crosswalk_v1pl.csv")
OUT_SUMMARY = _p("REVP_V1PL_OUT_SUMMARY", DATASETS / "dino_protocol_c_crosswalk_summary_v1pl.csv")
SCHEMA_CROSSWALK = _p("REVP_V1PL_SCHEMA_CROSSWALK", SCHEMAS / "dino_protocol_c_crosswalk_v1pl_schema.csv")
SCHEMA_SUMMARY = _p("REVP_V1PL_SCHEMA_SUMMARY", SCHEMAS / "dino_protocol_c_crosswalk_summary_v1pl_schema.csv")
DOC = _p("REVP_V1PL_DOC", DOCS / "revp_v1pl_dino_protocol_c_crosswalk.md")

CROSSWALK_FIELDS = [
    "crosswalk_id", "patch_id", "alias", "region", "embedding_id",
    "embedding_status", "protocol_c_event_id", "protocol_c_candidate_level",
    "evidence_tier", "temporal_status", "c_level_status",
    "dino_representation_status", "allowed_use", "dino_can_validate_event",
    "dino_can_create_label", "dino_can_train_model", "blocked_reason", "notes",
]
SUMMARY_FIELDS = ["stat_key", "stat_value"]


def _index_by_patch(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for r in rows:
        pid = (r.get("patch_id") or "").strip().upper()
        if pid:
            out.setdefault(pid, r)
    return out


def build_crosswalk() -> list[dict[str, Any]]:
    registry = read_csv(IN_REGISTRY)
    oy = _index_by_patch(read_csv(IN_V1OY))
    oz = _index_by_patch(read_csv(IN_V1OZ))
    ox = _index_by_patch(read_csv(IN_V1OX))

    rows: list[dict[str, Any]] = []
    for i, emb in enumerate(registry, 1):
        pid = (emb.get("patch_id") or "").strip().upper()
        dec = oy.get(pid, {})
        queue = oz.get(pid, {})
        link = ox.get(pid, {})
        event_id = dec.get("event_id") or queue.get("event_id") or link.get("event_id") or ""
        temporal = (
            dec.get("temporal_status")
            or link.get("temporal_linkage_status")
            or "NO_TEMPORAL_LINK"
        )
        c_level = dec.get("candidate_level") or "NOT_IN_PROTOCOL_C"
        rows.append({
            "crosswalk_id": f"V1PL_XW_{i:05d}",
            "patch_id": emb.get("patch_id", ""),
            "alias": emb.get("alias", ""),
            "region": normalize_region(emb.get("region", "")),
            "embedding_id": emb.get("embedding_id", ""),
            "embedding_status": emb.get("embedding_status", ""),
            "protocol_c_event_id": event_id,
            "protocol_c_candidate_level": c_level,
            "evidence_tier": dec.get("evidence_tier") or link.get("evidence_tier") or "",
            "temporal_status": temporal,
            "c_level_status": dec.get("decision_status") or "NOT_DECIDED",
            "dino_representation_status": emb.get("dino_allowed_use", ""),
            "allowed_use": "VISUAL_CONTEXT_ONLY",
            "dino_can_validate_event": "false",
            "dino_can_create_label": "false",
            "dino_can_train_model": "false",
            "blocked_reason": emb.get("blocked_reason", ""),
            "notes": "v1pf_present" if IN_V1PF.exists() else "",
        })
    return rows


def run() -> None:
    rows = build_crosswalk()
    require_no_abs_paths(rows, "v1pl_crosswalk")
    assert_no_forbidden_true(rows, "v1pl_crosswalk")
    matched = sum(1 for r in rows if r["protocol_c_event_id"])
    summary = [
        {"stat_key": "crosswalk_rows", "stat_value": str(len(rows))},
        {"stat_key": "rows_matched_to_protocol_c_event", "stat_value": str(matched)},
        {"stat_key": "events_validated_by_dino", "stat_value": "0"},
        {"stat_key": "labels_created", "stat_value": "0"},
        {"stat_key": "crosswalk_status",
         "stat_value": "CROSSWALK_READY_REVIEW_ONLY" if rows else "CROSSWALK_EMPTY_FAIL_CLOSED"},
    ]

    write_csv(OUT_CROSSWALK, rows, CROSSWALK_FIELDS)
    write_csv(OUT_SUMMARY, summary, SUMMARY_FIELDS)
    write_schema(SCHEMA_CROSSWALK, CROSSWALK_FIELDS, "v1pl_dino_protocol_c_crosswalk")
    write_schema(SCHEMA_SUMMARY, SUMMARY_FIELDS, "v1pl_dino_protocol_c_crosswalk_summary")
    write_doc(DOC, "v1pl — DINO ↔ Protocol C Crosswalk", [
        "## Objetivo",
        "Cruzar o registry de embeddings DINO com saídas observacionais do Protocolo C "
        "(v1oy, v1oz, v1ox e v1pf, se existir) por patch_id.",
        "## Guardrails",
        "DINO é representação visual review-only e NÃO valida evento observado. "
        "`dino_can_validate_event`, `dino_can_create_label` e `dino_can_train_model` "
        "são sempre false. O crosswalk é apenas contexto auditável.",
        f"## Resultado",
        f"Linhas de crosswalk: {len(rows)}. Casadas com evento Protocolo C: {matched}. "
        "Eventos validados por DINO: 0.",
    ])
    print(f"[v1pl] rows={len(rows)} matched={matched}")


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1pl dino protocol c crosswalk").parse_args()
    run()
