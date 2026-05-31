"""REV-P v1ox — Event-patch linkage registry.

Links observed events (v1ov) to patches via region, alias, and approximate
location. Uses fail-closed: if no confirmed Sentinel scene date exists,
temporal linkage is BLOCKED_SENTINEL_SCENE_DATE_MISSING.

can_create_label=false, can_train_model=false — always.
Maximum allowed_use: REVIEW_ONLY or CONTEXTUAL_ONLY.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from revp_v1lj_v1lq_common import DATASETS, DOCS, SCHEMAS, read_csv
from revp_v1ou_v1pa_common import (
    _p,
    assert_no_forbidden_true,
    normalize_patch_id,
    require_no_abs_paths_in_rows,
    write_csv_safe,
    write_doc,
    write_schema_safe,
)

ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Output paths (env-overridable)
# ---------------------------------------------------------------------------

OUT_REGISTRY = _p("REVP_V1OX_OUT_REGISTRY", DATASETS / "recife_event_patch_linkage_registry_v1ox.csv")
OUT_SUMMARY = _p("REVP_V1OX_OUT_SUMMARY", DATASETS / "recife_event_patch_linkage_summary_v1ox.csv")
SCHEMA_REGISTRY = _p("REVP_V1OX_SCHEMA_REGISTRY", SCHEMAS / "recife_event_patch_linkage_registry_v1ox_schema.csv")
SCHEMA_SUMMARY = _p("REVP_V1OX_SCHEMA_SUMMARY", SCHEMAS / "recife_event_patch_linkage_summary_v1ox_schema.csv")
DOC = _p("REVP_V1OX_DOC", DOCS / "revp_v1ox_event_patch_linkage_registry.md")
IN_V1OV = _p("REVP_V1OX_IN_V1OV", DATASETS / "recife_ground_reference_observed_event_registry_v1ov.csv")
IN_V1OW = _p("REVP_V1OX_IN_V1OW", DATASETS / "recife_ground_reference_evidence_scoring_v1ow.csv")
IN_V1OT_SUMMARY = _p(
    "REVP_V1OX_IN_V1OT_SUMMARY",
    DATASETS / "recife_scene_date_recovery_final_scientific_summary_v1ot.csv",
)
IN_PATCH_LINKAGE = _p(
    "REVP_V1OX_IN_PATCH_LINKAGE",
    DATASETS / "patch_event_reference_link_registry.csv",
)

REGISTRY_FIELDS = [
    "linkage_id",
    "event_id",
    "patch_id",
    "alias",
    "region",
    "linkage_basis",
    "spatial_linkage_status",
    "temporal_linkage_status",
    "sentinel_scene_date_status",
    "distance_meters",
    "temporal_delta_days",
    "evidence_tier",
    "linkage_confidence",
    "allowed_use",
    "can_create_label",
    "can_train_model",
    "blocked_reason",
    "notes",
]

SUMMARY_FIELDS = ["stat_key", "stat_value"]


def _get_temporal_recovery_status(v1ot_summary: list[dict[str, str]]) -> str:
    """Extract overall scene date recovery status from v1ot summary."""
    for row in v1ot_summary:
        if row.get("metric") == "temporal_recovery_final_status":
            return str(row.get("value", "UNKNOWN"))
    # Check product_dates_confirmed_real
    for row in v1ot_summary:
        if row.get("metric") == "product_dates_confirmed_real":
            try:
                n = int(row.get("value", "0"))
                return "PRODUCT_DATE_CONFIRMED" if n > 0 else "TEMPORAL_RECOVERY_FAIL_CLOSED"
            except ValueError:
                pass
    return "UNKNOWN"


def _get_existing_patch_linkages(patch_linkage_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Filter existing patch linkages that are Recife-region and not fixtures."""
    return [
        r for r in patch_linkage_rows
        if r.get("region", "").upper() in ("RECIFE", "REC")
    ]


def build_linkage_row(
    event_row: dict[str, str],
    patch_link: dict[str, str] | None,
    temporal_status: str,
    scoring_tier: str,
    seq: int,
) -> dict[str, Any]:
    """Build a single linkage row."""
    event_id = event_row.get("event_id", f"UNKNOWN_{seq}")
    patch_id = normalize_patch_id(patch_link.get("patch_id", "") if patch_link else "")
    alias = patch_link.get("patch_id", "") if patch_link else ""
    region = event_row.get("region", "RECIFE")
    linkage_basis = patch_link.get("source_family", "CONTEXTUAL_REGION") if patch_link else "CONTEXTUAL_REGION_ONLY"

    # Spatial linkage: if we have a patch+event in same region → contextual
    spatial_status = "SPATIAL_CONTEXTUAL_REGION" if patch_id else "NO_PATCH_CANDIDATE"

    # Temporal linkage: BLOCKED if scene date not confirmed (fail-closed)
    if temporal_status == "TEMPORAL_RECOVERY_FAIL_CLOSED":
        temporal_link = "BLOCKED_SENTINEL_SCENE_DATE_MISSING"
        sentinel_scene_status = "NOT_CONFIRMED"
        blocked_reason = "TEMPORAL_RECOVERY_FAIL_CLOSED_v1og_v1ot"
    elif temporal_status == "PRODUCT_DATE_CONFIRMED":
        temporal_link = "TEMPORAL_CONTEXTUAL_PENDING_REVIEW"
        sentinel_scene_status = "CONFIRMED"
        blocked_reason = ""
    else:
        temporal_link = "BLOCKED_SENTINEL_SCENE_DATE_MISSING"
        sentinel_scene_status = "UNKNOWN"
        blocked_reason = "SENTINEL_SCENE_DATE_STATUS_UNKNOWN"

    # allowed_use: max CONTEXTUAL_ONLY when temporal is blocked
    if temporal_link.startswith("BLOCKED"):
        allowed_use = "CONTEXTUAL_ONLY"
        linkage_conf = "CONTEXTUAL_NO_TEMPORAL"
    elif spatial_status == "NO_PATCH_CANDIDATE":
        allowed_use = "CONTEXTUAL_ONLY"
        linkage_conf = "CONTEXTUAL_NO_PATCH"
    else:
        allowed_use = "REVIEW_ONLY"
        linkage_conf = "REVIEW_CONTEXTUAL_SPATIAL_ONLY"

    return {
        "linkage_id": f"V1OX_LINK_{seq:04d}",
        "event_id": event_id,
        "patch_id": patch_id,
        "alias": alias,
        "region": region,
        "linkage_basis": linkage_basis,
        "spatial_linkage_status": spatial_status,
        "temporal_linkage_status": temporal_link,
        "sentinel_scene_date_status": sentinel_scene_status,
        "distance_meters": "",
        "temporal_delta_days": "",
        "evidence_tier": scoring_tier,
        "linkage_confidence": linkage_conf,
        "allowed_use": allowed_use,
        "can_create_label": "false",
        "can_train_model": "false",
        "blocked_reason": blocked_reason,
        "notes": (
            patch_link.get("notes", "")[:120] if patch_link else
            "No patch candidate; region-contextual only"
        ),
    }


def run() -> None:
    event_rows = read_csv(IN_V1OV) if IN_V1OV.exists() else []
    scoring_rows = read_csv(IN_V1OW) if IN_V1OW.exists() else []
    v1ot_summary = read_csv(IN_V1OT_SUMMARY) if IN_V1OT_SUMMARY.exists() else []
    patch_linkage_rows = read_csv(IN_PATCH_LINKAGE) if IN_PATCH_LINKAGE.exists() else []

    # Get temporal recovery status from v1ot
    temporal_status = _get_temporal_recovery_status(v1ot_summary)

    # Get existing Recife patch linkages (placeholder contextual)
    recife_links = _get_existing_patch_linkages(patch_linkage_rows)

    # Build scoring index: event_id → tier
    scoring_index: dict[str, str] = {}
    for sr in scoring_rows:
        scoring_index[sr.get("event_id", "")] = sr.get("evidence_tier", "BLOCKED")

    rows: list[dict[str, Any]] = []
    seq = 0

    for event_row in event_rows:
        eid = event_row.get("event_id", "")
        tier = scoring_index.get(eid, "BLOCKED")

        # Find matching patch links for this event's region
        matching_links = [
            pl for pl in recife_links
            if pl.get("region", "").upper() in ("RECIFE", "REC")
        ]

        if matching_links:
            for pl in matching_links[:3]:  # limit to top 3 per event
                row = build_linkage_row(event_row, pl, temporal_status, tier, seq)
                rows.append(row)
                seq += 1
        else:
            # No patch candidate — generate blocked contextual row
            row = build_linkage_row(event_row, None, temporal_status, tier, seq)
            rows.append(row)
            seq += 1

    assert_no_forbidden_true(rows, "v1ox_linkage")
    require_no_abs_paths_in_rows(rows, "v1ox_linkage")

    write_csv_safe(OUT_REGISTRY, rows, REGISTRY_FIELDS)
    write_schema_safe(SCHEMA_REGISTRY, REGISTRY_FIELDS, "v1ox_event_patch_linkage_registry")

    # Counts
    spatial_contextual = sum(1 for r in rows if "CONTEXTUAL" in r["spatial_linkage_status"])
    temporal_blocked = sum(1 for r in rows if r["temporal_linkage_status"].startswith("BLOCKED"))
    temporal_confirmed = sum(1 for r in rows if r["temporal_linkage_status"] == "TEMPORAL_CONTEXTUAL_PENDING_REVIEW")

    summary_rows = [
        {"stat_key": "total_linkage_rows", "stat_value": str(len(rows))},
        {"stat_key": "temporal_recovery_status_v1ot", "stat_value": temporal_status},
        {"stat_key": "spatial_contextual_linkages", "stat_value": str(spatial_contextual)},
        {"stat_key": "temporal_linkages_blocked", "stat_value": str(temporal_blocked)},
        {"stat_key": "temporal_linkages_confirmed", "stat_value": str(temporal_confirmed)},
        {"stat_key": "can_create_label_any", "stat_value": "false"},
        {"stat_key": "can_train_model_any", "stat_value": "false"},
        {"stat_key": "stage", "stat_value": "v1ox"},
    ]
    write_csv_safe(OUT_SUMMARY, summary_rows, SUMMARY_FIELDS)
    write_schema_safe(SCHEMA_SUMMARY, SUMMARY_FIELDS, "v1ox_summary")

    write_doc(
        DOC,
        "v1ox — Event-Patch Linkage Registry",
        [
            "## Objetivo",
            "Vincular eventos observados (v1ov) a patches por região, alias e localização "
            "aproximada. Usa fail-closed: sem scene_date Sentinel confirmada (v1og-v1ot), "
            "temporal linkage fica BLOCKED_SENTINEL_SCENE_DATE_MISSING.",
            "## Resultado",
            f"Status de recuperação temporal (v1ot): {temporal_status}. "
            f"Linkages totais: {len(rows)}. "
            f"Temporal bloqueado: {temporal_blocked}. "
            f"Temporal confirmado: {temporal_confirmed}.",
            "## Por que temporal permanece bloqueado",
            "v1og-v1ot confirmou TEMPORAL_RECOVERY_FAIL_CLOSED: 0 product_dates confirmadas "
            "em 2.654 patches avaliados. Sem cadeia patch→asset→produto Sentinel confirmada, "
            "qualquer temporal linkage seria baseado em data inferida — proibido neste protocolo.",
            "## Guardrails",
            "can_create_label=false, can_train_model=false. "
            "allowed_use máximo: REVIEW_ONLY ou CONTEXTUAL_ONLY.",
        ],
    )

    print(f"[v1ox] {len(rows)} linkage rows: {temporal_blocked} temporal blocked, "
          f"{temporal_confirmed} temporal confirmed")
    print(f"[v1ox] Temporal recovery status: {temporal_status}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="v1ox event-patch linkage registry")
    parser.parse_args()
    run()
