"""REV-P v1qv — Event/patch review sampling frame and sample.

Builds a sampling frame from existing Protocol C context (event-patch
linkages, DINO review queue, source requirements) and draws a stratified
human-review sample. DINO may prioritize review but never proves an event.
Blocked rows are included as a methodological control.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from revp_v1lj_v1lq_common import DATASETS, DOCS, SCHEMAS
from revp_v1qu_v1qz_ground_reference_common import (
    _p,
    assert_clean_rows,
    guardrail_row,
    hash_short,
    load_existing_protocol_c_context,
    normalize_alias,
    normalize_event_id,
    normalize_patch_id,
    normalize_region,
    write_csv_with_header,
    write_doc,
    write_schema_safe,
)

ROOT = Path(__file__).resolve().parents[2]

OUT_FRAME = _p("REVP_V1QV_OUT_FRAME", DATASETS / "protocol_c_event_patch_review_sampling_frame_v1qv.csv")
OUT_SAMPLE = _p("REVP_V1QV_OUT_SAMPLE", DATASETS / "protocol_c_event_patch_review_sample_v1qv.csv")
OUT_SUMMARY = _p("REVP_V1QV_OUT_SUMMARY", DATASETS / "protocol_c_event_patch_review_sampling_summary_v1qv.csv")
SCHEMA_FRAME = _p("REVP_V1QV_SCHEMA_FRAME", SCHEMAS / "protocol_c_event_patch_review_sampling_frame_v1qv_schema.csv")
SCHEMA_SAMPLE = _p("REVP_V1QV_SCHEMA_SAMPLE", SCHEMAS / "protocol_c_event_patch_review_sample_v1qv_schema.csv")
SCHEMA_SUMMARY = _p("REVP_V1QV_SCHEMA_SUMMARY", SCHEMAS / "protocol_c_event_patch_review_sampling_summary_v1qv_schema.csv")
DOC = _p("REVP_V1QV_DOC", DOCS / "revp_v1qv_event_patch_review_sampling_frame.md")

REQUIREMENTS_PATH = _p("REVP_V1QV_REQUIREMENTS", DATASETS / "protocol_c_official_evidence_source_requirements_v1qu.csv")

FRAME_FIELDS = [
    "frame_id", "event_id", "patch_id", "alias", "region", "hazard_type",
    "evidence_status", "temporal_status", "spatial_status", "dino_queue_status",
    "source_requirement_status", "sampling_stratum", "frame_priority",
    "review_only", "dino_validates_event", "can_create_operational_label",
    "can_train_model", "target_created", "ground_truth_operational", "notes",
]

SAMPLE_FIELDS = [
    "review_sample_id", "event_id", "patch_id", "alias", "region", "hazard_type",
    "evidence_status", "temporal_status", "spatial_status", "dino_queue_status",
    "source_requirement_status", "sampling_stratum", "sample_priority",
    "sample_reason", "review_only", "dino_validates_event",
    "can_create_operational_label", "can_train_model", "target_created",
    "ground_truth_operational", "notes",
]

SUMMARY_FIELDS = ["stat_key", "stat_value"]


def _hazard_for_region(region: str) -> str:
    return {"PET": "LANDSLIDE", "RECIFE": "FLOOD", "CURITIBA": "FLOOD"}.get(region, "UNKNOWN")


def _stratum(evidence_status: str, dino_status: str, blocked: bool) -> tuple[str, int]:
    """Return (stratum, priority) — higher priority sampled first."""
    if blocked:
        return ("BLOCKED_CONTROL", 1)
    ev = evidence_status.upper()
    if "C2" in ev or "REVIEW_ONLY" in ev:
        return ("C2_REVIEW_ONLY", 5)
    if "CONTEXTUAL" in ev or "C1" in ev:
        return ("C1_CONTEXTUAL_GAP", 4)
    if dino_status and dino_status.upper() not in ("", "NONE"):
        return ("DINO_REVIEW_QUEUE", 3)
    return ("GENERIC_CANDIDATE", 2)


def build_frame(context: dict[str, list[dict[str, str]]]) -> list[dict[str, Any]]:
    dino_patches = {
        normalize_patch_id(r.get("patch_id", "")): r.get("dino_allowed_use", "REVIEW_ONLY_REPRESENTATION")
        for r in context.get("dino_review_queue", [])
    }
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    for r in context.get("event_patch_linkages", []):
        event_id = normalize_event_id(r.get("event_id", ""))
        patch_id = normalize_patch_id(r.get("patch_id", ""))
        key = f"{event_id}|{patch_id}"
        if key in seen:
            continue
        seen.add(key)
        region = normalize_region(r.get("region", ""))
        evidence_status = r.get("allowed_use", "") or r.get("evidence_tier", "")
        blocked = bool(r.get("blocked_reason", "").strip()) or "BLOCKED" in evidence_status.upper()
        dino_status = dino_patches.get(patch_id, "")
        stratum, prio = _stratum(evidence_status, dino_status, blocked)
        row = {
            "frame_id": f"V1QV_FR_{hash_short(key, 10)}",
            "event_id": event_id,
            "patch_id": patch_id,
            "alias": normalize_alias(r.get("alias", "")),
            "region": region,
            "hazard_type": _hazard_for_region(region),
            "evidence_status": evidence_status or "UNKNOWN",
            "temporal_status": r.get("temporal_linkage_status", "") or r.get("sentinel_scene_date_status", "") or "UNKNOWN",
            "spatial_status": r.get("spatial_linkage_status", "") or "UNKNOWN",
            "dino_queue_status": dino_status or "NOT_IN_DINO_QUEUE",
            "source_requirement_status": "SEE_V1QU_REQUIREMENTS",
            "sampling_stratum": stratum,
            "frame_priority": str(prio),
            "notes": "blocked_control" if blocked else "",
        }
        row.update(guardrail_row())
        rows.append(row)

    # Include DINO queue patches not already in frame
    for r in context.get("dino_review_queue", []):
        event_id = normalize_event_id(r.get("event_id", ""))
        patch_id = normalize_patch_id(r.get("patch_id", ""))
        key = f"{event_id}|{patch_id}"
        if key in seen:
            continue
        seen.add(key)
        region = normalize_region(r.get("region", ""))
        row = {
            "frame_id": f"V1QV_FR_{hash_short(key, 10)}",
            "event_id": event_id, "patch_id": patch_id,
            "alias": normalize_alias(r.get("alias", "")),
            "region": region, "hazard_type": _hazard_for_region(region),
            "evidence_status": r.get("evidence_tier", "REVIEW_ONLY"),
            "temporal_status": r.get("linkage_status", "UNKNOWN"),
            "spatial_status": "UNKNOWN",
            "dino_queue_status": r.get("dino_allowed_use", "REVIEW_ONLY_REPRESENTATION"),
            "source_requirement_status": "SEE_V1QU_REQUIREMENTS",
            "sampling_stratum": "DINO_REVIEW_QUEUE", "frame_priority": "3",
            "notes": "from_dino_queue",
        }
        row.update(guardrail_row())
        rows.append(row)

    return rows


def draw_sample(frame: list[dict[str, Any]], sample_n: int, min_per_region: int) -> list[dict[str, Any]]:
    by_region: dict[str, list[dict[str, Any]]] = {}
    for r in frame:
        by_region.setdefault(r["region"], []).append(r)
    for region in by_region:
        by_region[region].sort(key=lambda x: (-int(x["frame_priority"]), x["frame_id"]))

    selected: list[dict[str, Any]] = []
    selected_keys: set[str] = set()

    # Pass 1: guarantee minimum per region
    for region, items in by_region.items():
        for r in items[:min_per_region]:
            k = r["frame_id"]
            if k not in selected_keys:
                selected.append(r)
                selected_keys.add(k)

    # Pass 2: fill remaining by global priority
    remaining = sorted(
        (r for r in frame if r["frame_id"] not in selected_keys),
        key=lambda x: (-int(x["frame_priority"]), x["frame_id"]),
    )
    for r in remaining:
        if len(selected) >= sample_n:
            break
        selected.append(r)
        selected_keys.add(r["frame_id"])

    selected.sort(key=lambda x: (x["region"], -int(x["frame_priority"]), x["frame_id"]))
    out: list[dict[str, Any]] = []
    for i, r in enumerate(selected):
        reason = {
            "BLOCKED_CONTROL": "included_as_methodological_control",
            "C2_REVIEW_ONLY": "c2_candidate_priority",
            "C1_CONTEXTUAL_GAP": "contextual_gap_priority",
            "DINO_REVIEW_QUEUE": "dino_review_only_priority_not_proof",
            "GENERIC_CANDIDATE": "generic_candidate",
        }.get(r["sampling_stratum"], "candidate")
        sample = {
            "review_sample_id": f"V1QV_SMP_{i:04d}",
            "event_id": r["event_id"], "patch_id": r["patch_id"], "alias": r["alias"],
            "region": r["region"], "hazard_type": r["hazard_type"],
            "evidence_status": r["evidence_status"], "temporal_status": r["temporal_status"],
            "spatial_status": r["spatial_status"], "dino_queue_status": r["dino_queue_status"],
            "source_requirement_status": r["source_requirement_status"],
            "sampling_stratum": r["sampling_stratum"], "sample_priority": r["frame_priority"],
            "sample_reason": reason, "notes": r.get("notes", ""),
        }
        sample.update(guardrail_row())
        out.append(sample)
    return out


def run(datasets: Path | None = None) -> dict[str, Any]:
    context = load_existing_protocol_c_context(datasets)
    sample_n = int(os.environ.get("REVP_PROTOCOL_C_REVIEW_SAMPLE_N", "24"))
    min_per_region = int(os.environ.get("REVP_PROTOCOL_C_MIN_PER_REGION", "4"))

    frame = build_frame(context)
    sample = draw_sample(frame, sample_n, min_per_region)

    assert_clean_rows(frame, "v1qv_frame")
    assert_clean_rows(sample, "v1qv_sample")

    write_csv_with_header(OUT_FRAME, frame, FRAME_FIELDS)
    write_csv_with_header(OUT_SAMPLE, sample, SAMPLE_FIELDS)
    write_schema_safe(SCHEMA_FRAME, FRAME_FIELDS, "v1qv_frame")
    write_schema_safe(SCHEMA_SAMPLE, SAMPLE_FIELDS, "v1qv_sample")

    strata: dict[str, int] = {}
    regions: dict[str, int] = {}
    for s in sample:
        strata[s["sampling_stratum"]] = strata.get(s["sampling_stratum"], 0) + 1
        regions[s["region"]] = regions.get(s["region"], 0) + 1

    summary = [
        {"stat_key": "frame_size", "stat_value": str(len(frame))},
        {"stat_key": "sample_size", "stat_value": str(len(sample))},
        {"stat_key": "sample_n_requested", "stat_value": str(sample_n)},
        {"stat_key": "min_per_region", "stat_value": str(min_per_region)},
        {"stat_key": "blocked_controls_in_sample", "stat_value": str(strata.get("BLOCKED_CONTROL", 0))},
        {"stat_key": "c2_in_sample", "stat_value": str(strata.get("C2_REVIEW_ONLY", 0))},
        {"stat_key": "dino_queue_in_sample", "stat_value": str(strata.get("DINO_REVIEW_QUEUE", 0))},
    ]
    for region, n in sorted(regions.items()):
        summary.append({"stat_key": f"sample_region_{region.lower()}", "stat_value": str(n)})
    summary.append({"stat_key": "stage", "stat_value": "v1qv"})
    write_csv_with_header(OUT_SUMMARY, summary, SUMMARY_FIELDS)
    write_schema_safe(SCHEMA_SUMMARY, SUMMARY_FIELDS, "v1qv_summary")

    write_doc(
        DOC,
        "v1qv — Event/Patch Review Sampling Frame",
        [
            "## Objetivo",
            "Construir o quadro amostral de unidades evento-patch e sortear uma amostra "
            "estratificada para revisao supervisora. Prioriza C2, lacunas contextuais, fila DINO "
            "review-only e lacunas de fonte. Inclui bloqueados como controle metodologico.",
            "## Parametros",
            f"REVP_PROTOCOL_C_REVIEW_SAMPLE_N={sample_n}; "
            f"REVP_PROTOCOL_C_MIN_PER_REGION={min_per_region}.",
            "## Resultado",
            f"Frame: {len(frame)} unidades. Amostra: {len(sample)}.",
            "## Guardrails",
            "DINO pode priorizar revisao mas nunca prova evento (dino_validates_event=false). "
            "Nenhuma linha cria label, target ou ground truth operacional.",
        ],
    )
    print(f"[v1qv] frame={len(frame)} sample={len(sample)}")
    return {"frame": len(frame), "sample": len(sample)}


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1qv review sampling frame").parse_args()
    run()
