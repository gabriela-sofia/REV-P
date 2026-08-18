from __future__ import annotations

import argparse
import csv
import json
from datetime import date, datetime
from pathlib import Path
from typing import Any


ALLOWED_CLAIM = "review-only observed evidence workbench; supports human audit but does not close operational ground truth, labels, negatives, training, detection, or prediction"
FORBIDDEN_CLAIM = "operational ground truth|binary label|formal negative|supervised dataset|training release|detection claim|prediction claim|automatic human decision"

GLOBAL_GUARDS = {
    "ground_truth_operational_status": "ABSENT",
    "formal_labels_available": "ABSENT",
    "formal_negatives_available": "ABSENT",
    "training_ready": "false",
    "supervised_model_allowed": "false",
    "prediction_claim_allowed": "false",
    "automatic_detection_claim_allowed": "false",
    "operational_validation_claim_allowed": "false",
    "negative_by_absence_allowed": "false",
    "random_background_negative_allowed": "false",
    "decision_locked": "false",
}

EVENT_FIELDS = [
    "observed_event_id",
    "region",
    "event_name",
    "event_start_date",
    "event_end_date",
    "event_date_precision",
    "source_family",
    "source_id",
    "source_url_or_reference",
    "documentary_evidence_available",
    "observed_geometry_available",
    "observed_geometry_uri",
    "crs",
    "crs_known",
    "provenance_available",
    "hash_available",
    "license_status",
    "manual_review_required",
    "registry_status",
    "blocking_reason",
    "allowed_claim",
    "forbidden_claim",
]

PACKET_FIELDS = [
    "evidence_packet_id",
    "observed_event_id",
    "canonical_patch_id",
    "region",
    "source_family",
    "documentary_source_id",
    "geometry_source_id",
    "observed_geometry_uri",
    "patch_boundary_uri",
    "event_date_available",
    "patch_acquisition_date_available",
    "temporal_alignment_auditable",
    "spatial_alignment_auditable",
    "provenance_complete",
    "hash_complete",
    "license_review_complete",
    "circularity_risk",
    "human_review_required",
    "packet_status",
    "blocking_reason",
    "allowed_claim",
    "forbidden_claim",
]

TEMPORAL_FIELDS = [
    "temporal_alignment_id",
    "evidence_packet_id",
    "observed_event_id",
    "canonical_patch_id",
    "region",
    "event_start_date",
    "event_end_date",
    "patch_acquisition_date",
    "days_from_event_start",
    "days_from_event_end",
    "temporal_window_class",
    "temporal_alignment_status",
    "requires_human_review",
    "blocking_reason",
    "allowed_claim",
    "forbidden_claim",
]

SPATIAL_FIELDS = [
    "spatial_binding_id",
    "evidence_packet_id",
    "observed_event_id",
    "canonical_patch_id",
    "region",
    "observed_geometry_available",
    "patch_boundary_available",
    "observed_crs",
    "patch_crs",
    "crs_compatible",
    "intersection_computed",
    "candidate_intersection_area_m2",
    "candidate_overlap_ratio",
    "spatial_binding_status",
    "requires_human_review",
    "blocking_reason",
    "allowed_claim",
    "forbidden_claim",
]

REVIEW_FIELDS = [
    "review_item_id",
    "evidence_packet_id",
    "observed_event_id",
    "canonical_patch_id",
    "region",
    "temporal_alignment_status",
    "spatial_binding_status",
    "circularity_risk",
    "review_priority",
    "review_state",
    "primary_reviewer_decision",
    "secondary_reviewer_decision",
    "adjudication_required",
    "adjudicated_decision",
    "decision_locked",
    "blocking_reason",
    "allowed_claim",
    "forbidden_claim",
]

DECISION_TEMPLATE_FIELDS = [
    "review_item_id",
    "decision_scope",
    "decision_options",
    "minimum_evidence_to_accept",
    "minimum_evidence_to_reject",
    "unknown_policy",
    "negative_policy",
    "requires_comment",
    "requires_reviewer_id",
    "requires_review_date",
]

LABEL_GATE_FIELDS = [
    "label_gate_id",
    "evidence_packet_id",
    "observed_event_id",
    "canonical_patch_id",
    "region",
    "positive_gate_closed",
    "negative_gate_closed",
    "unknown_gate_required",
    "excluded_gate_required",
    "formal_label_allowed",
    "formal_negative_allowed",
    "ground_truth_closure_allowed",
    "primary_blocking_gate",
    "all_blocking_gates",
    "label_gate_status",
    "allowed_claim",
    "forbidden_claim",
]

NEGATIVE_PROTOCOL_FIELDS = [
    "negative_gate_id",
    "evidence_packet_id",
    "explicit_non_occurrence_evidence",
    "symmetric_source_protocol",
    "controlled_temporal_window",
    "human_review_complete",
    "absence_only_rejected",
    "random_background_rejected",
    "negative_gate_status",
    "allowed_claim",
    "forbidden_claim",
]

DASHBOARD_FIELDS = [
    "region",
    "observed_event_id",
    "canonical_patch_id",
    "evidence_packet_status",
    "temporal_alignment_status",
    "spatial_binding_status",
    "human_review_status",
    "positive_gate_status",
    "negative_gate_status",
    "ground_truth_operational_status",
    "mv1_training_impact",
    "main_blocker",
    "next_action",
    "allowed_scientific_claim",
    "forbidden_scientific_claim",
]

NEXT_ACTION_FIELDS = [
    "next_action_id",
    "region",
    "observed_event_id",
    "canonical_patch_id",
    "priority_rank",
    "next_action",
    "blocker",
    "allowed_claim",
    "forbidden_claim",
]

ROLLUP_FIELDS = ["stage", "output", "status", "rows", "blocking_summary", "allowed_claim", "forbidden_claim"]
GUARD_FIELDS = ["stage", "guardrail", "value", "status"]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=True, indent=2), encoding="utf-8")


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def truth(value: str | bool | None) -> bool:
    return str(value).strip().lower() in {"true", "yes", "y", "1", "sim"}


def rel(root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def p(root: Path, *parts: str) -> Path:
    return root.joinpath(*parts)


def public_table(root: Path, name: str) -> Path:
    return p(root, "outputs_public", "tables", name)


def public_log(root: Path, name: str) -> Path:
    return p(root, "outputs_public", "logs_summary", name)


def public_report(root: Path, name: str) -> Path:
    return p(root, "outputs_public", "execution_reports", name)


def doc_path(root: Path, name: str) -> Path:
    return p(root, "docs", "metodologia_cientifica", name)


def guard_rows(stage: str) -> list[dict[str, str]]:
    return [{"stage": stage, "guardrail": key, "value": value, "status": "PASS"} for key, value in GLOBAL_GUARDS.items()]


def normalize_date(value: str) -> str:
    value = (value or "").strip()
    if not value:
        return ""
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(value, fmt).date().isoformat()
        except ValueError:
            pass
    return ""


def parse_date(value: str) -> date | None:
    norm = normalize_date(value)
    if not norm:
        return None
    return date.fromisoformat(norm)


def first_value(row: dict[str, str], *keys: str) -> str:
    for key in keys:
        value = row.get(key, "")
        if value:
            return value
    return ""


def status_for_event(row: dict[str, str]) -> tuple[str, str]:
    if not row["event_start_date"]:
        return "EVENT_BLOCKED_MISSING_DATE", "event date missing"
    if not row["source_id"] and not row["source_url_or_reference"]:
        return "EVENT_BLOCKED_MISSING_SOURCE", "source id/reference missing"
    if row["provenance_available"] != "true":
        return "EVENT_BLOCKED_MISSING_PROVENANCE", "provenance incomplete"
    if row["license_status"] == "UNKNOWN":
        return "EVENT_BLOCKED_LICENSE_UNKNOWN", "license status unknown"
    if row["observed_geometry_available"] == "true" and row["crs_known"] == "true":
        return "EVENT_READY_FOR_EVIDENCE_PACKET", "ready for packet review only"
    if row["observed_geometry_available"] == "true":
        return "EVENT_REGISTERED_GEOMETRY_UNVALIDATED", "geometry or coordinate evidence is unvalidated"
    return "EVENT_REGISTERED_GEOMETRY_MISSING", "observed geometry missing"


def normalize_event_row(raw: dict[str, str], source_family: str, idx: int) -> dict[str, str]:
    observed_id = first_value(raw, "event_id", "observed_event_id", "documented_event_unit_id", "candidate_id", "source_asset_id") or f"EVENT_SOURCE_ROW_{idx:04d}"
    date_value = first_value(raw, "event_date", "event_or_survey_date", "anchor_date")
    norm_date = normalize_date(date_value)
    source_id = first_value(raw, "source_asset_id", "source_event_unit_id", "documented_event_unit_id", "source_id", "source_institution")
    source_ref = first_value(raw, "official_source_url_reference", "source_document_sanitized", "source_document_name_sanitized", "source_repository", "source_name")
    coord = truth(first_value(raw, "coordinate_available")) or first_value(raw, "coordinate_status").startswith("EXPLICIT") or first_value(raw, "geometry_available").upper() in {"YES", "TRUE"}
    observed_uri = first_value(raw, "file_path", "observed_geometry_uri")
    crs = first_value(raw, "crs")
    crs_known = bool(crs and crs.upper() not in {"NO", "NOT_APPLICABLE", "UNKNOWN"})
    provenance = bool(source_id and source_ref)
    hash_available = truth(first_value(raw, "hash_available"))
    license_status = "REVIEW_REQUIRED" if source_ref else "UNKNOWN"
    event = {
        "observed_event_id": observed_id,
        "region": first_value(raw, "region", "municipality") or "UNKNOWN",
        "event_name": first_value(raw, "event_name", "source_title", "source_document_sanitized", "source_document_name_sanitized", "event_id") or observed_id,
        "event_start_date": norm_date,
        "event_end_date": norm_date,
        "event_date_precision": first_value(raw, "event_date_precision", "temporal_precision") or ("DAY_EXPLICIT" if norm_date else "UNKNOWN"),
        "source_family": source_family,
        "source_id": source_id,
        "source_url_or_reference": source_ref,
        "documentary_evidence_available": bool_text(bool(source_ref or source_id)),
        "observed_geometry_available": bool_text(coord or bool(observed_uri)),
        "observed_geometry_uri": observed_uri,
        "crs": crs,
        "crs_known": bool_text(crs_known),
        "provenance_available": bool_text(provenance),
        "hash_available": bool_text(hash_available),
        "license_status": license_status,
        "manual_review_required": "true",
        "registry_status": "",
        "blocking_reason": "",
        "allowed_claim": ALLOWED_CLAIM,
        "forbidden_claim": FORBIDDEN_CLAIM,
    }
    status, blocker = status_for_event(event)
    if status == "EVENT_REGISTERED_GEOMETRY_MISSING" and event["documentary_evidence_available"] == "true":
        status = "EVENT_REGISTERED_DOCUMENTARY_ONLY"
    event["registry_status"] = status
    event["blocking_reason"] = blocker
    return event


def build_observed_events(root: Path) -> list[dict[str, str]]:
    sources = [
        (p(root, "datasets", "ground_reference_event_registry.csv"), "ground_reference_event_registry"),
        (p(root, "datasets", "official_observed_event_vector_registry.csv"), "official_observed_event_vector_registry"),
        (p(root, "datasets", "official_documented_event_unit_registry.csv"), "official_documented_event_unit_registry"),
        (public_table(root, "revp_tp2_candidate_priority_v2cj.csv"), "tp2_candidate_priority"),
    ]
    seen: set[str] = set()
    events: list[dict[str, str]] = []
    idx = 1
    for path, family in sources:
        for raw in read_csv(path):
            event = normalize_event_row(raw, family, idx)
            idx += 1
            if event["observed_event_id"] in seen:
                continue
            seen.add(event["observed_event_id"])
            events.append(event)
    return events


def build_event_issues(events: list[dict[str, str]]) -> list[dict[str, str]]:
    rows = []
    for idx, event in enumerate(events, start=1):
        if event["registry_status"] == "EVENT_READY_FOR_EVIDENCE_PACKET":
            continue
        rows.append(
            {
                "issue_id": f"EVENT_ISSUE_v2dz_{idx:04d}",
                "observed_event_id": event["observed_event_id"],
                "region": event["region"],
                "registry_status": event["registry_status"],
                "blocking_reason": event["blocking_reason"],
                "next_action": next_action_for_blocker(event["blocking_reason"]),
                "allowed_claim": ALLOWED_CLAIM,
                "forbidden_claim": FORBIDDEN_CLAIM,
            }
        )
    return rows


def run_v2dz(root: Path, force: bool) -> list[Path]:
    events = build_observed_events(root)
    issues = build_event_issues(events)
    outputs = [
        public_table(root, "revp_observed_event_registry_v2dz.csv"),
        public_table(root, "revp_observed_event_registry_issues_v2dz.csv"),
        public_log(root, "revp_observed_event_registry_guardrails_v2dz.csv"),
        public_report(root, "revp_observed_event_registry_report_v2dz.md"),
    ]
    write_csv(outputs[0], events, EVENT_FIELDS)
    write_csv(outputs[1], issues, ["issue_id", "observed_event_id", "region", "registry_status", "blocking_reason", "next_action", "allowed_claim", "forbidden_claim"])
    write_csv(outputs[2], guard_rows("v2dz"), GUARD_FIELDS)
    write_text(outputs[3], report("v2dz", "observed event registry normalizer", len(events), "Existing observed/documentary event rows were normalized without creating synthetic events."))
    return outputs


def patch_index(root: Path) -> dict[str, dict[str, str]]:
    patches: dict[str, dict[str, str]] = {}
    for row in read_csv(p(root, "datasets", "official_anchor_sentinel_patch_registry.csv")):
        key = first_value(row, "source_documented_event_unit_id", "anchor_id", "reference_patch_id")
        if key:
            patches[key] = row
        if row.get("reference_patch_id"):
            patches[row["reference_patch_id"]] = row
    for row in read_csv(public_table(root, "revp_tp2_patch_pair_candidates_v2ci.csv")):
        key = first_value(row, "candidate_id", "patch_id", "pair_id")
        if key and key not in patches:
            patches[key] = row
    return patches


def find_patch_for_event(event: dict[str, str], patches: dict[str, dict[str, str]]) -> dict[str, str] | None:
    keys = [event["source_id"], event["observed_event_id"]]
    for key in keys:
        if key in patches:
            return patches[key]
    event_region = event["region"].lower()
    for row in patches.values():
        if first_value(row, "region").lower() == event_region:
            return row
    return None


def build_packets(root: Path) -> list[dict[str, str]]:
    events = read_csv(public_table(root, "revp_observed_event_registry_v2dz.csv")) or build_observed_events(root)
    patches = patch_index(root)
    rows = []
    for idx, event in enumerate(events, start=1):
        patch = find_patch_for_event(event, patches)
        patch_id = first_value(patch or {}, "reference_patch_id", "patch_id", "candidate_id", "pair_id")
        patch_boundary = truth(first_value(patch or {}, "patch_generated_locally", "patch_boundary_available"))
        patch_uri = patch_id if patch_boundary else ""
        patch_date = first_value(patch or {}, "scene_date", "post_scene_date", "pre_scene_date")
        event_date = bool(event["event_start_date"])
        provenance = truth(event["provenance_available"])
        hash_complete = truth(event["hash_available"])
        license_ok = event["license_status"] != "UNKNOWN"
        spatial = truth(event["observed_geometry_available"]) and patch_boundary and truth(event["crs_known"])
        temporal = event_date and bool(normalize_date(patch_date))
        if not patch_id:
            status, blocker = "PACKET_BLOCKED_NO_PATCH", "candidate patch missing"
        elif not event_date:
            status, blocker = "PACKET_BLOCKED_NO_EVENT_DATE", "event date missing"
        elif not patch_date:
            status, blocker = "PACKET_BLOCKED_NO_PATCH_DATE", "patch acquisition date missing"
        elif event["observed_geometry_available"] != "true":
            status, blocker = "PACKET_BLOCKED_NO_GEOMETRY", "observed geometry missing"
        elif not patch_boundary:
            status, blocker = "PACKET_BLOCKED_NO_PATCH_BOUNDARY", "patch boundary missing"
        elif not provenance:
            status, blocker = "PACKET_BLOCKED_PROVENANCE", "provenance incomplete"
        elif not hash_complete:
            status, blocker = "PACKET_BLOCKED_HASH", "hash incomplete"
        elif not license_ok:
            status, blocker = "PACKET_BLOCKED_LICENSE", "license review incomplete"
        elif spatial and temporal:
            status, blocker = "PACKET_READY_FOR_HUMAN_REVIEW", "ready for human review only"
        elif temporal:
            status, blocker = "PACKET_READY_FOR_TEMPORAL_AUDIT", "temporal audit candidate only"
        else:
            status, blocker = "PACKET_DOCUMENTARY_ONLY", "documentary packet only"
        rows.append(
            {
                "evidence_packet_id": f"PACKET_v2ea_{idx:04d}",
                "observed_event_id": event["observed_event_id"],
                "canonical_patch_id": patch_id,
                "region": event["region"],
                "source_family": event["source_family"],
                "documentary_source_id": event["source_id"],
                "geometry_source_id": event["source_id"] if event["observed_geometry_available"] == "true" else "",
                "observed_geometry_uri": event["observed_geometry_uri"],
                "patch_boundary_uri": patch_uri,
                "event_date_available": bool_text(event_date),
                "patch_acquisition_date_available": bool_text(bool(normalize_date(patch_date))),
                "temporal_alignment_auditable": bool_text(temporal),
                "spatial_alignment_auditable": bool_text(spatial),
                "provenance_complete": bool_text(provenance),
                "hash_complete": bool_text(hash_complete),
                "license_review_complete": bool_text(license_ok),
                "circularity_risk": "MEDIUM_REVIEW_REQUIRED",
                "human_review_required": "true",
                "packet_status": status,
                "blocking_reason": blocker,
                "allowed_claim": ALLOWED_CLAIM,
                "forbidden_claim": FORBIDDEN_CLAIM,
            }
        )
    return rows


def run_v2ea(root: Path, force: bool) -> list[Path]:
    packets = build_packets(root)
    blockers = [row for row in packets if row["packet_status"] != "PACKET_READY_FOR_HUMAN_REVIEW"]
    outputs = [
        public_table(root, "revp_evidence_packet_registry_v2ea.csv"),
        public_table(root, "revp_evidence_packet_blockers_v2ea.csv"),
        public_log(root, "revp_evidence_packet_guardrails_v2ea.csv"),
        public_report(root, "revp_evidence_packet_builder_report_v2ea.md"),
    ]
    write_csv(outputs[0], packets, PACKET_FIELDS)
    write_csv(outputs[1], blockers, PACKET_FIELDS)
    write_csv(outputs[2], guard_rows("v2ea"), GUARD_FIELDS)
    write_text(outputs[3], report("v2ea", "evidence packet builder", len(packets), "Packets are audit work items; ready-for-review packets are not labels."))
    return outputs


def patch_date_by_packet(root: Path) -> dict[str, str]:
    patches = patch_index(root)
    packets = read_csv(public_table(root, "revp_evidence_packet_registry_v2ea.csv"))
    result = {}
    for packet in packets:
        patch = patches.get(packet["canonical_patch_id"]) or {}
        result[packet["evidence_packet_id"]] = normalize_date(first_value(patch, "scene_date", "post_scene_date", "pre_scene_date"))
    return result


def temporal_class(days_start: int, days_end: int) -> str:
    if days_start < 0:
        return "PRE_EVENT_BASELINE_CANDIDATE"
    if days_start <= 0 <= days_end:
        return "WITHIN_EVENT_WINDOW"
    if days_end <= 30:
        return "POST_EVENT_SHORT_WINDOW"
    if days_end > 30:
        return "POST_EVENT_LONG_WINDOW"
    return "TEMPORAL_UNKNOWN"


def build_temporal(root: Path) -> list[dict[str, str]]:
    packets = read_csv(public_table(root, "revp_evidence_packet_registry_v2ea.csv")) or build_packets(root)
    events = {row["observed_event_id"]: row for row in (read_csv(public_table(root, "revp_observed_event_registry_v2dz.csv")) or build_observed_events(root))}
    patch_dates = patch_date_by_packet(root)
    rows = []
    for idx, packet in enumerate(packets, start=1):
        event = events.get(packet["observed_event_id"], {})
        start = event.get("event_start_date", "")
        end = event.get("event_end_date", "") or start
        patch_date = patch_dates.get(packet["evidence_packet_id"], "")
        start_date = parse_date(start)
        end_date = parse_date(end)
        acq_date = parse_date(patch_date)
        if not start_date or not end_date:
            status, window, blocker = "TEMPORAL_ALIGNMENT_BLOCKED_NO_EVENT_DATE", "TEMPORAL_BLOCKED", "event date missing"
            ds = de = ""
        elif not acq_date:
            status, window, blocker = "TEMPORAL_ALIGNMENT_BLOCKED_NO_PATCH_DATE", "TEMPORAL_BLOCKED", "patch acquisition date missing"
            ds = de = ""
        else:
            days_start = (acq_date - start_date).days
            days_end = (acq_date - end_date).days
            window = temporal_class(days_start, days_end)
            status = "TEMPORAL_ALIGNMENT_REQUIRES_REVIEW" if window == "PRE_EVENT_BASELINE_CANDIDATE" else "TEMPORAL_ALIGNMENT_CANDIDATE"
            blocker = "candidate temporal relation requires human review"
            ds, de = str(days_start), str(days_end)
        rows.append(
            {
                "temporal_alignment_id": f"TEMP_v2eb_{idx:04d}",
                "evidence_packet_id": packet["evidence_packet_id"],
                "observed_event_id": packet["observed_event_id"],
                "canonical_patch_id": packet["canonical_patch_id"],
                "region": packet["region"],
                "event_start_date": start,
                "event_end_date": end,
                "patch_acquisition_date": patch_date,
                "days_from_event_start": ds,
                "days_from_event_end": de,
                "temporal_window_class": window,
                "temporal_alignment_status": status,
                "requires_human_review": "true",
                "blocking_reason": blocker,
                "allowed_claim": ALLOWED_CLAIM,
                "forbidden_claim": FORBIDDEN_CLAIM,
            }
        )
    return rows


def run_v2eb(root: Path, force: bool) -> list[Path]:
    rows = build_temporal(root)
    outputs = [
        public_table(root, "revp_patch_event_temporal_alignment_v2eb.csv"),
        public_log(root, "revp_temporal_alignment_guardrails_v2eb.csv"),
        public_report(root, "revp_patch_event_temporal_alignment_report_v2eb.md"),
    ]
    write_csv(outputs[0], rows, TEMPORAL_FIELDS)
    write_csv(outputs[1], guard_rows("v2eb"), GUARD_FIELDS)
    write_text(outputs[2], report("v2eb", "patch-event temporal alignment auditor", len(rows), "Temporal candidates do not close ground truth; pre-event rows remain baseline candidates only."))
    return outputs


def build_spatial(root: Path) -> list[dict[str, str]]:
    packets = read_csv(public_table(root, "revp_evidence_packet_registry_v2ea.csv")) or build_packets(root)
    events = {row["observed_event_id"]: row for row in (read_csv(public_table(root, "revp_observed_event_registry_v2dz.csv")) or build_observed_events(root))}
    patches = patch_index(root)
    rows = []
    for idx, packet in enumerate(packets, start=1):
        event = events.get(packet["observed_event_id"], {})
        patch = patches.get(packet["canonical_patch_id"]) or {}
        observed = truth(event.get("observed_geometry_available"))
        boundary = bool(packet["patch_boundary_uri"]) or truth(first_value(patch, "patch_boundary_available", "patch_generated_locally"))
        observed_crs = event.get("crs", "")
        patch_crs = first_value(patch, "crs")
        crs_ok = bool(observed_crs and patch_crs and observed_crs == patch_crs)
        if not observed:
            status, blocker = "SPATIAL_BINDING_BLOCKED_NO_OBSERVED_GEOMETRY", "observed geometry missing"
        elif not boundary:
            status, blocker = "SPATIAL_BINDING_BLOCKED_NO_PATCH_BOUNDARY", "patch boundary missing"
        elif not observed_crs or not patch_crs:
            status, blocker = "SPATIAL_BINDING_BLOCKED_MISSING_CRS", "observed or patch CRS missing"
        elif not crs_ok:
            status, blocker = "SPATIAL_BINDING_BLOCKED_CRS_MISMATCH", "CRS values differ"
        else:
            status, blocker = "SPATIAL_BINDING_CANDIDATE_NO_INTERSECTION_COMPUTED", "GIS intersection not executed in this offline audit"
        rows.append(
            {
                "spatial_binding_id": f"SPACE_v2ec_{idx:04d}",
                "evidence_packet_id": packet["evidence_packet_id"],
                "observed_event_id": packet["observed_event_id"],
                "canonical_patch_id": packet["canonical_patch_id"],
                "region": packet["region"],
                "observed_geometry_available": bool_text(observed),
                "patch_boundary_available": bool_text(boundary),
                "observed_crs": observed_crs,
                "patch_crs": patch_crs,
                "crs_compatible": bool_text(crs_ok),
                "intersection_computed": "false",
                "candidate_intersection_area_m2": "",
                "candidate_overlap_ratio": "",
                "spatial_binding_status": status,
                "requires_human_review": "true",
                "blocking_reason": blocker,
                "allowed_claim": ALLOWED_CLAIM,
                "forbidden_claim": FORBIDDEN_CLAIM,
            }
        )
    return rows


def run_v2ec(root: Path, force: bool) -> list[Path]:
    rows = build_spatial(root)
    outputs = [
        public_table(root, "revp_patch_event_spatial_binding_v2ec.csv"),
        public_log(root, "revp_spatial_binding_guardrails_v2ec.csv"),
        public_report(root, "revp_patch_event_spatial_binding_report_v2ec.md"),
    ]
    write_csv(outputs[0], rows, SPATIAL_FIELDS)
    write_csv(outputs[1], guard_rows("v2ec"), GUARD_FIELDS)
    write_text(outputs[2], report("v2ec", "patch-event spatial binding auditor", len(rows), "Spatial checks audit prerequisites; missing or candidate intersections never create negatives or labels."))
    return outputs


def build_review(root: Path) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    packets = read_csv(public_table(root, "revp_evidence_packet_registry_v2ea.csv")) or build_packets(root)
    temporal = {row["evidence_packet_id"]: row for row in read_csv(public_table(root, "revp_patch_event_temporal_alignment_v2eb.csv"))}
    spatial = {row["evidence_packet_id"]: row for row in read_csv(public_table(root, "revp_patch_event_spatial_binding_v2ec.csv"))}
    review_rows = []
    adjudication = []
    templates = []
    for idx, packet in enumerate(packets, start=1):
        temp_status = temporal.get(packet["evidence_packet_id"], {}).get("temporal_alignment_status", "")
        space_status = spatial.get(packet["evidence_packet_id"], {}).get("spatial_binding_status", "")
        if "BLOCKED" in temp_status or "BLOCKED" in space_status or "BLOCKED" in packet["packet_status"]:
            state = "REVIEW_BLOCKED_INSUFFICIENT_EVIDENCE"
            priority = "LOW"
        elif "REQUIRES_REVIEW" in temp_status or "CANDIDATE" in space_status:
            state = "NEEDS_PRIMARY_REVIEW"
            priority = "HIGH"
        else:
            state = "REVIEW_READY_FOR_DECISION_TEMPLATE"
            priority = "MEDIUM"
        item_id = f"REVIEW_v2ed_{idx:04d}"
        row = {
            "review_item_id": item_id,
            "evidence_packet_id": packet["evidence_packet_id"],
            "observed_event_id": packet["observed_event_id"],
            "canonical_patch_id": packet["canonical_patch_id"],
            "region": packet["region"],
            "temporal_alignment_status": temp_status,
            "spatial_binding_status": space_status,
            "circularity_risk": packet["circularity_risk"],
            "review_priority": priority,
            "review_state": state,
            "primary_reviewer_decision": "",
            "secondary_reviewer_decision": "",
            "adjudication_required": "false",
            "adjudicated_decision": "",
            "decision_locked": "false",
            "blocking_reason": packet["blocking_reason"],
            "allowed_claim": ALLOWED_CLAIM,
            "forbidden_claim": FORBIDDEN_CLAIM,
        }
        review_rows.append(row)
        if state == "NEEDS_ADJUDICATION":
            adjudication.append(row)
        templates.append(
            {
                "review_item_id": item_id,
                "decision_scope": "observed_event_patch_candidate_review",
                "decision_options": "accept_candidate_positive|reject_candidate_positive|unknown|exclude_from_formal_use",
                "minimum_evidence_to_accept": "documented source, audited date, audited spatial prerequisites, provenance, hash, reviewer comment",
                "minimum_evidence_to_reject": "explicit reason; rejection of a positive candidate does not create a formal negative",
                "unknown_policy": "unknown remains unknown and does not become negative",
                "negative_policy": "formal negative requires explicit non-occurrence evidence and symmetric protocol",
                "requires_comment": "true",
                "requires_reviewer_id": "true",
                "requires_review_date": "true",
            }
        )
    return review_rows, adjudication, templates


def run_v2ed(root: Path, force: bool) -> list[Path]:
    review_rows, adjudication, templates = build_review(root)
    outputs = [
        public_table(root, "revp_human_review_queue_v2ed.csv"),
        public_table(root, "revp_adjudication_queue_v2ed.csv"),
        public_table(root, "revp_review_decision_template_v2ed.csv"),
        public_log(root, "revp_human_review_guardrails_v2ed.csv"),
        public_report(root, "revp_human_review_adjudication_queue_report_v2ed.md"),
    ]
    write_csv(outputs[0], review_rows, REVIEW_FIELDS)
    write_csv(outputs[1], adjudication, REVIEW_FIELDS)
    write_csv(outputs[2], templates, DECISION_TEMPLATE_FIELDS)
    write_csv(outputs[3], guard_rows("v2ed"), GUARD_FIELDS)
    write_text(outputs[4], report("v2ed", "human review and adjudication queue", len(review_rows), "All human decision fields are intentionally empty and decision_locked=false."))
    return outputs


def build_label_gates(root: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    reviews = read_csv(public_table(root, "revp_human_review_queue_v2ed.csv"))
    rows = []
    negative = []
    for idx, review in enumerate(reviews, start=1):
        blockers = [
            "primary_review_missing",
            "secondary_review_missing",
            "adjudication_missing_if_required",
            "formal_negative_explicit_evidence_missing",
            "ground_truth_closure_not_allowed_without locked human decision",
        ]
        rows.append(
            {
                "label_gate_id": f"GATE_v2ee_{idx:04d}",
                "evidence_packet_id": review["evidence_packet_id"],
                "observed_event_id": review["observed_event_id"],
                "canonical_patch_id": review["canonical_patch_id"],
                "region": review["region"],
                "positive_gate_closed": "false",
                "negative_gate_closed": "false",
                "unknown_gate_required": "true",
                "excluded_gate_required": "true",
                "formal_label_allowed": "false",
                "formal_negative_allowed": "false",
                "ground_truth_closure_allowed": "false",
                "primary_blocking_gate": "primary_review_missing",
                "all_blocking_gates": "|".join(blockers),
                "label_gate_status": "LABEL_GATE_BLOCKED_NO_REVIEW",
                "allowed_claim": ALLOWED_CLAIM,
                "forbidden_claim": FORBIDDEN_CLAIM,
            }
        )
        negative.append(
            {
                "negative_gate_id": f"NEG_GATE_v2ee_{idx:04d}",
                "evidence_packet_id": review["evidence_packet_id"],
                "explicit_non_occurrence_evidence": "false",
                "symmetric_source_protocol": "false",
                "controlled_temporal_window": "false",
                "human_review_complete": "false",
                "absence_only_rejected": "true",
                "random_background_rejected": "true",
                "negative_gate_status": "LABEL_GATE_BLOCKED_NO_FORMAL_NEGATIVE",
                "allowed_claim": ALLOWED_CLAIM,
                "forbidden_claim": FORBIDDEN_CLAIM,
            }
        )
    return rows, negative


def run_v2ee(root: Path, force: bool) -> list[Path]:
    rows, negative = build_label_gates(root)
    outputs = [
        public_table(root, "revp_formal_label_gate_evaluator_v2ee.csv"),
        public_table(root, "revp_negative_gate_protocol_v2ee.csv"),
        public_log(root, "revp_formal_label_gate_guardrails_v2ee.csv"),
        public_report(root, "revp_formal_label_gate_evaluator_report_v2ee.md"),
    ]
    write_csv(outputs[0], rows, LABEL_GATE_FIELDS)
    write_csv(outputs[1], negative, NEGATIVE_PROTOCOL_FIELDS)
    write_csv(outputs[2], guard_rows("v2ee"), GUARD_FIELDS)
    write_text(outputs[3], report("v2ee", "formal positive/negative gate evaluator", len(rows), "Formal positive and negative gates remain closed because human decisions and explicit negative evidence are absent."))
    return outputs


def next_action_for_blocker(blocker: str) -> str:
    text = (blocker or "").lower()
    if "geometry" in text:
        return "locate and validate observed geometry"
    if "crs" in text:
        return "validate CRS"
    if "hash" in text:
        return "register source hash"
    if "review" in text:
        return "complete human review"
    if "boundary" in text or "patch" in text:
        return "resolve patch boundary"
    if "date" in text:
        return "resolve event or patch date"
    if "license" in text:
        return "review license"
    if "adjudication" in text:
        return "adjudicate conflict"
    return "complete provenance and evidence audit"


def build_dashboard(root: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    packets = {row["evidence_packet_id"]: row for row in read_csv(public_table(root, "revp_evidence_packet_registry_v2ea.csv"))}
    temporal = {row["evidence_packet_id"]: row for row in read_csv(public_table(root, "revp_patch_event_temporal_alignment_v2eb.csv"))}
    spatial = {row["evidence_packet_id"]: row for row in read_csv(public_table(root, "revp_patch_event_spatial_binding_v2ec.csv"))}
    reviews = {row["evidence_packet_id"]: row for row in read_csv(public_table(root, "revp_human_review_queue_v2ed.csv"))}
    gates = {row["evidence_packet_id"]: row for row in read_csv(public_table(root, "revp_formal_label_gate_evaluator_v2ee.csv"))}
    rows = []
    next_rows = []
    for idx, (packet_id, packet) in enumerate(packets.items(), start=1):
        temp_status = temporal.get(packet_id, {}).get("temporal_alignment_status", "")
        space_status = spatial.get(packet_id, {}).get("spatial_binding_status", "")
        review_status = reviews.get(packet_id, {}).get("review_state", "")
        gate = gates.get(packet_id, {})
        blocker = gate.get("primary_blocking_gate") or packet.get("blocking_reason", "")
        next_action = next_action_for_blocker(blocker or packet.get("blocking_reason", ""))
        rows.append(
            {
                "region": packet["region"],
                "observed_event_id": packet["observed_event_id"],
                "canonical_patch_id": packet["canonical_patch_id"],
                "evidence_packet_status": packet["packet_status"],
                "temporal_alignment_status": temp_status,
                "spatial_binding_status": space_status,
                "human_review_status": review_status,
                "positive_gate_status": gate.get("label_gate_status", ""),
                "negative_gate_status": "LABEL_GATE_BLOCKED_NO_FORMAL_NEGATIVE",
                "ground_truth_operational_status": "ABSENT",
                "mv1_training_impact": "MV1 supervised training remains blocked because formal label, formal negative and closure gates are not allowed",
                "main_blocker": blocker,
                "next_action": next_action,
                "allowed_scientific_claim": ALLOWED_CLAIM,
                "forbidden_scientific_claim": FORBIDDEN_CLAIM,
            }
        )
        next_rows.append(
            {
                "next_action_id": f"NEXT_v2ef_{idx:04d}",
                "region": packet["region"],
                "observed_event_id": packet["observed_event_id"],
                "canonical_patch_id": packet["canonical_patch_id"],
                "priority_rank": str(idx),
                "next_action": next_action,
                "blocker": blocker,
                "allowed_claim": ALLOWED_CLAIM,
                "forbidden_claim": FORBIDDEN_CLAIM,
            }
        )
    return rows, next_rows


def run_v2ef(root: Path, force: bool) -> list[Path]:
    rows, next_rows = build_dashboard(root)
    outputs = [
        public_table(root, "revp_ground_truth_closure_dashboard_v2ef.csv"),
        public_table(root, "revp_ground_truth_next_action_plan_v2ef.csv"),
        public_log(root, "revp_ground_truth_closure_guardrails_v2ef.csv"),
        public_report(root, "revp_ground_truth_closure_dashboard_report_v2ef.md"),
        public_report(root, "revp_v2dz_to_v2ef_scientific_summary.md"),
    ]
    write_csv(outputs[0], rows, DASHBOARD_FIELDS)
    write_csv(outputs[1], next_rows, NEXT_ACTION_FIELDS)
    write_csv(outputs[2], guard_rows("v2ef"), GUARD_FIELDS)
    write_text(outputs[3], report("v2ef", "ground truth closure dashboard", len(rows), "Dashboard remains closure-blocked because this sprint does not fill human decisions."))
    write_text(outputs[4], scientific_summary(rows))
    return outputs


def report(stage: str, title: str, count: int, note: str) -> str:
    return f"""# REV-P {stage} {title}

Rows: {count}

{note}

Allowed claim: {ALLOWED_CLAIM}

Forbidden claim: {FORBIDDEN_CLAIM}

Global state: ground_truth_operational_status=ABSENT; formal_labels_available=ABSENT; formal_negatives_available=ABSENT; training_ready=false; decision_locked=false.
"""


def scientific_summary(rows: list[dict[str, str]]) -> str:
    regions = sorted({row["region"] for row in rows})
    return f"""# REV-P v2dz-to-v2ef Scientific Summary

Regions represented: {', '.join(regions)}

This workbench organizes observed-event evidence, packets, temporal/spatial audit states, human review queues, formal gate blockers and next actions. It does not close operational ground truth, create labels, create negatives, train models, or make detection/prediction claims.
"""


def write_docs(root: Path) -> list[Path]:
    docs = {
        "revp_v2dz_observed_event_registry_normalizer.md": "v2dz observed event registry normalizer",
        "revp_v2ea_evidence_packet_builder.md": "v2ea evidence packet builder",
        "revp_v2eb_patch_event_temporal_alignment.md": "v2eb patch-event temporal alignment auditor",
        "revp_v2ec_patch_event_spatial_binding.md": "v2ec patch-event spatial binding auditor",
        "revp_v2ed_human_review_adjudication_queue.md": "v2ed human review and adjudication queue",
        "revp_v2ee_formal_label_gate_evaluator.md": "v2ee formal positive/negative gate evaluator",
        "revp_v2ef_ground_truth_closure_dashboard.md": "v2ef ground truth closure dashboard",
    }
    paths = []
    for name, title in docs.items():
        path = doc_path(root, name)
        write_text(path, f"# {title}\n\nThis stage is review-only and fail-closed. It records missing event dates, sources, geometry, CRS, hashes, provenance, patch boundaries and human decisions as blockers. It does not create labels, negatives, model training data, operational detection claims, or prediction claims.\n")
        paths.append(path)
    return paths


def integrated_report(root: Path) -> Path:
    rows = read_csv(public_table(root, "revp_ground_truth_closure_dashboard_v2ef.csv"))
    text = "# REV-P v2dz-to-v2ef Integrated Report\n\n"
    text += "The ground truth workbench was generated as a review-only audit chain.\n\n"
    for row in rows[:50]:
        text += f"- {row['region']} / {row['observed_event_id']} / {row['canonical_patch_id']}: {row['ground_truth_operational_status']}; blocker={row['main_blocker']}; next={row['next_action']}\n"
    path = public_report(root, "revp_v2dz_to_v2ef_integrated_report.md")
    write_text(path, text)
    return path


def commit_checklist(root: Path) -> Path:
    path = public_report(root, "revp_v2dz_to_v2ef_commit_checklist.md")
    write_text(path, "# REV-P v2dz-to-v2ef Commit Checklist\n\nNo git add, commit, push or PR was performed by this script. Future reference message: analysis: estrutura workbench de ground truth observacional patch-level\n")
    return path


def run_integrated(root: Path, force: bool) -> list[Path]:
    outputs: list[Path] = []
    outputs += run_v2dz(root, force)
    outputs += run_v2ea(root, force)
    outputs += run_v2eb(root, force)
    outputs += run_v2ec(root, force)
    outputs += run_v2ed(root, force)
    outputs += run_v2ee(root, force)
    outputs += run_v2ef(root, force)
    outputs += write_docs(root)
    outputs += [integrated_report(root), commit_checklist(root)]
    rollup = []
    for stage in ["v2dz", "v2ea", "v2eb", "v2ec", "v2ed", "v2ee", "v2ef"]:
        stage_outputs = [rel(root, path) for path in outputs if stage in path.name]
        rollup.append({"stage": stage, "output": ";".join(stage_outputs), "status": "PASS", "rows": str(len(stage_outputs)), "blocking_summary": "ground truth closure remains absent", "allowed_claim": ALLOWED_CLAIM, "forbidden_claim": FORBIDDEN_CLAIM})
    test_rollup = public_log(root, "revp_v2dz_to_v2ef_test_rollup.csv")
    guard_rollup = public_log(root, "revp_v2dz_to_v2ef_guardrail_rollup.csv")
    write_csv(test_rollup, rollup, ROLLUP_FIELDS)
    guards = []
    for stage in ["v2dz", "v2ea", "v2eb", "v2ec", "v2ed", "v2ee", "v2ef", "v2dz_to_v2ef"]:
        guards.extend(guard_rows(stage))
    write_csv(guard_rollup, guards, GUARD_FIELDS)
    outputs += [test_rollup, guard_rollup]
    print(json.dumps({"stage": "v2dz_to_v2ef", "outputs": len(outputs), "ground_truth_operational_status": "ABSENT", "training_ready": False}, ensure_ascii=True, indent=2))
    return outputs


def add_repo_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--force", action="store_true")
    return parser


def parse_args() -> argparse.Namespace:
    return add_repo_args(argparse.ArgumentParser()).parse_args()
