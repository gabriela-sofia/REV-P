#!/usr/bin/env python3
"""v1um Recife Human Review package for locality-only evidence.

This module prepares locality-only candidates for auditable Human Review. It
does not execute overlay, geocode, infer coordinates, create operational truth,
or create training labels.
"""

import argparse
import csv
import hashlib
import os
import re
from collections import Counter, defaultdict

PROTOCOL_VERSION = "v1um"
EVENT_ID = "REC_2022_05_24_30"
DATASET_DIR = "datasets/protocolo_c"
DOCS_DIR = "docs/metodologia_cientifica"
CONFIG_DIR = "configs/protocolo_c"
REVIEW_DIR = "docs/review_packages/protocolo_c"
MAX_STATUS = "RECIFE_LOCALITY_ONLY_HUMAN_REVIEW_CANDIDATE"
HUMAN_REVIEW_STATUS = "PREPARED_NOT_OPERATIONAL"

LOCALITY_ROUTE = "ROUTE_LOCALITY_ONLY_REVIEW"

SAMPLE_COLUMNS = [
    "sample_id", "event_id", "candidate_row_id", "row_hash", "asset_id",
    "sample_type", "event_window_match", "hazard_signal", "hazard_strength",
    "locality_status", "has_neighborhood", "has_address", "address_hash",
    "locality_hash", "source_table", "review_priority", "inclusion_reason",
    "human_review_package_created", "human_review_status",
    "can_create_ground_reference", "can_create_training_label",
]

NORMALIZATION_COLUMNS = [
    "locality_norm_id", "event_id", "candidate_row_id", "raw_locality_hash",
    "normalized_locality_token", "normalization_status", "ambiguity_status",
    "locality_granularity", "can_support_human_review", "can_support_overlay",
    "human_review_package_created", "human_review_status", "notes",
]

HAZARD_COLUMNS = [
    "hazard_rank_id", "event_id", "candidate_row_id", "row_hash",
    "hazard_class", "hazard_strength_score", "matched_terms_hash",
    "matched_term_classes", "ambiguity_status", "review_priority_delta",
    "can_support_occurrence_review", "human_review_package_created",
    "human_review_status", "can_create_ground_reference",
    "can_create_training_label",
]

BATCH_COLUMNS = [
    "batch_id", "event_id", "batch_file", "sample_type", "candidate_count",
    "high_confidence_count", "locality_only_count", "ambiguous_count",
    "human_review_task", "human_review_decision_options",
    "human_review_package_created", "human_review_status",
    "supervisor_review_completed", "can_create_ground_reference",
    "can_create_training_label",
]

EVIDENCE_COLUMNS = [
    "evidence_package_id", "event_id", "candidate_row_id", "row_hash",
    "parsed_date", "hazard_class", "hazard_strength_score",
    "normalized_locality_token", "has_address", "has_neighborhood",
    "address_hash", "locality_hash", "source_table", "inclusion_reason",
    "public_redaction_status", "human_review_package_created",
    "human_review_status", "ground_truth_operational",
    "can_create_ground_reference", "can_create_training_label",
]

AGGREGATION_COLUMNS = [
    "aggregation_id", "event_id", "normalized_locality_token", "date",
    "candidate_count", "flood_strong_count", "rain_impact_count",
    "ambiguous_count", "source_table_count", "has_address_count",
    "review_priority", "can_support_contextual_review", "can_support_overlay",
    "can_create_ground_reference",
]

DECISION_COLUMNS = [
    "decision_id", "event_id", "candidate_row_id", "row_hash",
    "human_review_question", "evidence_signal", "suggested_decision",
    "operational_decision", "ground_reference_status", "label_status",
    "can_create_ground_reference", "can_create_training_label",
]

READINESS_COLUMNS = [
    "readiness_id", "event_id", "candidate_row_id", "row_hash",
    "non_overlay_readiness_status", "human_review_status",
    "has_event_window_match", "has_hazard_signal", "has_locality_text",
    "has_coordinates", "can_support_human_review", "can_support_overlay",
    "ground_truth_operational", "can_create_ground_reference",
    "can_create_training_label", "blocker", "required_next_action",
]

BLOCKER_COLUMNS = [
    "blocker_id", "event_id", "blocker", "status", "evidence_count",
    "ground_truth_operational", "can_create_ground_reference",
    "can_create_training_label", "human_review_status", "notes",
]

NEXT_ACTION_COLUMNS = [
    "action_id", "event_id", "action_type", "priority", "description",
    "target", "status", "notes",
]

MANIFEST_COLUMNS = [
    "artifact_id", "artifact_path", "artifact_type", "protocol_version",
    "sha256_prefix", "file_size_bytes", "is_versionable", "reason",
]

V1UM_ARTIFACTS = [
    "configs/protocolo_c/v1um_recife_review_sampling_policy.yaml",
    "configs/protocolo_c/v1um_recife_locality_normalization_policy.yaml",
    "configs/protocolo_c/v1um_recife_hazard_semantics_policy.yaml",
    "configs/protocolo_c/v1um_recife_redaction_policy.yaml",
    "configs/protocolo_c/v1um_recife_human_review_batch_policy.yaml",
    "configs/protocolo_c/v1um_recife_non_overlay_decision_policy.yaml",
    "datasets/protocolo_c/v1um_recife_locality_candidate_sample_registry.csv",
    "datasets/protocolo_c/v1um_recife_locality_normalization_registry.csv",
    "datasets/protocolo_c/v1um_recife_hazard_semantics_rank_registry.csv",
    "datasets/protocolo_c/v1um_recife_human_review_batch_registry.csv",
    "datasets/protocolo_c/v1um_recife_redacted_evidence_package_registry.csv",
    "datasets/protocolo_c/v1um_recife_neighborhood_signal_aggregation.csv",
    "datasets/protocolo_c/v1um_recife_human_review_decision_matrix.csv",
    "datasets/protocolo_c/v1um_recife_non_overlay_readiness_matrix.csv",
    "datasets/protocolo_c/v1um_recife_ground_reference_blocker_matrix.csv",
    "datasets/protocolo_c/v1um_next_actions_registry.csv",
    "datasets/protocolo_c/v1um_versionable_artifacts_manifest.csv",
    "docs/metodologia_cientifica/protocolo_c_v1um_recife_human_review_locality_only.md",
    "docs/metodologia_cientifica/protocolo_c_relatorio_v1um_recife_human_review_locality_only.md",
    "docs/metodologia_cientifica/protocolo_c_status_atual_v1um.md",
    "docs/review_packages/protocolo_c/v1um_recife_human_review_batch_01.md",
    "docs/review_packages/protocolo_c/v1um_recife_human_review_batch_02.md",
    "docs/review_packages/protocolo_c/v1um_recife_human_review_batch_03.md",
]

ABSOLUTE_PATH_RE = re.compile(r"([A-Za-z]:\\|\\\\|/home/|/Users/|/mnt/|local_only)")


def load_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, columns, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256_text(value, n=16):
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()[:n]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def bool_text(value):
    return "true" if bool(value) else "false"


def int_value(value):
    try:
        return int(value or 0)
    except ValueError:
        return 0


def artifact_path(path):
    return path.replace("\\", "/")


def source_table_map():
    result = {}
    for row in load_csv(os.path.join(DATASET_DIR, "v1uk_recife_occurrence_table_profile.csv")):
        result[row.get("asset_id", "")] = row.get("table_name", "")
    for row in load_csv(os.path.join(DATASET_DIR, "v1uk_recife_asset_schema_registry.csv")):
        result.setdefault(row.get("asset_id", ""), row.get("title", ""))
    return result


def joined_locality_candidates(router_path=None):
    router_path = router_path or os.path.join(DATASET_DIR, "v1ul_recife_candidate_review_router.csv")
    routes = [
        r for r in load_csv(router_path)
        if r.get("review_route") == LOCALITY_ROUTE
    ]
    matches = {
        r.get("row_hash", ""): r
        for r in load_csv(os.path.join(DATASET_DIR, "v1uk_recife_event_window_match_registry.csv"))
    }
    candidates = {
        r.get("candidate_row_id", ""): r
        for r in load_csv(os.path.join(DATASET_DIR, "v1uk_recife_candidate_row_registry.csv"))
    }
    tables = source_table_map()
    rows = []
    for route in routes:
        match = matches.get(route.get("row_hash", ""), {})
        cand = candidates.get(route.get("candidate_row_id", ""), {})
        locality_hash = match.get("neighborhood_hash") or match.get("address_hash") or sha256_text(route.get("candidate_row_id"))
        row = dict(route)
        row.update({
            "parsed_date": match.get("parsed_date", ""),
            "has_flood_term": match.get("has_flood_term", "false"),
            "has_rain_term": match.get("has_rain_term", "false"),
            "has_landslide_term": match.get("has_landslide_term", "false"),
            "has_neighborhood": match.get("has_neighborhood", "false"),
            "neighborhood_hash": match.get("neighborhood_hash", ""),
            "has_address": match.get("has_address", "false"),
            "address_hash": match.get("address_hash", ""),
            "locality_hash": locality_hash,
            "source_table": tables.get(route.get("asset_id", ""), route.get("asset_id", "")),
            "evidence_strength": cand.get("evidence_strength", ""),
        })
        rows.append(row)
    return rows


def hazard_class_for(row):
    flood = row.get("has_flood_term") == "true"
    rain = row.get("has_rain_term") == "true"
    landslide = row.get("has_landslide_term") == "true"
    hazard = row.get("hazard_signal") == "HAS_HAZARD_SIGNAL"
    if flood:
        return "FLOOD_STRONG", 90, "flood"
    if rain and landslide:
        return "LANDSLIDE_OR_SLOPE", 72, "rain|landslide"
    if landslide:
        return "LANDSLIDE_OR_SLOPE", 68, "landslide"
    if rain:
        return "RAIN_IMPACT", 60, "rain"
    if hazard:
        return "CIVIL_DEFENSE_GENERIC", 42, "generic_hazard"
    return "NO_HAZARD_SIGNAL", 0, "none"


def review_priority_delta(hazard_class):
    if hazard_class == "FLOOD_STRONG":
        return "-2"
    if hazard_class in {"RAIN_IMPACT", "LANDSLIDE_OR_SLOPE"}:
        return "-1"
    if hazard_class == "CIVIL_DEFENSE_GENERIC":
        return "0"
    return "+2"


def sample_type_for(row, idx, seen_diversity):
    hazard_class, score, _terms = hazard_class_for(row)
    key = (row.get("parsed_date", ""), row.get("asset_id", ""), row.get("locality_hash", ""))
    if score >= 60 and row.get("has_address") == "true":
        return "high_confidence_sample", "event_window_hazard_address_redacted"
    if key not in seen_diversity:
        seen_diversity.add(key)
        return "diversity_sample", "first_candidate_for_date_asset_locality_hash"
    if idx % 17 == 0:
        return "random_audit_sample", "deterministic_modulo_audit_selection"
    if idx % 19 == 0:
        return "full_queue_summary", "complete_locality_only_queue_member"
    if hazard_class in {"CIVIL_DEFENSE_GENERIC", "AMBIGUOUS", "NO_HAZARD_SIGNAL"}:
        return "edge_case_sample", "generic_or_ambiguous_hazard_for_review"
    return "full_queue_summary", "complete_locality_only_queue_member"


def normalized_locality_token(row):
    if row.get("has_neighborhood") == "true" and row.get("neighborhood_hash"):
        return "neighborhood_hash_" + row.get("neighborhood_hash")[:12], "NEIGHBORHOOD", "NORMALIZED_FROM_HASH"
    if row.get("has_address") == "true" and row.get("address_hash"):
        return "address_hash_" + row.get("address_hash")[:12], "ADDRESS_TEXT_REDACTED", "NORMALIZED_FROM_HASH"
    if row.get("locality_hash"):
        return "locality_hash_" + row.get("locality_hash")[:12], "AMBIGUOUS", "HASH_ONLY"
    return "missing_locality", "MISSING", "MISSING"


def run_locality_candidate_sampler(out_path=None, router_path=None):
    rows = []
    seen_diversity = set()
    for idx, row in enumerate(joined_locality_candidates(router_path)):
        hazard_class, score, _terms = hazard_class_for(row)
        sample_type, reason = sample_type_for(row, idx, seen_diversity)
        rows.append({
            "sample_id": f"SAMPLE_{PROTOCOL_VERSION}_{idx:06d}",
            "event_id": row.get("event_id") or EVENT_ID,
            "candidate_row_id": row.get("candidate_row_id", ""),
            "row_hash": row.get("row_hash", ""),
            "asset_id": row.get("asset_id", ""),
            "sample_type": sample_type,
            "event_window_match": row.get("event_window_match", ""),
            "hazard_signal": row.get("hazard_signal", ""),
            "hazard_strength": str(score),
            "locality_status": row.get("locality_status", ""),
            "has_neighborhood": row.get("has_neighborhood", "false"),
            "has_address": row.get("has_address", "false"),
            "address_hash": row.get("address_hash", ""),
            "locality_hash": row.get("locality_hash", ""),
            "source_table": row.get("source_table", ""),
            "review_priority": row.get("review_priority", "2"),
            "inclusion_reason": reason,
            "human_review_package_created": "true",
            "human_review_status": HUMAN_REVIEW_STATUS,
            "can_create_ground_reference": "false",
            "can_create_training_label": "false",
        })
    out_path = out_path or os.path.join(DATASET_DIR, "v1um_recife_locality_candidate_sample_registry.csv")
    write_csv(out_path, SAMPLE_COLUMNS, rows)
    print(f"[v1um sampler] rows={len(rows)} -> {out_path}")
    return rows


def run_locality_text_normalizer(out_path=None, router_path=None):
    rows = []
    for idx, row in enumerate(joined_locality_candidates(router_path)):
        token, granularity, status = normalized_locality_token(row)
        ambiguous = "AMBIGUOUS" if granularity in {"AMBIGUOUS", "MISSING"} else "NOT_AMBIGUOUS"
        rows.append({
            "locality_norm_id": f"LOCNORM_{PROTOCOL_VERSION}_{idx:06d}",
            "event_id": row.get("event_id") or EVENT_ID,
            "candidate_row_id": row.get("candidate_row_id", ""),
            "raw_locality_hash": row.get("locality_hash", ""),
            "normalized_locality_token": token,
            "normalization_status": status,
            "ambiguity_status": ambiguous,
            "locality_granularity": granularity,
            "can_support_human_review": bool_text(granularity != "MISSING"),
            "can_support_overlay": "false",
            "human_review_package_created": "true",
            "human_review_status": HUMAN_REVIEW_STATUS,
            "notes": "text_hash_normalization_only_no_geocoding_no_coordinate_inference",
        })
    out_path = out_path or os.path.join(DATASET_DIR, "v1um_recife_locality_normalization_registry.csv")
    write_csv(out_path, NORMALIZATION_COLUMNS, rows)
    print(f"[v1um locality normalizer] rows={len(rows)} -> {out_path}")
    return rows


def run_hazard_semantics_ranker(out_path=None, router_path=None):
    rows = []
    for idx, row in enumerate(joined_locality_candidates(router_path)):
        hazard_class, score, term_classes = hazard_class_for(row)
        rows.append({
            "hazard_rank_id": f"HAZRANK_{PROTOCOL_VERSION}_{idx:06d}",
            "event_id": row.get("event_id") or EVENT_ID,
            "candidate_row_id": row.get("candidate_row_id", ""),
            "row_hash": row.get("row_hash", ""),
            "hazard_class": hazard_class,
            "hazard_strength_score": str(score),
            "matched_terms_hash": sha256_text(term_classes, 16),
            "matched_term_classes": term_classes,
            "ambiguity_status": "AMBIGUOUS" if hazard_class == "CIVIL_DEFENSE_GENERIC" else "NOT_AMBIGUOUS",
            "review_priority_delta": review_priority_delta(hazard_class),
            "can_support_occurrence_review": bool_text(hazard_class not in {"NO_HAZARD_SIGNAL", "RISK_CONTEXT"}),
            "human_review_package_created": "true",
            "human_review_status": HUMAN_REVIEW_STATUS,
            "can_create_ground_reference": "false",
            "can_create_training_label": "false",
        })
    out_path = out_path or os.path.join(DATASET_DIR, "v1um_recife_hazard_semantics_rank_registry.csv")
    write_csv(out_path, HAZARD_COLUMNS, rows)
    print(f"[v1um hazard ranker] rows={len(rows)} -> {out_path}")
    return rows


def load_by_candidate(path):
    return {r.get("candidate_row_id", ""): r for r in load_csv(path)}


def match_by_row_hash():
    return {
        r.get("row_hash", ""): r
        for r in load_csv(os.path.join(DATASET_DIR, "v1uk_recife_event_window_match_registry.csv"))
    }


def batch_candidate_rows(sample_rows, hazard_by_id, norm_by_id, sample_type, limit=40):
    selected = [r for r in sample_rows if r.get("sample_type") == sample_type]
    if not selected:
        selected = sample_rows
    ordered = sorted(
        selected,
        key=lambda r: (
            -int_value(r.get("hazard_strength")),
            r.get("source_table", ""),
            r.get("candidate_row_id", ""),
        ),
    )
    result = []
    for row in ordered[:limit]:
        cid = row.get("candidate_row_id", "")
        result.append((row, hazard_by_id.get(cid, {}), norm_by_id.get(cid, {})))
    return result


def write_review_batch_md(path, batch_id, sample_type, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    hazard_counts = Counter(h.get("hazard_class", "") for _s, h, _n in rows)
    lines = [
        f"# v1um Recife Human Review Batch {batch_id}",
        "",
        "O pipeline REV-P organiza, redige, amostra, ranqueia e prepara os candidatos locality-only para Revisao Humana auditavel.",
        "",
        f"- event_id: {EVENT_ID}",
        f"- sample_type: {sample_type}",
        f"- candidate_count: {len(rows)}",
        f"- human_review_status: {HUMAN_REVIEW_STATUS}",
        "- operational_decision: NOT_OPERATIONAL",
        "- ground_reference_status: NOT_CREATED",
        "- label_status: NOT_CREATED",
        "- no_overlay_executed: true",
        "- no_coordinates_invented: true",
        "",
        "## Hazard Summary",
    ]
    for hazard, count in hazard_counts.most_common():
        lines.append(f"- {hazard or 'UNKNOWN'}: {count}")
    lines += [
        "",
        "## Redacted Candidate Rows",
        "",
        "| candidate_row_id | row_hash | date | hazard_class | locality_token | source_table | suggested_decision |",
        "|---|---|---|---|---|---|---|",
    ]
    for sample, hazard, norm in rows:
        lines.append(
            "| {cid} | {rh} | {date} | {haz} | {loc} | {src} | {dec} |".format(
                cid=sample.get("candidate_row_id", ""),
                rh=sample.get("row_hash", ""),
                date=sample.get("parsed_date", ""),
                haz=hazard.get("hazard_class", ""),
                loc=norm.get("normalized_locality_token", ""),
                src=sample.get("source_table", ""),
                dec="PREPARE_FOR_HUMAN_REVIEW_DO_NOT_PROMOTE",
            )
        )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def run_human_review_batch_builder(out_path=None, review_dir=None):
    sample_rows = load_csv(os.path.join(DATASET_DIR, "v1um_recife_locality_candidate_sample_registry.csv"))
    hazard_by_id = load_by_candidate(os.path.join(DATASET_DIR, "v1um_recife_hazard_semantics_rank_registry.csv"))
    norm_by_id = load_by_candidate(os.path.join(DATASET_DIR, "v1um_recife_locality_normalization_registry.csv"))
    matches = match_by_row_hash()
    review_dir = review_dir or REVIEW_DIR
    batch_defs = [
        ("BATCH_v1um_01", "high_confidence_sample", "v1um_recife_human_review_batch_01.md"),
        ("BATCH_v1um_02", "diversity_sample", "v1um_recife_human_review_batch_02.md"),
        ("BATCH_v1um_03", "edge_case_sample", "v1um_recife_human_review_batch_03.md"),
    ]
    rows = []
    options = "|".join([
        "ACCEPT_AS_LOCALITY_ONLY_OCCURRENCE_CANDIDATE",
        "REJECT_CONTEXT_ONLY",
        "REJECT_AMBIGUOUS",
        "REJECT_OUTSIDE_EVENT_SCOPE",
        "NEEDS_RAW_LOCAL_REVIEW",
        "DO_NOT_PROMOTE",
    ])
    for batch_id, sample_type, filename in batch_defs:
        selected = []
        for sample, hazard, norm in batch_candidate_rows(sample_rows, hazard_by_id, norm_by_id, sample_type):
            enriched = dict(sample)
            enriched["parsed_date"] = matches.get(sample.get("row_hash", ""), {}).get("parsed_date", "")
            selected.append((enriched, hazard, norm))
        batch_file = artifact_path(os.path.join(review_dir, filename))
        write_review_batch_md(batch_file, batch_id, sample_type, selected)
        hazard_counts = Counter(h.get("hazard_class", "") for _s, h, _n in selected)
        ambiguous = hazard_counts.get("CIVIL_DEFENSE_GENERIC", 0) + hazard_counts.get("AMBIGUOUS", 0)
        rows.append({
            "batch_id": batch_id,
            "event_id": EVENT_ID,
            "batch_file": artifact_path(os.path.join("docs/review_packages/protocolo_c", filename)),
            "sample_type": sample_type,
            "candidate_count": str(len(selected)),
            "high_confidence_count": str(sum(1 for s, _h, _n in selected if s.get("sample_type") == "high_confidence_sample")),
            "locality_only_count": str(len(selected)),
            "ambiguous_count": str(ambiguous),
            "human_review_task": "Revisao Humana estruturada pelo pipeline sobre evidencia locality-only redigida.",
            "human_review_decision_options": options,
            "human_review_package_created": "true",
            "human_review_status": HUMAN_REVIEW_STATUS,
            "supervisor_review_completed": "false",
            "can_create_ground_reference": "false",
            "can_create_training_label": "false",
        })
    out_path = out_path or os.path.join(DATASET_DIR, "v1um_recife_human_review_batch_registry.csv")
    write_csv(out_path, BATCH_COLUMNS, rows)
    print(f"[v1um human review batches] rows={len(rows)} -> {out_path}")
    return rows


def run_redacted_evidence_packager(out_path=None):
    samples = load_csv(os.path.join(DATASET_DIR, "v1um_recife_locality_candidate_sample_registry.csv"))
    hazards = load_by_candidate(os.path.join(DATASET_DIR, "v1um_recife_hazard_semantics_rank_registry.csv"))
    norms = load_by_candidate(os.path.join(DATASET_DIR, "v1um_recife_locality_normalization_registry.csv"))
    matches = match_by_row_hash()
    rows = []
    for idx, sample in enumerate(samples):
        cid = sample.get("candidate_row_id", "")
        hazard = hazards.get(cid, {})
        norm = norms.get(cid, {})
        match = matches.get(sample.get("row_hash", ""), {})
        rows.append({
            "evidence_package_id": f"EVIDPKG_{PROTOCOL_VERSION}_{idx:06d}",
            "event_id": sample.get("event_id") or EVENT_ID,
            "candidate_row_id": cid,
            "row_hash": sample.get("row_hash", ""),
            "parsed_date": match.get("parsed_date", ""),
            "hazard_class": hazard.get("hazard_class", ""),
            "hazard_strength_score": hazard.get("hazard_strength_score", ""),
            "normalized_locality_token": norm.get("normalized_locality_token", ""),
            "has_address": sample.get("has_address", "false"),
            "has_neighborhood": sample.get("has_neighborhood", "false"),
            "address_hash": sample.get("address_hash", ""),
            "locality_hash": sample.get("locality_hash", ""),
            "source_table": sample.get("source_table", ""),
            "inclusion_reason": sample.get("inclusion_reason", ""),
            "public_redaction_status": "HASHES_FLAGS_ONLY_NO_LITERAL_SENSITIVE_VALUES",
            "human_review_package_created": "true",
            "human_review_status": HUMAN_REVIEW_STATUS,
            "ground_truth_operational": "false",
            "can_create_ground_reference": "false",
            "can_create_training_label": "false",
        })
    out_path = out_path or os.path.join(DATASET_DIR, "v1um_recife_redacted_evidence_package_registry.csv")
    write_csv(out_path, EVIDENCE_COLUMNS, rows)
    print(f"[v1um redacted evidence] rows={len(rows)} -> {out_path}")
    return rows


def run_neighborhood_signal_aggregator(out_path=None):
    evidence = load_csv(os.path.join(DATASET_DIR, "v1um_recife_redacted_evidence_package_registry.csv"))
    samples = load_by_candidate(os.path.join(DATASET_DIR, "v1um_recife_locality_candidate_sample_registry.csv"))
    groups = defaultdict(list)
    for row in evidence:
        groups[(row.get("normalized_locality_token", ""), row.get("parsed_date", ""))].append(row)
    rows = []
    for idx, ((token, date), members) in enumerate(sorted(groups.items())):
        hazards = Counter(r.get("hazard_class", "") for r in members)
        source_tables = {r.get("source_table", "") for r in members}
        priority = "1" if hazards.get("FLOOD_STRONG", 0) else "2" if hazards.get("RAIN_IMPACT", 0) or hazards.get("LANDSLIDE_OR_SLOPE", 0) else "3"
        rows.append({
            "aggregation_id": f"AGG_{PROTOCOL_VERSION}_{idx:06d}",
            "event_id": EVENT_ID,
            "normalized_locality_token": token,
            "date": date,
            "candidate_count": str(len(members)),
            "flood_strong_count": str(hazards.get("FLOOD_STRONG", 0)),
            "rain_impact_count": str(hazards.get("RAIN_IMPACT", 0)),
            "ambiguous_count": str(hazards.get("CIVIL_DEFENSE_GENERIC", 0) + hazards.get("AMBIGUOUS", 0)),
            "source_table_count": str(len(source_tables)),
            "has_address_count": str(sum(1 for r in members if r.get("has_address") == "true")),
            "review_priority": priority,
            "can_support_contextual_review": "true",
            "can_support_overlay": "false",
            "can_create_ground_reference": "false",
        })
    out_path = out_path or os.path.join(DATASET_DIR, "v1um_recife_neighborhood_signal_aggregation.csv")
    write_csv(out_path, AGGREGATION_COLUMNS, rows)
    print(f"[v1um aggregation] rows={len(rows)} -> {out_path}")
    return rows


def suggested_decision_for(evidence):
    hazard = evidence.get("hazard_class", "")
    if hazard in {"FLOOD_STRONG", "RAIN_IMPACT", "LANDSLIDE_OR_SLOPE"}:
        return "ACCEPT_AS_LOCALITY_ONLY_OCCURRENCE_CANDIDATE"
    if hazard == "CIVIL_DEFENSE_GENERIC":
        return "NEEDS_RAW_LOCAL_REVIEW"
    return "REJECT_AMBIGUOUS"


def run_human_review_decision_matrix_builder(out_path=None):
    evidence = load_csv(os.path.join(DATASET_DIR, "v1um_recife_redacted_evidence_package_registry.csv"))
    rows = []
    question = "A linha tem data na janela, sinal textual de hazard e localidade textual suficiente para Revisao Humana sem overlay?"
    for idx, row in enumerate(evidence):
        rows.append({
            "decision_id": f"HRDEC_{PROTOCOL_VERSION}_{idx:06d}",
            "event_id": row.get("event_id") or EVENT_ID,
            "candidate_row_id": row.get("candidate_row_id", ""),
            "row_hash": row.get("row_hash", ""),
            "human_review_question": question,
            "evidence_signal": (
                f"hazard_class={row.get('hazard_class', '')};"
                f"locality_token={row.get('normalized_locality_token', '')};"
                f"has_address={row.get('has_address', '')}"
            ),
            "suggested_decision": suggested_decision_for(row),
            "operational_decision": "NOT_OPERATIONAL",
            "ground_reference_status": "NOT_CREATED",
            "label_status": "NOT_CREATED",
            "can_create_ground_reference": "false",
            "can_create_training_label": "false",
        })
    out_path = out_path or os.path.join(DATASET_DIR, "v1um_recife_human_review_decision_matrix.csv")
    write_csv(out_path, DECISION_COLUMNS, rows)
    print(f"[v1um decision matrix] rows={len(rows)} -> {out_path}")
    return rows


def run_non_overlay_readiness_matrix(out_path=None):
    evidence = load_csv(os.path.join(DATASET_DIR, "v1um_recife_redacted_evidence_package_registry.csv"))
    rows = []
    for idx, row in enumerate(evidence):
        hazard = row.get("hazard_class", "")
        has_locality = bool(row.get("normalized_locality_token", ""))
        if not has_locality:
            status, blocker = "BLOCKED_NO_LOCALITY", "no_locality"
        elif hazard in {"NO_HAZARD_SIGNAL", "AMBIGUOUS"}:
            status, blocker = "BLOCKED_AMBIGUOUS_HAZARD", "hazard_ambiguity"
        elif hazard == "CIVIL_DEFENSE_GENERIC":
            status, blocker = "NEEDS_RAW_LOCAL_REVIEW", "generic_hazard_requires_human_review"
        else:
            status, blocker = "READY_FOR_HUMAN_REVIEW_LOCALITY_ONLY", "locality_only_no_overlay"
        rows.append({
            "readiness_id": f"NONOVERLAY_{PROTOCOL_VERSION}_{idx:06d}",
            "event_id": row.get("event_id") or EVENT_ID,
            "candidate_row_id": row.get("candidate_row_id", ""),
            "row_hash": row.get("row_hash", ""),
            "non_overlay_readiness_status": status,
            "human_review_status": HUMAN_REVIEW_STATUS,
            "has_event_window_match": "true",
            "has_hazard_signal": bool_text(hazard != "NO_HAZARD_SIGNAL"),
            "has_locality_text": bool_text(has_locality),
            "has_coordinates": "false",
            "can_support_human_review": bool_text(status != "BLOCKED_NO_LOCALITY"),
            "can_support_overlay": "false",
            "ground_truth_operational": "false",
            "can_create_ground_reference": "false",
            "can_create_training_label": "false",
            "blocker": blocker,
            "required_next_action": "HUMAN_REVIEW" if status.startswith("READY") or status.startswith("NEEDS") else "DO_NOT_PROMOTE",
        })
    out_path = out_path or os.path.join(DATASET_DIR, "v1um_recife_non_overlay_readiness_matrix.csv")
    write_csv(out_path, READINESS_COLUMNS, rows)
    print(f"[v1um non-overlay readiness] rows={len(rows)} -> {out_path}")
    return rows


def run_ground_reference_blocker_matrix(out_path=None):
    readiness = load_csv(os.path.join(DATASET_DIR, "v1um_recife_non_overlay_readiness_matrix.csv"))
    hazards = load_csv(os.path.join(DATASET_DIR, "v1um_recife_hazard_semantics_rank_registry.csv"))
    counts = {
        "locality_only_no_geometry": len(readiness),
        "no_coordinates": len(readiness),
        "no_overlay": len(readiness),
        "no_supervisor_review": len(readiness),
        "sensitive_review_required": len(readiness),
        "hazard_ambiguity": sum(1 for r in hazards if r.get("ambiguity_status") == "AMBIGUOUS"),
        "label_forbidden": len(readiness),
        "patch_truth_forbidden": len(readiness),
    }
    rows = []
    for idx, (blocker, count) in enumerate(counts.items()):
        rows.append({
            "blocker_id": f"BLOCK_{PROTOCOL_VERSION}_{idx:04d}",
            "event_id": EVENT_ID,
            "blocker": blocker,
            "status": "ACTIVE" if count else "INACTIVE",
            "evidence_count": str(count),
            "ground_truth_operational": "false",
            "can_create_ground_reference": "false",
            "can_create_training_label": "false",
            "human_review_status": HUMAN_REVIEW_STATUS,
            "notes": "Revisao Humana locality-only sem overlay e sem promocao operacional.",
        })
    out_path = out_path or os.path.join(DATASET_DIR, "v1um_recife_ground_reference_blocker_matrix.csv")
    write_csv(out_path, BLOCKER_COLUMNS, rows)
    return rows


def write_policy_configs():
    os.makedirs(CONFIG_DIR, exist_ok=True)
    policies = {
        "v1um_recife_review_sampling_policy.yaml": [
            "protocol_version: v1um",
            "event_id: REC_2022_05_24_30",
            "human_review_status: PREPARED_NOT_OPERATIONAL",
            "sampling:",
            "  deterministic: true",
            "  include_sample_types:",
            "    - high_confidence_sample",
            "    - diversity_sample",
            "    - edge_case_sample",
            "    - random_audit_sample",
            "    - full_queue_summary",
        ],
        "v1um_recife_locality_normalization_policy.yaml": [
            "protocol_version: v1um",
            "normalization_scope: hashed_text_only",
            "geocoding_allowed: false",
            "centroid_allowed: false",
            "coordinate_inference_allowed: false",
        ],
        "v1um_recife_hazard_semantics_policy.yaml": [
            "protocol_version: v1um",
            "external_model_allowed: false",
            "ranking_method: rule_based_flags_from_v1uk",
            "classes: [FLOOD_STRONG, FLOOD_MODERATE, RAIN_IMPACT, LANDSLIDE_OR_SLOPE, CIVIL_DEFENSE_GENERIC, RISK_CONTEXT, AMBIGUOUS, NO_HAZARD_SIGNAL]",
        ],
        "v1um_recife_redaction_policy.yaml": [
            "protocol_version: v1um",
            "public_outputs: hashes_flags_counts_only",
            "literal_address_allowed: false",
            "personal_data_allowed: false",
            "raw_text_allowed: false",
        ],
        "v1um_recife_human_review_batch_policy.yaml": [
            "protocol_version: v1um",
            "batch_count_minimum: 3",
            "human_review_package_created: true",
            "human_review_status: PREPARED_NOT_OPERATIONAL",
            "supervisor_review_completed: false",
        ],
        "v1um_recife_non_overlay_decision_policy.yaml": [
            "protocol_version: v1um",
            "overlay_allowed: false",
            "ground_truth_operational: false",
            "can_create_ground_reference: false",
            "can_create_training_label: false",
            "max_status: RECIFE_LOCALITY_ONLY_HUMAN_REVIEW_CANDIDATE",
        ],
    }
    for name, lines in policies.items():
        with open(os.path.join(CONFIG_DIR, name), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")


def run_completion_report():
    write_policy_configs()
    run_ground_reference_blocker_matrix()
    evidence = load_csv(os.path.join(DATASET_DIR, "v1um_recife_redacted_evidence_package_registry.csv"))
    samples = load_csv(os.path.join(DATASET_DIR, "v1um_recife_locality_candidate_sample_registry.csv"))
    batches = load_csv(os.path.join(DATASET_DIR, "v1um_recife_human_review_batch_registry.csv"))
    aggregation = load_csv(os.path.join(DATASET_DIR, "v1um_recife_neighborhood_signal_aggregation.csv"))
    hazards = Counter(r.get("hazard_class", "") for r in evidence)
    sample_counts = Counter(r.get("sample_type", "") for r in samples)
    top_localities = Counter(r.get("normalized_locality_token", "") for r in evidence).most_common(10)
    ready_non_overlay = sum(
        1 for r in load_csv(os.path.join(DATASET_DIR, "v1um_recife_non_overlay_readiness_matrix.csv"))
        if r.get("non_overlay_readiness_status") == "READY_FOR_HUMAN_REVIEW_LOCALITY_ONLY"
    )
    strong_context = len(evidence) >= 1000 and bool(batches)
    next_action = "v1un - Human Review Evidence Consolidation Registry"
    action_rows = [{
        "action_id": "ACT_v1um_0000",
        "event_id": EVENT_ID,
        "action_type": "HUMAN_REVIEW_EVIDENCE_CONSOLIDATION",
        "priority": "1",
        "description": next_action,
        "target": "Recife locality-only Human Review packages",
        "status": "PENDING",
        "notes": "Nao implementar v1un nesta etapa.",
    }]
    write_csv(os.path.join(DATASET_DIR, "v1um_next_actions_registry.csv"), NEXT_ACTION_COLUMNS, action_rows)
    manifest = []
    for idx, path in enumerate(V1UM_ARTIFACTS):
        exists = os.path.exists(path)
        manifest.append({
            "artifact_id": f"ART_{PROTOCOL_VERSION}_{idx:04d}",
            "artifact_path": artifact_path(path),
            "artifact_type": "config" if path.startswith("configs/") else "doc" if path.startswith("docs/") else "dataset",
            "protocol_version": PROTOCOL_VERSION,
            "sha256_prefix": sha256_file(path)[:16] if exists else "MISSING",
            "file_size_bytes": str(os.path.getsize(path) if exists else 0),
            "is_versionable": bool_text(exists),
            "reason": "Safe Human Review metadata artifact" if exists else "File not found",
        })
    write_csv(os.path.join(DATASET_DIR, "v1um_versionable_artifacts_manifest.csv"), MANIFEST_COLUMNS, manifest)
    os.makedirs(DOCS_DIR, exist_ok=True)
    methodology = [
        "# Protocolo C v1um - Recife Human Review Locality-Only",
        "",
        "O pipeline REV-P organiza, redige, amostra, ranqueia e prepara os candidatos locality-only para Revisao Humana auditavel.",
        "",
        "## Escopo",
        f"- event_id: {EVENT_ID}",
        f"- status_maximo: {MAX_STATUS}",
        "- A Revisao Humana e uma etapa metodologica de organizacao e avaliacao de evidencias.",
        "- A etapa produz pacotes, filas, amostras, agregacoes e matrizes sem overlay.",
        "",
        "## Guardrails",
        "- ground_truth_operational=false",
        "- can_create_ground_reference=false",
        "- can_create_training_label=false",
        "- can_reopen_protocol_b=false",
        "- dino_usage=SUPPORT_ONLY",
        "- no_overlay_executed=true",
        "- no_coordinates_invented=true",
        "- human_review_package_created=true",
        "- human_review_queue_ready=true",
        f"- human_review_status={HUMAN_REVIEW_STATUS}",
        "- supervisor_review_completed=false",
    ]
    report = [
        "# Relatorio v1um - Recife Human Review Locality-Only",
        "",
        "O REV-P registra a preparacao de Revisao Humana sobre evidencia locality-only oficial e redigida.",
        "",
        "## Resultados",
        f"- candidatos_locality_only_processados: {len(evidence)}",
        f"- amostras_geradas: {len(samples)}",
        f"- batches_human_review_criados: {len(batches)}",
        f"- agregacoes_por_localidade_data: {len(aggregation)}",
        f"- candidatos_ready_non_overlay: {ready_non_overlay}",
        f"- evidencia_contextual_forte: {bool_text(strong_context)}",
        "",
        "## Hazards Dominantes",
    ]
    report += [f"- {hazard}: {count}" for hazard, count in hazards.most_common()]
    report += [
        "",
        "## Amostras",
        *(f"- {sample_type}: {count}" for sample_type, count in sample_counts.most_common()),
        "",
        "## Localidades Textuais Redigidas Mais Frequentes",
        *(f"- {token}: {count}" for token, count in top_localities),
        "",
        "## Bloqueios",
        "- overlay: bloqueado por evidencia locality-only e ausencia de coordenada observada.",
        "- referencia operacional: nao criada por ausencia de coordenada, ausencia de overlay e Revisao Humana pendente.",
        "- treinamento supervisionado: nao liberado.",
        "",
        "## Proxima Etapa",
        f"- {next_action}",
    ]
    status = [
        "# Status Atual - Protocolo C v1um",
        "",
        f"event_id={EVENT_ID}",
        f"locality_only_candidates_processed={len(evidence)}",
        f"sample_rows={len(samples)}",
        f"human_review_batches={len(batches)}",
        f"aggregation_rows={len(aggregation)}",
        f"dominant_hazard={hazards.most_common(1)[0][0] if hazards else 'none'}",
        f"contextual_evidence_strong={bool_text(strong_context)}",
        "coordinates_available=false",
        "overlay_blocked=true",
        "ground_truth_operational=false",
        "can_create_ground_reference=false",
        "can_create_training_label=false",
        "can_reopen_protocol_b=false",
        "dino_usage=SUPPORT_ONLY",
        "no_overlay_executed=true",
        "no_coordinates_invented=true",
        "human_review_package_created=true",
        "human_review_queue_ready=true",
        f"human_review_status={HUMAN_REVIEW_STATUS}",
        "supervisor_review_completed=false",
        f"max_status={MAX_STATUS}",
        f"next_action={next_action}",
    ]
    with open(os.path.join(DOCS_DIR, "protocolo_c_v1um_recife_human_review_locality_only.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(methodology) + "\n")
    with open(os.path.join(DOCS_DIR, "protocolo_c_relatorio_v1um_recife_human_review_locality_only.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")
    with open(os.path.join(DOCS_DIR, "protocolo_c_status_atual_v1um.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(status) + "\n")
    print(f"[v1um completion] next_action={next_action}")
    return {
        "locality_only_candidates_processed": len(evidence),
        "sample_rows": len(samples),
        "human_review_batches": len(batches),
        "aggregation_rows": len(aggregation),
        "dominant_hazard": hazards.most_common(1)[0][0] if hazards else "none",
        "contextual_evidence_strong": strong_context,
        "next_action": next_action,
    }


def simple_main(fn):
    parser = argparse.ArgumentParser()
    parser.parse_args()
    fn()
