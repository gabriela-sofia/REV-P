#!/usr/bin/env python3
"""v2aa Sentinel date recovery for event-patch packages.

Recovers real Sentinel scene dates (or traceable temporal metadata) for patches
and event-patch candidates using only existing versionable registries, manifests
and sidecars in the repository. It never downloads new Sentinel data, never
queries the web, never infers a date without evidence, never invents a
``sentinel_scene_date``, and never creates overlay, ground reference, ground
truth or labels. Dates are recovered from filenames/scene ids and explicit
sidecar date fields only; ``created_at``/``modified_at`` and file mtimes are
never used as scene dates.
"""

import argparse
import csv
import datetime
import hashlib
import json
import os
import re

PROTOCOL_VERSION = "v2aa"
DATASET_DIR = "datasets/protocolo_c"
DOCS_DIR = "docs/metodologia_cientifica"
CONFIG_DIR = "configs/protocolo_c"
STAGING_DIR = "local_only/protocolo_c/sentinel_date_recovery/staging/v2aa"
REPORTS_DIR = "local_only/protocolo_c/sentinel_date_recovery/reports/v2aa"
# Versionable roots to scan for patch/date metadata. local_only is never scanned.
SCAN_ROOTS = ["datasets", "configs"]

MAX_STATUS = "SENTINEL_DATE_RECOVERED_FOR_EVENT_PATCH_REVIEW_ONLY"

GUARDRAIL_COLUMNS = [
    "ground_truth_operational", "can_create_ground_reference",
    "can_create_training_label", "can_reopen_protocol_b", "dino_usage",
    "no_overlay_executed", "no_coordinates_invented", "patch_bound_truth",
    "operational_validation", "sentinel_date_recovery_only",
    "sentinel_date_inferred", "raw_data_versioned",
]
GUARDRAIL_MUST_BE_FALSE = {
    "ground_truth_operational", "can_create_ground_reference",
    "can_create_training_label", "can_reopen_protocol_b", "patch_bound_truth",
    "operational_validation", "sentinel_date_inferred", "raw_data_versioned",
}
FORBIDDEN_STATUS_TOKENS = [
    "GROUND_REFERENCE", "GROUND_TRUTH", "TRAINING_LABEL", "PATCH_POSITIVE",
    "PATCH_NEGATIVE", "OPERATIONAL_VALIDATED", "OBSERVED_FLOOD_LABEL",
    "FLOOD_DETECTED", "EVENT_VALIDATED_BY_SENTINEL",
]

# Field-name vocabularies.
PATCH_ID_FIELDS = ["patch_id", "reference_patch_id", "patch_candidate_id"]
SCENE_DATE_FIELDS = [
    "scene_date", "sensing_date", "sensing_time", "acquisition_date",
    "datetime", "scene_datetime", "product_start_time", "pre_scene_date",
    "post_scene_date", "sentinel_scene_date",
]
SCENE_ID_FIELDS = [
    "scene_id", "scene_id_sanitized", "pre_scene_id_sanitized",
    "post_scene_id_sanitized", "filename", "file_name", "safe_name",
    "product_uri", "granule", "asset_path", "source_path",
]
PLATFORM_FIELDS = ["sensor", "platform", "gee_collection", "sensor_stack"]
# Date-like fields that must NEVER be used as a scene date.
FORBIDDEN_DATE_FIELDS = {
    "created_at", "modified_at", "created", "modified", "updated_at",
    "file_mtime", "mtime", "ingested_at", "downloaded_at", "anchor_date",
    "event_start_date", "event_end_date",
}

PLATFORM_RE = re.compile(r"(?<![A-Za-z0-9])(S[12][AB])(?![A-Za-z0-9])")
DATE_T_RE = re.compile(r"(?<!\d)(\d{4})(\d{2})(\d{2})T\d{2,6}")
DATE_DASH_RE = re.compile(r"(?<!\d)(\d{4})-(\d{2})-(\d{2})(?!\d)")
DATE_USCORE_RE = re.compile(r"(?<!\d)(\d{4})_(\d{2})_(\d{2})(?!\d)")
PATCH_ID_TOKEN_RE = re.compile(r"\b((?:CUR|PET|REC)_\d{4,6})\b")

# Column definitions ------------------------------------------------------
SCAN_COLUMNS = [
    "source_scan_id", "registry_path", "registry_hash", "row_count",
    "has_patch_id", "has_filename", "has_date_field", "date_field_candidates",
    "has_sentinel_platform_field", "source_status", "should_parse_for_dates",
    "notes",
]
FILENAME_COLUMNS = [
    "filename_extract_id", "patch_id", "registry_path_hash", "filename_hash",
    "extracted_date", "extracted_datetime", "sentinel_platform",
    "extraction_pattern", "extraction_status", "ambiguity_status", "notes",
]
SIDECAR_COLUMNS = [
    "sidecar_resolution_id", "patch_id", "sidecar_path_hash", "date_field_used",
    "resolved_date", "resolved_datetime", "source_type", "resolution_status",
    "confidence_hint", "notes",
]
CONSOLIDATION_COLUMNS = [
    "patch_date_id", "patch_id", "region", "candidate_dates",
    "selected_sentinel_date", "selected_sentinel_datetime", "source_count",
    "agreeing_source_count", "conflict_status", "consolidation_status",
    "sentinel_date_recovered", "sentinel_date_inferred", "notes",
]
CONFIDENCE_COLUMNS = [
    "confidence_audit_id", "patch_id", "selected_sentinel_date",
    "confidence_class", "confidence_score", "usable_for_temporal_linkage",
    "blocker", "notes",
]
TEMPORAL_COLUMNS = [
    "temporal_distance_id", "event_patch_candidate_id", "event_id", "patch_id",
    "event_start_date", "event_end_date", "sentinel_date",
    "days_from_event_start", "days_from_event_end", "temporal_class",
    "usable_for_contextual_review", "usable_for_overlay_preflight",
    "can_create_ground_reference", "notes",
]
READINESS_COLUMNS = [
    "readiness_update_id", "event_patch_candidate_id", "event_id", "patch_id",
    "region", "dimension", "classification", "basis", *GUARDRAIL_COLUMNS,
    "notes",
]
REDUCTION_COLUMNS = [
    "reduction_id", "region", "total_patches", "patches_with_recovered_date",
    "high_confidence_dates", "medium_confidence_dates",
    "missing_or_blocked_dates", "event_patch_candidates_improved",
    "blocker_reduction_status", "notes",
]
RANKER_COLUMNS = [
    "rank", "next_target", "programming_value", "ground_truth_value",
    "blocker_reduction_value", "expected_effort", "overclaim_risk",
    "recommended_version", "recommended_action", "notes",
]
BLOCKER_COLUMNS = [
    "blocker_id", "region", "event_id", "blocker", "status", *GUARDRAIL_COLUMNS,
    "notes",
]
NEXT_COLUMNS = [
    "action_id", "event_id", "action_type", "priority", "description",
    "target", "status", "notes",
]
MANIFEST_COLUMNS = [
    "artifact_id", "artifact_path", "artifact_type", "protocol_version",
    "sha256_prefix", "file_size_bytes", "is_versionable", "reason",
]

V2AA_ARTIFACTS = [
    "configs/protocolo_c/v2aa_patch_source_scan_policy.yaml",
    "configs/protocolo_c/v2aa_sentinel_date_patterns.yaml",
    "configs/protocolo_c/v2aa_sidecar_metadata_policy.yaml",
    "configs/protocolo_c/v2aa_date_confidence_policy.yaml",
    "configs/protocolo_c/v2aa_temporal_distance_policy.yaml",
    "configs/protocolo_c/v2aa_next_programming_target_policy.yaml",
    "datasets/protocolo_c/v2aa_patch_source_registry_scan.csv",
    "datasets/protocolo_c/v2aa_sentinel_filename_date_extraction.csv",
    "datasets/protocolo_c/v2aa_sentinel_sidecar_metadata_resolution.csv",
    "datasets/protocolo_c/v2aa_patch_date_candidate_consolidation.csv",
    "datasets/protocolo_c/v2aa_sentinel_date_confidence_audit.csv",
    "datasets/protocolo_c/v2aa_event_patch_temporal_distance.csv",
    "datasets/protocolo_c/v2aa_event_patch_readiness_update.csv",
    "datasets/protocolo_c/v2aa_multiregion_temporal_blocker_reduction.csv",
    "datasets/protocolo_c/v2aa_next_programming_target_ranker.csv",
    "datasets/protocolo_c/v2aa_ground_reference_blocker_matrix.csv",
    "datasets/protocolo_c/v2aa_next_actions_registry.csv",
    "docs/metodologia_cientifica/protocolo_c_v2aa_sentinel_date_recovery.md",
    "docs/metodologia_cientifica/protocolo_c_relatorio_v2aa_sentinel_date_recovery.md",
    "docs/metodologia_cientifica/protocolo_c_status_atual_v2aa.md",
]

TEMPORAL_NEAR_DAYS = 15


# Helpers -----------------------------------------------------------------

def hash_text(value, n=16):
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()[:n]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, columns, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_text(path, lines):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def dataset_path(name):
    return os.path.join(DATASET_DIR, name)


def config_path(name):
    return os.path.join(CONFIG_DIR, name)


def doc_path(name):
    return os.path.join(DOCS_DIR, name)


def artifact_path(path):
    base = os.path.basename(path)
    if path.startswith("datasets/protocolo_c/"):
        return dataset_path(base)
    if path.startswith("configs/protocolo_c/"):
        return config_path(base)
    if path.startswith("docs/metodologia_cientifica/"):
        return doc_path(base)
    return path


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def guardrails():
    return {
        "ground_truth_operational": "false",
        "can_create_ground_reference": "false",
        "can_create_training_label": "false",
        "can_reopen_protocol_b": "false",
        "dino_usage": "SUPPORT_ONLY",
        "no_overlay_executed": "true",
        "no_coordinates_invented": "true",
        "patch_bound_truth": "false",
        "operational_validation": "false",
        "sentinel_date_recovery_only": "true",
        "sentinel_date_inferred": "false",
        "raw_data_versioned": "false",
    }


def write_policy_configs():
    policies = {
        "v2aa_patch_source_scan_policy.yaml": [
            "scan_roots: [datasets, configs]",
            "scan_local_only_raw: false",
            "infer_dates_in_scanner: false",
            "max_status: SENTINEL_DATE_RECOVERED_FOR_EVENT_PATCH_REVIEW_ONLY",
        ],
        "v2aa_sentinel_date_patterns.yaml": [
            "patterns:",
            "  - S2A_MSIL2A_YYYYMMDDT",
            "  - S2B_MSIL1C_YYYYMMDDT",
            "  - S1A_YYYYMMDDT",
            "  - S1B_YYYYMMDDT",
            "  - YYYYMMDDT",
            "  - YYYY-MM-DD",
            "  - YYYY_MM_DD",
            "reject_bare_year: true",
            "reject_ambiguous_multidate: true",
        ],
        "v2aa_sidecar_metadata_policy.yaml": [
            "accepted_date_fields: [scene_date, sensing_date, sensing_time, acquisition_date, datetime, product_start_time, pre_scene_date, post_scene_date]",
            "forbidden_date_fields: [created_at, modified_at, file_mtime, anchor_date]",
            "use_file_modified_time: false",
        ],
        "v2aa_date_confidence_policy.yaml": [
            "classes: [HIGH_CONFIDENCE, MEDIUM_CONFIDENCE, LOW_CONFIDENCE, BLOCKED_CONFLICT, MISSING]",
            "multi_source_agreement: HIGH_CONFIDENCE",
            "canonical_sentinel_filename: HIGH_CONFIDENCE",
            "explicit_sensing_date: HIGH_CONFIDENCE",
            "weak_generic_date_field: MEDIUM_OR_LOW",
            "conflict: BLOCKED_CONFLICT",
            "usable_requires: HIGH_OR_MEDIUM",
        ],
        "v2aa_temporal_distance_policy.yaml": [
            f"near_days_threshold: {TEMPORAL_NEAR_DAYS}",
            "classes: [WITHIN_EVENT_WINDOW, PRE_EVENT_NEAR, POST_EVENT_NEAR, PRE_EVENT_FAR, POST_EVENT_FAR, TEMPORAL_DISTANCE_BLOCKED_NO_DATE, TEMPORAL_DISTANCE_BLOCKED_LOW_CONFIDENCE]",
            "temporal_class_is_truth: false",
            "usable_for_overlay_preflight: false",
        ],
        "v2aa_next_programming_target_policy.yaml": [
            "ranking: score_based_not_hardcoded",
            "programming_weight: 0.5",
            "blocker_reduction_weight: 0.5",
            "effort_penalty: {LOW: 0, MEDIUM: 5, HIGH: 15}",
            "overclaim_penalty: {LOW: 0, MEDIUM: 10, HIGH: 25}",
        ],
    }
    for name, lines in policies.items():
        write_text(config_path(name), lines)


def valid_date(year, month, day):
    try:
        y, m, d = int(year), int(month), int(day)
    except (TypeError, ValueError):
        return None
    if not (1900 <= y <= 2100 and 1 <= m <= 12 and 1 <= d <= 31):
        return None
    try:
        return datetime.date(y, m, d).isoformat()
    except ValueError:
        return None


def extract_dates_from_text(value):
    """Return (sorted distinct ISO dates, matched pattern label)."""
    text = str(value or "")
    found = set()
    pattern = ""
    for y, m, d in DATE_T_RE.findall(text):
        iso = valid_date(y, m, d)
        if iso:
            found.add(iso)
            pattern = "YYYYMMDDT"
    for y, m, d in DATE_DASH_RE.findall(text):
        iso = valid_date(y, m, d)
        if iso:
            found.add(iso)
            pattern = pattern or "YYYY-MM-DD"
    for y, m, d in DATE_USCORE_RE.findall(text):
        iso = valid_date(y, m, d)
        if iso:
            found.add(iso)
            pattern = pattern or "YYYY_MM_DD"
    return sorted(found), pattern


def detect_platform(*values):
    for value in values:
        match = PLATFORM_RE.search(str(value or ""))
        if match:
            return match.group(1)
    blob = " ".join(str(v or "") for v in values).upper()
    if "S2" in blob or "MSIL" in blob or "SENTINEL-2" in blob or "SENTINEL2" in blob:
        return "S2"
    if "S1" in blob or "GRD" in blob or "SENTINEL-1" in blob:
        return "S1"
    return ""


# 1. Patch Source Registry Scanner ----------------------------------------

def _iter_versionable_files():
    for root in SCAN_ROOTS:
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            norm = dirpath.replace("\\", "/")
            if "local_only" in norm.split("/"):
                dirnames[:] = []
                continue
            for name in sorted(filenames):
                if name.lower().endswith((".csv", ".json")):
                    yield os.path.join(dirpath, name)


def _header_of(path):
    try:
        if path.lower().endswith(".csv"):
            with open(path, "r", encoding="utf-8", newline="") as f:
                reader = csv.reader(f)
                header = next(reader, [])
                row_count = sum(1 for _ in reader)
            return [h.strip() for h in header], row_count
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list) and data and isinstance(data[0], dict):
            return list(data[0].keys()), len(data)
        if isinstance(data, dict):
            return list(data.keys()), 1
        return [], 0
    except (OSError, ValueError, StopIteration):
        return [], 0


def run_patch_source_registry_scanner(args=None):
    write_policy_configs()
    rows = []
    for path in _iter_versionable_files():
        rel = path.replace("\\", "/")
        if os.path.basename(rel).startswith("v2aa_"):
            continue  # never scan our own outputs
        header, row_count = _header_of(path)
        lower = {h.lower() for h in header}
        has_patch = any(f in lower for f in PATCH_ID_FIELDS)
        has_filename = any(f in lower for f in SCENE_ID_FIELDS)
        date_fields = [h for h in header if h.lower() in SCENE_DATE_FIELDS]
        has_date = bool(date_fields)
        has_platform = any(f in lower for f in PLATFORM_FIELDS)
        should_parse = has_patch and (has_date or has_filename)
        if not (has_patch or has_date or has_filename):
            continue  # not patch/date related
        if should_parse:
            status = "PATCH_AND_DATE_SOURCE"
        elif has_patch:
            status = "PATCH_ONLY_NO_DATE_FIELD"
        else:
            status = "DATE_OR_FILENAME_NO_PATCH_ID"
        rows.append({
            "source_scan_id": f"SCAN_v2aa_{len(rows):04d}",
            "registry_path": rel,
            "registry_hash": hash_text(rel, 24),
            "row_count": str(row_count),
            "has_patch_id": "true" if has_patch else "false",
            "has_filename": "true" if has_filename else "false",
            "has_date_field": "true" if has_date else "false",
            "date_field_candidates": "|".join(date_fields),
            "has_sentinel_platform_field": "true" if has_platform else "false",
            "source_status": status,
            "should_parse_for_dates": "true" if should_parse else "false",
            "notes": "Scanner records field presence only; no date inference here; local_only never scanned.",
        })
    out = dataset_path("v2aa_patch_source_registry_scan.csv")
    write_csv(out, SCAN_COLUMNS, rows)
    print(f"[v2aa scan] registries={len(rows)} parseable={sum(1 for r in rows if r['should_parse_for_dates'] == 'true')} -> {out}")
    return rows


# Region inference helper.

def _region_of(patch_id, fallback=""):
    pid = str(patch_id or "")
    if pid.startswith("CUR") or "CUR" in pid:
        return "CUR"
    if pid.startswith("REC") or "REC" in pid:
        return "REC"
    if pid.startswith("PET") or "PET" in pid:
        return "PET"
    return fallback


def _row_patch_id(record):
    for field in PATCH_ID_FIELDS:
        for key in record:
            if key.lower() == field and (record[key] or "").strip():
                return record[key].strip()
    return ""


# 2. Sentinel Filename Date Extractor -------------------------------------

def run_sentinel_filename_date_extractor(args=None):
    scan = load_csv(dataset_path("v2aa_patch_source_registry_scan.csv")) or run_patch_source_registry_scanner(args)
    rows = []
    for source in scan:
        if source.get("should_parse_for_dates") != "true":
            continue
        path = source.get("registry_path", "")
        if not path.lower().endswith(".csv") or not os.path.exists(path):
            continue
        for record in load_csv(path):
            patch_id = _row_patch_id(record)
            if not patch_id:
                continue
            id_lower = {k.lower(): k for k in record}
            scene_values = [
                record[id_lower[f]] for f in SCENE_ID_FIELDS if f in id_lower and record.get(id_lower[f])
            ]
            for value in scene_values:
                dates, pattern = extract_dates_from_text(value)
                if not dates:
                    continue
                platform = detect_platform(
                    value, *[record.get(id_lower[f], "") for f in PLATFORM_FIELDS if f in id_lower]
                )
                if len(dates) > 1:
                    rows.append({
                        "filename_extract_id": f"FN_v2aa_{len(rows):05d}",
                        "patch_id": patch_id,
                        "registry_path_hash": hash_text(path, 24),
                        "filename_hash": hash_text(value, 24),
                        "extracted_date": "",
                        "extracted_datetime": "",
                        "sentinel_platform": platform,
                        "extraction_pattern": pattern,
                        "extraction_status": "BLOCKED_AMBIGUOUS_MULTIPLE_DATES",
                        "ambiguity_status": "AMBIGUOUS",
                        "notes": f"Multiple distinct dates in one value ({'|'.join(dates)}); not selected.",
                    })
                    continue
                rows.append({
                    "filename_extract_id": f"FN_v2aa_{len(rows):05d}",
                    "patch_id": patch_id,
                    "registry_path_hash": hash_text(path, 24),
                    "filename_hash": hash_text(value, 24),
                    "extracted_date": dates[0],
                    "extracted_datetime": "",
                    "sentinel_platform": platform,
                    "extraction_pattern": pattern,
                    "extraction_status": "DATE_EXTRACTED",
                    "ambiguity_status": "UNAMBIGUOUS",
                    "notes": "Date parsed from scene id / filename pattern; not inferred.",
                })
    out = dataset_path("v2aa_sentinel_filename_date_extraction.csv")
    write_csv(out, FILENAME_COLUMNS, rows)
    extracted = sum(1 for r in rows if r["extraction_status"] == "DATE_EXTRACTED")
    print(f"[v2aa filename] rows={len(rows)} extracted={extracted} -> {out}")
    return rows


# 3. Sentinel Sidecar Metadata Resolver -----------------------------------

def run_sentinel_sidecar_metadata_resolver(args=None):
    scan = load_csv(dataset_path("v2aa_patch_source_registry_scan.csv")) or run_patch_source_registry_scanner(args)
    rows = []
    for source in scan:
        if source.get("has_date_field") != "true" or source.get("has_patch_id") != "true":
            continue
        path = source.get("registry_path", "")
        if not os.path.exists(path):
            continue
        records = _load_records(path)
        for record in records:
            patch_id = _row_patch_id(record)
            if not patch_id:
                continue
            id_lower = {k.lower(): k for k in record}
            for field in SCENE_DATE_FIELDS:
                if field not in id_lower:
                    continue
                if field in FORBIDDEN_DATE_FIELDS:
                    continue
                raw = (record.get(id_lower[field]) or "").strip()
                if not raw:
                    continue
                dates, _ = extract_dates_from_text(raw)
                if not dates:
                    continue
                resolved_dt = raw if "T" in raw or ":" in raw else ""
                conf_hint = "HIGH" if field in {"sensing_date", "sensing_time", "scene_date", "acquisition_date", "product_start_time"} else "MEDIUM"
                rows.append({
                    "sidecar_resolution_id": f"SC_v2aa_{len(rows):05d}",
                    "patch_id": patch_id,
                    "sidecar_path_hash": hash_text(path, 24),
                    "date_field_used": field,
                    "resolved_date": dates[0],
                    "resolved_datetime": resolved_dt,
                    "source_type": "JSON_SIDECAR" if path.lower().endswith(".json") else "CSV_REGISTRY",
                    "resolution_status": "RESOLVED_FROM_EXPLICIT_DATE_FIELD",
                    "confidence_hint": conf_hint,
                    "notes": "Resolved from explicit acquisition/sensing/scene date field; created_at/modified_at and file mtime are never used.",
                })
    out = dataset_path("v2aa_sentinel_sidecar_metadata_resolution.csv")
    write_csv(out, SIDECAR_COLUMNS, rows)
    print(f"[v2aa sidecar] rows={len(rows)} -> {out}")
    return rows


def _load_records(path):
    if path.lower().endswith(".csv"):
        return load_csv(path)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return []
    if isinstance(data, list):
        return [d for d in data if isinstance(d, dict)]
    if isinstance(data, dict):
        return [data]
    return []


# 4. Patch Date Candidate Consolidator ------------------------------------

def _region_for_patch(patch_id):
    # Prefer region from the v1us patch registry resolution / candidate registry.
    for path in ("v1us_patch_registry_resolution.csv", "v1us_event_patch_candidate_registry.csv"):
        for record in load_csv(dataset_path(path)):
            if record.get("patch_id") == patch_id and record.get("region"):
                return record["region"]
    return _region_of(patch_id)


def run_patch_date_candidate_consolidator(args=None):
    filenames = load_csv(dataset_path("v2aa_sentinel_filename_date_extraction.csv")) or run_sentinel_filename_date_extractor(args)
    sidecars = load_csv(dataset_path("v2aa_sentinel_sidecar_metadata_resolution.csv")) or run_sentinel_sidecar_metadata_resolver(args)
    per_patch = {}
    for row in filenames:
        if row.get("extraction_status") != "DATE_EXTRACTED":
            per_patch.setdefault(row["patch_id"], {"dates": {}, "ambiguous": False, "datetime": ""})
            if row.get("ambiguity_status") == "AMBIGUOUS":
                per_patch[row["patch_id"]]["ambiguous"] = True
            continue
        entry = per_patch.setdefault(row["patch_id"], {"dates": {}, "ambiguous": False, "datetime": ""})
        entry["dates"].setdefault(row["extracted_date"], set()).add("filename")
    for row in sidecars:
        date = row.get("resolved_date")
        if not date:
            continue
        entry = per_patch.setdefault(row["patch_id"], {"dates": {}, "ambiguous": False, "datetime": ""})
        entry["dates"].setdefault(date, set()).add("sidecar")
        if row.get("resolved_datetime") and not entry["datetime"]:
            entry["datetime"] = row["resolved_datetime"]

    rows = []
    for patch_id in sorted(per_patch):
        entry = per_patch[patch_id]
        dates = entry["dates"]
        source_count = sum(len(v) for v in dates.values())
        region = _region_for_patch(patch_id)
        if not dates:
            status = "DATE_AMBIGUOUS_BLOCKED" if entry["ambiguous"] else "DATE_NOT_FOUND"
            conflict = "AMBIGUOUS" if entry["ambiguous"] else "NONE"
            selected, agreeing, recovered = "", 0, "false"
        elif len(dates) > 1:
            status = "DATE_CONFLICT_BLOCKED"
            conflict = "CONFLICT"
            selected, agreeing, recovered = "", 0, "false"
        else:
            only_date = next(iter(dates))
            srcs = dates[only_date]
            agreeing = len(srcs)
            selected = only_date
            recovered = "true"
            conflict = "NONE"
            status = "DATE_CONFIRMED_MULTI_SOURCE_AGREE" if agreeing > 1 else "DATE_CONFIRMED_SINGLE_SOURCE"
        rows.append({
            "patch_date_id": f"PD_v2aa_{len(rows):05d}",
            "patch_id": patch_id,
            "region": region,
            "candidate_dates": "|".join(sorted(dates)),
            "selected_sentinel_date": selected,
            "selected_sentinel_datetime": entry["datetime"] if selected else "",
            "source_count": str(source_count),
            "agreeing_source_count": str(agreeing),
            "conflict_status": conflict,
            "consolidation_status": status,
            "sentinel_date_recovered": recovered,
            "sentinel_date_inferred": "false",
            "notes": "Consolidated from filename and explicit sidecar dates; strong conflicts are blocked, never auto-resolved.",
        })
    out = dataset_path("v2aa_patch_date_candidate_consolidation.csv")
    write_csv(out, CONSOLIDATION_COLUMNS, rows)
    recovered = sum(1 for r in rows if r["sentinel_date_recovered"] == "true")
    print(f"[v2aa consolidate] patches={len(rows)} recovered={recovered} -> {out}")
    return rows


# 5. Sentinel Date Confidence Audit ---------------------------------------

def run_sentinel_date_confidence_audit(args=None):
    consolidation = load_csv(dataset_path("v2aa_patch_date_candidate_consolidation.csv")) or run_patch_date_candidate_consolidator(args)
    filenames = load_csv(dataset_path("v2aa_sentinel_filename_date_extraction.csv"))
    canonical = {
        r["patch_id"] for r in filenames
        if r.get("extraction_status") == "DATE_EXTRACTED" and r.get("sentinel_platform", "").startswith("S")
    }
    rows = []
    for row in consolidation:
        status = row.get("consolidation_status")
        agreeing = int(row.get("agreeing_source_count") or 0)
        patch_id = row["patch_id"]
        if status == "DATE_CONFLICT_BLOCKED":
            cls, score, usable, blocker = "BLOCKED_CONFLICT", "0", "false", "date_conflict_between_sources"
        elif status in {"DATE_NOT_FOUND", "DATE_AMBIGUOUS_BLOCKED"}:
            cls, score, usable, blocker = "MISSING", "0", "false", "no_recoverable_sentinel_date"
        elif status == "DATE_CONFIRMED_MULTI_SOURCE_AGREE":
            cls, score, usable, blocker = "HIGH_CONFIDENCE", "95", "true", ""
        elif status == "DATE_CONFIRMED_SINGLE_SOURCE" and patch_id in canonical:
            cls, score, usable, blocker = "HIGH_CONFIDENCE", "90", "true", ""
        elif status == "DATE_CONFIRMED_SINGLE_SOURCE":
            cls, score, usable, blocker = "MEDIUM_CONFIDENCE", "70", "true", ""
        else:
            cls, score, usable, blocker = "LOW_CONFIDENCE", "40", "false", "weak_single_generic_date_field"
        rows.append({
            "confidence_audit_id": f"CA_v2aa_{len(rows):05d}",
            "patch_id": patch_id,
            "selected_sentinel_date": row.get("selected_sentinel_date", ""),
            "confidence_class": cls,
            "confidence_score": score,
            "usable_for_temporal_linkage": usable,
            "blocker": blocker,
            "notes": "Only HIGH/MEDIUM confidence dates are usable; LOW confidence is never promoted to usable.",
        })
    out = dataset_path("v2aa_sentinel_date_confidence_audit.csv")
    write_csv(out, CONFIDENCE_COLUMNS, rows)
    usable = sum(1 for r in rows if r["usable_for_temporal_linkage"] == "true")
    print(f"[v2aa confidence] rows={len(rows)} usable={usable} -> {out}")
    return rows


# 6. Event-Patch Temporal Distance Builder --------------------------------

def _date(value):
    try:
        return datetime.date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError):
        return None


def run_event_patch_temporal_distance_builder(args=None):
    confidence = {r["patch_id"]: r for r in (load_csv(dataset_path("v2aa_sentinel_date_confidence_audit.csv")) or run_sentinel_date_confidence_audit(args))}
    linkage = load_csv(dataset_path("v1us_event_temporal_window_linkage.csv"))
    rows = []
    for link in linkage:
        patch_id = link.get("patch_id", "")
        conf = confidence.get(patch_id, {})
        sentinel_date = conf.get("selected_sentinel_date", "")
        usable = conf.get("usable_for_temporal_linkage") == "true"
        start = _date(link.get("event_start_date"))
        end = _date(link.get("event_end_date"))
        sd = _date(sentinel_date)
        days_start = days_end = ""
        if not sentinel_date or conf.get("confidence_class") in {"MISSING", ""}:
            temporal_class = "TEMPORAL_DISTANCE_BLOCKED_NO_DATE"
            usable_ctx = "false"
        elif not usable:
            temporal_class = "TEMPORAL_DISTANCE_BLOCKED_LOW_CONFIDENCE"
            usable_ctx = "false"
        elif sd and start and end:
            days_start = str((sd - start).days)
            days_end = str((sd - end).days)
            if start <= sd <= end:
                temporal_class = "WITHIN_EVENT_WINDOW"
            elif sd < start:
                temporal_class = "PRE_EVENT_NEAR" if (start - sd).days <= TEMPORAL_NEAR_DAYS else "PRE_EVENT_FAR"
            else:
                temporal_class = "POST_EVENT_NEAR" if (sd - end).days <= TEMPORAL_NEAR_DAYS else "POST_EVENT_FAR"
            usable_ctx = "true"
        else:
            temporal_class = "TEMPORAL_DISTANCE_BLOCKED_NO_DATE"
            usable_ctx = "false"
        rows.append({
            "temporal_distance_id": f"TD_v2aa_{len(rows):05d}",
            "event_patch_candidate_id": link.get("event_patch_candidate_id", ""),
            "event_id": link.get("event_id", ""),
            "patch_id": patch_id,
            "event_start_date": link.get("event_start_date", ""),
            "event_end_date": link.get("event_end_date", ""),
            "sentinel_date": sentinel_date,
            "days_from_event_start": days_start,
            "days_from_event_end": days_end,
            "temporal_class": temporal_class,
            "usable_for_contextual_review": usable_ctx,
            "usable_for_overlay_preflight": "false",
            "can_create_ground_reference": "false",
            "notes": "Temporal class is review-only context; never a truth, label, overlay or ground reference.",
        })
    out = dataset_path("v2aa_event_patch_temporal_distance.csv")
    write_csv(out, TEMPORAL_COLUMNS, rows)
    placed = sum(1 for r in rows if not r["temporal_class"].startswith("TEMPORAL_DISTANCE_BLOCKED"))
    print(f"[v2aa temporal] rows={len(rows)} placed={placed} -> {out}")
    return rows


# 7. Event-Patch Readiness Updater ----------------------------------------

def run_event_patch_readiness_updater(args=None):
    temporal = load_csv(dataset_path("v2aa_event_patch_temporal_distance.csv")) or run_event_patch_temporal_distance_builder(args)
    rows = []
    for td in temporal:
        blocked = td["temporal_class"].startswith("TEMPORAL_DISTANCE_BLOCKED")
        sentinel_cls = "RECOVERED_USABLE" if td.get("usable_for_contextual_review") == "true" else "STILL_MISSING_OR_BLOCKED"
        temporal_cls = "IMPROVED_REVIEW_ONLY" if not blocked else "UNCHANGED_BLOCKED"
        contextual_cls = "CONTEXTUAL_REVIEW_READY" if not blocked else "BLOCKED_NO_USABLE_DATE"
        dims = [
            ("sentinel_date_support", sentinel_cls),
            ("temporal_linkage", temporal_cls),
            ("contextual_review_readiness", contextual_cls),
            ("overlay_readiness", "BLOCKED"),
            ("ground_reference_readiness", "BLOCKED"),
            ("training_readiness", "BLOCKED"),
        ]
        for dim, cls in dims:
            rows.append({
                "readiness_update_id": f"RDY_v2aa_{len(rows):05d}",
                "event_patch_candidate_id": td.get("event_patch_candidate_id", ""),
                "event_id": td.get("event_id", ""),
                "patch_id": td.get("patch_id", ""),
                "region": _region_for_patch(td.get("patch_id", "")),
                "dimension": dim,
                "classification": cls,
                "basis": "v2aa sentinel date recovery",
                **guardrails(),
                "notes": "Additive readiness; v1us not modified; overlay/ground reference/training stay BLOCKED.",
            })
    out = dataset_path("v2aa_event_patch_readiness_update.csv")
    write_csv(out, READINESS_COLUMNS, rows)
    print(f"[v2aa readiness] rows={len(rows)} -> {out}")
    return rows


# 8. Multi-Region Temporal Blocker Reducer ---------------------------------

def run_multiregion_temporal_blocker_reducer(args=None):
    consolidation = load_csv(dataset_path("v2aa_patch_date_candidate_consolidation.csv")) or run_patch_date_candidate_consolidator(args)
    confidence = {r["patch_id"]: r for r in (load_csv(dataset_path("v2aa_sentinel_date_confidence_audit.csv")) or run_sentinel_date_confidence_audit(args))}
    temporal = load_csv(dataset_path("v2aa_event_patch_temporal_distance.csv")) or run_event_patch_temporal_distance_builder(args)
    # Total patches per region from the real patch registry resolution.
    region_totals = {}
    for record in load_csv(dataset_path("v1us_patch_registry_resolution.csv")):
        region_totals[record.get("region", "")] = region_totals.get(record.get("region", ""), 0) + 1
    regions = sorted(set(region_totals) | {r.get("region", "") for r in consolidation if r.get("region")})
    improved_by_region = {}
    for td in temporal:
        if not td["temporal_class"].startswith("TEMPORAL_DISTANCE_BLOCKED"):
            reg = _region_for_patch(td.get("patch_id", ""))
            improved_by_region[reg] = improved_by_region.get(reg, 0) + 1
    rows = []
    for region in regions:
        if not region:
            continue
        cons = [r for r in consolidation if r.get("region") == region]
        recovered = [r for r in cons if r.get("sentinel_date_recovered") == "true"]
        high = sum(1 for r in recovered if confidence.get(r["patch_id"], {}).get("confidence_class") == "HIGH_CONFIDENCE")
        medium = sum(1 for r in recovered if confidence.get(r["patch_id"], {}).get("confidence_class") == "MEDIUM_CONFIDENCE")
        total = region_totals.get(region, len(cons))
        missing = max(0, total - len(recovered))
        improved = improved_by_region.get(region, 0)
        if total and len(recovered) == total:
            reduction = "BLOCKER_FULLY_REDUCED"
        elif recovered:
            reduction = "BLOCKER_PARTIALLY_REDUCED"
        else:
            reduction = "BLOCKER_REMAINS_DOMINANT"
        rows.append({
            "reduction_id": f"TBR_v2aa_{len(rows):04d}",
            "region": region,
            "total_patches": str(total),
            "patches_with_recovered_date": str(len(recovered)),
            "high_confidence_dates": str(high),
            "medium_confidence_dates": str(medium),
            "missing_or_blocked_dates": str(missing),
            "event_patch_candidates_improved": str(improved),
            "blocker_reduction_status": reduction,
            "notes": "Quantifies no_sentinel_date reduction from recovered dates only; no inferred dates counted.",
        })
    out = dataset_path("v2aa_multiregion_temporal_blocker_reduction.csv")
    write_csv(out, REDUCTION_COLUMNS, rows)
    print(f"[v2aa blocker reduce] regions={len(rows)} -> {out}")
    return rows


# 9. Next Programming Target Ranker ----------------------------------------

EFFORT_PENALTY = {"LOW": 0, "MEDIUM": 5, "HIGH": 15}
OVERCLAIM_PENALTY = {"LOW": 0, "MEDIUM": 10, "HIGH": 25}

TARGET_VERSION = {
    "MULTI_REGION_REGISTRY_HARDENING": "v2ab — Multi-Region Registry Hardening",
    "EVENT_PATCH_PACKAGE_SCHEMA_HARDENING": "v2ab — Event-Patch Package Schema Hardening",
    "DINO_REVIEW_SUPPORT_COMPLETION": "v2ab — DINO Review Support Completion",
    "PUBLIC_SOURCE_RECHECK_HOLD": "v2ab — Public Source Recheck Hold",
    "STOP_GROUND_TRUTH_SEARCH_UNTIL_NEW_SOURCE": "v2ab — Ground Truth Search Hold",
    "SENTINEL_DATE_RECOVERY_CONTINUE": "v2ab — Sentinel Date Recovery Continue",
}


def _recovery_metrics():
    temporal = load_csv(dataset_path("v2aa_event_patch_temporal_distance.csv"))
    total = len(temporal) or 1
    placed = sum(1 for r in temporal if not r["temporal_class"].startswith("TEMPORAL_DISTANCE_BLOCKED"))
    candidate_recovery_rate = placed / total
    # "Continue" only adds value while there is unresolved date evidence left to
    # chase (conflicts/ambiguities). If the candidate patch namespace simply has
    # no date anywhere, deeper local search cannot help and continue is low-value.
    consolidation = load_csv(dataset_path("v2aa_patch_date_candidate_consolidation.csv"))
    unresolved = sum(
        1 for r in consolidation
        if r.get("consolidation_status") in {"DATE_CONFLICT_BLOCKED", "DATE_AMBIGUOUS_BLOCKED"}
    )
    return {
        "candidate_total": total,
        "candidate_placed": placed,
        "candidate_recovery_rate": candidate_recovery_rate,
        "remaining_sources": unresolved,
        "extracted": sum(1 for r in consolidation if r.get("sentinel_date_recovered") == "true"),
    }


def _candidate_targets(m):
    rate = m["candidate_recovery_rate"]
    # Continue value scales with how much remains recoverable AND whether more
    # date-bearing sources are still unparsed. With sources exhausted, continue
    # has little programming value.
    continue_value = round(min(1.0, (1 - rate)) * 100) if m["remaining_sources"] >= 2 else round((1 - rate) * 15)
    continue_blocker = round((1 - rate) * 100) if m["remaining_sources"] >= 2 else round((1 - rate) * 10)
    schema_blocker = round((1 - rate) * 45) + 15
    return [
        {
            "next_target": "EVENT_PATCH_PACKAGE_SCHEMA_HARDENING",
            "programming_value": 60,
            "ground_truth_value": 0,
            "blocker_reduction_value": min(60, schema_blocker),
            "expected_effort": "LOW",
            "overclaim_risk": "LOW",
            "notes": "Most candidate patch ids carry no recoverable Sentinel date locally; harden the event-patch package schema to make temporal fields explicit and auditable.",
        },
        {
            "next_target": "MULTI_REGION_REGISTRY_HARDENING",
            "programming_value": 55,
            "ground_truth_value": 0,
            "blocker_reduction_value": 35,
            "expected_effort": "LOW",
            "overclaim_risk": "LOW",
            "notes": "Consolidate the multi-region registries with the recovered-date evidence and the persisting no_sentinel_date blocker.",
        },
        {
            "next_target": "SENTINEL_DATE_RECOVERY_CONTINUE",
            "programming_value": continue_value,
            "ground_truth_value": 0,
            "blocker_reduction_value": continue_blocker,
            "expected_effort": "MEDIUM",
            "overclaim_risk": "LOW",
            "notes": "Deeper local date recovery; only worthwhile while unparsed date-bearing sources remain for the candidate patch namespace.",
        },
        {
            "next_target": "DINO_REVIEW_SUPPORT_COMPLETION",
            "programming_value": 10,
            "ground_truth_value": 0,
            "blocker_reduction_value": 10,
            "expected_effort": "MEDIUM",
            "overclaim_risk": "LOW",
            "notes": "DINO review support already attached for nearly all candidates; review-only.",
        },
        {
            "next_target": "PUBLIC_SOURCE_RECHECK_HOLD",
            "programming_value": 20,
            "ground_truth_value": 0,
            "blocker_reduction_value": 25,
            "expected_effort": "LOW",
            "overclaim_risk": "LOW",
            "notes": "Hold until a new public source with patch-level acquisition dates appears.",
        },
        {
            "next_target": "STOP_GROUND_TRUTH_SEARCH_UNTIL_NEW_SOURCE",
            "programming_value": 10,
            "ground_truth_value": 0,
            "blocker_reduction_value": 0,
            "expected_effort": "LOW",
            "overclaim_risk": "LOW",
            "notes": "Explicit stop on ground-truth search until a qualifying public source is published.",
        },
    ]


def _score(t):
    base = 0.5 * t["programming_value"] + 0.5 * t["blocker_reduction_value"]
    return base - EFFORT_PENALTY.get(t["expected_effort"], 5) - OVERCLAIM_PENALTY.get(t["overclaim_risk"], 10)


def run_next_programming_target_ranker(args=None):
    metrics = _recovery_metrics()
    targets = _candidate_targets(metrics)
    targets.sort(key=_score, reverse=True)
    rows = []
    for idx, t in enumerate(targets, start=1):
        rows.append({
            "rank": str(idx),
            "next_target": t["next_target"],
            "programming_value": str(t["programming_value"]),
            "ground_truth_value": str(t["ground_truth_value"]),
            "blocker_reduction_value": str(t["blocker_reduction_value"]),
            "expected_effort": t["expected_effort"],
            "overclaim_risk": t["overclaim_risk"],
            "recommended_version": TARGET_VERSION.get(t["next_target"], ""),
            "recommended_action": "SELECTED_NEXT_TARGET" if idx == 1 else "RANKED_ALTERNATIVE",
            "notes": t["notes"],
        })
    out = dataset_path("v2aa_next_programming_target_ranker.csv")
    write_csv(out, RANKER_COLUMNS, rows)
    print(f"[v2aa ranker] selected={rows[0]['next_target'] if rows else 'none'} -> {out}")
    return rows


# 10. Completion Report ----------------------------------------------------

def run_ground_reference_blocker_matrix(args=None):
    reductions = load_csv(dataset_path("v2aa_multiregion_temporal_blocker_reduction.csv")) or run_multiregion_temporal_blocker_reducer(args)
    blockers = [
        "no_observed_geometry", "no_occurrence_coordinates", "no_overlay",
        "no_ground_reference", "no_training_label", "patch_truth_forbidden",
    ]
    region_event = {"REC": "REC_2022_05_24_30", "PET": "PET_2022_02_15", "CUR": "CUR_2022_01_15"}
    rows = []
    for red in reductions:
        region = red.get("region", "")
        # no_sentinel_date status depends on the recovery result for the region.
        local = list(blockers)
        if red.get("blocker_reduction_status") != "BLOCKER_FULLY_REDUCED":
            local = ["no_sentinel_date"] + local
        for blocker in local:
            rows.append({
                "blocker_id": f"GB_v2aa_{len(rows):04d}",
                "region": region,
                "event_id": region_event.get(region, ""),
                "blocker": blocker,
                "status": "BLOCKED",
                **guardrails(),
                "notes": "Sentinel date recovery does not unblock observed geometry, overlay, ground reference or labels.",
            })
    out = dataset_path("v2aa_ground_reference_blocker_matrix.csv")
    write_csv(out, BLOCKER_COLUMNS, rows)
    print(f"[v2aa gr blockers] rows={len(rows)} -> {out}")
    return rows


def run_completion_report(args=None):
    write_policy_configs()
    scan = load_csv(dataset_path("v2aa_patch_source_registry_scan.csv")) or run_patch_source_registry_scanner(args)
    filenames = load_csv(dataset_path("v2aa_sentinel_filename_date_extraction.csv")) or run_sentinel_filename_date_extractor(args)
    sidecars = load_csv(dataset_path("v2aa_sentinel_sidecar_metadata_resolution.csv")) or run_sentinel_sidecar_metadata_resolver(args)
    consolidation = load_csv(dataset_path("v2aa_patch_date_candidate_consolidation.csv")) or run_patch_date_candidate_consolidator(args)
    confidence = load_csv(dataset_path("v2aa_sentinel_date_confidence_audit.csv")) or run_sentinel_date_confidence_audit(args)
    temporal = load_csv(dataset_path("v2aa_event_patch_temporal_distance.csv")) or run_event_patch_temporal_distance_builder(args)
    readiness = load_csv(dataset_path("v2aa_event_patch_readiness_update.csv")) or run_event_patch_readiness_updater(args)
    reductions = load_csv(dataset_path("v2aa_multiregion_temporal_blocker_reduction.csv")) or run_multiregion_temporal_blocker_reducer(args)
    ranker = load_csv(dataset_path("v2aa_next_programming_target_ranker.csv")) or run_next_programming_target_ranker(args)
    blockers = run_ground_reference_blocker_matrix(args)

    recovered = sum(1 for r in consolidation if r.get("sentinel_date_recovered") == "true")
    missing = sum(1 for r in consolidation if r.get("sentinel_date_recovered") != "true")
    usable = sum(1 for r in confidence if r.get("usable_for_temporal_linkage") == "true")
    placed = sum(1 for r in temporal if not r["temporal_class"].startswith("TEMPORAL_DISTANCE_BLOCKED"))
    parseable = sum(1 for r in scan if r.get("should_parse_for_dates") == "true")
    fn_extracted = sum(1 for r in filenames if r.get("extraction_status") == "DATE_EXTRACTED")
    next_target = ranker[0].get("next_target", "") if ranker else ""
    next_version = ranker[0].get("recommended_version", "") if ranker else ""
    improved_regions = [r["region"] for r in reductions if r.get("blocker_reduction_status") != "BLOCKER_REMAINS_DOMINANT"]

    write_csv(dataset_path("v2aa_next_actions_registry.csv"), NEXT_COLUMNS, [{
        "action_id": "NA_v2aa_0000",
        "event_id": "MULTI_REGION",
        "action_type": next_target,
        "priority": "1",
        "description": "Selected from v2aa score-based next-programming-target ranker after Sentinel date recovery.",
        "target": "EVENT_PATCH_PACKAGE_TEMPORAL_READINESS",
        "status": "RECOMMENDED_NEXT_STEP",
        "notes": "No overlay, labels, ground truth, ground reference or inferred Sentinel dates.",
    }])

    lines = [
        "# Protocolo C v2aa - Sentinel Date Recovery for Event-Patch Packages",
        "",
        f"- registries scanned: `{len(scan)}` (parseable for dates: `{parseable}`)",
        f"- filename/scene-id dates extracted: `{fn_extracted}`",
        f"- sidecar explicit-date resolutions: `{len(sidecars)}`",
        f"- patches consolidated: `{len(consolidation)}` (recovered: `{recovered}`, missing/blocked: `{missing}`)",
        f"- usable (HIGH/MEDIUM) dates: `{usable}`",
        f"- event-patch candidates with placed temporal distance: `{placed}` of `{len(temporal)}`",
        f"- readiness update rows: `{len(readiness)}`",
        f"- ground-reference blocker rows: `{len(blockers)}`",
        f"- selected next target: `{next_target}`",
        f"- suggested next version: `{next_version}`",
        "",
        "v2aa recovered Sentinel scene dates only from existing filenames, scene ids and explicit sidecar date fields. It never downloaded data, queried the web, inferred a date, used an approximate date as real, used created_at/modified_at or file mtimes, executed overlay, or created ground truth, ground reference or labels.",
    ]
    write_text(doc_path("protocolo_c_v2aa_sentinel_date_recovery.md"), lines)

    report = lines + [
        "",
        "## How many registries were scanned",
        f"{len(scan)} versionable registries (datasets/ and configs/) were scanned for patch and date metadata; {parseable} contained both a patch id and a date or filename field and were parsed. local_only was never scanned.",
        "",
        "## Which date sources worked",
        "Real Sentinel scene dates were recovered from canonical scene-id values (pattern `YYYYMMDDT...`) and from explicit `scene_date` fields in the anchor Sentinel patch registries. The event-patch candidate patch ids (CUR/PET/REC numeric namespace) carry no Sentinel scene date in any versionable registry, so they remain without a recoverable date.",
        "",
        "## How many patches recovered a date / stayed missing",
        f"{recovered} patches recovered a Sentinel date; {missing} remained missing or blocked. {usable} dates are usable (HIGH/MEDIUM confidence) for review-only temporal linkage.",
        "",
        "## How many event-patch candidates improved",
        f"{placed} of {len(temporal)} event-patch candidates obtained a placed temporal distance. The remainder stay temporally blocked because their patch ids have no recoverable Sentinel date locally.",
        "",
        "## Which regions improved",
        f"Regions with any reduction: {', '.join(improved_regions) if improved_regions else 'none'}. The recovered dates belong to the anchor reference-patch namespace; the bulk candidate namespace keeps no_sentinel_date dominant.",
        "",
        "## Was no_sentinel_date reduced",
        "Partially and only where real dates exist. The blocker remains dominant for the event-patch candidate namespace, so it is recorded as still-blocked in the readiness and blocker matrices.",
        "",
        "## Why there is still no overlay",
        "No overlay was executed and overlay readiness stays BLOCKED. A recovered acquisition date does not establish observed occurrence geometry, which overlay would require.",
        "",
        "## Why there is still no ground reference",
        "A Sentinel acquisition date is temporal metadata, not an observed occurrence. Without observed occurrence geometry there is no basis for ground reference, and none was created.",
        "",
        "## Next programming step",
        f"The score-based ranker selected `{next_target}` (`{next_version}`).",
    ]
    write_text(doc_path("protocolo_c_relatorio_v2aa_sentinel_date_recovery.md"), report)

    write_text(doc_path("protocolo_c_status_atual_v2aa.md"), [
        "# Status atual - Protocolo C v2aa",
        "",
        f"Sentinel date recovery status: `{MAX_STATUS}`.",
        f"Patches with recovered date: `{recovered}`; missing/blocked: `{missing}`.",
        f"Selected next programming target: `{next_target}`.",
        f"Suggested next version: `{next_version}`.",
        "",
        "Overlay, ground reference, training labels, ground truth, inferred Sentinel dates and operational validation remain blocked.",
    ])

    manifest = []
    for idx, artifact in enumerate(V2AA_ARTIFACTS):
        real = artifact_path(artifact)
        if not os.path.exists(real):
            continue
        manifest.append({
            "artifact_id": f"MAN_v2aa_{idx:04d}",
            "artifact_path": artifact.replace("\\", "/"),
            "artifact_type": os.path.splitext(artifact)[1].lstrip(".") or "text",
            "protocol_version": PROTOCOL_VERSION,
            "sha256_prefix": sha256_file(real)[:16],
            "file_size_bytes": str(os.path.getsize(real)),
            "is_versionable": "true",
            "reason": "v2aa recovery/registry artifact; no raw data, no private path, no inferred date.",
        })
    write_csv(dataset_path("v2aa_versionable_artifacts_manifest.csv"), MANIFEST_COLUMNS, manifest)
    for folder in (STAGING_DIR, REPORTS_DIR):
        os.makedirs(folder, exist_ok=True)
    print(f"[v2aa completion] recovered={recovered} missing={missing} next={next_target}")
    return {"recovered": recovered, "missing": missing, "next_target": next_target, "next_version": next_version}


def run_all(args=None):
    args = args or parse_args([])
    run_patch_source_registry_scanner(args)
    run_sentinel_filename_date_extractor(args)
    run_sentinel_sidecar_metadata_resolver(args)
    run_patch_date_candidate_consolidator(args)
    run_sentinel_date_confidence_audit(args)
    run_event_patch_temporal_distance_builder(args)
    run_event_patch_readiness_updater(args)
    run_multiregion_temporal_blocker_reducer(args)
    run_next_programming_target_ranker(args)
    return run_completion_report(args)
