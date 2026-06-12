#!/usr/bin/env python3
"""v2at Evidence Fact Hardening and Observational Readiness Audit.

Classifies source/event/patch evidence without creating labels, ground truth,
overlays, training readiness, or negative evidence. Network access is opt-in
through V2AT_NETWORK=1 and may only populate the ignored evidence cache.
"""

import argparse
import csv
import hashlib
import json
import os
import re
import urllib.request

PROTOCOL_VERSION = "v2at"
DATASET_ROOT = os.environ.get("DATASET_ROOT", "datasets")
DATASET_DIR = os.environ.get("DATASET_DIR", os.path.join(DATASET_ROOT, "protocolo_c"))
DOCS_DIR = os.environ.get("DOCS_DIR", "docs/protocolo_c/v2at_evidence_fact_hardening")
CACHE_DIR = os.environ.get("CACHE_DIR", os.path.join(DOCS_DIR, "evidence_cache"))
CONFIG_DIR = os.environ.get("CONFIG_DIR", "configs/protocolo_c")
NETWORK_ENV = "V2AT_NETWORK"
HTTP_TIMEOUT = 8
MAX_CACHE_BYTES = 2 * 1024 * 1024
ABSOLUTE_PATH_RE = re.compile(r"(?:[A-Za-z]:\\|/Users/|/home/|/mnt/|\\\\)")

TARGET_SOURCES = [
    {"source_id": "INMET", "source_name": "INMET historical and automatic stations",
     "source_class": "OFFICIAL_METEO", "authority_level": "OFFICIAL_NATIONAL",
     "target_url": "https://portal.inmet.gov.br/dadoshistoricos",
     "default_role": "OBSERVED_MEASUREMENT"},
    {"source_id": "CEMADEN", "source_name": "CEMADEN monitoring stations and monthly downloads",
     "source_class": "OFFICIAL_MONITORING", "authority_level": "OFFICIAL_NATIONAL",
     "target_url": "https://www.cemaden.gov.br/mapainterativo/",
     "default_role": "OBSERVED_MEASUREMENT"},
    {"source_id": "ANA_HIDROWEB", "source_name": "ANA HidroWeb and telemetry",
     "source_class": "OFFICIAL_HYDRO", "authority_level": "OFFICIAL_NATIONAL",
     "target_url": "https://www.snirh.gov.br/hidroweb/",
     "default_role": "OBSERVED_MEASUREMENT"},
    {"source_id": "SGB_CPRM", "source_name": "SGB CPRM risk and susceptibility products",
     "source_class": "OFFICIAL_GEOCIENCE", "authority_level": "OFFICIAL_NATIONAL",
     "target_url": "https://www.sgb.gov.br/",
     "default_role": "SUSCEPTIBILITY_CONTEXT"},
    {"source_id": "COPERNICUS_EMS", "source_name": "Copernicus Emergency Management Service",
     "source_class": "OFFICIAL_INTERNATIONAL", "authority_level": "OFFICIAL_INTERNATIONAL",
     "target_url": "https://emergency.copernicus.eu/",
     "default_role": "PRODUCT"},
    {"source_id": "INTERNATIONAL_CHARTER", "source_name": "International Charter Space and Major Disasters",
     "source_class": "OFFICIAL_INTERNATIONAL", "authority_level": "OFFICIAL_INTERNATIONAL",
     "target_url": "https://disasterscharter.org/",
     "default_role": "QUICKVIEW_OR_PRODUCT"},
]

FACT_COLUMNS = [
    "assertion_id", "candidate_id", "event_id", "patch_id", "source_id", "source_name",
    "evidence_role", "source_identified", "license_explicit", "crs_resolved",
    "observed_event", "temporal_compatible", "hazard_typed", "geometry_or_measurement_compatible",
    "human_review_complete", "independent_corroboration", "fact_classification",
    "critical_blocker", "quickview_can_promote", "susceptibility_can_promote",
    "absence_can_create_negative", "can_create_ground_truth", "can_create_training_label",
]


def parse_args(argv=None):
    return argparse.ArgumentParser().parse_args(argv)


def clean(value):
    return str(value or "").strip()


def is_true(value):
    return clean(value).lower() == "true"


def boolean(value):
    return "true" if bool(value) else "false"


def safe_slug(value):
    return re.sub(r"[^a-z0-9]+", "-", clean(value).lower()).strip("-") or "item"


def dataset_path(name):
    return os.path.join(DATASET_DIR, name)


def root_dataset_path(name):
    return os.path.join(DATASET_ROOT, name)


def doc_path(name):
    return os.path.join(DOCS_DIR, name)


def cache_path(name):
    return os.path.join(CACHE_DIR, name)


def rel_dataset(name):
    return f"datasets/protocolo_c/{name}"


def rel_doc(name):
    return f"docs/protocolo_c/v2at_evidence_fact_hardening/{name}"


def load_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, columns, rows):
    base = os.path.basename(path)
    if not base.startswith("v2at_"):
        raise ValueError(f"Refusing non-v2at output: {path}")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_text(path):
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def write_markdown(path, lines):
    if not os.path.basename(path).startswith("v2at_") and os.path.basename(path) != "README.md":
        raise ValueError(f"Refusing non-v2at documentation output: {path}")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def sha256_file(path):
    if not os.path.exists(path):
        return ""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_cache_policy():
    os.makedirs(CACHE_DIR, exist_ok=True)
    marker = cache_path(".gitignore")
    with open(marker, "w", encoding="utf-8") as handle:
        handle.write("*\n!.gitignore\n")
    return marker


def validate_cache_policy():
    marker = ensure_cache_policy()
    entries = sorted(os.listdir(CACHE_DIR))
    return {
        "marker_content_valid": boolean(read_text(marker) == "*\n!.gitignore\n"),
        "tracked_candidate_count": str(sum(1 for name in entries if name != ".gitignore")),
        "cache_policy_valid": boolean(read_text(marker) == "*\n!.gitignore\n"),
    }


def is_network_enabled():
    return clean(os.environ.get(NETWORK_ENV)) == "1"


def fetch_to_cache(url, source_id):
    ensure_cache_policy()
    if not is_network_enabled():
        return {"download_status": "NETWORK_DISABLED_DETERMINISTIC_RUN", "cache_path": "",
                "cache_sha256": "", "raw_data_versioned": "false"}
    request = urllib.request.Request(clean(url), headers={"User-Agent": "REV-P-v2at/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT) as response:
            payload = response.read(MAX_CACHE_BYTES + 1)
        if len(payload) > MAX_CACHE_BYTES:
            return {"download_status": "PAYLOAD_TOO_LARGE_NOT_CACHED", "cache_path": "",
                    "cache_sha256": "", "raw_data_versioned": "false"}
        name = f"{safe_slug(source_id)}-{hashlib.sha256(clean(url).encode()).hexdigest()[:12]}.cache"
        path = cache_path(name)
        with open(path, "wb") as handle:
            handle.write(payload)
        return {"download_status": "CACHED_FOR_LOCAL_REVIEW", "cache_path": rel_doc(f"evidence_cache/{name}"),
                "cache_sha256": sha256_file(path), "raw_data_versioned": "false"}
    except Exception as exc:
        return {"download_status": f"DOWNLOAD_FAILED_{type(exc).__name__.upper()}", "cache_path": "",
                "cache_sha256": "", "raw_data_versioned": "false"}


def normalize_license(value):
    upper = clean(value).upper()
    explicit = any(token in upper for token in ("PUBLIC_OPEN", "OPEN_LICENSE", "CC-", "EXPLICIT"))
    unknown = not upper or any(token in upper for token in ("UNKNOWN", "NEEDS_", "NOT_DOCUMENTED", "PENDING"))
    return "LICENSE_EXPLICIT" if explicit and not unknown else "LICENSE_UNKNOWN_BLOCKING"


def normalize_crs(value, role=""):
    upper = clean(value).upper()
    if role in {"OBSERVED_MEASUREMENT", "DOCUMENTARY_EVENT"}:
        return "CRS_IRRELEVANT_FOR_NON_GEOMETRIC_ASSERTION"
    if upper and not any(token in upper for token in ("UNKNOWN", "NOT_DOCUMENTED", "PENDING", "NEEDS_")):
        return "CRS_EXPLICIT"
    return "CRS_UNKNOWN_BLOCKING"


def role_is_context(role):
    upper = clean(role).upper()
    return any(token in upper for token in ("SUSCEPTIBILITY", "RISK", "STATIC", "DINO", "GIS_PROXY", "CONTEXT"))


def classify_fact_assertion(row):
    role = clean(row.get("evidence_role")).upper()
    if role_is_context(role):
        return "CONTEXT_ONLY", "CONTEXT_SOURCE_NOT_OBSERVED_OCCURRENCE"
    if "QUICKVIEW" in role:
        return "REVIEW_ONLY_SIGNAL", "QUICKVIEW_NEVER_PROMOTES"
    checks = [
        ("source_identified", "BLOCKED_SOURCE_UNIDENTIFIED"),
        ("license_explicit", "BLOCKED_LICENSE_UNKNOWN"),
        ("crs_resolved", "BLOCKED_CRS_UNKNOWN"),
        ("observed_event", "BLOCKED_NO_OBSERVED_EVENT"),
        ("temporal_compatible", "BLOCKED_TEMPORAL_MISMATCH"),
        ("hazard_typed", "BLOCKED_HAZARD_AMBIGUOUS"),
        ("geometry_or_measurement_compatible", "BLOCKED_GEOMETRY_OR_MEASUREMENT_MISSING"),
    ]
    for field, status in checks:
        if not is_true(row.get(field)):
            return status, status
    if not is_true(row.get("human_review_complete")) or not is_true(row.get("independent_corroboration")):
        return "SUPPORTED_CANDIDATE", "HUMAN_REVIEW_OR_INDEPENDENCE_PENDING"
    return "FACT_VERIFIED", ""


def source_authority_rows():
    rows = []
    for rank, source in enumerate(TARGET_SOURCES, 1):
        rows.append({
            "authority_id": f"AUTH_v2at_{rank:03d}", **source,
            "source_identified": "true", "official_authority": "true",
            "observed_occurrence_default": boolean(source["default_role"] == "OBSERVED_MEASUREMENT"),
            "quickview_can_promote": "false", "susceptibility_can_promote": "false",
            "absence_can_create_negative": "false", "notes": "Authority does not by itself establish an event fact.",
        })
    return rows


def _index(rows, key):
    return {clean(row.get(key)): row for row in rows if clean(row.get(key))}


def derive_assertion_rows():
    priorities = load_csv(dataset_path("v2as_deep_probe_priority.csv"))
    geometries = _index(load_csv(dataset_path("v2as_geojson_candidate_index.csv")), "candidate_id")
    sources = load_csv(dataset_path("v2ar_official_geometry_source_registry.csv"))
    licenses = load_csv(dataset_path("v2ar_license_crs_checklist.csv"))
    observed = _index(load_csv(root_dataset_path("observed_event_reference_candidate_registry.csv")), "observed_event_id")
    source_by_candidate = {}
    for source in sources:
        if clean(source.get("source_role")).lower() == "primary":
            source_by_candidate[clean(source.get("candidate_id"))] = source
    license_by_candidate = {}
    for item in licenses:
        if clean(item.get("checklist_id")).lower().endswith("_primary"):
            license_by_candidate[clean(item.get("candidate_id"))] = item
    rows = []
    for number, priority in enumerate(priorities, 1):
        cid = clean(priority.get("candidate_id"))
        source = source_by_candidate.get(cid, {})
        license_row = license_by_candidate.get(cid, {})
        event = observed.get(cid, {})
        geometry = geometries.get(cid, {})
        event_type = clean(event.get("event_type"))
        role = "DOCUMENTARY_EVENT"
        row = {
            "assertion_id": f"FACT_v2at_{number:04d}", "candidate_id": cid, "event_id": cid,
            "patch_id": "", "source_id": clean(source.get("source_registry_id")) or f"SOURCE_{cid}",
            "source_name": clean(source.get("source_name")) or clean(event.get("primary_source_name")),
            "evidence_role": role, "source_identified": boolean(bool(source or event)),
            "license_explicit": boolean(normalize_license(license_row.get("license_status")) == "LICENSE_EXPLICIT"),
            "crs_resolved": boolean(normalize_crs(license_row.get("crs_status"), role).startswith("CRS_")
                                    and "UNKNOWN" not in normalize_crs(license_row.get("crs_status"), role)),
            "observed_event": boolean(is_true(event.get("observed_event_confirmed"))),
            "temporal_compatible": boolean(clean(event.get("temporal_alignment_status")) == "CLOSED"),
            "hazard_typed": boolean(bool(event_type)),
            "geometry_or_measurement_compatible": boolean(is_true(geometry.get("geometry_present"))),
            "human_review_complete": "false", "independent_corroboration": boolean(bool(event.get("secondary_source_name"))),
            "quickview_can_promote": "false", "susceptibility_can_promote": "false",
            "absence_can_create_negative": "false", "can_create_ground_truth": "false",
            "can_create_training_label": "false",
        }
        row["fact_classification"], row["critical_blocker"] = classify_fact_assertion(row)
        rows.append(row)
    return rows


def _simple_doc(title, rows, columns, filename):
    lines = [f"# {title}", "", f"Rows: {len(rows)}.", ""]
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in rows:
        lines.append("| " + " | ".join(clean(row.get(c)).replace("|", "\\|") for c in columns) + " |")
    write_markdown(doc_path(filename), lines)


def run_build_source_authority_matrix(args=None):
    rows = source_authority_rows()
    columns = list(rows[0].keys())
    write_csv(dataset_path("v2at_source_authority_matrix.csv"), columns, rows)
    _simple_doc("v2at source authority matrix", rows, ["source_id", "authority_level", "default_role"],
                "v2at_source_authority_matrix.md")
    return rows


def run_build_fact_assertion_registry(args=None):
    rows = derive_assertion_rows()
    write_csv(dataset_path("v2at_fact_assertion_registry.csv"), FACT_COLUMNS, rows)
    _simple_doc("v2at fact assertion registry", rows,
                ["candidate_id", "source_name", "fact_classification", "critical_blocker"],
                "v2at_fact_assertion_registry.md")
    return rows


def run_audit_license_crs(args=None):
    facts = load_csv(dataset_path("v2at_fact_assertion_registry.csv"))
    rows = []
    for fact in facts:
        rows.append({
            "audit_id": fact["assertion_id"].replace("FACT_", "LC_"), "assertion_id": fact["assertion_id"],
            "candidate_id": fact["candidate_id"],
            "license_status": "LICENSE_EXPLICIT" if is_true(fact["license_explicit"]) else "LICENSE_UNKNOWN_BLOCKING",
            "crs_status": "CRS_RESOLVED_OR_IRRELEVANT" if is_true(fact["crs_resolved"]) else "CRS_UNKNOWN_BLOCKING",
            "reuse_or_promotion_allowed": "false", "review_required": "true",
        })
    columns = list(rows[0].keys()) if rows else ["audit_id"]
    write_csv(dataset_path("v2at_license_crs_audit.csv"), columns, rows)
    return rows


def run_audit_temporal_alignment(args=None):
    facts = load_csv(dataset_path("v2at_fact_assertion_registry.csv"))
    rows = [{"audit_id": f["assertion_id"].replace("FACT_", "TIME_"), "assertion_id": f["assertion_id"],
             "candidate_id": f["candidate_id"],
             "temporal_status": "TEMPORAL_COMPATIBLE" if is_true(f["temporal_compatible"]) else "TEMPORAL_MISMATCH_OR_UNKNOWN",
             "can_promote": "false"} for f in facts]
    write_csv(dataset_path("v2at_temporal_alignment_audit.csv"), list(rows[0].keys()) if rows else ["audit_id"], rows)
    return rows


def run_audit_hazard_typing(args=None):
    facts = load_csv(dataset_path("v2at_fact_assertion_registry.csv"))
    rows = [{"audit_id": f["assertion_id"].replace("FACT_", "HAZ_"), "assertion_id": f["assertion_id"],
             "candidate_id": f["candidate_id"],
             "hazard_status": "HAZARD_TYPED" if is_true(f["hazard_typed"]) else "HAZARD_AMBIGUOUS_BLOCKING",
             "can_promote": "false"} for f in facts]
    write_csv(dataset_path("v2at_hazard_typing_audit.csv"), list(rows[0].keys()) if rows else ["audit_id"], rows)
    return rows


def run_audit_observed_geometry_status(args=None):
    facts = load_csv(dataset_path("v2at_fact_assertion_registry.csv"))
    rows = [{"audit_id": f["assertion_id"].replace("FACT_", "GEOM_"), "assertion_id": f["assertion_id"],
             "candidate_id": f["candidate_id"],
             "observed_geometry_status": ("COMPATIBLE_GEOMETRY_OR_MEASUREMENT" if
                                          is_true(f["geometry_or_measurement_compatible"]) else
                                          "GEOMETRY_OR_MEASUREMENT_MISSING_BLOCKING"),
             "geometry_inferred": "false", "can_promote": "false"} for f in facts]
    write_csv(dataset_path("v2at_observed_geometry_status.csv"), list(rows[0].keys()) if rows else ["audit_id"], rows)
    return rows


def run_build_event_patch_packages(args=None):
    facts = load_csv(dataset_path("v2at_fact_assertion_registry.csv"))
    rows = []
    for fact in facts:
        rows.append({
            "package_id": fact["assertion_id"].replace("FACT_", "PKG_"), "event_id": fact["event_id"],
            "patch_id": fact["patch_id"], "source_id": fact["source_id"],
            "fact_classification": fact["fact_classification"], "manual_review_packet_ready": "true",
            "patch_truth_allowed": "false", "can_create_ground_truth": "false",
            "can_create_training_label": "false", "blocking_reason": fact["critical_blocker"],
        })
    write_csv(dataset_path("v2at_event_patch_package_index.csv"), list(rows[0].keys()) if rows else ["package_id"], rows)
    return rows


def run_build_download_target_manifest(args=None):
    rows = []
    for number, source in enumerate(TARGET_SOURCES, 1):
        attempt = fetch_to_cache(source["target_url"], source["source_id"])
        rows.append({
            "target_id": f"DL_v2at_{number:03d}", "source_id": source["source_id"],
            "target_url": source["target_url"], "network_enabled": boolean(is_network_enabled()),
            **attempt, "promotion_allowed": "false",
        })
    write_csv(dataset_path("v2at_download_target_manifest.csv"), list(rows[0].keys()), rows)
    return rows


def run_validate_cache_policy(args=None):
    status = validate_cache_policy()
    rows = []
    for fact in load_csv(dataset_path("v2at_fact_assertion_registry.csv")):
        rows.append({
            "blocker_id": fact["assertion_id"].replace("FACT_", "BLOCK_"),
            "scope": fact["assertion_id"], "marker_content_valid": "",
            "tracked_candidate_count": "", "cache_policy_valid": "",
            "promotion_allowed": "false", "can_create_ground_truth": "false",
            "can_create_training_label": "false",
            "blocking_reason": fact["critical_blocker"] or "OBSERVATIONAL_FACT_IS_NOT_OPERATIONAL_GROUND_TRUTH",
        })
    rows.append({"blocker_id": "CACHE_v2at_001", "scope": "evidence_cache", **status,
                 "promotion_allowed": "false", "can_create_ground_truth": "false",
                 "can_create_training_label": "false",
                 "blocking_reason": "Cache is local review support only."})
    write_csv(dataset_path("v2at_promotion_blocker_audit.csv"), list(rows[0].keys()), rows)
    return rows


def evidence_score(row):
    fields = ["source_identified", "license_explicit", "crs_resolved", "observed_event",
              "temporal_compatible", "hazard_typed", "geometry_or_measurement_compatible",
              "human_review_complete", "independent_corroboration"]
    return sum(1 for field in fields if is_true(row.get(field))) * 10


def run_compute_evidence_strength(args=None):
    facts = load_csv(dataset_path("v2at_fact_assertion_registry.csv"))
    rows = [{"score_id": f["assertion_id"].replace("FACT_", "SCORE_"), "assertion_id": f["assertion_id"],
             "candidate_id": f["candidate_id"], "evidence_strength_score": str(evidence_score(f)),
             "fact_classification": f["fact_classification"], "promotion_allowed": "false"} for f in facts]
    write_csv(dataset_path("v2at_evidence_strength_scores.csv"), list(rows[0].keys()) if rows else ["score_id"], rows)
    return rows


def run_generate_non_fact_gap_report(args=None):
    facts = load_csv(dataset_path("v2at_fact_assertion_registry.csv"))
    rows = []
    for fact in facts:
        if fact["fact_classification"] != "FACT_VERIFIED":
            rows.append({
                "gap_id": fact["assertion_id"].replace("FACT_", "GAP_"), "assertion_id": fact["assertion_id"],
                "candidate_id": fact["candidate_id"], "fact_classification": fact["fact_classification"],
                "blocking_gap": fact["critical_blocker"], "do_not_infer": "true",
                "recommended_action": ("RESOLVE_LICENSE_CRS_AND_TEMPORAL_OBSERVATION_FOR_HIGH_PRIORITY_SOURCES"
                                       if "LICENSE" in fact["critical_blocker"] or "CRS" in fact["critical_blocker"]
                                       else "BUILD_MANUAL_REVIEW_PACKETS_FROM_FACT_HARDENED_EVIDENCE"),
            })
    write_csv(dataset_path("v2at_non_fact_gap_report.csv"), list(rows[0].keys()) if rows else
              ["gap_id", "assertion_id", "candidate_id", "fact_classification", "blocking_gap",
               "do_not_infer", "recommended_action"], rows)
    return rows


def _v2at_artifacts():
    artifacts = []
    if os.path.isdir(DATASET_DIR):
        artifacts.extend(dataset_path(name) for name in sorted(os.listdir(DATASET_DIR))
                         if name.startswith("v2at_") and name.endswith(".csv"))
    if os.path.isdir(DOCS_DIR):
        artifacts.extend(doc_path(name) for name in sorted(os.listdir(DOCS_DIR))
                         if name.startswith("v2at_") and name.endswith(".md"))
    return artifacts


def scan_guardrail(path):
    text = read_text(path)
    checks = {
        "absolute_path": bool(ABSOLUTE_PATH_RE.search(text)),
        "ground_truth_promotion": False,
        "training_label_promotion": False,
        "quickview_promotion": False,
        "susceptibility_promotion": False,
        "absence_negative": False,
        "raw_versioned": False,
    }
    if path.endswith(".csv"):
        fields = {
            "ground_truth_promotion": "can_create_ground_truth",
            "training_label_promotion": "can_create_training_label",
            "quickview_promotion": "quickview_can_promote",
            "susceptibility_promotion": "susceptibility_can_promote",
            "absence_negative": "absence_can_create_negative",
            "raw_versioned": "raw_data_versioned",
        }
        for row in load_csv(path):
            for check, field in fields.items():
                checks[check] = checks[check] or is_true(row.get(field))
    return checks


def run_guardrail_regression(args=None):
    rows = []
    failures = 0
    for path in _v2at_artifacts():
        rel = rel_dataset(os.path.basename(path)) if path.startswith(DATASET_DIR) else rel_doc(os.path.basename(path))
        for check, failed in scan_guardrail(path).items():
            failures += int(failed)
            rows.append({"regression_id": f"GR_v2at_{len(rows):05d}", "artifact_path": rel,
                         "check_type": check, "status": "FAIL" if failed else "PASS",
                         "violation_count": "1" if failed else "0"})
    cache = validate_cache_policy()
    failed = not is_true(cache["cache_policy_valid"])
    failures += int(failed)
    rows.append({"regression_id": f"GR_v2at_{len(rows):05d}", "artifact_path": rel_doc("evidence_cache/.gitignore"),
                 "check_type": "cache_policy", "status": "FAIL" if failed else "PASS",
                 "violation_count": "1" if failed else "0"})
    write_csv(dataset_path("v2at_guardrail_regression.csv"), list(rows[0].keys()), rows)
    if failures:
        raise ValueError(f"v2at guardrail regression failed with {failures} violation(s)")
    return rows


def _write_readmes():
    ensure_cache_policy()
    write_markdown(doc_path("README.md"), [
        "# v2at Evidence Fact Hardening",
        "",
        "Audit-only, offline deterministic by default. No label, training, overlay, ground truth,",
        "negative evidence, or prior-output modification is performed.",
    ])
    write_markdown(os.path.join(DOCS_DIR, "download_attempts", "README.md"), [
        "# Download attempts", "", "Network attempts require V2AT_NETWORK=1. Raw payloads stay in the ignored cache."
    ])
    write_markdown(os.path.join(DOCS_DIR, "manual_review_packets", "README.md"), [
        "# Manual review packets", "", "Packages are review-only and never authorize promotion."
    ])


_ORCHESTRATION = [
    ("source_authority_matrix", run_build_source_authority_matrix, ["v2at_source_authority_matrix.csv"]),
    ("fact_assertion_registry", run_build_fact_assertion_registry, ["v2at_fact_assertion_registry.csv"]),
    ("license_crs_audit", run_audit_license_crs, ["v2at_license_crs_audit.csv"]),
    ("temporal_alignment", run_audit_temporal_alignment, ["v2at_temporal_alignment_audit.csv"]),
    ("hazard_typing", run_audit_hazard_typing, ["v2at_hazard_typing_audit.csv"]),
    ("observed_geometry_status", run_audit_observed_geometry_status, ["v2at_observed_geometry_status.csv"]),
    ("event_patch_packages", run_build_event_patch_packages, ["v2at_event_patch_package_index.csv"]),
    ("download_target_manifest", run_build_download_target_manifest, ["v2at_download_target_manifest.csv"]),
    ("cache_policy", run_validate_cache_policy, ["v2at_promotion_blocker_audit.csv"]),
    ("evidence_strength", run_compute_evidence_strength, ["v2at_evidence_strength_scores.csv"]),
    ("non_fact_gap_report", run_generate_non_fact_gap_report, ["v2at_non_fact_gap_report.csv"]),
    ("guardrail_regression", run_guardrail_regression, ["v2at_guardrail_regression.csv"]),
]


def run_orchestrator(args=None):
    _write_readmes()
    rows = []
    for order, (name, function, outputs) in enumerate(_ORCHESTRATION, 1):
        try:
            function(args)
            status, notes = "OK", "Completed."
        except Exception as exc:
            status, notes = "FAIL", f"{type(exc).__name__}: {exc}"
        rows.append({
            "step_order": str(order), "step_name": name, "status": status,
            "outputs": "|".join(rel_dataset(output) for output in outputs),
            "output_hashes": "|".join(sha256_file(dataset_path(output))[:16] for output in outputs),
            "notes": notes,
        })
        if status == "FAIL":
            write_csv(dataset_path("v2at_orchestrator_manifest.csv"), list(rows[0].keys()), rows)
            raise ValueError(notes)
    write_csv(dataset_path("v2at_orchestrator_manifest.csv"), list(rows[0].keys()), rows)
    _simple_doc("v2at orchestrator manifest", rows, ["step_order", "step_name", "status"],
                "v2at_orchestrator_manifest.md")
    return rows


def run_all(args=None):
    return run_orchestrator(args)
