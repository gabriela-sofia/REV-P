#!/usr/bin/env python3
"""v2ag Sentinel date crosswalk discovery.

This stage searches versionable registries for explicit, documented links
between event-patch package v2 patch IDs and Sentinel-dated anchor namespaces.
It is discovery-only: no Sentinel date is applied to package registries, no
overlay is executed, and no ground reference or label is created.
"""

import argparse
import csv
import hashlib
import json
import os
import re
from pathlib import Path

PROTOCOL_VERSION = "v2ag"
DATASET_DIR = "datasets/protocolo_c"
DOCS_DIR = "docs/metodologia_cientifica"
CONFIG_DIR = "configs/protocolo_c"
STAGING_DIR = "local_only/protocolo_c/sentinel_date_crosswalk_discovery/staging/v2ag"
REPORTS_DIR = "local_only/protocolo_c/sentinel_date_crosswalk_discovery/reports/v2ag"

ROOT_SCAN_DIRS = ["datasets", "configs", "docs"]
SCAN_EXTENSIONS = {".csv", ".json", ".yaml", ".yml"}
DOC_EXTENSIONS = {".csv", ".json", ".yaml", ".yml"}

EVENT_PATCH_ID_RE = re.compile(r"^(REC|PET|CUR)_\d{5}$")
EVENT_PATCH_CANDIDATE_RE = re.compile(r"^EPC")
ANCHOR_ID_RE = re.compile(r"^(REFPATCH|REC_PATCH|ANCHOR|OFFICIAL_ANCHOR)")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
ABSOLUTE_PATH_RE = re.compile(r"(?:[A-Za-z]:\\|/Users/|/home/|/mnt/|\\\\)")
TOOL_NAME_RE = re.compile(r"\b(claude|codex|llm|assistant|chatgpt|openai|anthropic|copilot|gemini)\b", re.I)

KEY_FIELDS = [
    "patch_id", "event_patch_candidate_id", "refpatch_id", "reference_patch_id",
    "anchor_patch_id", "source_patch_id", "visual_patch_id", "dino_patch_id",
    "scene_id", "manifest_id", "source_asset_hash", "asset_hash",
    "source_file_hash", "geotiff_hash", "scene_date", "sentinel_scene_date",
]
EVENT_FIELDS = ["event_patch_candidate_id", "patch_id", "event_patch_patch_id"]
ANCHOR_FIELDS = ["refpatch_id", "reference_patch_id", "anchor_patch_id", "source_patch_id"]
SCENE_FIELDS = ["scene_id"]
HASH_FIELDS = ["source_asset_hash", "asset_hash", "source_file_hash", "geotiff_hash"]
DATE_FIELDS = ["scene_date", "sentinel_scene_date", "selected_sentinel_date", "recovered_date"]
DINO_FIELDS = ["dino_patch_id", "visual_patch_id", "explicit_crosswalk_id"]

SOURCE_INVENTORY_COLUMNS = [
    "source_inventory_id", "registry_path", "registry_hash", "row_count",
    "candidate_key_fields", "contains_event_patch_ids", "contains_anchor_ids",
    "contains_scene_ids", "contains_hash_fields", "contains_date_fields",
    "source_status", "should_extract_keys", "notes",
]
KEY_EXTRACTION_COLUMNS = [
    "key_extract_id", "registry_path_hash", "row_hash", "key_type",
    "key_value_hash", "key_value_class", "raw_value_versioned", "region_hint",
    "namespace_hint", "notes",
]
EXPLICIT_COLUMNS = [
    "explicit_crosswalk_id", "event_patch_candidate_id", "event_patch_patch_id",
    "source_namespace", "target_namespace", "target_patch_or_scene_hash",
    "crosswalk_type", "evidence_registry_hash", "evidence_row_hash",
    "explicit_crosswalk_found", "crosswalk_inferred", "can_link_sentinel_date",
    "blocker", "notes",
]
LINEAGE_COLUMNS = [
    "lineage_candidate_id", "event_patch_candidate_id", "event_patch_patch_id",
    "candidate_target_namespace", "candidate_target_id_hash",
    "lineage_evidence_type", "evidence_strength", "accepted_as_explicit_crosswalk",
    "rejected_reason", "crosswalk_inferred", "can_link_sentinel_date", "notes",
]
EVIDENCE_COLUMNS = [
    "evidence_strength_id", "event_patch_candidate_id", "crosswalk_or_candidate_id",
    "evidence_class", "evidence_strength_score", "can_enable_date_linkability",
    "can_update_package_v2", "requires_future_migration", "blocker", "notes",
]
LINKABILITY_COLUMNS = [
    "linkability_audit_id", "event_patch_candidate_id", "patch_id",
    "recovered_date_source_namespace", "recovered_date", "explicit_crosswalk_id",
    "linkability_status", "can_link_sentinel_date", "can_update_v2_package",
    "sentinel_date_inferred", "notes",
]
GUARD_UPDATE_COLUMNS = [
    "guard_update_id", "event_patch_candidate_id", "patch_id",
    "previous_guard_status", "new_guard_status", "prohibited_use", "safe_use",
    "future_allowed_action", "sentinel_date_inferred", "crosswalk_inferred", "notes",
]
TEMPORAL_PREVIEW_COLUMNS = [
    "temporal_preview_id", "event_patch_candidate_id", "event_id", "patch_id",
    "preview_sentinel_date", "preview_temporal_class", "preview_status",
    "applied_to_package", "can_create_ground_reference", "notes",
]
RANKER_COLUMNS = [
    "rank", "next_target", "score", "programming_value", "ground_truth_value",
    "blocker_reduction_value", "expected_effort", "overclaim_risk",
    "recommended_version", "recommended_action", "notes",
]
BLOCKER_MATRIX_COLUMNS = [
    "blocker_id", "blocker", "status", "ground_truth_operational",
    "can_create_ground_reference", "can_create_training_label",
    "can_reopen_protocol_b", "dino_usage", "no_overlay_executed",
    "no_coordinates_invented", "patch_bound_truth", "operational_validation",
    "sentinel_date_crosswalk_discovery_only", "crosswalk_inferred",
    "sentinel_date_inferred", "raw_data_versioned", "notes",
]
NEXT_ACTION_COLUMNS = [
    "action_id", "next_version", "action_type", "priority", "status",
    "requires_explicit_crosswalk", "prohibited_until_new_source", "notes",
]
MANIFEST_COLUMNS = [
    "artifact_id", "artifact_path", "artifact_type", "protocol_version",
    "sha256_prefix", "file_size_bytes", "is_versionable", "reason",
]

V2AG_ARTIFACTS = [
    "scripts/protocolo_c/revp_v2ag_common.py",
    "scripts/protocolo_c/revp_v2ag_crosswalk_source_inventory.py",
    "scripts/protocolo_c/revp_v2ag_patch_identity_key_extractor.py",
    "scripts/protocolo_c/revp_v2ag_explicit_crosswalk_detector.py",
    "scripts/protocolo_c/revp_v2ag_lineage_crosswalk_candidate_builder.py",
    "scripts/protocolo_c/revp_v2ag_crosswalk_evidence_strength_auditor.py",
    "scripts/protocolo_c/revp_v2ag_sentinel_date_linkability_auditor.py",
    "scripts/protocolo_c/revp_v2ag_unlinkable_date_guard_updater.py",
    "scripts/protocolo_c/revp_v2ag_event_patch_temporal_preview_builder.py",
    "scripts/protocolo_c/revp_v2ag_next_programming_target_ranker.py",
    "scripts/protocolo_c/revp_v2ag_completion_report.py",
    "configs/protocolo_c/v2ag_crosswalk_source_inventory_policy.yaml",
    "configs/protocolo_c/v2ag_identity_key_policy.yaml",
    "configs/protocolo_c/v2ag_explicit_crosswalk_policy.yaml",
    "configs/protocolo_c/v2ag_lineage_candidate_policy.yaml",
    "configs/protocolo_c/v2ag_linkability_policy.yaml",
    "configs/protocolo_c/v2ag_next_programming_target_policy.yaml",
    "datasets/protocolo_c/v2ag_crosswalk_source_inventory.csv",
    "datasets/protocolo_c/v2ag_patch_identity_key_extraction.csv",
    "datasets/protocolo_c/v2ag_explicit_crosswalk_detection.csv",
    "datasets/protocolo_c/v2ag_lineage_crosswalk_candidate_registry.csv",
    "datasets/protocolo_c/v2ag_crosswalk_evidence_strength_audit.csv",
    "datasets/protocolo_c/v2ag_sentinel_date_linkability_audit.csv",
    "datasets/protocolo_c/v2ag_unlinkable_date_guard_update.csv",
    "datasets/protocolo_c/v2ag_event_patch_temporal_preview.csv",
    "datasets/protocolo_c/v2ag_next_programming_target_ranker.csv",
    "datasets/protocolo_c/v2ag_ground_reference_blocker_matrix.csv",
    "datasets/protocolo_c/v2ag_next_actions_registry.csv",
    "datasets/protocolo_c/v2ag_versionable_artifacts_manifest.csv",
    "docs/metodologia_cientifica/protocolo_c_v2ag_sentinel_date_crosswalk_discovery.md",
    "docs/metodologia_cientifica/protocolo_c_relatorio_v2ag_sentinel_date_crosswalk_discovery.md",
    "docs/metodologia_cientifica/protocolo_c_status_atual_v2ag.md",
]


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    return parser.parse_args(argv)


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


def relpath(path):
    return Path(path).as_posix()


def sha256_text(value):
    return hashlib.sha256((value or "").strip().encode("utf-8")).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def row_hash(row):
    payload = json.dumps(row, sort_keys=True, ensure_ascii=True)
    return sha256_text(payload)


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


def bool_text(value):
    return "true" if value else "false"


def as_bool(value):
    return str(value).strip().lower() == "true"


def clean_value(value):
    return str(value or "").strip()


def split_values(value):
    raw = clean_value(value)
    if not raw:
        return []
    parts = re.split(r"[|;,]", raw)
    return [p.strip() for p in parts if p.strip()]


def detect_region(value, row=None):
    value = clean_value(value)
    if row:
        for field in ("event_region", "region", "city_code"):
            hint = clean_value(row.get(field))
            if hint in {"REC", "PET", "CUR"}:
                return hint
    match = re.match(r"^(REC|PET|CUR)_", value)
    if match:
        return match.group(1)
    if "PET" in value:
        return "PET"
    if "REC" in value:
        return "REC"
    if "CUR" in value:
        return "CUR"
    return ""


def key_value_class(value, key_type=""):
    value = clean_value(value)
    if not value:
        return "EMPTY"
    if EVENT_PATCH_ID_RE.match(value):
        return "EVENT_PATCH_NUMERIC_ID"
    if EVENT_PATCH_CANDIDATE_RE.match(value):
        return "EVENT_PATCH_CANDIDATE_ID"
    if ANCHOR_ID_RE.match(value):
        return "ANCHOR_OR_RECOVERY_PATCH_ID"
    if DATE_RE.match(value):
        return "DATE_VALUE"
    if key_type in HASH_FIELDS or re.fullmatch(r"[0-9a-fA-F]{16,128}", value):
        return "HASH_VALUE"
    if key_type in SCENE_FIELDS or "S2" in value or "SENTINEL" in value.upper():
        return "SCENE_ID"
    return "OTHER_VERSIONED_IDENTIFIER"


def namespace_hint(value, key_type=""):
    cls = key_value_class(value, key_type)
    if cls == "EVENT_PATCH_NUMERIC_ID" or cls == "EVENT_PATCH_CANDIDATE_ID":
        return "EVENT_PATCH_CANDIDATE_NAMESPACE"
    if clean_value(value).startswith("REFPATCH"):
        return "ANCHOR_REFPATCH_NAMESPACE"
    if clean_value(value).startswith("REC_PATCH"):
        return "RECOVERY_SCAFFOLDING_NAMESPACE"
    if key_type in DINO_FIELDS or clean_value(value).startswith("XW_DINO"):
        return "DINO_REVIEW_NAMESPACE"
    if key_type in SCENE_FIELDS:
        return "SENTINEL_SCENE_NAMESPACE"
    if key_type in HASH_FIELDS:
        return "SOURCE_ASSET_HASH_NAMESPACE"
    return "UNKNOWN_NAMESPACE"


def read_tabular_rows(path):
    suffix = Path(path).suffix.lower()
    try:
        if suffix == ".csv":
            rows = load_csv(path)
            return rows, list(rows[0].keys()) if rows else _csv_header(path)
        if suffix == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                data = data.get("rows", data.get("records", [data]))
            rows = [r for r in data if isinstance(r, dict)] if isinstance(data, list) else []
            keys = sorted({k for row in rows for k in row.keys()})
            return rows, keys
        if suffix in {".yaml", ".yml"}:
            keys = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    m = re.match(r"\s*([A-Za-z0-9_]+)\s*:", line)
                    if m:
                        keys.append(m.group(1))
            return [], sorted(set(keys))
    except (UnicodeDecodeError, csv.Error, json.JSONDecodeError, OSError):
        return [], []
    return [], []


def _csv_header(path):
    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            return next(csv.reader(f), [])
    except (OSError, StopIteration, UnicodeDecodeError, csv.Error):
        return []


def iter_candidate_files():
    for root in ROOT_SCAN_DIRS:
        if not os.path.exists(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            parts = Path(dirpath).parts
            if "local_only" in parts or "local_runs" in parts:
                dirnames[:] = []
                continue
            if root == "docs":
                filenames = [f for f in filenames if Path(f).suffix.lower() in DOC_EXTENSIONS]
            for name in filenames:
                path = os.path.join(dirpath, name)
                suffix = Path(path).suffix.lower()
                if suffix in SCAN_EXTENSIONS:
                    yield relpath(path)


def relevant_fields(fields):
    lowered = {f.lower(): f for f in fields}
    return [lowered[k] for k in KEY_FIELDS if k in lowered]


def file_inventory_record(idx, path):
    rows, fields = read_tabular_rows(path)
    candidates = relevant_fields(fields)
    values = []
    for row in rows[:250]:
        values.extend(clean_value(v) for v in row.values())
    joined_values = "|".join(values)
    contains_event = any(f in fields for f in EVENT_FIELDS) or bool(EVENT_PATCH_ID_RE.search(joined_values))
    contains_anchor = any(f in fields for f in ANCHOR_FIELDS) or bool(ANCHOR_ID_RE.search(joined_values))
    contains_scene = any(f in fields for f in SCENE_FIELDS)
    contains_hash = any(f in fields for f in HASH_FIELDS)
    contains_date = any(f in fields for f in DATE_FIELDS)
    should_extract = bool(candidates and (contains_event or contains_anchor or contains_scene or contains_hash or contains_date))
    return {
        "source_inventory_id": f"CSI_v2ag_{idx:05d}",
        "registry_path": path,
        "registry_hash": sha256_file(path) if os.path.exists(path) else "",
        "row_count": str(len(rows)),
        "candidate_key_fields": "|".join(candidates),
        "contains_event_patch_ids": bool_text(contains_event),
        "contains_anchor_ids": bool_text(contains_anchor),
        "contains_scene_ids": bool_text(contains_scene),
        "contains_hash_fields": bool_text(contains_hash),
        "contains_date_fields": bool_text(contains_date),
        "source_status": "CANDIDATE_REGISTRY_FOUND" if should_extract else "NO_RELEVANT_KEY_FIELDS",
        "should_extract_keys": bool_text(should_extract),
        "notes": "Versionable registry scanned; non-versionable local directories excluded.",
    }


def run_crosswalk_source_inventory(args=None):
    rows = [file_inventory_record(i, path) for i, path in enumerate(sorted(iter_candidate_files()))]
    write_csv(dataset_path("v2ag_crosswalk_source_inventory.csv"), SOURCE_INVENTORY_COLUMNS, rows)
    return rows


def load_inventory():
    path = dataset_path("v2ag_crosswalk_source_inventory.csv")
    if not os.path.exists(path):
        return run_crosswalk_source_inventory(parse_args([]))
    return load_csv(path)


def all_source_rows(inventory=None):
    inventory = inventory if inventory is not None else load_inventory()
    for item in inventory:
        if not as_bool(item.get("should_extract_keys")):
            continue
        path = item["registry_path"]
        if not os.path.exists(path):
            continue
        rows, _ = read_tabular_rows(path)
        for idx, row in enumerate(rows):
            yield path, idx, row


def run_patch_identity_key_extractor(args=None):
    rows = []
    seen = set()
    for path, _, row in all_source_rows():
        reg_hash = sha256_text(path)
        rhash = row_hash(row)
        for key in KEY_FIELDS:
            if key not in row:
                continue
            for value in split_values(row.get(key)):
                value_hash = sha256_text(value)
                dedupe = (reg_hash, rhash, key, value_hash)
                if dedupe in seen:
                    continue
                seen.add(dedupe)
                rows.append({
                    "key_extract_id": f"KEY_v2ag_{len(rows):06d}",
                    "registry_path_hash": reg_hash,
                    "row_hash": rhash,
                    "key_type": key,
                    "key_value_hash": value_hash,
                    "key_value_class": key_value_class(value, key),
                    "raw_value_versioned": "false",
                    "region_hint": detect_region(value, row),
                    "namespace_hint": namespace_hint(value, key),
                    "notes": "Raw key value hashed for public output.",
                })
    write_csv(dataset_path("v2ag_patch_identity_key_extraction.csv"), KEY_EXTRACTION_COLUMNS, rows)
    return rows


def package_rows():
    rows = load_csv(dataset_path("v2ac_event_patch_v2_package_registry.csv"))
    if not rows:
        rows = load_csv(dataset_path("v2ae_canonical_event_patch_registry.csv"))
    return rows


def date_rows():
    return load_csv(dataset_path("v2aa_patch_date_candidate_consolidation.csv"))


def date_by_hash():
    out = {}
    for row in date_rows():
        pid = clean_value(row.get("patch_id") or row.get("date_patch_id"))
        if not pid:
            continue
        out[sha256_text(pid)] = row
    return out


def known_event_patch_ids():
    return {clean_value(r.get("patch_id")) for r in package_rows() if EVENT_PATCH_ID_RE.match(clean_value(r.get("patch_id")))}


def event_patch_for_row(row):
    patch_id = clean_value(row.get("patch_id") or row.get("event_patch_patch_id"))
    epc = clean_value(row.get("event_patch_candidate_id"))
    if EVENT_PATCH_ID_RE.match(patch_id):
        return epc, patch_id
    source_patch = clean_value(row.get("source_patch_id"))
    if EVENT_PATCH_ID_RE.match(source_patch):
        return epc, source_patch
    return "", ""


def _explicit_row(base, target_value, target_namespace, ctype, can_link, blocker, note):
    return {
        "explicit_crosswalk_id": f"XW_v2ag_{base['idx']:06d}",
        "event_patch_candidate_id": base["epc"],
        "event_patch_patch_id": base["patch_id"],
        "source_namespace": "EVENT_PATCH_CANDIDATE_NAMESPACE",
        "target_namespace": target_namespace,
        "target_patch_or_scene_hash": sha256_text(target_value),
        "crosswalk_type": ctype,
        "evidence_registry_hash": sha256_text(base["path"]),
        "evidence_row_hash": row_hash(base["row"]),
        "explicit_crosswalk_found": "true",
        "crosswalk_inferred": "false",
        "can_link_sentinel_date": bool_text(can_link),
        "blocker": blocker,
        "notes": note,
    }


def run_explicit_crosswalk_detector(args=None):
    dated = date_by_hash()
    rows = []
    idx = 0
    for path, _, row in all_source_rows():
        epc, patch_id = event_patch_for_row(row)
        if not patch_id:
            continue
        base = {"idx": idx, "epc": epc, "patch_id": patch_id, "path": path, "row": row}
        for field in ("refpatch_id", "reference_patch_id"):
            target = clean_value(row.get(field))
            if target:
                target_hash = sha256_text(target)
                rows.append(_explicit_row(
                    base, target, namespace_hint(target, field), "PATCH_TO_REFPATCH_EXPLICIT",
                    target_hash in dated, "" if target_hash in dated else "TARGET_DATE_NOT_RECOVERED",
                    "Same versionable row contains event patch and reference patch key.",
                ))
                idx += 1
                base["idx"] = idx
        for field in ("anchor_patch_id", "source_patch_id"):
            target = clean_value(row.get(field))
            if target and target != patch_id:
                target_hash = sha256_text(target)
                rows.append(_explicit_row(
                    base, target, namespace_hint(target, field), "PATCH_TO_ANCHOR_EXPLICIT",
                    target_hash in dated, "" if target_hash in dated else "TARGET_DATE_NOT_RECOVERED",
                    "Same versionable row contains event patch and anchor/source patch key.",
                ))
                idx += 1
                base["idx"] = idx
        scene = clean_value(row.get("scene_id"))
        if scene:
            scene_hash = sha256_text(scene)
            rows.append(_explicit_row(
                base, scene, "SENTINEL_SCENE_NAMESPACE", "PATCH_TO_SCENE_ID_EXPLICIT",
                scene_hash in dated, "" if scene_hash in dated else "SCENE_DATE_NOT_RECOVERED",
                "Same versionable row contains event patch and Sentinel scene key.",
            ))
            idx += 1
            base["idx"] = idx
        for field in HASH_FIELDS:
            target = clean_value(row.get(field))
            if target:
                rows.append(_explicit_row(
                    base, target, "SOURCE_ASSET_HASH_NAMESPACE", "PATCH_TO_SOURCE_HASH_EXPLICIT",
                    False, "HASH_LINKAGE_REQUIRES_FUTURE_MIGRATION",
                    "Same versionable row contains event patch and source asset hash; no date applied.",
                ))
                idx += 1
                base["idx"] = idx
        for field in DINO_FIELDS:
            target = clean_value(row.get(field))
            if target and ("DINO" in target.upper() or field != "explicit_crosswalk_id"):
                rows.append(_explicit_row(
                    base, target, "DINO_REVIEW_NAMESPACE", "PATCH_TO_DINO_EXPLICIT",
                    False, "DINO_CROSSWALK_NOT_SENTINEL_DATE_CROSSWALK",
                    "DINO linkage is explicit but cannot link Sentinel date.",
                ))
                idx += 1
                base["idx"] = idx
    write_csv(dataset_path("v2ag_explicit_crosswalk_detection.csv"), EXPLICIT_COLUMNS, rows)
    return rows


def load_explicit():
    path = dataset_path("v2ag_explicit_crosswalk_detection.csv")
    if not os.path.exists(path):
        return run_explicit_crosswalk_detector(parse_args([]))
    return load_csv(path)


def run_lineage_crosswalk_candidate_builder(args=None):
    explicit = load_explicit()
    rows = []
    explicit_by_patch = {r["event_patch_patch_id"] for r in explicit if as_bool(r.get("can_link_sentinel_date"))}
    for item in explicit:
        evidence_type = {
            "PATCH_TO_REFPATCH_EXPLICIT": "SAME_REGISTRY_WITH_BOTH_KEYS",
            "PATCH_TO_ANCHOR_EXPLICIT": "SAME_REGISTRY_WITH_BOTH_KEYS",
            "PATCH_TO_SCENE_ID_EXPLICIT": "SHARED_SCENE_ID",
            "PATCH_TO_SOURCE_HASH_EXPLICIT": "SHARED_SOURCE_ASSET_HASH",
            "PATCH_TO_DINO_EXPLICIT": "DINO_SIMILARITY_REJECTED",
        }.get(item["crosswalk_type"], "SAME_REGISTRY_WITH_BOTH_KEYS")
        accepted = as_bool(item.get("can_link_sentinel_date"))
        rows.append({
            "lineage_candidate_id": f"LC_v2ag_{len(rows):06d}",
            "event_patch_candidate_id": item["event_patch_candidate_id"],
            "event_patch_patch_id": item["event_patch_patch_id"],
            "candidate_target_namespace": item["target_namespace"],
            "candidate_target_id_hash": item["target_patch_or_scene_hash"],
            "lineage_evidence_type": evidence_type,
            "evidence_strength": "STRONG_EXPLICIT" if accepted else "WEAK_NON_DATE_OR_UNLINKED",
            "accepted_as_explicit_crosswalk": bool_text(accepted),
            "rejected_reason": "" if accepted else item.get("blocker", "NOT_STRONG_DATE_CROSSWALK"),
            "crosswalk_inferred": "false",
            "can_link_sentinel_date": bool_text(accepted),
            "notes": "Candidate recorded without applying any date.",
        })
    dated_regions = {detect_region(r.get("patch_id"), r) for r in date_rows()}
    for pkg in package_rows():
        patch_id = clean_value(pkg.get("patch_id"))
        if not patch_id or patch_id in explicit_by_patch:
            continue
        region = clean_value(pkg.get("event_region") or pkg.get("region") or detect_region(patch_id, pkg))
        if region in dated_regions:
            rows.append({
                "lineage_candidate_id": f"LC_v2ag_{len(rows):06d}",
                "event_patch_candidate_id": clean_value(pkg.get("event_patch_candidate_id")),
                "event_patch_patch_id": patch_id,
                "candidate_target_namespace": "REGION_ONLY_REJECTED_NAMESPACE",
                "candidate_target_id_hash": sha256_text(region),
                "lineage_evidence_type": "REGION_ONLY_REJECTED",
                "evidence_strength": "REJECTED_INFERENTIAL_MATCH",
                "accepted_as_explicit_crosswalk": "false",
                "rejected_reason": "REGION_ONLY_NOT_A_CROSSWALK_KEY",
                "crosswalk_inferred": "false",
                "can_link_sentinel_date": "false",
                "notes": "Region overlap is recorded only as rejected evidence.",
            })
    write_csv(dataset_path("v2ag_lineage_crosswalk_candidate_registry.csv"), LINEAGE_COLUMNS, rows)
    return rows


def load_lineage():
    path = dataset_path("v2ag_lineage_crosswalk_candidate_registry.csv")
    if not os.path.exists(path):
        return run_lineage_crosswalk_candidate_builder(parse_args([]))
    return load_csv(path)


def run_crosswalk_evidence_strength_auditor(args=None):
    rows = []
    for item in load_explicit():
        dino = item["crosswalk_type"] == "PATCH_TO_DINO_EXPLICIT"
        strong_date = as_bool(item.get("can_link_sentinel_date")) and not dino
        rows.append({
            "evidence_strength_id": f"ES_v2ag_{len(rows):06d}",
            "event_patch_candidate_id": item["event_patch_candidate_id"],
            "crosswalk_or_candidate_id": item["explicit_crosswalk_id"],
            "evidence_class": "STRONG_EXPLICIT_CROSSWALK" if strong_date else "WEAK_AMBIGUOUS_CANDIDATE",
            "evidence_strength_score": "100" if strong_date else ("35" if dino else "50"),
            "can_enable_date_linkability": bool_text(strong_date),
            "can_update_package_v2": "false",
            "requires_future_migration": bool_text(strong_date),
            "blocker": "" if strong_date else item.get("blocker", "NO_DATE_LINKABILITY"),
            "notes": "Evidence audited; package v2 update is forbidden in v2ag.",
        })
    explicit_ids = {r["explicit_crosswalk_id"] for r in load_explicit()}
    for item in load_lineage():
        if item["lineage_candidate_id"] in explicit_ids or as_bool(item.get("accepted_as_explicit_crosswalk")):
            continue
        rows.append({
            "evidence_strength_id": f"ES_v2ag_{len(rows):06d}",
            "event_patch_candidate_id": item["event_patch_candidate_id"],
            "crosswalk_or_candidate_id": item["lineage_candidate_id"],
            "evidence_class": "REJECTED_INFERENTIAL_MATCH" if "REJECTED" in item["lineage_evidence_type"] else "WEAK_AMBIGUOUS_CANDIDATE",
            "evidence_strength_score": "0" if "REJECTED" in item["lineage_evidence_type"] else "25",
            "can_enable_date_linkability": "false",
            "can_update_package_v2": "false",
            "requires_future_migration": "false",
            "blocker": item.get("rejected_reason") or "NO_STRONG_EXPLICIT_CROSSWALK",
            "notes": "Candidate cannot enable date linkability.",
        })
    if not rows:
        rows.append({
            "evidence_strength_id": "ES_v2ag_000000",
            "event_patch_candidate_id": "",
            "crosswalk_or_candidate_id": "",
            "evidence_class": "NO_CROSSWALK_EVIDENCE",
            "evidence_strength_score": "0",
            "can_enable_date_linkability": "false",
            "can_update_package_v2": "false",
            "requires_future_migration": "false",
            "blocker": "NO_CROSSWALK_EVIDENCE",
            "notes": "No explicit or candidate crosswalk evidence found.",
        })
    write_csv(dataset_path("v2ag_crosswalk_evidence_strength_audit.csv"), EVIDENCE_COLUMNS, rows)
    return rows


def load_evidence():
    path = dataset_path("v2ag_crosswalk_evidence_strength_audit.csv")
    if not os.path.exists(path):
        return run_crosswalk_evidence_strength_auditor(parse_args([]))
    return load_csv(path)


def run_sentinel_date_linkability_auditor(args=None):
    explicit = [r for r in load_explicit() if as_bool(r.get("can_link_sentinel_date"))]
    explicit_by_patch = {}
    for row in explicit:
        explicit_by_patch.setdefault(row["event_patch_patch_id"], []).append(row)
    dates = date_by_hash()
    rows = []
    for pkg in package_rows():
        patch_id = clean_value(pkg.get("patch_id"))
        matches = explicit_by_patch.get(patch_id, [])
        if matches:
            match = matches[0]
            date = dates.get(match["target_patch_or_scene_hash"], {})
            recovered = clean_value(date.get("selected_sentinel_date") or date.get("recovered_date"))
            conflict = clean_value(date.get("conflict_status")) == "CONFLICT"
            status = "DATE_CONFLICT_BLOCKED" if conflict else "DATE_LINKABILITY_CONFIRMED_BY_EXPLICIT_CROSSWALK"
            rows.append({
                "linkability_audit_id": f"DLA_v2ag_{len(rows):06d}",
                "event_patch_candidate_id": clean_value(pkg.get("event_patch_candidate_id")),
                "patch_id": patch_id,
                "recovered_date_source_namespace": namespace_hint(date.get("patch_id")),
                "recovered_date": "" if conflict else recovered,
                "explicit_crosswalk_id": match["explicit_crosswalk_id"],
                "linkability_status": status,
                "can_link_sentinel_date": bool_text(not conflict and bool(recovered)),
                "can_update_v2_package": "false",
                "sentinel_date_inferred": "false",
                "notes": "Date is reported only as linkability audit; package v2 is unchanged.",
            })
        else:
            missing = clean_value(pkg.get("sentinel_date_status")) == "SENTINEL_DATE_MISSING_WITH_BLOCKER"
            rows.append({
                "linkability_audit_id": f"DLA_v2ag_{len(rows):06d}",
                "event_patch_candidate_id": clean_value(pkg.get("event_patch_candidate_id")),
                "patch_id": patch_id,
                "recovered_date_source_namespace": clean_value(pkg.get("date_source_namespace")) or "UNLINKABLE_NAMESPACE",
                "recovered_date": "",
                "explicit_crosswalk_id": "",
                "linkability_status": "DATE_MISSING" if missing else "DATE_REMAINS_UNLINKABLE",
                "can_link_sentinel_date": "false",
                "can_update_v2_package": "false",
                "sentinel_date_inferred": "false",
                "notes": "No strong explicit crosswalk was found for this event patch.",
            })
    write_csv(dataset_path("v2ag_sentinel_date_linkability_audit.csv"), LINKABILITY_COLUMNS, rows)
    return rows


def load_linkability():
    path = dataset_path("v2ag_sentinel_date_linkability_audit.csv")
    if not os.path.exists(path):
        return run_sentinel_date_linkability_auditor(parse_args([]))
    return load_csv(path)


def run_unlinkable_date_guard_updater(args=None):
    link = {r["patch_id"]: r for r in load_linkability()}
    lineage_by_patch = {r["event_patch_patch_id"]: r for r in load_lineage() if not as_bool(r.get("can_link_sentinel_date"))}
    rows = []
    for pkg in package_rows():
        patch_id = clean_value(pkg.get("patch_id"))
        item = link.get(patch_id, {})
        can_link = as_bool(item.get("can_link_sentinel_date"))
        has_candidate = patch_id in lineage_by_patch
        if can_link:
            new_status = "EXPLICIT_CROSSWALK_FOUND_FUTURE_MIGRATION_ONLY"
            safe_use = "future_package_migration_precheck_only"
            future = "EVENT_PATCH_PACKAGE_V2_CROSSWALK_MIGRATION"
        elif has_candidate:
            new_status = "DATE_LINKABILITY_CANDIDATE_NEEDS_REVIEW_GUARD_RETAINED"
            safe_use = "review_candidate_only_no_date_application"
            future = "manual_registry_lineage_review"
        else:
            new_status = "DATE_REMAINS_UNLINKABLE_GUARD_RETAINED"
            safe_use = "document_blocker_only"
            future = "wait_for_new_explicit_source"
        rows.append({
            "guard_update_id": f"GU_v2ag_{len(rows):06d}",
            "event_patch_candidate_id": clean_value(pkg.get("event_patch_candidate_id")),
            "patch_id": patch_id,
            "previous_guard_status": clean_value(pkg.get("date_linkability_status")) or "UNLINKABLE_NAMESPACE",
            "new_guard_status": new_status,
            "prohibited_use": "apply_date_by_region|apply_date_by_name_similarity|apply_date_by_file_order|apply_date_without_explicit_crosswalk",
            "safe_use": safe_use,
            "future_allowed_action": future,
            "sentinel_date_inferred": "false",
            "crosswalk_inferred": "false",
            "notes": "Guard update is non-operational and does not modify earlier package registries.",
        })
    write_csv(dataset_path("v2ag_unlinkable_date_guard_update.csv"), GUARD_UPDATE_COLUMNS, rows)
    return rows


def run_event_patch_temporal_preview_builder(args=None):
    link = {r["patch_id"]: r for r in load_linkability()}
    rows = []
    for pkg in package_rows():
        patch_id = clean_value(pkg.get("patch_id"))
        item = link.get(patch_id, {})
        can_link = as_bool(item.get("can_link_sentinel_date"))
        rows.append({
            "temporal_preview_id": f"TP_v2ag_{len(rows):06d}",
            "event_patch_candidate_id": clean_value(pkg.get("event_patch_candidate_id")),
            "event_id": clean_value(pkg.get("event_id")),
            "patch_id": patch_id,
            "preview_sentinel_date": clean_value(item.get("recovered_date")) if can_link else "",
            "preview_temporal_class": "EXPLICIT_CROSSWALK_DATE_AVAILABLE" if can_link else "TEMPORAL_LINKAGE_BLOCKED",
            "preview_status": "PREVIEW_ONLY_NOT_APPLIED" if can_link else "NO_TEMPORAL_UPDATE_AVAILABLE",
            "applied_to_package": "false",
            "can_create_ground_reference": "false",
            "notes": "Temporal impact preview only; no package field is updated.",
        })
    write_csv(dataset_path("v2ag_event_patch_temporal_preview.csv"), TEMPORAL_PREVIEW_COLUMNS, rows)
    return rows


def run_next_programming_target_ranker(args=None):
    strong = sum(1 for r in load_linkability() if as_bool(r.get("can_link_sentinel_date")))
    dino = sum(1 for r in load_explicit() if r.get("crosswalk_type") == "PATCH_TO_DINO_EXPLICIT")
    candidates = sum(1 for r in load_lineage() if not as_bool(r.get("accepted_as_explicit_crosswalk")))
    options = [
        {
            "next_target": "EVENT_PATCH_PACKAGE_V2_CROSSWALK_MIGRATION",
            "score": 100 if strong else 5,
            "programming_value": "high" if strong else "low",
            "blocker_reduction_value": "high" if strong else "none",
            "expected_effort": "medium",
            "overclaim_risk": "medium",
            "recommended_action": "migrate_only_after_explicit_crosswalk_review" if strong else "hold",
        },
        {
            "next_target": "DINO_REVIEW_SUPPORT_COMPLETION",
            "score": 60 if dino and not strong else 30,
            "programming_value": "medium",
            "blocker_reduction_value": "support_only",
            "expected_effort": "medium",
            "overclaim_risk": "medium",
            "recommended_action": "keep_support_only_no_execution",
        },
        {
            "next_target": "PUBLIC_SOURCE_RECHECK_HOLD",
            "score": 50 if not strong and candidates else 25,
            "programming_value": "medium",
            "blocker_reduction_value": "uncertain",
            "expected_effort": "low",
            "overclaim_risk": "low",
            "recommended_action": "hold_until_new_versionable_source",
        },
        {
            "next_target": "MULTI_REGION_REGISTRY_MAINTENANCE",
            "score": 35,
            "programming_value": "low",
            "blocker_reduction_value": "none",
            "expected_effort": "low",
            "overclaim_risk": "low",
            "recommended_action": "defer",
        },
        {
            "next_target": "STOP_GROUND_TRUTH_SEARCH_UNTIL_NEW_SOURCE",
            "score": 70 if not strong else 10,
            "programming_value": "high",
            "blocker_reduction_value": "prevents_invalid_promotion",
            "expected_effort": "low",
            "overclaim_risk": "low",
            "recommended_action": "stop_operational_search_until_explicit_source",
        },
    ]
    options.sort(key=lambda r: (-int(r["score"]), r["next_target"]))
    rows = []
    for idx, item in enumerate(options, 1):
        rows.append({
            "rank": str(idx),
            "next_target": item["next_target"],
            "score": str(item["score"]),
            "programming_value": item["programming_value"],
            "ground_truth_value": "none",
            "blocker_reduction_value": item["blocker_reduction_value"],
            "expected_effort": item["expected_effort"],
            "overclaim_risk": item["overclaim_risk"],
            "recommended_version": "v2ah",
            "recommended_action": item["recommended_action"],
            "notes": "Ranked from v2ag evidence counts, not hardcoded final status.",
        })
    write_csv(dataset_path("v2ag_next_programming_target_ranker.csv"), RANKER_COLUMNS, rows)
    return rows


def artifact_type(path):
    if path.startswith("datasets/"):
        return "dataset"
    if path.startswith("configs/"):
        return "config"
    if path.startswith("docs/"):
        return "technical_doc"
    if path.startswith("scripts/"):
        return "script"
    return "artifact"


def build_manifest_rows():
    rows = []
    for path in V2AG_ARTIFACTS:
        resolved = artifact_path(path)
        exists = os.path.exists(resolved)
        rows.append({
            "artifact_id": f"ART_v2ag_{len(rows):03d}",
            "artifact_path": path,
            "artifact_type": artifact_type(path),
            "protocol_version": PROTOCOL_VERSION,
            "sha256_prefix": sha256_file(resolved)[:16] if exists else "",
            "file_size_bytes": str(os.path.getsize(resolved)) if exists else "0",
            "is_versionable": "true",
            "reason": "v2ag discovery artifact; no raw data versioned.",
        })
    return rows


def run_completion_report(args=None):
    inventory = load_csv(dataset_path("v2ag_crosswalk_source_inventory.csv"))
    keys = load_csv(dataset_path("v2ag_patch_identity_key_extraction.csv"))
    explicit = load_explicit()
    lineage = load_lineage()
    evidence = load_evidence()
    linkability = load_linkability()
    guards = load_csv(dataset_path("v2ag_unlinkable_date_guard_update.csv"))
    preview = load_csv(dataset_path("v2ag_event_patch_temporal_preview.csv"))
    ranker = run_next_programming_target_ranker(parse_args([]))

    strong = [r for r in explicit if as_bool(r.get("can_link_sentinel_date")) and r["crosswalk_type"] != "PATCH_TO_DINO_EXPLICIT"]
    dino = [r for r in explicit if r.get("crosswalk_type") == "PATCH_TO_DINO_EXPLICIT"]
    linked = [r for r in linkability if as_bool(r.get("can_link_sentinel_date"))]
    top = ranker[0] if ranker else {}

    blocker_rows = [
        {
            "blocker_id": "BM_v2ag_000",
            "blocker": "unlinkable_missing_sentinel_date",
            "status": "SENTINEL_DATE_CROSSWALK_CANDIDATE_AUDITED_NON_OPERATIONAL",
            "ground_truth_operational": "false",
            "can_create_ground_reference": "false",
            "can_create_training_label": "false",
            "can_reopen_protocol_b": "false",
            "dino_usage": "SUPPORT_ONLY",
            "no_overlay_executed": "true",
            "no_coordinates_invented": "true",
            "patch_bound_truth": "false",
            "operational_validation": "false",
            "sentinel_date_crosswalk_discovery_only": "true",
            "crosswalk_inferred": "false",
            "sentinel_date_inferred": "false",
            "raw_data_versioned": "false",
            "notes": "No operational promotion in v2ag.",
        },
        {
            "blocker_id": "BM_v2ag_001",
            "blocker": "no_ground_reference",
            "status": "BLOCKED_NO_OBSERVED_GEOMETRY_OR_OPERATIONAL_VALIDATION",
            "ground_truth_operational": "false",
            "can_create_ground_reference": "false",
            "can_create_training_label": "false",
            "can_reopen_protocol_b": "false",
            "dino_usage": "SUPPORT_ONLY",
            "no_overlay_executed": "true",
            "no_coordinates_invented": "true",
            "patch_bound_truth": "false",
            "operational_validation": "false",
            "sentinel_date_crosswalk_discovery_only": "true",
            "crosswalk_inferred": "false",
            "sentinel_date_inferred": "false",
            "raw_data_versioned": "false",
            "notes": "Date crosswalk discovery is insufficient for ground reference.",
        },
    ]
    write_csv(dataset_path("v2ag_ground_reference_blocker_matrix.csv"), BLOCKER_MATRIX_COLUMNS, blocker_rows)
    next_rows = [
        {
            "action_id": "NA_v2ag_000",
            "next_version": "v2ah",
            "action_type": top.get("next_target", "STOP_GROUND_TRUTH_SEARCH_UNTIL_NEW_SOURCE"),
            "priority": "1",
            "status": top.get("recommended_action", "hold"),
            "requires_explicit_crosswalk": bool_text(top.get("next_target") == "EVENT_PATCH_PACKAGE_V2_CROSSWALK_MIGRATION"),
            "prohibited_until_new_source": bool_text(top.get("next_target") == "STOP_GROUND_TRUTH_SEARCH_UNTIL_NEW_SOURCE"),
            "notes": "Next action follows v2ag discovery evidence.",
        }
    ]
    write_csv(dataset_path("v2ag_next_actions_registry.csv"), NEXT_ACTION_COLUMNS, next_rows)

    method_lines = [
        "# Protocolo C v2ag Sentinel Date Crosswalk Discovery",
        "",
        "## Scope",
        "",
        "v2ag scans versionable registries for explicit crosswalk evidence between event-patch package v2 IDs and Sentinel-dated anchor namespaces.",
        "",
        "## Allowed Evidence",
        "",
        "- Same-row event patch plus reference patch, anchor patch, scene id, or source hash.",
        "- Documented lineage candidates are recorded without date application.",
        "- Region-only, row-order, name-only, visual similarity, and date-only joins are rejected.",
        "",
        "## Guardrails",
        "",
        "- No overlay execution.",
        "- No coordinate invention.",
        "- No package v2 date update.",
        "- No ground reference or label creation.",
    ]
    write_text(doc_path("protocolo_c_v2ag_sentinel_date_crosswalk_discovery.md"), method_lines)

    report_lines = [
        "# Relatorio tecnico v2ag Sentinel Date Crosswalk Discovery",
        "",
        f"Fontes escaneadas: {len(inventory)}.",
        f"Chaves extraidas: {len(keys)}.",
        f"Crosswalks explicitos de data: {len(strong)}.",
        f"Crosswalks DINO somente suporte: {len(dino)}.",
        f"Lineage candidates registrados: {len(lineage)}.",
        f"Datas linkable por crosswalk explicito: {len(linked)}.",
        f"Guard updates registrados: {len(guards)}.",
        f"Temporal previews registrados: {len(preview)}.",
        "",
        "Resultado: a v2ag nao aplica datas aos packages v2 e nao altera registries anteriores.",
        "Overlay continua bloqueado porque a etapa nao produz geometria observada.",
        "Ground reference continua bloqueado porque nao ha validacao operacional patch-bound.",
        f"Proximo alvo recomendado: {top.get('next_target', '')}.",
        "Proxima versao recomendada: v2ah.",
    ]
    write_text(doc_path("protocolo_c_relatorio_v2ag_sentinel_date_crosswalk_discovery.md"), report_lines)

    status_lines = [
        "# Status atual Protocolo C v2ag",
        "",
        "Gate: SENTINEL_DATE_CROSSWALK_CANDIDATE_AUDITED_NON_OPERATIONAL.",
        f"Explicit date crosswalk count: {len(strong)}.",
        f"Date linkability confirmed count: {len(linked)}.",
        "Package v2 update: false.",
        "Overlay executed: false.",
        "Coordinates invented: false.",
        "Sentinel date inferred: false.",
        "Crosswalk inferred: false.",
        f"Next programming target: {top.get('next_target', '')}.",
    ]
    write_text(doc_path("protocolo_c_status_atual_v2ag.md"), status_lines)

    manifest = build_manifest_rows()
    write_csv(dataset_path("v2ag_versionable_artifacts_manifest.csv"), MANIFEST_COLUMNS, manifest)
    return {
        "inventory": inventory,
        "keys": keys,
        "explicit": explicit,
        "lineage": lineage,
        "evidence": evidence,
        "linkability": linkability,
        "ranker": ranker,
        "manifest": manifest,
    }


def run_all(args=None):
    run_crosswalk_source_inventory(args)
    run_patch_identity_key_extractor(args)
    run_explicit_crosswalk_detector(args)
    run_lineage_crosswalk_candidate_builder(args)
    run_crosswalk_evidence_strength_auditor(args)
    run_sentinel_date_linkability_auditor(args)
    run_unlinkable_date_guard_updater(args)
    run_event_patch_temporal_preview_builder(args)
    run_next_programming_target_ranker(args)
    return run_completion_report(args)
