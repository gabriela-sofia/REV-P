#!/usr/bin/env python3
"""v1us Event-Patch Package Linkage Engine.

Builds auditable, non-operational event-patch packages for Recife, Petropolis
and Curitiba using only consolidated evidence and existing registries. It never
creates ground truth, ground reference, training labels, overlay, inferred
coordinates, patch-bound truth, or new downloads.
"""

import argparse
import csv
import hashlib
import os

PROTOCOL_VERSION = "v1us"
DATASET_DIR = "datasets/protocolo_c"
DATASETS_ROOT = "datasets"
DOCS_DIR = "docs/metodologia_cientifica"
CONFIG_DIR = "configs/protocolo_c"
MAX_STATUS = "EVENT_PATCH_PACKAGE_CANDIDATE_NON_OPERATIONAL"

EVENT_REGISTRY = "v1uo_multiregion_event_registry.csv"
PATCH_REGISTRY_SOURCES = [
    "dino_patch_visual_linkage_registry_v1pv.csv",
]
DINO_REGISTRY_SOURCES = [
    "dino_patch_visual_linkage_registry_v1pv.csv",
]

REGION_NORMALIZE = {
    "CURITIBA": "CUR", "CTB": "CUR", "CUR": "CUR",
    "RECIFE": "REC", "REC": "REC",
    "PETROPOLIS": "PET", "PETRÓPOLIS": "PET", "PET": "PET",
}
REGION_META = {
    "REC": ("Recife", "PE"),
    "PET": ("Petrópolis", "RJ"),
    "CUR": ("Curitiba", "PR"),
}

RESOLUTION_COLUMNS = [
    "patch_resolution_id", "patch_id", "region", "city", "uf",
    "source_registry", "sentinel_scene_date", "has_sentinel_date",
    "has_patch_geometry", "has_patch_bounds", "resolution_status", "notes",
]
CANDIDATE_COLUMNS = [
    "event_patch_candidate_id", "event_id", "region", "patch_id",
    "linkage_basis", "linkage_status", "event_patch_candidate_only",
    "patch_bound_truth", "can_create_ground_reference",
    "can_create_training_label", "blocker", "notes",
]
TEMPORAL_COLUMNS = [
    "temporal_linkage_id", "event_patch_candidate_id", "event_id", "patch_id",
    "region", "event_start_date", "event_end_date", "sentinel_scene_date",
    "has_sentinel_date", "temporal_linkage_class", "notes",
]
EVIDENCE_COLUMNS = [
    "attachment_id", "event_patch_candidate_id", "event_id", "patch_id",
    "evidence_source", "evidence_type", "evidence_strength", "evidence_status",
    "evidence_limitations", "can_support_contextual_review",
    "can_support_overlay", "can_create_ground_reference", "notes",
]
PHENOMENON_COLUMNS = [
    "phenomenon_attachment_id", "event_patch_candidate_id", "event_id",
    "patch_id", "region", "phenomenon_class", "phenomenon_support",
    "phenomenon_basis", "is_observed_label", "can_create_training_label",
    "notes",
]
GEOMETRY_BLOCKER_COLUMNS = [
    "geometry_blocker_id", "event_patch_candidate_id", "event_id", "patch_id",
    "region", "coordinate_status", "geometry_status", "overlay_blocker",
    "ground_reference_blocker", "label_blocker", "no_overlay_executed",
    "no_coordinates_invented", "notes",
]
READINESS_COLUMNS = [
    "readiness_id", "event_patch_candidate_id", "event_id", "patch_id",
    "region", "dimension", "classification", "basis",
    "ground_truth_operational", "can_create_ground_reference",
    "can_create_training_label", "notes",
]
DINO_COLUMNS = [
    "dino_attachment_id", "event_patch_candidate_id", "event_id", "patch_id",
    "region", "dino_registry", "dino_review_support_status", "dino_usage",
    "can_create_training_label", "notes",
]
RANKER_COLUMNS = [
    "rank", "event_id", "region", "candidate_count", "main_blocker",
    "next_action", "action_basis", "expected_programming_value",
    "overclaim_risk", "recommended_next_version", "notes",
]
NEXT_ACTION_COLUMNS = [
    "action_id", "event_id", "action_type", "priority", "description",
    "target", "status", "notes",
]
BLOCKER_MATRIX_COLUMNS = [
    "blocker_id", "event_id", "region", "gate", "gate_status",
    "blocking_reason", "ground_truth_operational", "can_create_ground_reference",
    "can_create_training_label", "notes",
]
MANIFEST_COLUMNS = [
    "artifact_id", "artifact_path", "artifact_type", "protocol_version",
    "sha256_prefix", "file_size_bytes", "is_versionable", "reason",
]

V1US_ARTIFACTS = [
    "configs/protocolo_c/v1us_patch_registry_resolution_policy.yaml",
    "configs/protocolo_c/v1us_event_patch_linkage_policy.yaml",
    "configs/protocolo_c/v1us_temporal_window_policy.yaml",
    "configs/protocolo_c/v1us_external_evidence_attachment_policy.yaml",
    "configs/protocolo_c/v1us_readiness_matrix_policy.yaml",
    "configs/protocolo_c/v1us_next_action_policy.yaml",
    "datasets/protocolo_c/v1us_patch_registry_resolution.csv",
    "datasets/protocolo_c/v1us_event_patch_candidate_registry.csv",
    "datasets/protocolo_c/v1us_event_temporal_window_linkage.csv",
    "datasets/protocolo_c/v1us_external_evidence_attachment_registry.csv",
    "datasets/protocolo_c/v1us_phenomenon_status_attachment.csv",
    "datasets/protocolo_c/v1us_geometry_blocker_attachment.csv",
    "datasets/protocolo_c/v1us_event_patch_readiness_matrix.csv",
    "datasets/protocolo_c/v1us_dino_review_support_attachment.csv",
    "datasets/protocolo_c/v1us_next_action_ranker.csv",
    "datasets/protocolo_c/v1us_ground_reference_blocker_matrix.csv",
    "datasets/protocolo_c/v1us_versionable_artifacts_manifest.csv",
    "datasets/protocolo_c/v1us_next_actions_registry.csv",
    "docs/metodologia_cientifica/protocolo_c_v1us_event_patch_package_linkage_engine.md",
    "docs/metodologia_cientifica/protocolo_c_relatorio_v1us_event_patch_package_linkage_engine.md",
    "docs/metodologia_cientifica/protocolo_c_status_atual_v1us.md",
]


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


def write_text(path, lines):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def bool_text(value):
    return "true" if bool(value) else "false"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_region(value):
    return REGION_NORMALIZE.get((value or "").strip().upper(), (value or "").strip().upper())


def dataset_path(name):
    return os.path.join(DATASET_DIR, name)


def root_dataset_path(name):
    return os.path.join(DATASETS_ROOT, name)


def event_registry():
    return load_csv(dataset_path(EVENT_REGISTRY))


def first_existing(sources):
    for name in sources:
        path = root_dataset_path(name)
        if os.path.exists(path):
            return name, path
    return "", ""


# ---------------------------------------------------------------------------
# 1. Patch registry resolver
# ---------------------------------------------------------------------------
def run_patch_registry_resolver():
    source_name, source_path = first_existing(PATCH_REGISTRY_SOURCES)
    rows = []
    if not source_path:
        rows.append({
            "patch_resolution_id": "PR_v1us_00000",
            "patch_id": "", "region": "", "city": "", "uf": "",
            "source_registry": "", "sentinel_scene_date": "",
            "has_sentinel_date": "false", "has_patch_geometry": "false",
            "has_patch_bounds": "false",
            "resolution_status": "PATCH_REGISTRY_MISSING",
            "notes": "No real patch registry found; patch_id not invented.",
        })
    else:
        seen = set()
        for r in load_csv(source_path):
            patch_id = (r.get("patch_id") or "").strip()
            region = normalize_region(r.get("region"))
            if not patch_id or (patch_id, region) in seen:
                continue
            seen.add((patch_id, region))
            city, uf = REGION_META.get(region, ("", ""))
            rows.append({
                "patch_resolution_id": f"PR_v1us_{len(rows):05d}",
                "patch_id": patch_id,
                "region": region,
                "city": city,
                "uf": uf,
                "source_registry": source_name,
                "sentinel_scene_date": "",
                "has_sentinel_date": "false",
                "has_patch_geometry": "false",
                "has_patch_bounds": "false",
                "resolution_status": "RESOLVED_PATCH_SENTINEL_DATE_MISSING",
                "notes": "Patch resolved from real registry; SENTINEL_DATE_MISSING; geometry not present.",
            })
    out = dataset_path("v1us_patch_registry_resolution.csv")
    write_csv(out, RESOLUTION_COLUMNS, rows)
    print(f"[v1us patch resolver] rows={len(rows)} source={source_name or 'MISSING'} -> {out}")
    return rows


def patch_resolution():
    return load_csv(dataset_path("v1us_patch_registry_resolution.csv"))


# ---------------------------------------------------------------------------
# 2. Event-patch candidate builder
# ---------------------------------------------------------------------------
def is_clear_event(event):
    event_id = event.get("event_id", "")
    region = normalize_region(event.get("region"))
    if region == "CUR":
        return False
    if "EVENT_REGISTRY_MISSING" in event_id:
        return False
    return bool(event.get("start_date"))


def run_event_patch_candidate_builder():
    events = event_registry()
    patches = patch_resolution()
    by_region = {}
    for p in patches:
        by_region.setdefault(p.get("region", ""), []).append(p)
    rows = []
    for event in events:
        event_id = event.get("event_id", "")
        region = normalize_region(event.get("region"))
        if not is_clear_event(event):
            rows.append({
                "event_patch_candidate_id": f"EPC_v1us_{len(rows):05d}",
                "event_id": event_id,
                "region": region,
                "patch_id": "",
                "linkage_basis": "EVENT_REGISTRY_MISSING",
                "linkage_status": "BLOCKED_NO_CLEAR_EVENT",
                "event_patch_candidate_only": "true",
                "patch_bound_truth": "false",
                "can_create_ground_reference": "false",
                "can_create_training_label": "false",
                "blocker": "CURITIBA_EVENT_REGISTRY_MISSING" if region == "CUR" else "EVENT_REGISTRY_MISSING",
                "notes": "No clear event registry; patches not linked to avoid inventing event-patch truth.",
            })
            continue
        region_patches = by_region.get(region, [])
        if not region_patches:
            rows.append({
                "event_patch_candidate_id": f"EPC_v1us_{len(rows):05d}",
                "event_id": event_id,
                "region": region,
                "patch_id": "",
                "linkage_basis": "REGION_PATCHES_MISSING",
                "linkage_status": "BLOCKED_PATCH_REGISTRY_MISSING",
                "event_patch_candidate_only": "true",
                "patch_bound_truth": "false",
                "can_create_ground_reference": "false",
                "can_create_training_label": "false",
                "blocker": "PATCH_EVENT_LINKAGE_NOT_AVAILABLE",
                "notes": "No region patches resolved for this event.",
            })
            continue
        for p in region_patches:
            rows.append({
                "event_patch_candidate_id": f"EPC_v1us_{len(rows):05d}",
                "event_id": event_id,
                "region": region,
                "patch_id": p.get("patch_id", ""),
                "linkage_basis": "REGION_ONLY_CANDIDATE_NO_SPATIAL_DISTANCE",
                "linkage_status": "CANDIDATE_NON_OPERATIONAL",
                "event_patch_candidate_only": "true",
                "patch_bound_truth": "false",
                "can_create_ground_reference": "false",
                "can_create_training_label": "false",
                "blocker": "SENTINEL_DATE_AND_GEOMETRY_MISSING",
                "notes": "Region-only candidate linkage; no overlay, no coordinate inference, no distance.",
            })
    out = dataset_path("v1us_event_patch_candidate_registry.csv")
    write_csv(out, CANDIDATE_COLUMNS, rows)
    print(f"[v1us candidates] rows={len(rows)} -> {out}")
    return rows


def candidates():
    return load_csv(dataset_path("v1us_event_patch_candidate_registry.csv"))


def event_index():
    return {e.get("event_id", ""): e for e in event_registry()}


# ---------------------------------------------------------------------------
# 3. Event temporal window linker
# ---------------------------------------------------------------------------
def run_event_temporal_window_linker():
    events = event_index()
    patch_dates = {(p.get("patch_id", ""), p.get("region", "")): p for p in patch_resolution()}
    rows = []
    for cand in candidates():
        event = events.get(cand.get("event_id", ""), {})
        region = cand.get("region", "")
        patch_id = cand.get("patch_id", "")
        patch = patch_dates.get((patch_id, region), {})
        scene_date = patch.get("sentinel_scene_date", "") if patch else ""
        has_date = bool(scene_date)
        if not patch_id:
            cls = "PATCH_EVENT_LINKAGE_NOT_AVAILABLE"
        elif not has_date:
            cls = "SENTINEL_DATE_MISSING"
        else:
            cls = "TEMPORAL_DISTANCE_UNKNOWN"
        rows.append({
            "temporal_linkage_id": f"TWL_v1us_{len(rows):05d}",
            "event_patch_candidate_id": cand.get("event_patch_candidate_id", ""),
            "event_id": cand.get("event_id", ""),
            "patch_id": patch_id,
            "region": region,
            "event_start_date": event.get("start_date", ""),
            "event_end_date": event.get("end_date", ""),
            "sentinel_scene_date": scene_date,
            "has_sentinel_date": bool_text(has_date),
            "temporal_linkage_class": cls,
            "notes": "Temporal class recorded only; Sentinel date never invented; no auto-exclusion.",
        })
    out = dataset_path("v1us_event_temporal_window_linkage.csv")
    write_csv(out, TEMPORAL_COLUMNS, rows)
    print(f"[v1us temporal] rows={len(rows)} -> {out}")
    return rows


# ---------------------------------------------------------------------------
# Region-level consolidated facts (derived from confirmed registries)
# ---------------------------------------------------------------------------
def region_evidence_profile(event_id, region):
    if region == "REC":
        return {
            "evidence_source": "v1un_recife_human_review_evidence_consolidation",
            "evidence_type": "LOCALITY_ONLY_HUMAN_REVIEW",
            "evidence_strength": "STRONG_CONTEXTUAL_LOCALITY_ONLY",
            "evidence_status": "CONSOLIDATED_NON_OPERATIONAL",
            "evidence_limitations": "locality_only_no_observed_coordinates",
            "can_support_contextual_review": "true",
        }
    if event_id == "PET_2022_02_15":
        return {
            "evidence_source": "v1up_petropolis_sgb_rigeo_registry;v1uq_phenomenon_separation_decision_matrix;v1ur_petropolis_event_status_registry",
            "evidence_type": "OFFICIAL_DOCUMENT_SGB_RIGEO_PARTIAL_PHENOMENON_SEPARATION",
            "evidence_strength": "MODERATE_DOCUMENTARY_NO_GEODATA",
            "evidence_status": "DOCUMENT_ONLY_NO_GEODATA",
            "evidence_limitations": "phenomenon_separation_partial_textual_no_geometry",
            "can_support_contextual_review": "true",
        }
    if event_id == "PET_2024_03_21_28":
        return {
            "evidence_source": "v1up_petropolis_event_status_registry;v1ur_petropolis_event_status_registry",
            "evidence_type": "OFFICIAL_DOCUMENT_HYDROMET_CONTEXT",
            "evidence_strength": "WEAK_DOCUMENTARY_NO_PUBLIC_PATH",
            "evidence_status": "BLOCKED_NO_PUBLIC_PATH",
            "evidence_limitations": "document_only_no_geometry_no_public_geodata_path",
            "can_support_contextual_review": "true",
        }
    if region == "CUR":
        return {
            "evidence_source": "",
            "evidence_type": "EVENT_REGISTRY_MISSING",
            "evidence_strength": "ABSENT",
            "evidence_status": "EVENT_REGISTRY_MISSING",
            "evidence_limitations": "no_clear_event_registry_no_official_source_bound",
            "can_support_contextual_review": "false",
        }
    return {
        "evidence_source": "",
        "evidence_type": "UNKNOWN",
        "evidence_strength": "UNKNOWN",
        "evidence_status": "UNKNOWN",
        "evidence_limitations": "no_consolidated_evidence_attached",
        "can_support_contextual_review": "false",
    }


def region_phenomenon_profile(event_id, region):
    if region == "REC":
        return ("URBAN_FLOOD", "MODERATE_TEXTUAL", "locality_only_hazard_textual_support")
    if event_id == "PET_2022_02_15":
        return ("MIXED_FLOOD_AND_MASS_MOVEMENT", "PARTIAL_TEXTUAL", "phenomenon_separation_partial_textual_no_geometry")
    if event_id == "PET_2024_03_21_28":
        return ("MIXED_HYDROMET_CONTEXT", "WEAK", "document_only_no_geometry")
    if region == "CUR":
        return ("UNKNOWN", "UNKNOWN", "event_registry_missing")
    return ("UNKNOWN", "UNKNOWN", "no_phenomenon_status_available")


def region_geometry_profile(region):
    if region == "REC":
        coord = "LOCALITY_ONLY_NO_COORDINATES"
        geom = "NO_OBSERVED_GEOMETRY"
        overlay = "no_coordinates_locality_only"
    elif region == "PET":
        coord = "HYDROMET_CONTEXT_COORDINATE_ONLY"
        geom = "GEOMETRY_STILL_MISSING"
        overlay = "document_only_geometry_missing"
    elif region == "CUR":
        coord = "EVENT_REGISTRY_MISSING"
        geom = "EVENT_REGISTRY_MISSING"
        overlay = "event_registry_missing"
    else:
        coord = "UNKNOWN"
        geom = "UNKNOWN"
        overlay = "no_geometry_available"
    return coord, geom, overlay


# ---------------------------------------------------------------------------
# 4. External evidence attacher
# ---------------------------------------------------------------------------
def run_external_evidence_attacher():
    rows = []
    for cand in candidates():
        event_id = cand.get("event_id", "")
        region = cand.get("region", "")
        prof = region_evidence_profile(event_id, region)
        rows.append({
            "attachment_id": f"EVA_v1us_{len(rows):05d}",
            "event_patch_candidate_id": cand.get("event_patch_candidate_id", ""),
            "event_id": event_id,
            "patch_id": cand.get("patch_id", ""),
            "evidence_source": prof["evidence_source"],
            "evidence_type": prof["evidence_type"],
            "evidence_strength": prof["evidence_strength"],
            "evidence_status": prof["evidence_status"],
            "evidence_limitations": prof["evidence_limitations"],
            "can_support_contextual_review": prof["can_support_contextual_review"],
            "can_support_overlay": "false",
            "can_create_ground_reference": "false",
            "notes": "Consolidated evidence attached as context only; no promotion.",
        })
    out = dataset_path("v1us_external_evidence_attachment_registry.csv")
    write_csv(out, EVIDENCE_COLUMNS, rows)
    print(f"[v1us evidence] rows={len(rows)} -> {out}")
    return rows


# ---------------------------------------------------------------------------
# 5. Phenomenon status attacher
# ---------------------------------------------------------------------------
def run_phenomenon_status_attacher():
    rows = []
    for cand in candidates():
        event_id = cand.get("event_id", "")
        region = cand.get("region", "")
        phen_class, support, basis = region_phenomenon_profile(event_id, region)
        rows.append({
            "phenomenon_attachment_id": f"PHE_v1us_{len(rows):05d}",
            "event_patch_candidate_id": cand.get("event_patch_candidate_id", ""),
            "event_id": event_id,
            "patch_id": cand.get("patch_id", ""),
            "region": region,
            "phenomenon_class": phen_class,
            "phenomenon_support": support,
            "phenomenon_basis": basis,
            "is_observed_label": "false",
            "can_create_training_label": "false",
            "notes": "Phenomenon status is contextual support only; never an observed label.",
        })
    out = dataset_path("v1us_phenomenon_status_attachment.csv")
    write_csv(out, PHENOMENON_COLUMNS, rows)
    print(f"[v1us phenomenon] rows={len(rows)} -> {out}")
    return rows


# ---------------------------------------------------------------------------
# 6. Geometry blocker attacher
# ---------------------------------------------------------------------------
def run_geometry_blocker_attacher():
    rows = []
    for cand in candidates():
        event_id = cand.get("event_id", "")
        region = cand.get("region", "")
        coord, geom, overlay = region_geometry_profile(region)
        rows.append({
            "geometry_blocker_id": f"GEO_v1us_{len(rows):05d}",
            "event_patch_candidate_id": cand.get("event_patch_candidate_id", ""),
            "event_id": event_id,
            "patch_id": cand.get("patch_id", ""),
            "region": region,
            "coordinate_status": coord,
            "geometry_status": geom,
            "overlay_blocker": overlay,
            "ground_reference_blocker": "ground_reference_forbidden_no_observed_geometry",
            "label_blocker": "training_label_forbidden",
            "no_overlay_executed": "true",
            "no_coordinates_invented": "true",
            "notes": "Overlay and ground reference remain blocked; geometry not inferred.",
        })
    out = dataset_path("v1us_geometry_blocker_attachment.csv")
    write_csv(out, GEOMETRY_BLOCKER_COLUMNS, rows)
    print(f"[v1us geometry blockers] rows={len(rows)} -> {out}")
    return rows


# ---------------------------------------------------------------------------
# 7. Event-patch readiness matrix builder
# ---------------------------------------------------------------------------
def readiness_dimensions(event_id, region, has_patch, has_sentinel_date, dino_available):
    temporal = "UNKNOWN" if has_patch and not has_sentinel_date else ("MODERATE" if has_sentinel_date else "ABSENT")
    if region == "REC":
        base = {
            "temporal_linkage": temporal,
            "official_source_support": "STRONG",
            "phenomenon_support": "MODERATE",
            "locality_support": "STRONG",
            "coordinate_support": "BLOCKED",
            "geometry_support": "BLOCKED",
        }
    elif event_id == "PET_2022_02_15":
        base = {
            "temporal_linkage": temporal,
            "official_source_support": "STRONG",
            "phenomenon_support": "MODERATE",
            "locality_support": "WEAK",
            "coordinate_support": "MODERATE",
            "geometry_support": "BLOCKED",
        }
    elif event_id == "PET_2024_03_21_28":
        base = {
            "temporal_linkage": temporal,
            "official_source_support": "STRONG",
            "phenomenon_support": "WEAK",
            "locality_support": "WEAK",
            "coordinate_support": "MODERATE",
            "geometry_support": "BLOCKED",
        }
    elif region == "CUR":
        base = {
            "temporal_linkage": "ABSENT",
            "official_source_support": "ABSENT",
            "phenomenon_support": "ABSENT",
            "locality_support": "ABSENT",
            "coordinate_support": "ABSENT",
            "geometry_support": "ABSENT",
        }
    else:
        base = {
            "temporal_linkage": "UNKNOWN",
            "official_source_support": "UNKNOWN",
            "phenomenon_support": "UNKNOWN",
            "locality_support": "UNKNOWN",
            "coordinate_support": "UNKNOWN",
            "geometry_support": "UNKNOWN",
        }
    base["overlay_readiness"] = "BLOCKED"
    base["ground_reference_readiness"] = "BLOCKED"
    base["training_readiness"] = "BLOCKED"
    base["dino_review_support"] = "MODERATE" if dino_available else ("ABSENT" if region == "CUR" else "ABSENT")
    return base


def run_event_patch_readiness_matrix_builder():
    temporal_index = {r.get("event_patch_candidate_id", ""): r for r in load_csv(dataset_path("v1us_event_temporal_window_linkage.csv"))}
    # DINO support is derivable directly: patches are resolved from the DINO
    # registry, so a candidate with a patch_id has review-only support when that
    # registry exists. This removes any dependency on the DINO attacher having
    # run first (the documented step order builds readiness before DINO).
    _, dino_source = first_existing(DINO_REGISTRY_SOURCES)
    rows = []
    for cand in candidates():
        epc = cand.get("event_patch_candidate_id", "")
        event_id = cand.get("event_id", "")
        region = cand.get("region", "")
        patch_id = cand.get("patch_id", "")
        tw = temporal_index.get(epc, {})
        has_sentinel = tw.get("has_sentinel_date", "false") == "true"
        dino_available = bool(patch_id) and bool(dino_source)
        dims = readiness_dimensions(event_id, region, bool(patch_id), has_sentinel, dino_available)
        for dim, cls in dims.items():
            rows.append({
                "readiness_id": f"RDY_v1us_{len(rows):05d}",
                "event_patch_candidate_id": epc,
                "event_id": event_id,
                "patch_id": patch_id,
                "region": region,
                "dimension": dim,
                "classification": cls,
                "basis": cand.get("linkage_basis", ""),
                "ground_truth_operational": "false",
                "can_create_ground_reference": "false",
                "can_create_training_label": "false",
                "notes": "Readiness dimension; overlay/ground_reference/training remain BLOCKED.",
            })
    out = dataset_path("v1us_event_patch_readiness_matrix.csv")
    write_csv(out, READINESS_COLUMNS, rows)
    print(f"[v1us readiness] rows={len(rows)} -> {out}")
    return rows


# ---------------------------------------------------------------------------
# 8. DINO review support attacher
# ---------------------------------------------------------------------------
def run_dino_review_support_attacher():
    source_name, source_path = first_existing(DINO_REGISTRY_SOURCES)
    dino_patches = set()
    if source_path:
        for r in load_csv(source_path):
            pid = (r.get("patch_id") or "").strip()
            if pid:
                dino_patches.add((pid, normalize_region(r.get("region"))))
    rows = []
    for cand in candidates():
        patch_id = cand.get("patch_id", "")
        region = cand.get("region", "")
        if not patch_id:
            status = "DINO_NOT_APPLICABLE"
            registry = ""
        elif not source_path:
            status = "DINO_REGISTRY_MISSING"
            registry = ""
        elif (patch_id, region) in dino_patches:
            status = "DINO_REVIEW_SUPPORT_AVAILABLE"
            registry = source_name
        else:
            status = "DINO_REVIEW_SUPPORT_MISSING"
            registry = source_name
        rows.append({
            "dino_attachment_id": f"DINO_v1us_{len(rows):05d}",
            "event_patch_candidate_id": cand.get("event_patch_candidate_id", ""),
            "event_id": cand.get("event_id", ""),
            "patch_id": patch_id,
            "region": region,
            "dino_registry": registry,
            "dino_review_support_status": status,
            "dino_usage": "SUPPORT_ONLY",
            "can_create_training_label": "false",
            "notes": "DINO is review-only structural support; never label or truth; no new extraction.",
        })
    out = dataset_path("v1us_dino_review_support_attachment.csv")
    write_csv(out, DINO_COLUMNS, rows)
    print(f"[v1us dino support] rows={len(rows)} -> {out}")
    return rows


# ---------------------------------------------------------------------------
# 9. Next action ranker
# ---------------------------------------------------------------------------
def event_action_profile(region):
    if region == "REC":
        return ("coordinate_support_blocked", "RECIFE_COORDINATE_RECOVERY",
                "strong_consolidated_evidence_single_missing_dimension_coordinates",
                90, "MEDIUM", "v1ut - Recife Coordinate Recovery from Public CKAN")
    if region == "PET":
        return ("geometry_still_missing_public_path_exhausted",
                "PETROPOLIS_GEOMETRY_SEARCH_EXHAUSTED_SWITCH_REGION",
                "public_geodata_path_exhausted_after_v1ur_no_geometry",
                60, "HIGH", "v1ut - Sentinel Date Recovery for Event-Patch Packages")
    if region == "CUR":
        return ("event_registry_missing", "CURITIBA_EVENT_REGISTRY_DISCOVERY",
                "no_clear_event_registry_discovery_is_prerequisite",
                70, "MEDIUM", "v1ut - Curitiba Event Registry and Public Source Discovery")
    return ("no_geometry", "HOLD_NO_GEOMETRY", "no_actionable_blocker_resolved",
            10, "HIGH", "v1ut - Event-Patch Package Hold")


def run_next_action_ranker():
    cand_rows = candidates()
    counts = {}
    order = []
    for c in cand_rows:
        key = c.get("event_id", "")
        if key not in counts:
            counts[key] = {"region": c.get("region", ""), "count": 0}
            order.append(key)
        counts[key]["count"] += 1
    scored = []
    for event_id in order:
        region = counts[event_id]["region"]
        blocker, action, basis, value, risk, next_version = event_action_profile(region)
        scored.append((value, event_id, region, counts[event_id]["count"], blocker, action, basis, risk, next_version))
    scored.sort(key=lambda item: (-item[0], item[1]))
    rows = []
    for rank, (value, event_id, region, count, blocker, action, basis, risk, next_version) in enumerate(scored, start=1):
        rows.append({
            "rank": str(rank),
            "event_id": event_id,
            "region": region,
            "candidate_count": str(count),
            "main_blocker": blocker,
            "next_action": action,
            "action_basis": basis,
            "expected_programming_value": str(value),
            "overclaim_risk": risk,
            "recommended_next_version": next_version,
            "notes": "Ranked by programming value from real blockers; no operational promotion.",
        })
    out = dataset_path("v1us_next_action_ranker.csv")
    write_csv(out, RANKER_COLUMNS, rows)
    print(f"[v1us next action ranker] rows={len(rows)} -> {out}")
    return rows


# ---------------------------------------------------------------------------
# 10. Completion report
# ---------------------------------------------------------------------------
def write_policy_configs():
    os.makedirs(CONFIG_DIR, exist_ok=True)
    policies = {
        "v1us_patch_registry_resolution_policy.yaml": [
            "protocol_version: v1us",
            "invent_patch_id_allowed: false",
            "invent_sentinel_date_allowed: false",
            "patch_registry_sources:",
        ] + [f"  - {s}" for s in PATCH_REGISTRY_SOURCES],
        "v1us_event_patch_linkage_policy.yaml": [
            "protocol_version: v1us",
            "linkage_basis: region_only_candidate_no_spatial_distance",
            "event_patch_candidate_only: true",
            "patch_bound_truth: false",
            "use_spatial_distance: false",
            "use_inferred_coordinate: false",
            "curitiba_requires_clear_event_registry: true",
        ],
        "v1us_temporal_window_policy.yaml": [
            "protocol_version: v1us",
            "invent_sentinel_date_allowed: false",
            "auto_exclude_on_temporal_mismatch: false",
            "temporal_classes: [WITHIN_EVENT_WINDOW, PRE_EVENT_WINDOW, POST_EVENT_WINDOW, TEMPORAL_DISTANCE_UNKNOWN, SENTINEL_DATE_MISSING]",
        ],
        "v1us_external_evidence_attachment_policy.yaml": [
            "protocol_version: v1us",
            "can_support_overlay: false",
            "can_create_ground_reference: false",
            "recife_evidence: locality_only_human_review",
            "petropolis_evidence: document_only_no_geodata",
        ],
        "v1us_readiness_matrix_policy.yaml": [
            "protocol_version: v1us",
            "classifications: [STRONG, MODERATE, WEAK, ABSENT, BLOCKED, UNKNOWN]",
            "ground_reference_blocked_without_geometry: true",
            "overlay_readiness: BLOCKED",
            "training_readiness: BLOCKED",
        ],
        "v1us_next_action_policy.yaml": [
            "protocol_version: v1us",
            "ranking_basis: programming_value_from_real_blockers",
            "petropolis_after_v1ur: do_not_insist_switch_region_or_date_recovery",
            "options: [CURITIBA_EVENT_REGISTRY_DISCOVERY, RECIFE_COORDINATE_RECOVERY, PETROPOLIS_GEOMETRY_SEARCH_EXHAUSTED_SWITCH_REGION, EVENT_PATCH_SENTINEL_DATE_RECOVERY, DINO_REVIEW_SUPPORT_COMPLETION, HOLD_NO_GEOMETRY]",
        ],
    }
    for name, lines in policies.items():
        write_text(os.path.join(CONFIG_DIR, name), lines)


def build_ground_reference_blocker_matrix():
    rows = []
    seen = set()
    for cand in candidates():
        event_id = cand.get("event_id", "")
        region = cand.get("region", "")
        if event_id in seen:
            continue
        seen.add(event_id)
        for gate, reason in [
            ("overlay_preflight", "overlay_forbidden_no_observed_geometry"),
            ("ground_reference", "ground_reference_forbidden_no_observed_geometry"),
            ("training_label", "training_label_forbidden"),
            ("observed_geometry", "observed_geometry_missing"),
            ("sentinel_scene_date", "sentinel_date_missing"),
        ]:
            rows.append({
                "blocker_id": f"BLOCK_v1us_{event_id}_{gate}",
                "event_id": event_id,
                "region": region,
                "gate": gate,
                "gate_status": "BLOCKED",
                "blocking_reason": reason,
                "ground_truth_operational": "false",
                "can_create_ground_reference": "false",
                "can_create_training_label": "false",
                "notes": "Ground reference and overlay remain blocked in v1us.",
            })
    write_csv(dataset_path("v1us_ground_reference_blocker_matrix.csv"), BLOCKER_MATRIX_COLUMNS, rows)
    return rows


def run_completion_report():
    write_policy_configs()
    resolution = patch_resolution()
    cand_rows = candidates()
    temporal = load_csv(dataset_path("v1us_event_temporal_window_linkage.csv"))
    evidence = load_csv(dataset_path("v1us_external_evidence_attachment_registry.csv"))
    phenomenon = load_csv(dataset_path("v1us_phenomenon_status_attachment.csv"))
    geometry = load_csv(dataset_path("v1us_geometry_blocker_attachment.csv"))
    readiness = load_csv(dataset_path("v1us_event_patch_readiness_matrix.csv"))
    dino = load_csv(dataset_path("v1us_dino_review_support_attachment.csv"))
    ranker = load_csv(dataset_path("v1us_next_action_ranker.csv"))

    build_ground_reference_blocker_matrix()

    top = ranker[0] if ranker else {}
    next_version = top.get("recommended_next_version", "v1ut - Event-Patch Package Hold")
    write_csv(dataset_path("v1us_next_actions_registry.csv"), NEXT_ACTION_COLUMNS, [{
        "action_id": "ACT_v1us_0000",
        "event_id": top.get("event_id", ""),
        "action_type": "PROGRAMMING_DEEPENING",
        "priority": "1",
        "description": next_version,
        "target": top.get("region", ""),
        "status": "PENDING",
        "notes": "Selected by v1us next-action ranker; non-operational.",
    }])

    manifest = []
    for idx, path in enumerate(V1US_ARTIFACTS):
        exists = os.path.exists(path)
        manifest.append({
            "artifact_id": f"ART_v1us_{idx:04d}",
            "artifact_path": path.replace("\\", "/"),
            "artifact_type": "config" if path.startswith("configs/") else "doc" if path.startswith("docs/") else "dataset",
            "protocol_version": PROTOCOL_VERSION,
            "sha256_prefix": sha256_file(path)[:16] if exists else "MISSING",
            "file_size_bytes": str(os.path.getsize(path) if exists else 0),
            "is_versionable": bool_text(exists),
            "reason": "Safe v1us engineering artifact" if exists else "File not found",
        })
    write_csv(dataset_path("v1us_versionable_artifacts_manifest.csv"), MANIFEST_COLUMNS, manifest)

    def region_status(region):
        statuses = {c.get("event_id"): c.get("linkage_status") for c in cand_rows if c.get("region") == region}
        return "; ".join(f"{k}={v}" for k, v in statuses.items()) or "NO_CANDIDATE"

    with_temporal = sum(1 for r in temporal if r.get("temporal_linkage_class") not in {"SENTINEL_DATE_MISSING", "PATCH_EVENT_LINKAGE_NOT_AVAILABLE"})
    with_official = sum(1 for r in evidence if r.get("evidence_strength") not in {"ABSENT", "UNKNOWN"})
    with_phenomenon = sum(1 for r in phenomenon if r.get("phenomenon_support") not in {"UNKNOWN", "ABSENT"})
    with_geometry = sum(1 for r in geometry if r.get("geometry_status") not in {"NO_OBSERVED_GEOMETRY", "GEOMETRY_STILL_MISSING", "EVENT_REGISTRY_MISSING", "UNKNOWN"})
    overlay_blocked = len(geometry)
    ground_ref_blocked = len(geometry)
    dino_available = sum(1 for r in dino if r.get("dino_review_support_status") == "DINO_REVIEW_SUPPORT_AVAILABLE")

    method = [
        "# Protocolo C v1us - Event-Patch Package Linkage Engine",
        "",
        "## Engineering Scope",
        "- Builds auditable, non-operational event-patch packages for Recife, Petropolis and Curitiba.",
        "- Uses only consolidated evidence and existing registries; no new download and no web search.",
        "- Does not execute overlay, infer coordinates, geocode localities, or create ground truth, ground reference, or labels.",
        "",
        "## Components",
        "- patch registry resolver (real registries only; no invented patch_id or Sentinel date)",
        "- event-patch candidate builder (region-only candidate linkage, no spatial distance)",
        "- event temporal window linker (records temporal class; never invents Sentinel date)",
        "- external evidence attacher (Recife locality-only; Petropolis document-only)",
        "- phenomenon status attacher (contextual support, never a label)",
        "- geometry blocker attacher (overlay and ground reference blocked)",
        "- event-patch readiness matrix builder",
        "- DINO review support attacher (review-only)",
        "- next action ranker (programming value from real blockers)",
    ]
    write_text(os.path.join(DOCS_DIR, "protocolo_c_v1us_event_patch_package_linkage_engine.md"), method)

    report = [
        "# Relatorio tecnico v1us - Event-Patch Package Linkage Engine",
        "",
        f"- patches_resolved: {len(resolution)}",
        f"- event_patch_candidates: {len(cand_rows)}",
        f"- candidates_with_temporal_linkage: {with_temporal}",
        f"- candidates_with_official_evidence: {with_official}",
        f"- candidates_with_phenomenon_support: {with_phenomenon}",
        f"- candidates_with_observed_geometry: {with_geometry}",
        f"- candidates_overlay_blocked: {overlay_blocked}",
        f"- candidates_ground_reference_blocked: {ground_ref_blocked}",
        f"- dino_review_support_available: {dino_available}",
        f"- readiness_matrix_rows: {len(readiness)}",
        "",
        "## Region Status",
        f"- Recife: {region_status('REC')}",
        f"- Petropolis: {region_status('PET')}",
        f"- Curitiba: {region_status('CUR')}",
        "",
        f"- recommended_next_step: {next_version}",
        "",
        "## Guardrails",
        "- ground_truth_operational=false",
        "- can_create_ground_reference=false",
        "- can_create_training_label=false",
        "- can_reopen_protocol_b=false",
        "- dino_usage=SUPPORT_ONLY",
        "- no_overlay_executed=true",
        "- no_coordinates_invented=true",
        "- patch_bound_truth=false",
        "- operational_validation=false",
        "- event_patch_linkage_candidate_only=true",
    ]
    write_text(os.path.join(DOCS_DIR, "protocolo_c_relatorio_v1us_event_patch_package_linkage_engine.md"), report)

    status = [
        "# Status Atual - Protocolo C v1us",
        "",
        f"status_max={MAX_STATUS}",
        f"patches_resolved={len(resolution)}",
        f"event_patch_candidates={len(cand_rows)}",
        f"recife_status={region_status('REC')}",
        f"petropolis_status={region_status('PET')}",
        f"curitiba_status={region_status('CUR')}",
        f"dino_review_support_available={dino_available}",
        f"recommended_next_action={next_version}",
        "ground_truth_operational=false",
        "can_create_ground_reference=false",
        "can_create_training_label=false",
        "can_reopen_protocol_b=false",
        "dino_usage=SUPPORT_ONLY",
        "no_overlay_executed=true",
        "no_coordinates_invented=true",
        "patch_bound_truth=false",
        "operational_validation=false",
        "event_patch_linkage_candidate_only=true",
    ]
    write_text(os.path.join(DOCS_DIR, "protocolo_c_status_atual_v1us.md"), status)
    print(f"[v1us completion] next_action={next_version}")
    return {
        "patches_resolved": len(resolution),
        "event_patch_candidates": len(cand_rows),
        "readiness_rows": len(readiness),
        "dino_available": dino_available,
        "next_action": next_version,
    }


def main_for(kind):
    argparse.ArgumentParser(description=f"v1us {kind}").parse_args()
    dispatch = {
        "patch_registry_resolver": run_patch_registry_resolver,
        "event_patch_candidate_builder": run_event_patch_candidate_builder,
        "event_temporal_window_linker": run_event_temporal_window_linker,
        "external_evidence_attacher": run_external_evidence_attacher,
        "phenomenon_status_attacher": run_phenomenon_status_attacher,
        "geometry_blocker_attacher": run_geometry_blocker_attacher,
        "event_patch_readiness_matrix_builder": run_event_patch_readiness_matrix_builder,
        "dino_review_support_attacher": run_dino_review_support_attacher,
        "next_action_ranker": run_next_action_ranker,
        "completion_report": run_completion_report,
    }
    if kind not in dispatch:
        raise ValueError(kind)
    return dispatch[kind]()
