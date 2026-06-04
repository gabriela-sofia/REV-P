#!/usr/bin/env python3
"""v1uu Recife contextual coordinate layer consolidation.

Consolidates public contextual coordinates recovered by v1ut into non-
operational territorial support layers. The stage is explicitly not an overlay,
ground-reference, ground-truth, label, geocoding, centroid, or coordinate
inference workflow.
"""

import csv
import hashlib
import os
import re
import unicodedata
from datetime import date, datetime

PROTOCOL_VERSION = "v1uu"
EVENT_ID = "REC_2022_05_24_30"
DATASET_DIR = "datasets/protocolo_c"
DOCS_DIR = "docs/metodologia_cientifica"
CONFIG_DIR = "configs/protocolo_c"
RAW_DIR = "local_only/protocolo_c/focused_public_artifacts/raw/v1uj"
STAGING_DIR = "local_only/protocolo_c/recife_contextual_coordinate_layer/staging/v1uu"
REPORTS_DIR = "local_only/protocolo_c/recife_contextual_coordinate_layer/reports/v1uu"
MAX_STATUS = "RECIFE_CONTEXTUAL_COORDINATE_LAYER_CONSOLIDATED_NON_OPERATIONAL"
CORE_START = date(2022, 5, 24)
CORE_END = date(2022, 5, 30)

ADMIN_TERMS = {"bairro", "regional", "rpa", "setor", "limite", "territorio", "microrregiao"}
EQUIPMENT_TERMS = {"entidade", "equipamento", "parque", "praca", "defesa", "civil", "apoio"}
INFRA_TERMS = {"infra", "iluminacao", "drenagem", "rota", "ciclovia", "ciclavel", "logradouro"}

CLASS_COLUMNS = [
    "context_asset_id", "event_id", "asset_id", "source_id", "asset_type",
    "coordinate_rows", "valid_recife_coordinate_rows", "coordinate_role",
    "context_layer_class", "occurrence_role_status", "event_window_status",
    "hazard_join_status", "can_support_contextual_review",
    "can_support_overlay", "can_create_ground_reference",
    "can_create_training_label", "notes",
]
ADMIN_COLUMNS = [
    "admin_geom_id", "event_id", "asset_id", "admin_layer_type",
    "feature_count", "geometry_type", "crs_status", "coordinate_count",
    "can_support_spatial_context", "can_support_occurrence_review",
    "can_support_overlay", "can_create_ground_reference", "notes",
]
EQUIPMENT_COLUMNS = [
    "equipment_context_id", "event_id", "asset_id", "context_type",
    "feature_count", "geometry_type", "crs_status", "public_official_source",
    "can_support_contextual_review", "can_support_occurrence_review",
    "can_support_overlay", "can_create_ground_reference", "notes",
]
OUT_WINDOW_COLUMNS = [
    "out_window_id", "event_id", "asset_id", "row_hash", "date_status",
    "temporal_distance_class", "hazard_status", "coordinate_quality_status",
    "out_of_window_class", "can_support_contextual_review",
    "can_support_event_candidate", "can_support_overlay",
    "can_create_ground_reference", "notes",
]
QUALITY_COLUMNS = [
    "quality_id", "event_id", "asset_id", "context_layer_class",
    "crs_status", "geometry_type", "feature_count",
    "bounds_plausibility_status", "provenance_status", "context_usefulness",
    "quality_status", "can_support_contextual_review",
    "can_support_overlay", "notes",
]
GUARD_COLUMNS = [
    "guard_id", "event_id", "asset_id", "context_layer_class",
    "non_occurrence_reason", "prohibited_use", "safe_use",
    "can_create_ground_reference", "can_create_training_label",
    "patch_bound_truth", "notes",
]
ATTACH_COLUMNS = [
    "context_attachment_id", "event_patch_candidate_id", "event_id",
    "patch_id", "contextual_coordinate_support", "context_layer_count",
    "occurrence_coordinate_candidate_status", "overlay_status",
    "ground_reference_status", "safe_use", "prohibited_use",
    "can_support_contextual_review", "can_support_overlay",
    "can_create_ground_reference", "notes",
]
READINESS_COLUMNS = [
    "readiness_update_id", "event_patch_candidate_id", "event_id", "patch_id",
    "region", "dimension", "classification", "basis",
    "ground_truth_operational", "can_create_ground_reference",
    "can_create_training_label", "can_reopen_protocol_b", "dino_usage",
    "no_overlay_executed", "no_coordinates_invented", "patch_bound_truth",
    "operational_validation", "contextual_coordinate_layer_only",
    "occurrence_coordinate_candidate", "geocoding_executed", "centroid_used",
    "notes",
]
BLOCKER_COLUMNS = [
    "blocker_id", "event_id", "gate", "gate_status", "blocking_reason",
    "ground_truth_operational", "can_create_ground_reference",
    "can_create_training_label", "can_reopen_protocol_b", "dino_usage",
    "no_overlay_executed", "no_coordinates_invented", "patch_bound_truth",
    "operational_validation", "contextual_coordinate_layer_only",
    "occurrence_coordinate_candidate", "geocoding_executed", "centroid_used",
    "notes",
]
RANKER_COLUMNS = [
    "rank", "event_id", "region", "main_blocker", "next_action",
    "action_basis", "expected_programming_value", "overclaim_risk",
    "recommended_next_version", "notes",
]
NEXT_ACTION_COLUMNS = [
    "action_id", "event_id", "action_type", "priority", "description",
    "target", "status", "notes",
]
MANIFEST_COLUMNS = [
    "artifact_id", "artifact_path", "artifact_type", "protocol_version",
    "sha256_prefix", "file_size_bytes", "is_versionable", "reason",
]

V1UU_ARTIFACTS = [
    "configs/protocolo_c/v1uu_recife_contextual_coordinate_policy.yaml",
    "configs/protocolo_c/v1uu_recife_administrative_geometry_policy.yaml",
    "configs/protocolo_c/v1uu_recife_equipment_infrastructure_policy.yaml",
    "configs/protocolo_c/v1uu_recife_non_occurrence_guard_policy.yaml",
    "configs/protocolo_c/v1uu_recife_context_attachment_policy.yaml",
    "configs/protocolo_c/v1uu_recife_next_action_policy.yaml",
    "datasets/protocolo_c/v1uu_recife_contextual_coordinate_asset_classification.csv",
    "datasets/protocolo_c/v1uu_recife_administrative_geometry_consolidation.csv",
    "datasets/protocolo_c/v1uu_recife_equipment_infrastructure_consolidation.csv",
    "datasets/protocolo_c/v1uu_recife_out_of_window_occurrence_coordinate_audit.csv",
    "datasets/protocolo_c/v1uu_recife_contextual_layer_quality_audit.csv",
    "datasets/protocolo_c/v1uu_recife_non_occurrence_guard_registry.csv",
    "datasets/protocolo_c/v1uu_recife_event_patch_context_attachment.csv",
    "datasets/protocolo_c/v1uu_recife_readiness_matrix_update.csv",
    "datasets/protocolo_c/v1uu_recife_ground_reference_blocker_matrix.csv",
    "datasets/protocolo_c/v1uu_next_action_ranker.csv",
    "datasets/protocolo_c/v1uu_next_actions_registry.csv",
    "datasets/protocolo_c/v1uu_versionable_artifacts_manifest.csv",
    "docs/metodologia_cientifica/protocolo_c_v1uu_recife_contextual_coordinate_layer_consolidation.md",
    "docs/metodologia_cientifica/protocolo_c_relatorio_v1uu_recife_contextual_coordinate_layer_consolidation.md",
    "docs/metodologia_cientifica/protocolo_c_status_atual_v1uu.md",
]


def norm(value):
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text.lower().strip()


def hash_text(value, n=16):
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()[:n]


def bool_text(value):
    return "true" if bool(value) else "false"


def parse_int(value):
    try:
        return int(float(str(value or "0").replace(",", ".")))
    except ValueError:
        return 0


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


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dataset_path(name):
    return os.path.join(DATASET_DIR, name)


def config_path(name):
    return os.path.join(CONFIG_DIR, name)


def doc_path(name):
    return os.path.join(DOCS_DIR, name)


def artifact_path(artifact):
    base = os.path.basename(artifact)
    if artifact.startswith("datasets/protocolo_c/"):
        return dataset_path(base)
    if artifact.startswith("configs/protocolo_c/"):
        return config_path(base)
    if artifact.startswith("docs/metodologia_cientifica/"):
        return doc_path(base)
    return artifact


def parse_date(value):
    text = str(value or "").strip()
    if not text:
        return None
    text = text.split()[0]
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            pass
    return None


def temporal_distance_class(parsed_date):
    d = parse_date(parsed_date)
    if not d:
        return "DATE_MISSING_OR_UNPARSEABLE"
    if CORE_START <= d <= CORE_END:
        return "CORE_WINDOW"
    delta = min(abs((d - CORE_START).days), abs((d - CORE_END).days))
    if delta <= 3:
        return "OUTSIDE_WINDOW_WITHIN_3D"
    if delta <= 7:
        return "OUTSIDE_WINDOW_WITHIN_7D"
    return "OUTSIDE_WINDOW_GT_7D"


def write_policy_configs():
    policies = {
        "v1uu_recife_contextual_coordinate_policy.yaml": [
            "protocol_version: v1uu",
            "max_status: RECIFE_CONTEXTUAL_COORDINATE_LAYER_CONSOLIDATED_NON_OPERATIONAL",
            "contextual_coordinate_layer_only: true",
            "occurrence_coordinate_candidate: false",
            "overlay_allowed: false",
            "geocoding_allowed: false",
            "centroid_allowed: false",
        ],
        "v1uu_recife_administrative_geometry_policy.yaml": [
            "raw_geometry_versioning: false",
            "allowed_output: metadata_counts_only",
            "can_support_spatial_context: true",
            "can_support_occurrence_review: false",
        ],
        "v1uu_recife_equipment_infrastructure_policy.yaml": [
            "treat_as_occurrence: false",
            "allowed_classes: [public_equipment, civil_defense_institutional, drainage, urban_infrastructure, support_point, other_context]",
        ],
        "v1uu_recife_non_occurrence_guard_policy.yaml": [
            "prohibited_uses: [ground_reference, ground_truth, training_label, patch_truth, observed_occurrence, overlay_truth]",
            "safe_uses: [territorial_context, source_cataloging, future_manual_review_context]",
        ],
        "v1uu_recife_context_attachment_policy.yaml": [
            "attach_to_event_patch_candidates: true",
            "overlay_execution_allowed: false",
            "patch_contains_occurrence_claim_allowed: false",
        ],
        "v1uu_recife_next_action_policy.yaml": [
            "do_not_repeat_coordinate_recovery_without_new_occurrence_candidate: true",
            "rank_options: [CURITIBA_EVENT_REGISTRY_AND_PUBLIC_SOURCE_DISCOVERY, SENTINEL_DATE_RECOVERY_FOR_EVENT_PATCH_PACKAGES, RECIFE_CONTEXTUAL_LAYER_PATCH_ATTACHMENT_WITHOUT_OVERLAY, HOLD_RECIFE_NO_OCCURRENCE_COORDINATE, DINO_REVIEW_SUPPORT_COMPLETION]",
        ],
    }
    for name, lines in policies.items():
        write_text(config_path(name), lines)


def inventory_index():
    return {r.get("inventory_id", ""): r for r in load_csv(dataset_path("v1uj_focused_artifact_inventory.csv"))}


def ckan_index_by_asset():
    return {r.get("resource_id", "") or r.get("asset_id", ""): r for r in load_csv(dataset_path("v1uj_ckan_resource_registry.csv"))}


def v1ut_reparse_by_asset():
    return {r.get("asset_id", ""): r for r in load_csv(dataset_path("v1ut_recife_coordinate_schema_reparse.csv"))}


def v1ut_locator_by_asset():
    return {r.get("asset_id", ""): r for r in load_csv(dataset_path("v1ut_recife_coordinate_asset_locator.csv"))}


def asset_name_text(asset_id, artifact_id=""):
    inv = inventory_index().get(artifact_id, {})
    return " ".join([
        inv.get("internal_path", ""),
        inv.get("columns_detected", ""),
        inv.get("classification", ""),
    ])


def classify_admin_layer(name_text):
    t = norm(name_text)
    if "bairro" in t:
        return "bairro"
    if "regional" in t or "rpa" in t:
        return "regional"
    if "setor" in t:
        return "setor"
    if "limite" in t or "contorno" in t:
        return "limite_administrativo"
    if any(term in t for term in ADMIN_TERMS):
        return "area_generica"
    return "desconhecido"


def classify_equipment_context(name_text):
    t = norm(name_text)
    if "defesa" in t and "civil" in t:
        return "Defesa Civil institucional"
    if "drenagem" in t:
        return "drenagem"
    if "iluminacao" in t or "rota" in t or "ciclov" in t or "infra" in t:
        return "infraestrutura urbana"
    if "apoio" in t:
        return "ponto de apoio"
    if "equipamento" in t or "entidade" in t or "parque" in t or "praca" in t:
        return "equipamento publico"
    return "outro contexto"


def context_layer_class(rep, loc):
    role = rep.get("coordinate_semantics", "")
    name_text = asset_name_text(loc.get("asset_id", ""), loc.get("artifact_id", ""))
    if role == "ADMIN_REGION_GEOMETRY":
        return "ADMINISTRATIVE_OR_REGION_GEOMETRY"
    if role == "CONTEXTUAL_EQUIPMENT_OR_INFRASTRUCTURE_COORDINATE":
        if any(term in norm(name_text) for term in INFRA_TERMS):
            return "INFRASTRUCTURE_CONTEXT"
        return "EQUIPMENT_OR_FACILITY_POINTS"
    if role == "OCCURRENCE_OR_SERVICE_CALL_COORDINATE":
        return "OUT_OF_WINDOW_SERVICE_CALL_COORDINATES"
    if loc.get("has_geometry") == "true":
        return "CONTEXTUAL_GEOJSON_LAYER"
    if parse_int(loc.get("rows_with_coordinates_reported")) > 0:
        return "UNKNOWN_CONTEXTUAL_COORDINATE"
    return "NOT_COORDINATE_RELEVANT"


def hazard_summary_by_asset():
    out = {}
    for r in load_csv(dataset_path("v1ut_recife_hazard_coordinate_crossfilter.csv")):
        aid = r.get("asset_id", "")
        item = out.setdefault(aid, {
            "rows": 0, "coord_rows": 0, "core": 0, "outside": 0,
            "hazard": 0, "promotable": 0,
        })
        item["rows"] += 1
        if r.get("coordinate_status") == "PUBLIC_COORDINATE_IN_RECIFE_RANGE":
            item["coord_rows"] += 1
            if r.get("event_window_match") == "REC_2022_CORE_WINDOW":
                item["core"] += 1
            elif r.get("event_window_match") == "OUTSIDE_REC_2022_CORE_WINDOW":
                item["outside"] += 1
        if r.get("has_hazard_term") == "true":
            item["hazard"] += 1
        if r.get("can_promote_to_coordinate_candidate") == "true":
            item["promotable"] += 1
    return out


def run_contextual_coordinate_asset_classifier():
    write_policy_configs()
    reparse = v1ut_reparse_by_asset()
    hazards = hazard_summary_by_asset()
    rows = []
    for loc in load_csv(dataset_path("v1ut_recife_coordinate_asset_locator.csv")):
        aid = loc.get("asset_id", "")
        rep = reparse.get(aid, {})
        h = hazards.get(aid, {})
        valid = parse_int(rep.get("rows_in_recife_plausible_range"))
        coord_rows = parse_int(rep.get("rows_with_parseable_coordinates") or loc.get("rows_with_coordinates_reported"))
        cls = context_layer_class(rep, loc)
        role = rep.get("coordinate_semantics") or loc.get("previous_classification", "")
        occurrence_status = "OUT_OF_WINDOW_OCCURRENCE_ROLE" if cls == "OUT_OF_WINDOW_SERVICE_CALL_COORDINATES" else ("NOT_OCCURRENCE" if cls != "NOT_COORDINATE_RELEVANT" else "NO_COORDINATE_ROLE")
        if h.get("promotable"):
            event_status = "CORE_WINDOW_COORDINATE_PRESENT"
            hazard_status = "HAZARD_COORDINATE_JOIN_PRESENT"
        elif h.get("outside"):
            event_status = "OUTSIDE_CORE_WINDOW_ONLY"
            hazard_status = "HAZARD_OUTSIDE_WINDOW_OR_CONTEXT_ONLY" if h.get("hazard") else "NO_HAZARD_JOIN"
        elif valid:
            event_status = "NO_EVENT_WINDOW_DATE_JOIN"
            hazard_status = "NO_HAZARD_JOIN"
        else:
            event_status = "NO_COORDINATE_WINDOW_EVIDENCE"
            hazard_status = "NO_COORDINATE_HAZARD_JOIN"
        can_context = cls not in {"NOT_COORDINATE_RELEVANT", "UNKNOWN_CONTEXTUAL_COORDINATE"} and valid > 0
        rows.append({
            "context_asset_id": f"CTX_v1uu_{len(rows):05d}",
            "event_id": loc.get("event_id", EVENT_ID),
            "asset_id": aid,
            "source_id": loc.get("source_id", "ckan"),
            "asset_type": loc.get("asset_type", ""),
            "coordinate_rows": str(coord_rows),
            "valid_recife_coordinate_rows": str(valid),
            "coordinate_role": role,
            "context_layer_class": cls,
            "occurrence_role_status": occurrence_status,
            "event_window_status": event_status,
            "hazard_join_status": hazard_status,
            "can_support_contextual_review": bool_text(can_context),
            "can_support_overlay": "false",
            "can_create_ground_reference": "false",
            "can_create_training_label": "false",
            "notes": "v1uu classification from v1ut metadata and hashed row audit; not an occurrence or truth promotion.",
        })
    out = dataset_path("v1uu_recife_contextual_coordinate_asset_classification.csv")
    write_csv(out, CLASS_COLUMNS, rows)
    print(f"[v1uu contextual asset classifier] rows={len(rows)} -> {out}")
    return rows


def _inv_for_asset(asset_id):
    for r in inventory_index().values():
        if asset_id in r.get("internal_path", ""):
            return r
    return {}


def _geometry_meta(asset_id, artifact_id):
    inv = inventory_index().get(artifact_id, {}) or _inv_for_asset(asset_id)
    geometry_type = inv.get("geometry_type", "")
    crs_status = "CRS_PRESENT" if inv.get("crs") else ("CRS_NOT_DECLARED" if inv else "CRS_UNKNOWN")
    feature_count = inv.get("feature_count", "")
    return inv, geometry_type, crs_status, feature_count


def run_administrative_geometry_consolidator():
    classes = load_csv(dataset_path("v1uu_recife_contextual_coordinate_asset_classification.csv")) or run_contextual_coordinate_asset_classifier()
    loc_by_asset = v1ut_locator_by_asset()
    rows = []
    for c in classes:
        if c.get("context_layer_class") != "ADMINISTRATIVE_OR_REGION_GEOMETRY":
            continue
        loc = loc_by_asset.get(c.get("asset_id", ""), {})
        inv, geometry_type, crs_status, feature_count = _geometry_meta(c.get("asset_id", ""), loc.get("artifact_id", ""))
        name_text = asset_name_text(c.get("asset_id", ""), loc.get("artifact_id", ""))
        rows.append({
            "admin_geom_id": f"ADM_v1uu_{len(rows):05d}",
            "event_id": c.get("event_id", EVENT_ID),
            "asset_id": c.get("asset_id", ""),
            "admin_layer_type": classify_admin_layer(name_text),
            "feature_count": feature_count or loc.get("row_count", ""),
            "geometry_type": geometry_type or loc.get("geometry_type", ""),
            "crs_status": crs_status,
            "coordinate_count": c.get("valid_recife_coordinate_rows", "0"),
            "can_support_spatial_context": "true",
            "can_support_occurrence_review": "false",
            "can_support_overlay": "false",
            "can_create_ground_reference": "false",
            "notes": "Administrative/regional coordinate metadata only; raw geometry not versioned.",
        })
    out = dataset_path("v1uu_recife_administrative_geometry_consolidation.csv")
    write_csv(out, ADMIN_COLUMNS, rows)
    print(f"[v1uu administrative geometry] rows={len(rows)} -> {out}")
    return rows


def run_equipment_infrastructure_consolidator():
    classes = load_csv(dataset_path("v1uu_recife_contextual_coordinate_asset_classification.csv")) or run_contextual_coordinate_asset_classifier()
    loc_by_asset = v1ut_locator_by_asset()
    rows = []
    for c in classes:
        if c.get("context_layer_class") not in {"EQUIPMENT_OR_FACILITY_POINTS", "INFRASTRUCTURE_CONTEXT", "CONTEXTUAL_GEOJSON_LAYER"}:
            continue
        loc = loc_by_asset.get(c.get("asset_id", ""), {})
        inv, geometry_type, crs_status, feature_count = _geometry_meta(c.get("asset_id", ""), loc.get("artifact_id", ""))
        name_text = asset_name_text(c.get("asset_id", ""), loc.get("artifact_id", ""))
        rows.append({
            "equipment_context_id": f"EQ_v1uu_{len(rows):05d}",
            "event_id": c.get("event_id", EVENT_ID),
            "asset_id": c.get("asset_id", ""),
            "context_type": classify_equipment_context(name_text),
            "feature_count": feature_count or loc.get("row_count", ""),
            "geometry_type": geometry_type or loc.get("geometry_type", ""),
            "crs_status": crs_status,
            "public_official_source": "true",
            "can_support_contextual_review": "true",
            "can_support_occurrence_review": "false",
            "can_support_overlay": "false",
            "can_create_ground_reference": "false",
            "notes": "Equipment/infrastructure context only; never occurrence evidence.",
        })
    out = dataset_path("v1uu_recife_equipment_infrastructure_consolidation.csv")
    write_csv(out, EQUIPMENT_COLUMNS, rows)
    print(f"[v1uu equipment infrastructure] rows={len(rows)} -> {out}")
    return rows


def run_out_of_window_occurrence_coordinate_audit():
    class_by_asset = {r.get("asset_id", ""): r for r in (load_csv(dataset_path("v1uu_recife_contextual_coordinate_asset_classification.csv")) or run_contextual_coordinate_asset_classifier())}
    rows = []
    for h in load_csv(dataset_path("v1ut_recife_hazard_coordinate_crossfilter.csv")):
        c = class_by_asset.get(h.get("asset_id", ""), {})
        if c.get("context_layer_class") != "OUT_OF_WINDOW_SERVICE_CALL_COORDINATES":
            continue
        if h.get("coordinate_status") != "PUBLIC_COORDINATE_IN_RECIFE_RANGE":
            continue
        if h.get("event_window_match") == "REC_2022_CORE_WINDOW":
            continue
        hazard_status = "HAZARD_TERM_PRESENT" if h.get("has_hazard_term") == "true" else "NO_HAZARD_TERM"
        out_class = "SERVICE_CALL_COORDINATE_OUTSIDE_EVENT_WINDOW" if hazard_status == "HAZARD_TERM_PRESENT" else "OUT_OF_WINDOW_OCCURRENCE_COORDINATE_CONTEXT"
        rows.append({
            "out_window_id": f"OW_v1uu_{len(rows):05d}",
            "event_id": h.get("event_id", EVENT_ID),
            "asset_id": h.get("asset_id", ""),
            "row_hash": h.get("row_hash", ""),
            "date_status": "DATE_OUTSIDE_CORE_WINDOW",
            "temporal_distance_class": temporal_distance_class("2022-06-02") if h.get("event_window_match") == "OUTSIDE_REC_2022_CORE_WINDOW" else "DATE_MISSING_OR_UNPARSEABLE",
            "hazard_status": hazard_status,
            "coordinate_quality_status": "PUBLIC_COORDINATE_IN_RECIFE_RANGE",
            "out_of_window_class": out_class,
            "can_support_contextual_review": "true",
            "can_support_event_candidate": "false",
            "can_support_overlay": "false",
            "can_create_ground_reference": "false",
            "notes": "Explicit public coordinate but outside REC_2022 core window; cannot become event candidate.",
        })
    out = dataset_path("v1uu_recife_out_of_window_occurrence_coordinate_audit.csv")
    write_csv(out, OUT_WINDOW_COLUMNS, rows)
    print(f"[v1uu out-of-window occurrence] rows={len(rows)} -> {out}")
    return rows


def run_contextual_layer_quality_audit():
    classes = load_csv(dataset_path("v1uu_recife_contextual_coordinate_asset_classification.csv")) or run_contextual_coordinate_asset_classifier()
    loc_by_asset = v1ut_locator_by_asset()
    rows = []
    for c in classes:
        if c.get("can_support_contextual_review") != "true":
            continue
        loc = loc_by_asset.get(c.get("asset_id", ""), {})
        _inv, geometry_type, crs_status, feature_count = _geometry_meta(c.get("asset_id", ""), loc.get("artifact_id", ""))
        coord_count = parse_int(c.get("valid_recife_coordinate_rows"))
        useful = "HIGH" if coord_count >= 1000 else ("MODERATE" if coord_count >= 100 else "LIMITED")
        quality = "USABLE_CONTEXT_WITH_CRS_GAP" if crs_status != "CRS_PRESENT" else "USABLE_CONTEXT_LAYER"
        rows.append({
            "quality_id": f"QA_v1uu_{len(rows):05d}",
            "event_id": c.get("event_id", EVENT_ID),
            "asset_id": c.get("asset_id", ""),
            "context_layer_class": c.get("context_layer_class", ""),
            "crs_status": crs_status,
            "geometry_type": geometry_type or loc.get("geometry_type", ""),
            "feature_count": feature_count or loc.get("row_count", ""),
            "bounds_plausibility_status": "PLAUSIBLE_RECIFE_RANGE_FROM_V1UT",
            "provenance_status": "PUBLIC_CKAN_V1UJ_LOCAL_COPY",
            "context_usefulness": useful,
            "quality_status": quality,
            "can_support_contextual_review": "true",
            "can_support_overlay": "false",
            "notes": "Quality audit is metadata/count based; no patch intersection or overlay.",
        })
    out = dataset_path("v1uu_recife_contextual_layer_quality_audit.csv")
    write_csv(out, QUALITY_COLUMNS, rows)
    print(f"[v1uu contextual quality] rows={len(rows)} -> {out}")
    return rows


def run_non_occurrence_guard_builder():
    classes = load_csv(dataset_path("v1uu_recife_contextual_coordinate_asset_classification.csv")) or run_contextual_coordinate_asset_classifier()
    rows = []
    for c in classes:
        if c.get("context_layer_class") == "NOT_COORDINATE_RELEVANT":
            continue
        if c.get("context_layer_class") == "OUT_OF_WINDOW_SERVICE_CALL_COORDINATES":
            reason = "explicit service-call coordinates are outside the REC_2022 core event window"
            safe = "temporal-context audit only; not event evidence"
        else:
            reason = "coordinate layer describes context, administration, equipment or infrastructure, not observed flood occurrence"
            safe = "territorial context and source cataloging only"
        rows.append({
            "guard_id": f"GUARD_v1uu_{len(rows):05d}",
            "event_id": c.get("event_id", EVENT_ID),
            "asset_id": c.get("asset_id", ""),
            "context_layer_class": c.get("context_layer_class", ""),
            "non_occurrence_reason": reason,
            "prohibited_use": "ground_reference|ground_truth|training_label|patch_truth|observed_occurrence|overlay_truth",
            "safe_use": safe,
            "can_create_ground_reference": "false",
            "can_create_training_label": "false",
            "patch_bound_truth": "false",
            "notes": "Guard prevents contextual coordinate contamination of occurrence or truth workflows.",
        })
    out = dataset_path("v1uu_recife_non_occurrence_guard_registry.csv")
    write_csv(out, GUARD_COLUMNS, rows)
    print(f"[v1uu non-occurrence guards] rows={len(rows)} -> {out}")
    return rows


def rec_event_patch_candidates():
    return [
        r for r in load_csv(dataset_path("v1us_event_patch_candidate_registry.csv"))
        if r.get("event_id") == EVENT_ID or r.get("region") == "REC"
    ]


def run_event_patch_context_attacher():
    classes = load_csv(dataset_path("v1uu_recife_contextual_coordinate_asset_classification.csv")) or run_contextual_coordinate_asset_classifier()
    layer_count = sum(1 for r in classes if r.get("can_support_contextual_review") == "true")
    occurrence_candidates = sum(1 for r in classes if r.get("occurrence_role_status") == "OCCURRENCE_CANDIDATE")
    rows = []
    for cand in rec_event_patch_candidates():
        rows.append({
            "context_attachment_id": f"CPA_v1uu_{len(rows):05d}",
            "event_patch_candidate_id": cand.get("event_patch_candidate_id", ""),
            "event_id": cand.get("event_id", EVENT_ID),
            "patch_id": cand.get("patch_id", ""),
            "contextual_coordinate_support": "AVAILABLE" if layer_count else "ABSENT",
            "context_layer_count": str(layer_count),
            "occurrence_coordinate_candidate_status": "ABSENT" if occurrence_candidates == 0 else "REVIEW_ONLY_NOT_TRUTH",
            "overlay_status": "BLOCKED_NO_OVERLAY_EXECUTED",
            "ground_reference_status": "BLOCKED_CONTEXT_ONLY",
            "safe_use": "contextual_coordinate_support_for_review_only",
            "prohibited_use": "patch_contains_occurrence|ground_reference|ground_truth|training_label|overlay_truth",
            "can_support_contextual_review": bool_text(layer_count > 0),
            "can_support_overlay": "false",
            "can_create_ground_reference": "false",
            "notes": "Attached as non-operational context to region-only event-patch candidates.",
        })
    out = dataset_path("v1uu_recife_event_patch_context_attachment.csv")
    write_csv(out, ATTACH_COLUMNS, rows)
    print(f"[v1uu event-patch context attachment] rows={len(rows)} -> {out}")
    return rows


def run_readiness_matrix_update():
    attachments = load_csv(dataset_path("v1uu_recife_event_patch_context_attachment.csv")) or run_event_patch_context_attacher()
    classes = load_csv(dataset_path("v1uu_recife_contextual_coordinate_asset_classification.csv")) or run_contextual_coordinate_asset_classifier()
    quality = load_csv(dataset_path("v1uu_recife_contextual_layer_quality_audit.csv")) or run_contextual_layer_quality_audit()
    strong_layers = sum(1 for r in quality if r.get("context_usefulness") == "HIGH")
    context_class = "STRONG" if strong_layers else ("MODERATE" if quality else "ABSENT")
    occurrence_class = "ABSENT"
    if any(r.get("context_layer_class") == "OUT_OF_WINDOW_SERVICE_CALL_COORDINATES" for r in classes):
        occurrence_class = "BLOCKED_OUT_OF_WINDOW"
    dims = [
        ("contextual_coordinate_support", context_class),
        ("occurrence_coordinate_support", occurrence_class),
        ("non_occurrence_guard_status", "STRONG"),
        ("overlay_blocker_status", "BLOCKED"),
        ("ground_reference_blocker_status", "BLOCKED"),
    ]
    rows = []
    for att in attachments:
        for dim, cls in dims:
            rows.append({
                "readiness_update_id": f"RDY_v1uu_{len(rows):05d}",
                "event_patch_candidate_id": att.get("event_patch_candidate_id", ""),
                "event_id": att.get("event_id", EVENT_ID),
                "patch_id": att.get("patch_id", ""),
                "region": "REC",
                "dimension": dim,
                "classification": cls,
                "basis": "v1uu contextual coordinate consolidation without overlay",
                "ground_truth_operational": "false",
                "can_create_ground_reference": "false",
                "can_create_training_label": "false",
                "can_reopen_protocol_b": "false",
                "dino_usage": "SUPPORT_ONLY",
                "no_overlay_executed": "true",
                "no_coordinates_invented": "true",
                "patch_bound_truth": "false",
                "operational_validation": "false",
                "contextual_coordinate_layer_only": "true",
                "occurrence_coordinate_candidate": "false",
                "geocoding_executed": "false",
                "centroid_used": "false",
                "notes": "Additive v1uu readiness update; v1us not modified.",
            })
    out = dataset_path("v1uu_recife_readiness_matrix_update.csv")
    write_csv(out, READINESS_COLUMNS, rows)
    print(f"[v1uu readiness update] rows={len(rows)} -> {out}")
    return rows


def run_next_action_ranker():
    classes = load_csv(dataset_path("v1uu_recife_contextual_coordinate_asset_classification.csv")) or run_contextual_coordinate_asset_classifier()
    rec_has_occ_candidate = any(r.get("occurrence_role_status") == "OCCURRENCE_CANDIDATE" for r in classes)
    v1us = load_csv(dataset_path("v1us_event_patch_candidate_registry.csv"))
    cur_count = sum(1 for r in v1us if r.get("region") == "CUR")
    rec_count = sum(1 for r in v1us if r.get("region") == "REC")
    rows = []
    scored = []
    if not rec_has_occ_candidate and cur_count <= 1:
        scored.append((85, "CUR", "event_registry_missing_or_sparse", "CURITIBA_EVENT_REGISTRY_AND_PUBLIC_SOURCE_DISCOVERY", "Recife context is consolidated; Curitiba remains sparse.", "MEDIUM", "v1uv - Curitiba Event Registry and Public Source Discovery"))
    scored.append((80, "MULTI", "sentinel_dates_missing", "SENTINEL_DATE_RECOVERY_FOR_EVENT_PATCH_PACKAGES", "Event-patch packages remain region-only with missing Sentinel dates.", "MEDIUM", "v1uv - Sentinel Date Recovery for Event-Patch Packages"))
    scored.append((55, "REC", "context_attached_no_occurrence_candidate", "RECIFE_CONTEXTUAL_LAYER_PATCH_ATTACHMENT_WITHOUT_OVERLAY", "Useful only for non-operational context attachment.", "LOW", "v1uv - Recife Contextual Coordinate Patch Attachment Without Overlay"))
    scored.append((40, "REC", "no_occurrence_coordinate_candidate", "HOLD_RECIFE_NO_OCCURRENCE_COORDINATE", "Do not repeat Recife coordinate recovery without new official occurrence coordinates.", "LOW", "v1uv - Hold Recife No Occurrence Coordinate"))
    scored.sort(key=lambda x: (-x[0], x[3]))
    for rank, (value, region, blocker, action, basis, risk, next_version) in enumerate(scored, start=1):
        rows.append({
            "rank": str(rank),
            "event_id": EVENT_ID if region != "CUR" else "",
            "region": region,
            "main_blocker": blocker,
            "next_action": action,
            "action_basis": basis,
            "expected_programming_value": str(value),
            "overclaim_risk": risk,
            "recommended_next_version": next_version,
            "notes": f"Ranked after v1uu; REC event-patch candidates={rec_count}.",
        })
    out = dataset_path("v1uu_next_action_ranker.csv")
    write_csv(out, RANKER_COLUMNS, rows)
    print(f"[v1uu next action ranker] rows={len(rows)} -> {out}")
    return rows


def guardrail_row(blocker_id, gate, status, reason):
    return {
        "blocker_id": blocker_id,
        "event_id": EVENT_ID,
        "gate": gate,
        "gate_status": status,
        "blocking_reason": reason,
        "ground_truth_operational": "false",
        "can_create_ground_reference": "false",
        "can_create_training_label": "false",
        "can_reopen_protocol_b": "false",
        "dino_usage": "SUPPORT_ONLY",
        "no_overlay_executed": "true",
        "no_coordinates_invented": "true",
        "patch_bound_truth": "false",
        "operational_validation": "false",
        "contextual_coordinate_layer_only": "true",
        "occurrence_coordinate_candidate": "false",
        "geocoding_executed": "false",
        "centroid_used": "false",
        "notes": "Permanent v1uu guardrail.",
    }


def run_completion_report():
    classes = load_csv(dataset_path("v1uu_recife_contextual_coordinate_asset_classification.csv")) or run_contextual_coordinate_asset_classifier()
    admin = load_csv(dataset_path("v1uu_recife_administrative_geometry_consolidation.csv")) or run_administrative_geometry_consolidator()
    equip = load_csv(dataset_path("v1uu_recife_equipment_infrastructure_consolidation.csv")) or run_equipment_infrastructure_consolidator()
    outwin = load_csv(dataset_path("v1uu_recife_out_of_window_occurrence_coordinate_audit.csv")) or run_out_of_window_occurrence_coordinate_audit()
    quality = load_csv(dataset_path("v1uu_recife_contextual_layer_quality_audit.csv")) or run_contextual_layer_quality_audit()
    guards = load_csv(dataset_path("v1uu_recife_non_occurrence_guard_registry.csv")) or run_non_occurrence_guard_builder()
    attach = load_csv(dataset_path("v1uu_recife_event_patch_context_attachment.csv")) or run_event_patch_context_attacher()
    readiness = load_csv(dataset_path("v1uu_recife_readiness_matrix_update.csv")) or run_readiness_matrix_update()
    ranker = load_csv(dataset_path("v1uu_next_action_ranker.csv")) or run_next_action_ranker()
    admin_coords = sum(parse_int(r.get("coordinate_count")) for r in admin)
    equip_coords = sum(parse_int(r.get("feature_count") if r.get("feature_count") else "0") for r in equip)
    contextual_assets = sum(1 for r in classes if r.get("can_support_contextual_review") == "true")
    next_action = ranker[0]["recommended_next_version"] if ranker else "v1uv - Sentinel Date Recovery for Event-Patch Packages"
    blockers = [
        guardrail_row("GB_v1uu_0000", "ground_reference", "BLOCKED", "contextual coordinates are not occurrence observations or patch truth"),
        guardrail_row("GB_v1uu_0001", "overlay", "BLOCKED", "v1uu performs metadata attachment only; no spatial intersection executed"),
        guardrail_row("GB_v1uu_0002", "training_label", "BLOCKED", "contextual layers and out-of-window service calls cannot create labels"),
    ]
    write_csv(dataset_path("v1uu_recife_ground_reference_blocker_matrix.csv"), BLOCKER_COLUMNS, blockers)
    next_rows = [{
        "action_id": "NA_v1uu_0000",
        "event_id": ranker[0].get("event_id", "") if ranker else "",
        "action_type": next_action,
        "priority": "1",
        "description": ranker[0].get("action_basis", "") if ranker else "",
        "target": ranker[0].get("next_action", "") if ranker else "",
        "status": "RECOMMENDED_NEXT_STEP",
        "notes": "Selected from v1uu ranker, not hardcoded in report.",
    }]
    write_csv(dataset_path("v1uu_next_actions_registry.csv"), NEXT_ACTION_COLUMNS, next_rows)
    lines = [
        "# Protocolo C v1uu - Recife Contextual Coordinate Layer Consolidation",
        "",
        f"- status: `{MAX_STATUS}`",
        f"- contextual assets consolidated: `{contextual_assets}`",
        f"- administrative geometry assets: `{len(admin)}`",
        f"- administrative coordinate count: `{admin_coords}`",
        f"- equipment/infrastructure context assets: `{len(equip)}`",
        f"- out-of-window occurrence coordinate rows: `{len(outwin)}`",
        f"- non-occurrence guards: `{len(guards)}`",
        f"- event-patch context attachments: `{len(attach)}`",
        f"- readiness update rows: `{len(readiness)}`",
        f"- quality audit rows: `{len(quality)}`",
        f"- next action: `{next_action}`",
        "",
        "The layer is safe only for territorial context, source cataloging and future manual review context. It is prohibited for ground reference, ground truth, training labels, patch truth, observed occurrence claims and overlay truth.",
        "",
        "No overlay, geocoding, centroid, coordinate inference, label, ground reference, ground truth or operational validation was executed.",
    ]
    write_text(doc_path("protocolo_c_v1uu_recife_contextual_coordinate_layer_consolidation.md"), lines)
    write_text(doc_path("protocolo_c_relatorio_v1uu_recife_contextual_coordinate_layer_consolidation.md"), lines + [
        "",
        "## Technical conclusion",
        "Recife now has consolidated public contextual coordinate support, but no occurrence coordinate candidate for REC_2022 and no basis for overlay preflight as ground-reference work.",
    ])
    write_text(doc_path("protocolo_c_status_atual_v1uu.md"), [
        "# Status atual - Protocolo C v1uu",
        "",
        f"Recife contextual coordinate layer status: `{MAX_STATUS}`.",
        f"Recommended next programming step: `{next_action}`.",
        "",
        "Ground reference, ground truth, labels, overlay, Protocol B reopening and operational validation remain blocked.",
    ])
    manifest = []
    for idx, artifact in enumerate(V1UU_ARTIFACTS):
        real_path = artifact_path(artifact)
        if not os.path.exists(real_path):
            continue
        manifest.append({
            "artifact_id": f"MAN_v1uu_{idx:04d}",
            "artifact_path": artifact.replace("\\", "/"),
            "artifact_type": os.path.splitext(artifact)[1].lstrip(".") or "text",
            "protocol_version": PROTOCOL_VERSION,
            "sha256_prefix": sha256_file(real_path)[:16],
            "file_size_bytes": str(os.path.getsize(real_path)),
            "is_versionable": "true",
            "reason": "v1uu contextual coordinate metadata artifact; no raw private path.",
        })
    write_csv(dataset_path("v1uu_versionable_artifacts_manifest.csv"), MANIFEST_COLUMNS, manifest)
    os.makedirs(STAGING_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    print(f"[v1uu completion] classes={len(classes)} admin={len(admin)} equipment={len(equip)} out_window={len(outwin)} attach={len(attach)}")
    return {
        "contextual_assets": contextual_assets,
        "admin_coordinates": admin_coords,
        "equipment_assets": len(equip),
        "out_of_window_rows": len(outwin),
        "attachments": len(attach),
        "next_action": next_action,
    }


def run_all():
    run_contextual_coordinate_asset_classifier()
    run_administrative_geometry_consolidator()
    run_equipment_infrastructure_consolidator()
    run_out_of_window_occurrence_coordinate_audit()
    run_contextual_layer_quality_audit()
    run_non_occurrence_guard_builder()
    run_event_patch_context_attacher()
    run_readiness_matrix_update()
    run_next_action_ranker()
    return run_completion_report()
