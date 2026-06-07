#!/usr/bin/env python3
"""v2aq Event Geometry Acquisition and Patch-Link Review Sprint.

Attacks the dominant blocker left by v2ap: missing event observational geometry for
patch-level linkage. Builds event-geometry candidates and patch-link review candidates
from explicit evidence only -- never inventing coordinates/geometry, never executing
operational overlay, never inferring Sentinel dates/crosswalks by visual/DINO similarity.

GeoJSON candidates are written with real geometry only when an explicit coordinate/bbox
source exists; otherwise ``geometry: null`` plus collection properties. Only ``v2aq_*``
artifacts are written. Operational ground truth, labels and Protocol B stay blocked and
``patch_truth_allowed`` stays false.
"""

import argparse
import csv
import hashlib
import json
import os
import re

PROTOCOL_VERSION = "v2aq"
DATASET_ROOT = os.environ.get("DATASET_ROOT", "datasets")
DATASET_DIR = os.environ.get("DATASET_DIR", os.path.join(DATASET_ROOT, "protocolo_c"))
DOCS_DIR = os.environ.get("DOCS_DIR", "docs/protocolo_c/v2aq_event_geometry_patch_link")
GEOJSON_DIR = os.environ.get("GEOJSON_DIR", os.path.join(DOCS_DIR, "geojson_candidates"))
CONFIG_DIR = os.environ.get("CONFIG_DIR", "configs/protocolo_c")

V2AP_INPUTS = {
    "selection": "v2ap_candidate_selection.csv",
    "geometry": "v2ap_spatial_geometry_readiness.csv",
    "patch_registry": "v2ap_patch_registry_inventory.csv",
    "crosswalk": "v2ap_sentinel_crosswalk_candidates.csv",
    "link": "v2ap_patch_event_link_readiness.csv",
    "geom_packet": "v2ap_geometry_collection_packet.csv",
    "cross_packet": "v2ap_crosswalk_collection_packet.csv",
    "scores": "v2ap_patch_reference_readiness_scores.csv",
    "boundary": "v2ap_patch_truth_boundary_update.csv",
}
V2AO_INPUTS = {
    "levels": "v2ao_reference_level_classification.csv",
    "ranking": "v2ao_final_candidate_ranking.csv",
    "trace": "v2ao_human_review_trace.csv",
}
V2AN_OPTIONAL = {"spatial": "v2an_spatial_anchor_registry.csv"}
OBSERVED_REGISTRY = "observed_event_reference_candidate_registry.csv"

REGION_BY_PREFIX = {"REC": "Recife", "PET": "Petropolis", "CTB": "Curitiba"}

ANCHOR_SPECIFICITY = {
    "ponto_de_alagamento": 4, "rua": 4, "mapa_ou_laudo": 4, "area_tecnica": 4,
    "corredor_de_rio": 2, "bairro": 2, "municipio": 1,
}

ALLOWED_GEOMETRY_STATUS = {
    "EXPLICIT_EVENT_GEOMETRY_AVAILABLE", "EXPLICIT_POINT_OR_COORDINATE_AVAILABLE",
    "OFFICIAL_MAP_DIGITIZATION_REQUIRED", "TEXTUAL_ANCHOR_ONLY", "INSUFFICIENT_GEOMETRY",
}
EXPLICIT_GEOMETRY_STATUS = {"EXPLICIT_EVENT_GEOMETRY_AVAILABLE",
                            "EXPLICIT_POINT_OR_COORDINATE_AVAILABLE"}

# --- guardrail vocabulary --------------------------------------------------
FORBIDDEN_TRUE_FIELDS = {
    "ground_truth_created", "ground_reference_created", "label_created",
    "operational_ground_truth", "training_ready", "training_use_allowed",
    "overlay_ready", "overlay_executed", "prediction_ready", "promotion_allowed",
    "can_create_ground_truth", "can_create_label", "can_use_for_ground_truth",
    "raw_data_versioned", "sentinel_date_inferred", "crosswalk_inferred",
    "geometry_inferred", "coordinate_invented", "geometry_invented",
    "protocol_b_open", "protocol_b_reopened", "operational_use_allowed",
    "patch_truth_allowed", "flood_mask_created",
}
FORBIDDEN_STATUS_VALUES = {
    "GROUND_TRUTH_VALIDATED", "GROUND_REFERENCE_CREATED", "GROUND_REFERENCE_TRUE",
    "LABEL_READY", "LABEL_POSITIVE", "LABEL_NEGATIVE", "TRAINING_READY",
    "PROTOCOL_B_OPEN", "PROTOCOL_B_REOPENED", "OPERATIONAL_VALIDATION",
    "PATCH_POSITIVE", "PATCH_NEGATIVE", "FLOOD_DETECTED", "PROMOTION_ALLOWED",
    "OVERLAY_EXECUTED",
}
FORBIDDEN_KV_MARKERS = [
    "ground_truth=true", "ground_reference=true", "label=true", "training=true",
    "overlay=true", "prediction=true", "protocol_b_open=true", "protocol_b_reopen=true",
    "sentinel_date_inferred=true", "crosswalk_inferred=true", "geometry_inferred=true",
    "operational_validation=true", "promotion_allowed=true",
    "can_create_ground_truth=true", "can_create_label=true", "raw_data_versioned=true",
    "patch_truth_allowed=true",
]
UNSAFE_LANGUAGE = [
    "ground truth validado", "classe positiva", "classe negativa", "label operacional",
    "deteccao de enchente", "deteccao de inundacao", "predicao de inundacao",
    "mascara de inundacao", "modelo preditivo", "validacao operacional",
    "treinamento supervisionado pronto", "similaridade visual confirma",
    "dino confirma evento", "overlay operacional executado", "coordenada inventada",
]
SAFE_UNSAFE_FIELDS = {
    "forbidden_use", "forbidden_decisions", "do_not_infer", "blocking_reason",
    "why_not_ground_truth", "dominant_blocker", "notes", "priority_reason",
    "geometry_null_reason", "geometry_status", "geometry_candidate_type",
    "geometry_source_type", "geometry_precision_level", "event_geometry_status",
    "event_geometry_candidate_status", "patch_geometry_status", "crosswalk_status",
    "match_review_status", "join_status", "patch_reference_candidate_status",
    "patch_reference_band", "anchor_strength_band", "anchor_specificity",
    "review_question", "evidence_to_check", "required_external_validation",
    "allowed_decisions", "review_status", "task_type", "source_to_open",
    "geometry_to_digitize", "expected_output", "acceptance_criteria", "safe_use",
    "recommended_next_step", "required_next_action", "reference_level",
    "human_review_decision", "v2ap_readiness_band", "geometry_readiness_status",
    "strongest_anchor_text", "strongest_anchor_type", "patch_link_review_status",
    "geometry_source_artifact", "geometry_source_field", "region", "patch_id",
    "reason", "purpose", "status", "patch_truth_allowed", "review_packet_id",
}
SAFE_CONTEXT_MARKERS = [
    "nao pode dizer", "nao usar", "nao afirmar", "nao ha", "nao deve", "nao temos",
    "nao realiza", "nao detecta", "nao cria", "nao existe", "nao produz", "nao inferir",
    "nao inventar", "nao significa", "nao implica", "nao foi", "nao e ", "nao ", "do_not",
    "do not", "proibid", "forbidden", "limitation", "limitacao", "blocker", "blocked",
    "bloque", "does not", "not ", "no ", "sem ", "evitar", "ausencia", "pendente",
    "candidato", "review-only", "needs_", "missing", "not_established", "not_created",
    "insufficient", "digitization_required", "digitization", "manual", "null",
    "anchor_only", "textual_anchor", "external_validation", "explicit",
]
ABSOLUTE_PATH_RE = re.compile(r"(?:[A-Za-z]:\\|/Users/|/home/|/mnt/|\\\\)")
LOCAL_ONLY_MARKER = "local" + "_" + "only"


def parse_args(argv=None):
    return argparse.ArgumentParser().parse_args(argv)


# --- path helpers ----------------------------------------------------------
def dataset_path(name):
    return os.path.join(DATASET_DIR, name)


def root_dataset_path(name):
    return os.path.join(DATASET_ROOT, name)


def doc_path(name):
    return os.path.join(DOCS_DIR, name)


def geojson_path(name):
    return os.path.join(GEOJSON_DIR, name)


def rel_dataset(name):
    return f"datasets/protocolo_c/{name}"


def rel_doc(name):
    return f"docs/protocolo_c/v2aq_event_geometry_patch_link/{name}"


def rel_geojson(name):
    return f"docs/protocolo_c/v2aq_event_geometry_patch_link/geojson_candidates/{name}"


def repo_relative_path(path):
    raw = str(path)
    if ABSOLUTE_PATH_RE.search(raw):
        raise ValueError(f"Refusing absolute path: {path}")
    return raw.replace("\\", "/")


# --- value helpers ---------------------------------------------------------
def clean(value):
    return str(value or "").strip()


def is_true(value):
    return clean(value).lower() == "true"


def normalize_bool(value):
    return "true" if is_true(value) else "false"


def safe_slug(text):
    return re.sub(r"[^a-z0-9]+", "-", clean(text).lower()).strip("-") or "item"


def short_fragment(text, limit=160):
    return re.sub(r"\s+", " ", clean(text))[:limit].strip()


def normalize_region(value, candidate_id=""):
    prefix = clean(candidate_id)[:3].upper()
    if prefix in REGION_BY_PREFIX:
        return REGION_BY_PREFIX[prefix]
    low = clean(value).lower()
    if "recife" in low or low == "rec":
        return "Recife"
    if "petrop" in low or low == "pet":
        return "Petropolis"
    if "curitiba" in low or low == "ctb":
        return "Curitiba"
    return clean(value) or "Unspecified"


def normalize_candidate_id(value):
    return clean(value).upper()


def normalize_patch_id(value):
    return re.sub(r"\s+", "", clean(value)).upper()


# --- io helpers ------------------------------------------------------------
def load_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def assert_output_is_v2aq(path):
    base = os.path.basename(str(path))
    if LOCAL_ONLY_MARKER in str(path).lower():
        raise ValueError(f"Refusing local_only output path: {path}")
    if not base.startswith("v2aq_"):
        raise ValueError(f"Refusing to write non-v2aq output: {path}")
    return True


def write_csv(path, columns, rows):
    assert_output_is_v2aq(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_text(path):
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write_markdown(path, lines):
    assert_output_is_v2aq(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_json(path, payload):
    assert_output_is_v2aq(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def write_geojson(path, features):
    payload = {"type": "FeatureCollection", "features": features}
    write_json(path, payload)


def sha256_file(path):
    if not os.path.exists(path):
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text):
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def write_markdown_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        cells = [clean(c).replace("|", "\\|").replace("\n", " ") for c in row]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


# --- schema / loaders ------------------------------------------------------
def assert_min_schema(rows, required, artifact):
    if not rows:
        raise FileNotFoundError(f"Required artifact is missing or empty: {artifact}")
    missing = [c for c in required if c not in rows[0]]
    if missing:
        raise ValueError(f"{artifact} missing required columns: {','.join(missing)}")
    return True


def load_v2ap_stack():
    missing = [name for name in V2AP_INPUTS.values()
               if not os.path.exists(dataset_path(name))]
    if missing:
        raise FileNotFoundError("v2aq requires v2ap outputs; missing: " + ",".join(missing))
    stack = {key: load_csv(dataset_path(name)) for key, name in V2AP_INPUTS.items()}
    assert_min_schema(stack["geometry"], ["candidate_id", "geometry_readiness_status"],
                      V2AP_INPUTS["geometry"])
    return stack


def load_v2ao_stack():
    stack = {key: load_csv(dataset_path(name)) for key, name in V2AO_INPUTS.items()}
    stack["spatial"] = load_csv(dataset_path(V2AN_OPTIONAL["spatial"]))
    stack["observed"] = load_csv(root_dataset_path(OBSERVED_REGISTRY))
    return stack


# --- guardrail assertions --------------------------------------------------
def _iter_values(rows_or_text):
    if isinstance(rows_or_text, list):
        for idx, row in enumerate(rows_or_text):
            values = row.values() if isinstance(row, dict) else [row]
            for v in values:
                yield idx, v
    else:
        yield 0, rows_or_text


def assert_no_absolute_paths_in_content(rows_or_text):
    for idx, value in _iter_values(rows_or_text):
        if ABSOLUTE_PATH_RE.search(clean(value)):
            raise ValueError(f"Absolute path in content at row {idx}: {value}")
    return True


def assert_no_local_only(rows_or_text):
    for idx, value in _iter_values(rows_or_text):
        if LOCAL_ONLY_MARKER in clean(value).lower():
            raise ValueError(f"local_only marker in content at row {idx}: {value}")
    return True


def _field_allows_unsafe(key, value):
    key_l = clean(key).lower()
    value_l = clean(value).lower()
    return key_l in SAFE_UNSAFE_FIELDS or any(m in value_l for m in SAFE_CONTEXT_MARKERS)


def assert_no_operational_promotion(rows):
    violations = []
    for idx, row in enumerate(rows):
        for key, value in row.items():
            key_l = clean(key).lower()
            value_s = clean(value)
            value_l = value_s.lower()
            if key_l in FORBIDDEN_TRUE_FIELDS and is_true(value_s):
                violations.append((idx, key, "forbidden_true"))
            if value_s in FORBIDDEN_STATUS_VALUES:
                violations.append((idx, key, "forbidden_status"))
            if ABSOLUTE_PATH_RE.search(value_s):
                violations.append((idx, key, "absolute_path"))
            if LOCAL_ONLY_MARKER in value_l:
                violations.append((idx, key, "local_only"))
            squashed = re.sub(r"\s*=\s*", "=", value_l)
            for marker in FORBIDDEN_KV_MARKERS:
                if marker in squashed:
                    violations.append((idx, key, f"forbidden_kv:{marker}"))
            for phrase in UNSAFE_LANGUAGE:
                if phrase in value_l and not _field_allows_unsafe(key, value_s):
                    violations.append((idx, key, f"unsafe_language:{phrase}"))
    if violations:
        sample = "; ".join(f"row={r[0]} field={r[1]} type={r[2]}" for r in violations[:5])
        raise ValueError(f"Operational promotion violation: {sample}")
    return True


def assert_no_label_creation(rows):
    for idx, row in enumerate(rows):
        for key, value in row.items():
            key_l = clean(key).lower()
            if key_l in {"can_create_label", "label_created", "training_use_allowed"} and is_true(value):
                raise ValueError(f"label/training creation at row {idx}: {key_l}")
            if clean(value) in {"LABEL_READY", "TRAINING_READY"}:
                raise ValueError(f"label/training status at row {idx}: {value}")
    return True


def assert_no_fake_ground_truth(rows):
    for idx, row in enumerate(rows):
        for key, value in row.items():
            key_l = clean(key).lower()
            value_s = clean(value)
            if key_l in {"operational_ground_truth_status", "ground_truth_status"}:
                if value_s.upper() not in {"NOT_ESTABLISHED", "", "NOT_CREATED"}:
                    raise ValueError(f"ground truth must stay NOT_ESTABLISHED, got {value_s}")
            if key_l in {"can_create_ground_truth", "can_use_for_ground_truth",
                         "ground_truth_created"} and is_true(value_s):
                raise ValueError(f"{key_l}=true forbidden at row {idx}.")
    return True


def assert_no_fake_geometry(rows):
    for idx, row in enumerate(rows):
        for key, value in row.items():
            key_l = clean(key).lower()
            if key_l in {"coordinate_invented", "geometry_invented", "geometry_inferred"} and is_true(value):
                raise ValueError(f"invented/inferred geometry at row {idx}: {key_l}")
            if key_l == "geometry_status" and clean(value).upper() in EXPLICIT_GEOMETRY_STATUS:
                src = clean(row.get("geometry_source_type")).lower()
                if src in {"textual_anchor", "", "guess", "inferred"}:
                    raise ValueError(
                        f"explicit geometry status without explicit source at row {idx}.")
    return True


def assert_no_fake_overlay(rows):
    for idx, row in enumerate(rows):
        for key, value in row.items():
            if clean(key).lower() in {"overlay_executed", "overlay_ready", "flood_mask_created"} and is_true(value):
                raise ValueError(f"overlay/mask flagged at row {idx}: {key}")
    return True


def scan_text_violations(text):
    counts = {"absolute_path": 0, "local_only": 0, "forbidden_kv": 0,
              "unsafe_language": 0, "forbidden_true_flag": 0, "forbidden_status": 0}
    for line in text.splitlines():
        line_l = line.lower()
        safe_context = any(m in line_l for m in SAFE_CONTEXT_MARKERS)
        if ABSOLUTE_PATH_RE.search(line):
            counts["absolute_path"] += 1
        if LOCAL_ONLY_MARKER in line_l:
            counts["local_only"] += 1
        squashed = re.sub(r"\s*=\s*", "=", line_l)
        for marker in FORBIDDEN_KV_MARKERS:
            if marker in squashed:
                counts["forbidden_kv"] += 1
        if not safe_context and any(s in line for s in FORBIDDEN_STATUS_VALUES):
            counts["forbidden_status"] += 1
        for phrase in UNSAFE_LANGUAGE:
            if phrase in line_l and not safe_context:
                counts["unsafe_language"] += 1
    return counts


def assert_safe_text(text):
    counts = scan_text_violations(text)
    bad = {k: v for k, v in counts.items() if v}
    if bad:
        raise ValueError(f"Unsafe text detected: {bad}")
    return True


# --- record constructors ---------------------------------------------------
def build_event_geometry_candidate_row(candidate_id, region, geometry_candidate_type,
                                        geometry_source_type, source_artifact, source_field,
                                        status, precision, confidence, manual_digit,
                                        can_patch_link, notes, explicit_geometry_geojson=""):
    return {
        "event_geometry_id": f"EG_v2aq_{candidate_id}",
        "candidate_id": candidate_id,
        "region": region,
        "geometry_candidate_type": geometry_candidate_type,
        "geometry_source_type": geometry_source_type,
        "geometry_source_artifact": source_artifact,
        "geometry_source_field": source_field,
        "geometry_status": status,
        "geometry_precision_level": precision,
        "geometry_confidence": confidence,
        "manual_digitization_required": normalize_bool(manual_digit),
        "can_use_for_patch_link_review": normalize_bool(can_patch_link),
        "can_use_for_ground_truth": "false",
        "notes": short_fragment(notes, 160),
        "explicit_geometry_geojson": explicit_geometry_geojson,
    }


def build_geojson_feature(props, geometry):
    return {"type": "Feature", "geometry": geometry, "properties": props}


def build_patch_link_review_row(candidate_id, region, patch_id, event_geometry_status,
                                crosswalk_status, review_question, evidence, required_ext,
                                allowed_decisions, forbidden_decisions):
    return {
        "review_packet_id": f"PLR_v2aq_{candidate_id}",
        "candidate_id": candidate_id,
        "region": region,
        "patch_id": patch_id,
        "event_geometry_status": event_geometry_status,
        "crosswalk_status": crosswalk_status,
        "review_question": review_question,
        "evidence_to_check": evidence,
        "required_external_validation": required_ext,
        "allowed_decisions": allowed_decisions,
        "forbidden_decisions": forbidden_decisions,
        "review_status": "PENDING_PATCH_LINK_REVIEW",
    }


# --- derivation ------------------------------------------------------------
_LEVEL_RANK = {
    "C0_REJECTED_OR_INSUFFICIENT": 0, "C1_CONTEXTUAL_OBSERVED_EVENT": 1,
    "C2_DOCUMENTED_OBSERVED_EVENT": 2, "C3_STRONG_REFERENCE_CANDIDATE": 3,
    "C4_READY_FOR_EXTERNAL_VALIDATION_REVIEW": 4,
}
ALLOWED_REVIEW_DECISIONS = ("PATCH_LINK_REVIEW_CANDIDATE|NEEDS_EVENT_GEOMETRY_DIGITIZATION|"
                            "NEEDS_SENTINEL_CROSSWALK|REMAINS_EVENT_REFERENCE_ONLY|"
                            "REJECT_FOR_PATCH_LINK")
FORBIDDEN_REVIEW_DECISIONS = ("GROUND_TRUTH_VALIDATED|LABEL_READY|PROTOCOL_B_OPEN|"
                              "TRAINING_READY|OPERATIONAL_VALIDATION|PATCH_TRUTH_CONFIRMED")


def derive_candidates():
    v2ap = load_v2ap_stack()
    v2ao = load_v2ao_stack()
    geometry = {r["candidate_id"]: r for r in v2ap["geometry"]}
    selection = {r["candidate_id"]: r for r in v2ap["selection"]}
    scores = {r["candidate_id"]: r for r in v2ap["scores"]}
    levels = {r["candidate_id"]: clean(r.get("reference_level")) for r in v2ao["levels"]}
    ranking = {r["candidate_id"]: r for r in v2ao["ranking"]}
    observed = {clean(r.get("observed_event_id")): r for r in v2ao["observed"]}
    anchors = {}
    for r in v2ao["spatial"]:
        anchors.setdefault(r["candidate_id"], []).append(
            (clean(r.get("anchor_type")), clean(r.get("anchor_text"))))
    crosswalk = {}
    for r in v2ap["crosswalk"]:
        crosswalk.setdefault(r["candidate_id"], []).append(r)
    patch_has_geom = any(is_true(r.get("has_geometry")) or is_true(r.get("has_bbox"))
                         for r in v2ap["patch_registry"])
    patch_id_for = {}
    for r in v2ap["crosswalk"]:
        pid = normalize_patch_id(r.get("patch_id_detected"))
        if pid and r["candidate_id"] not in patch_id_for:
            patch_id_for[r["candidate_id"]] = pid

    out = []
    for cid in selection:
        geo = geometry.get(cid, {})
        cand_anchors = anchors.get(cid, [])
        spec = max((ANCHOR_SPECIFICITY.get(t, 0) for t, _ in cand_anchors), default=0)
        strongest = max(cand_anchors, key=lambda at: ANCHOR_SPECIFICITY.get(at[0], 0),
                        default=("", ""))
        cc = crosswalk.get(cid, [])
        has_cc = any(is_true(r.get("within_temporal_window")) or
                     is_true(r.get("can_be_used_as_explicit_crosswalk")) for r in cc)
        explicit_cc = any(is_true(r.get("can_be_used_as_explicit_crosswalk")) for r in cc)
        level = levels.get(cid, clean(selection[cid].get("reference_level")))
        out.append({
            "candidate_id": cid,
            "region": normalize_region(selection[cid].get("region"), cid),
            "reference_level": level,
            "level_rank": _LEVEL_RANK.get(level, 0),
            "human_review_decision": clean(selection[cid].get("human_review_decision")),
            "v2ap_readiness_band": clean(scores.get(cid, {}).get("readiness_band")),
            "geometry_readiness_status": clean(geo.get("geometry_readiness_status")),
            "has_event_geometry": is_true(geo.get("has_event_geometry")),
            "has_patch_geometry": is_true(geo.get("has_patch_geometry")) or patch_has_geom,
            "anchor_count": len(cand_anchors),
            "anchor_types": [t for t, _ in cand_anchors],
            "strongest_anchor_type": strongest[0],
            "strongest_anchor_text": strongest[1] or strongest[0],
            "anchor_specificity_value": spec,
            "has_crosswalk_candidate": has_cc,
            "has_explicit_crosswalk": explicit_cc,
            "crosswalk_count": len(cc),
            "patch_id": patch_id_for.get(cid, ""),
            "event_date_start": clean(observed.get(cid, {}).get("date_start")),
            "readiness_score": int(clean(ranking.get(cid, {}).get("readiness_score")) or 0),
        })
    out.sort(key=lambda d: (-d["level_rank"], -d["readiness_score"], d["candidate_id"]))
    return out


def _anchor_band(spec):
    return "HIGH" if spec >= 4 else "MEDIUM" if spec >= 2 else "LOW"


def _event_geometry_status(d):
    """Status from explicit evidence only; never invents geometry."""
    spec = d["anchor_specificity_value"]
    has_map = d["has_event_geometry"] or "mapa_ou_laudo" in d["anchor_types"]
    has_point = "ponto_de_alagamento" in d["anchor_types"] or "rua" in d["anchor_types"]
    if has_map or has_point:
        return "OFFICIAL_MAP_DIGITIZATION_REQUIRED"
    if spec >= 2:
        return "TEXTUAL_ANCHOR_ONLY"
    return "INSUFFICIENT_GEOMETRY"


# --- column schemas --------------------------------------------------------
PRIORITY_COLUMNS = [
    "priority_id", "candidate_id", "region", "reference_level", "human_review_decision",
    "v2ap_readiness_band", "geometry_readiness_status", "crosswalk_candidate_count",
    "priority_rank", "priority_reason", "included_in_v2aq",
]
ANCHOR_COLUMNS = [
    "anchor_strength_id", "candidate_id", "region", "anchor_count", "strongest_anchor_text",
    "strongest_anchor_type", "anchor_specificity", "geometry_derivation_allowed",
    "manual_digitization_required", "anchor_strength_band", "blocking_reason",
]
EVENT_GEOMETRY_COLUMNS = [
    "event_geometry_id", "candidate_id", "region", "geometry_candidate_type",
    "geometry_source_type", "geometry_source_artifact", "geometry_source_field",
    "geometry_status", "geometry_precision_level", "geometry_confidence",
    "manual_digitization_required", "can_use_for_patch_link_review",
    "can_use_for_ground_truth", "notes", "explicit_geometry_geojson",
]
GEOJSON_INDEX_COLUMNS = [
    "geojson_id", "candidate_id", "geojson_path", "geometry_present", "geometry_null_reason",
    "manual_digitization_required", "safe_use", "forbidden_use",
]
MATCH_COLUMNS = [
    "match_candidate_id", "candidate_id", "region", "event_geometry_status",
    "patch_registry_item_id", "patch_id", "patch_geometry_status", "match_review_status",
    "overlay_executed", "manual_review_required", "blocking_reason",
]
JOIN_COLUMNS = [
    "join_id", "candidate_id", "region", "event_geometry_id", "patch_id",
    "crosswalk_candidate_id", "has_event_geometry_candidate", "has_patch_geometry",
    "has_crosswalk_candidate", "join_status", "patch_level_review_ready",
    "patch_truth_allowed", "blocking_reason",
]
REVIEW_PACKET_COLUMNS = [
    "review_packet_id", "candidate_id", "region", "patch_id", "event_geometry_status",
    "crosswalk_status", "review_question", "evidence_to_check", "required_external_validation",
    "allowed_decisions", "forbidden_decisions", "review_status",
]
SCORE_COLUMNS = [
    "score_id", "candidate_id", "region", "event_geometry_score", "patch_geometry_score",
    "sentinel_crosswalk_score", "source_trace_score", "overall_patch_reference_score",
    "patch_reference_band", "dominant_blocker", "patch_reference_candidate_status",
    "can_create_ground_truth", "can_create_label", "protocol_b_status",
]
TASK_COLUMNS = [
    "task_id", "candidate_id", "region", "task_type", "source_to_open", "geometry_to_digitize",
    "expected_output", "priority", "do_not_infer", "acceptance_criteria",
]
BOUNDARY_COLUMNS = [
    "boundary_id", "candidate_id", "event_geometry_candidate_status", "patch_geometry_status",
    "crosswalk_status", "patch_link_review_status", "ground_truth_blocked", "label_blocked",
    "protocol_b_blocked", "why_not_ground_truth",
]
REGRESSION_COLUMNS = [
    "regression_id", "artifact_path", "check_type", "violation_count", "status", "severity", "notes",
]
NEXT_COLUMNS = [
    "rank", "next_action", "score", "allowed", "blocked_operational_use", "required_input",
    "recommended_artifact", "notes",
]
MANIFEST_COLUMNS = ["step_order", "step_name", "status", "outputs", "output_hashes", "notes"]
COMPLETION_COLUMNS = ["completion_id", "metric", "value", "status", "notes"]
FORBIDDEN_USE = "ground_truth|label|training|overlay|prediction|protocol_b_reopen"


# --- runners ---------------------------------------------------------------
def run_priority_candidate_builder(args=None):
    rows = []
    for rank, d in enumerate(derive_candidates(), 1):
        high = d["level_rank"] >= _LEVEL_RANK["C3_STRONG_REFERENCE_CANDIDATE"]
        reasons = []
        if d["level_rank"] == _LEVEL_RANK["C4_READY_FOR_EXTERNAL_VALIDATION_REVIEW"]:
            reasons.append("C4")
        elif high:
            reasons.append("C3")
        if d["geometry_readiness_status"] == "EVENT_AND_PATCH_GEOMETRY_READY":
            reasons.append("EVENT_AND_PATCH_GEOMETRY_READY")
        if d["has_explicit_crosswalk"]:
            reasons.append("explicit_crosswalk")
        if d["anchor_specificity_value"] >= 4:
            reasons.append("strong_spatial_anchor")
        rows.append({
            "priority_id": f"PRI_v2aq_{d['candidate_id']}",
            "candidate_id": d["candidate_id"],
            "region": d["region"],
            "reference_level": d["reference_level"],
            "human_review_decision": d["human_review_decision"],
            "v2ap_readiness_band": d["v2ap_readiness_band"],
            "geometry_readiness_status": d["geometry_readiness_status"],
            "crosswalk_candidate_count": str(d["crosswalk_count"]),
            "priority_rank": str(rank),
            "priority_reason": "|".join(reasons) or "lower_priority_geometry_collection",
            "included_in_v2aq": "true",
        })
    assert_no_operational_promotion(rows)
    write_csv(dataset_path("v2aq_priority_candidates.csv"), PRIORITY_COLUMNS, rows)
    lines = [
        "# v2aq - candidatos priorizados para aquisicao geometrica",
        "",
        "Prioriza C4/C3, EVENT_AND_PATCH_GEOMETRY_READY, crosswalk explicito e ancora forte.",
        "",
    ]
    lines.extend(write_markdown_table(
        ["priority_rank", "candidate_id", "reference_level", "geometry_readiness_status", "priority_reason"],
        [(r["priority_rank"], r["candidate_id"], r["reference_level"],
          r["geometry_readiness_status"], r["priority_reason"]) for r in rows]))
    write_markdown(doc_path("v2aq_priority_candidates.md"), lines)
    return rows


def run_spatial_anchor_strength_builder(args=None):
    rows = []
    for d in derive_candidates():
        spec = d["anchor_specificity_value"]
        band = _anchor_band(spec)
        derivation_allowed = spec >= 4  # only map/point/area can derive geometry
        rows.append({
            "anchor_strength_id": f"AS_v2aq_{d['candidate_id']}",
            "candidate_id": d["candidate_id"],
            "region": d["region"],
            "anchor_count": str(d["anchor_count"]),
            "strongest_anchor_text": short_fragment(d["strongest_anchor_text"], 100),
            "strongest_anchor_type": d["strongest_anchor_type"] or "none",
            "anchor_specificity": band,
            "geometry_derivation_allowed": normalize_bool(derivation_allowed),
            "manual_digitization_required": "true",
            "anchor_strength_band": band,
            "blocking_reason": ("Ancora de mapa/ponto/area permite digitalizacao manual; "
                                "bairro/municipio amplo nao deriva geometria pronta. Nao inventar."),
        })
    assert_no_operational_promotion(rows)
    write_csv(dataset_path("v2aq_spatial_anchor_strength.csv"), ANCHOR_COLUMNS, rows)
    lines = [
        "# v2aq - forca das ancoras espaciais",
        "",
        "geometry_derivation_allowed=true so para mapa/ponto/area. Bairro/municipio amplo",
        "exige manual_digitization_required e nao vira geometria pronta. Nada e inventado.",
        "",
    ]
    lines.extend(write_markdown_table(
        ["candidate_id", "strongest_anchor_type", "anchor_specificity", "geometry_derivation_allowed"],
        [(r["candidate_id"], r["strongest_anchor_type"], r["anchor_specificity"],
          r["geometry_derivation_allowed"]) for r in rows]))
    write_markdown(doc_path("v2aq_spatial_anchor_strength.md"), lines)
    return rows


def run_event_geometry_candidate_builder(args=None):
    rows = []
    for d in derive_candidates():
        status = _event_geometry_status(d)
        if status == "OFFICIAL_MAP_DIGITIZATION_REQUIRED":
            src_type = "official_map_documented"
            precision = "sector_or_affected_area_pending_digitization"
            confidence = "MEDIUM"
            can_link = True
            cand_type = "polygon_area_pending_digitization"
        elif status == "TEXTUAL_ANCHOR_ONLY":
            src_type = "textual_anchor"
            precision = "neighborhood_or_corridor_textual"
            confidence = "LOW"
            can_link = False
            cand_type = "textual_anchor_no_geometry"
        else:
            src_type = "textual_anchor"
            precision = "municipality_only"
            confidence = "LOW"
            can_link = False
            cand_type = "insufficient"
        rows.append(build_event_geometry_candidate_row(
            d["candidate_id"], d["region"], cand_type, src_type,
            rel_dataset(V2AN_OPTIONAL["spatial"]), f"anchor_type={d['strongest_anchor_type']}",
            status, precision, confidence,
            manual_digit=True, can_patch_link=can_link,
            notes="Geometria nao inventada; digitalizacao manual quando ha mapa/ponto oficial."))
    assert_no_operational_promotion(rows)
    assert_no_fake_geometry(rows)
    for r in rows:
        if r["geometry_status"] not in ALLOWED_GEOMETRY_STATUS:
            raise ValueError(f"Illegal geometry_status: {r['geometry_status']}")
        if is_true(r["can_use_for_ground_truth"]):
            raise ValueError("can_use_for_ground_truth must stay false.")
    write_csv(dataset_path("v2aq_event_geometry_candidates.csv"), EVENT_GEOMETRY_COLUMNS, rows)
    lines = [
        "# v2aq - candidatos de geometria de evento",
        "",
        "can_use_for_ground_truth=false sempre. Geometria explicita so de fonte explicita;",
        "caso contrario, OFFICIAL_MAP_DIGITIZATION_REQUIRED ou TEXTUAL_ANCHOR_ONLY (sem inventar).",
        "",
    ]
    lines.extend(write_markdown_table(
        ["candidate_id", "geometry_status", "geometry_source_type", "can_use_for_patch_link_review"],
        [(r["candidate_id"], r["geometry_status"], r["geometry_source_type"],
          r["can_use_for_patch_link_review"]) for r in rows]))
    write_markdown(doc_path("v2aq_event_geometry_candidates.md"), lines)
    return rows


def run_geojson_candidate_exporter(args=None):
    candidates = load_csv(dataset_path("v2aq_event_geometry_candidates.csv"))
    index = []
    for c in candidates:
        cid = c["candidate_id"]
        status = clean(c.get("geometry_status"))
        explicit_raw = clean(c.get("explicit_geometry_geojson"))
        geometry = None
        geometry_null_reason = ""
        if status in EXPLICIT_GEOMETRY_STATUS and explicit_raw:
            try:
                parsed = json.loads(explicit_raw)
                if isinstance(parsed, dict) and parsed.get("type") and parsed.get("coordinates") is not None:
                    geometry = parsed
            except (ValueError, TypeError):
                geometry = None
        if geometry is None:
            geometry_null_reason = ("manual_digitization_required" if "DIGITIZATION" in status
                                    else "textual_anchor_only_no_explicit_geometry"
                                    if status == "TEXTUAL_ANCHOR_ONLY"
                                    else "insufficient_geometry")
        props = {
            "candidate_id": cid,
            "region": clean(c.get("region")),
            "geometry_status": status,
            "source_artifact": clean(c.get("geometry_source_artifact")),
            "manual_digitization_required": normalize_bool(c.get("manual_digitization_required")),
            "not_ground_truth": True,
            "not_label": True,
            "patch_truth_allowed": False,
        }
        feature = build_geojson_feature(props, geometry)
        fname = f"v2aq_event_geometry_{safe_slug(cid)}.geojson"
        write_geojson(geojson_path(fname), [feature])
        index.append({
            "geojson_id": f"GJ_v2aq_{cid}",
            "candidate_id": cid,
            "geojson_path": rel_geojson(fname),
            "geometry_present": normalize_bool(geometry is not None),
            "geometry_null_reason": geometry_null_reason,
            "manual_digitization_required": normalize_bool(c.get("manual_digitization_required")),
            "safe_use": "Referencia observacional candidata para revisao/digitalizacao externa.",
            "forbidden_use": FORBIDDEN_USE,
        })
    assert_no_operational_promotion(index)
    assert_no_absolute_paths_in_content(index)
    write_csv(dataset_path("v2aq_geojson_candidate_index.csv"), GEOJSON_INDEX_COLUMNS, index)
    present = sum(1 for r in index if r["geometry_present"] == "true")
    lines = [
        "# v2aq - indice de GeoJSON candidatos",
        "",
        f"GeoJSONs criados: {len(index)}; com geometria real: {present}; geometry null: {len(index) - present}.",
        "Sem geometria explicita -> geometry: null; nenhuma coordenada e inventada.",
        "",
    ]
    lines.extend(write_markdown_table(
        ["candidate_id", "geojson_path", "geometry_present", "geometry_null_reason"],
        [(r["candidate_id"], short_fragment(r["geojson_path"], 70), r["geometry_present"],
          r["geometry_null_reason"]) for r in index]))
    write_markdown(doc_path("v2aq_geojson_candidate_index.md"), lines)
    return index


def run_patch_geometry_match_builder(args=None):
    v2ap = load_v2ap_stack()
    geom = {r["candidate_id"]: r for r in load_csv(dataset_path("v2aq_event_geometry_candidates.csv"))}
    geom_registries = [r for r in v2ap["patch_registry"]
                       if is_true(r.get("has_geometry")) or is_true(r.get("has_bbox"))]
    rows = []
    for d in derive_candidates():
        eg = geom.get(d["candidate_id"], {})
        reg = geom_registries[0] if geom_registries else {}
        patch_geom_status = "PATCH_GEOMETRY_COLUMNS_AVAILABLE" if geom_registries else "NO_PATCH_GEOMETRY"
        match_status = ("EVENT_AND_PATCH_GEOMETRY_PENDING_MANUAL_REVIEW"
                        if is_true(eg.get("can_use_for_patch_link_review")) and geom_registries
                        else "NEEDS_EVENT_GEOMETRY" if geom_registries
                        else "NEEDS_PATCH_GEOMETRY")
        rows.append({
            "match_candidate_id": f"MT_v2aq_{d['candidate_id']}",
            "candidate_id": d["candidate_id"],
            "region": d["region"],
            "event_geometry_status": clean(eg.get("geometry_status")),
            "patch_registry_item_id": clean(reg.get("registry_item_id")),
            "patch_id": d["patch_id"] or clean(reg.get("patch_id")),
            "patch_geometry_status": patch_geom_status,
            "match_review_status": match_status,
            "overlay_executed": "false",
            "manual_review_required": "true",
            "blocking_reason": ("Sem overlay nem interseccao; match exige revisao manual de "
                                "geometria de evento e de patch."),
        })
    assert_no_operational_promotion(rows)
    assert_no_fake_overlay(rows)
    write_csv(dataset_path("v2aq_patch_geometry_match_candidates.csv"), MATCH_COLUMNS, rows)
    lines = [
        "# v2aq - candidatos de match geometria evento-patch",
        "",
        "overlay_executed=false e manual_review_required=true para todos. Nenhuma interseccao calculada.",
        "",
    ]
    lines.extend(write_markdown_table(
        ["candidate_id", "event_geometry_status", "patch_geometry_status", "match_review_status", "overlay_executed"],
        [(r["candidate_id"], r["event_geometry_status"], r["patch_geometry_status"],
          r["match_review_status"], r["overlay_executed"]) for r in rows]))
    write_markdown(doc_path("v2aq_patch_geometry_match_candidates.md"), lines)
    return rows


def run_crosswalk_geometry_join_builder(args=None):
    geom = {r["candidate_id"]: r for r in load_csv(dataset_path("v2aq_event_geometry_candidates.csv"))}
    rows = []
    for d in derive_candidates():
        eg = geom.get(d["candidate_id"], {})
        has_eg = is_true(eg.get("can_use_for_patch_link_review"))
        has_pg = d["has_patch_geometry"]
        has_cc = d["has_crosswalk_candidate"]
        ready = has_eg and has_pg and has_cc
        if ready:
            status = "PATCH_LEVEL_REVIEW_READY_PENDING_EXTERNAL_VALIDATION"
            blocking = ""
        elif not has_eg:
            status = "NEEDS_EVENT_GEOMETRY"
            blocking = "Geometria de evento usavel ausente (digitalizar mapa/ponto oficial)."
        elif not has_cc:
            status = "NEEDS_SENTINEL_CROSSWALK"
            blocking = "Sem crosswalk Sentinel candidato; nao inferir por similaridade visual."
        else:
            status = "NEEDS_PATCH_GEOMETRY"
            blocking = "Sem geometria de patch disponivel."
        rows.append({
            "join_id": f"JN_v2aq_{d['candidate_id']}",
            "candidate_id": d["candidate_id"],
            "region": d["region"],
            "event_geometry_id": f"EG_v2aq_{d['candidate_id']}",
            "patch_id": d["patch_id"],
            "crosswalk_candidate_id": (f"present:{d['crosswalk_count']}" if has_cc else "none"),
            "has_event_geometry_candidate": normalize_bool(has_eg),
            "has_patch_geometry": normalize_bool(has_pg),
            "has_crosswalk_candidate": normalize_bool(has_cc),
            "join_status": status,
            "patch_level_review_ready": normalize_bool(ready),
            "patch_truth_allowed": "false",
            "blocking_reason": blocking,
        })
    assert_no_operational_promotion(rows)
    for r in rows:
        if is_true(r["patch_truth_allowed"]):
            raise ValueError("patch_truth_allowed must stay false in v2aq.")
    write_csv(dataset_path("v2aq_crosswalk_geometry_join_candidates.csv"), JOIN_COLUMNS, rows)
    lines = [
        "# v2aq - join geometria-crosswalk",
        "",
        "patch_level_review_ready=true exige geometria candidata + patch geometry + crosswalk candidate.",
        "patch_truth_allowed=false sempre.",
        "",
    ]
    lines.extend(write_markdown_table(
        ["candidate_id", "event_geometry", "patch_geometry", "crosswalk", "join_status", "review_ready"],
        [(r["candidate_id"], r["has_event_geometry_candidate"], r["has_patch_geometry"],
          r["has_crosswalk_candidate"], r["join_status"], r["patch_level_review_ready"]) for r in rows]))
    write_markdown(doc_path("v2aq_crosswalk_geometry_join_candidates.md"), lines)
    return rows


def run_patch_link_review_packet_builder(args=None):
    geom = {r["candidate_id"]: r for r in load_csv(dataset_path("v2aq_event_geometry_candidates.csv"))}
    rows = []
    for d in derive_candidates():
        eg = geom.get(d["candidate_id"], {})
        # only build packets for prioritized/strong candidates
        if d["level_rank"] < _LEVEL_RANK["C3_STRONG_REFERENCE_CANDIDATE"]:
            continue
        rows.append(build_patch_link_review_row(
            d["candidate_id"], d["region"], d["patch_id"],
            clean(eg.get("geometry_status")),
            "EXPLICIT" if d["has_explicit_crosswalk"] else "CANDIDATE_OR_MISSING",
            "O evento pode ser ligado a um patch como referencia observacional candidata, sem virar ground truth?",
            "Geometria de evento (mapa/ponto), geometria de patch, crosswalk Sentinel e source trace.",
            "true",
            ALLOWED_REVIEW_DECISIONS,
            FORBIDDEN_REVIEW_DECISIONS))
    if not rows:
        rows.append(build_patch_link_review_row(
            "none_ready", "", "", "INSUFFICIENT_GEOMETRY", "MISSING",
            "Nenhum candidato forte pronto; coletar geometria primeiro.",
            "Geometria de evento e crosswalk.", "true",
            ALLOWED_REVIEW_DECISIONS, FORBIDDEN_REVIEW_DECISIONS))
    assert_no_operational_promotion(rows)
    for r in rows:
        if r["review_status"] != "PENDING_PATCH_LINK_REVIEW":
            raise ValueError("review_status must stay PENDING_PATCH_LINK_REVIEW.")
    write_csv(dataset_path("v2aq_patch_link_review_packet.csv"), REVIEW_PACKET_COLUMNS, rows)
    lines = [
        "# v2aq - pacote de revisao patch-link",
        "",
        "review_status=PENDING_PATCH_LINK_REVIEW. Decisoes proibidas a evitar incluem",
        "ground-truth/label/protocolo B. Nenhum ground truth e criado.",
        "",
    ]
    lines.extend(write_markdown_table(
        ["candidate_id", "patch_id", "event_geometry_status", "crosswalk_status", "review_status"],
        [(r["candidate_id"], r["patch_id"], r["event_geometry_status"], r["crosswalk_status"],
          r["review_status"]) for r in rows]))
    write_markdown(doc_path("v2aq_patch_link_review_packet.md"), lines)
    return rows


def run_patch_reference_candidate_scorer(args=None):
    geom = {r["candidate_id"]: r for r in load_csv(dataset_path("v2aq_event_geometry_candidates.csv"))}
    joins = {r["candidate_id"]: r for r in load_csv(dataset_path("v2aq_crosswalk_geometry_join_candidates.csv"))}
    trace_ids = {r.get("candidate_id") for r in load_v2ao_stack()["trace"]}
    rows = []
    for d in derive_candidates():
        eg = geom.get(d["candidate_id"], {})
        jn = joins.get(d["candidate_id"], {})
        event_geom_score = (40 if clean(eg.get("geometry_status")) in EXPLICIT_GEOMETRY_STATUS
                            else 20 if is_true(eg.get("can_use_for_patch_link_review")) else 0)
        patch_geom_score = 25 if d["has_patch_geometry"] else 0
        crosswalk_score = (25 if d["has_explicit_crosswalk"] else 10 if d["has_crosswalk_candidate"] else 0)
        trace_score = 10 if d["candidate_id"] in trace_ids else 0
        overall = event_geom_score + patch_geom_score + crosswalk_score + trace_score
        band = "HIGH" if overall >= 70 else "MEDIUM" if overall >= 40 else "LOW"
        if event_geom_score == 0:
            blocker = "missing_usable_event_geometry"
        elif crosswalk_score == 0:
            blocker = "missing_sentinel_crosswalk"
        else:
            blocker = "needs_external_validation"
        if is_true(jn.get("patch_level_review_ready")):
            status = "PATCH_REFERENCE_CANDIDATE_READY_FOR_EXTERNAL_VALIDATION"
        elif overall >= 40:
            status = "PARTIAL_PATCH_REFERENCE_NEEDS_GEOMETRY_OR_CROSSWALK"
        else:
            status = "EVENT_REFERENCE_ONLY"
        rows.append({
            "score_id": f"PRS_v2aq_{d['candidate_id']}",
            "candidate_id": d["candidate_id"],
            "region": d["region"],
            "event_geometry_score": str(event_geom_score),
            "patch_geometry_score": str(patch_geom_score),
            "sentinel_crosswalk_score": str(crosswalk_score),
            "source_trace_score": str(trace_score),
            "overall_patch_reference_score": str(overall),
            "patch_reference_band": band,
            "dominant_blocker": blocker,
            "patch_reference_candidate_status": status,
            "can_create_ground_truth": "false",
            "can_create_label": "false",
            "protocol_b_status": "BLOCKED",
        })
    assert_no_operational_promotion(rows)
    assert_no_fake_ground_truth(rows)
    write_csv(dataset_path("v2aq_patch_reference_candidate_scores.csv"), SCORE_COLUMNS, rows)
    lines = [
        "# v2aq - scores de candidato a referencia patch-level",
        "",
        "can_create_ground_truth/label=false; protocol_b=BLOCKED para todos.",
        "",
    ]
    lines.extend(write_markdown_table(
        ["candidate_id", "event_geom", "patch_geom", "crosswalk", "trace", "overall", "band", "status"],
        [(r["candidate_id"], r["event_geometry_score"], r["patch_geometry_score"],
          r["sentinel_crosswalk_score"], r["source_trace_score"], r["overall_patch_reference_score"],
          r["patch_reference_band"], r["patch_reference_candidate_status"]) for r in rows]))
    write_markdown(doc_path("v2aq_patch_reference_candidate_scores.md"), lines)
    return rows


def run_manual_digitization_task_builder(args=None):
    geom = {r["candidate_id"]: r for r in load_csv(dataset_path("v2aq_event_geometry_candidates.csv"))}
    rows = []
    for d in derive_candidates():
        eg = geom.get(d["candidate_id"], {})
        status = clean(eg.get("geometry_status"))
        if status in EXPLICIT_GEOMETRY_STATUS:
            continue
        priority = ("HIGH" if d["level_rank"] >= _LEVEL_RANK["C4_READY_FOR_EXTERNAL_VALIDATION_REVIEW"]
                    else "HIGH" if d["level_rank"] >= _LEVEL_RANK["C3_STRONG_REFERENCE_CANDIDATE"]
                    else "MEDIUM" if status == "TEXTUAL_ANCHOR_ONLY" else "LOW")
        task_type = ("digitize_official_map_sector" if status == "OFFICIAL_MAP_DIGITIZATION_REQUIRED"
                     else "collect_specific_locality_geometry")
        rows.append({
            "task_id": f"DT_v2aq_{d['candidate_id']}",
            "candidate_id": d["candidate_id"],
            "region": d["region"],
            "task_type": task_type,
            "source_to_open": "mapa/laudo oficial (Defesa Civil/DRM/Prefeitura) ja documentado",
            "geometry_to_digitize": "setor/area atingida ou ponto especifico do evento",
            "expected_output": "GeoJSON externo com CRS explicito (revisao manual), nao versionar bruto pesado",
            "priority": priority,
            "do_not_infer": "Nao inferir geometria pelo patch; nao inventar coordenada; nao usar DINO/similaridade visual.",
            "acceptance_criteria": "Geometria de fonte oficial explicita, CRS registrado, licenca verificada, sem overlay.",
        })
    if not rows:
        rows.append({
            "task_id": "DT_v2aq_NONE", "candidate_id": "all_have_explicit_geometry", "region": "",
            "task_type": "none", "source_to_open": "", "geometry_to_digitize": "",
            "expected_output": "", "priority": "LOW",
            "do_not_infer": "Nao inferir geometria; nao inventar coordenada.",
            "acceptance_criteria": "n/a",
        })
    assert_no_operational_promotion(rows)
    for r in rows:
        if "nao inferir" not in r["do_not_infer"].lower() and "nao inventar" not in r["do_not_infer"].lower():
            raise ValueError("manual digitization task must carry do_not_infer guidance.")
    write_csv(dataset_path("v2aq_manual_digitization_tasks.csv"), TASK_COLUMNS, rows)
    lines = [
        "# v2aq - tarefas de digitalizacao manual",
        "",
        "Cada tarefa carrega do_not_infer. Nenhuma geometria e inferida ou inventada.",
        "",
    ]
    lines.extend(write_markdown_table(
        ["candidate_id", "task_type", "priority", "do_not_infer"],
        [(r["candidate_id"], r["task_type"], r["priority"], short_fragment(r["do_not_infer"], 50)) for r in rows]))
    write_markdown(doc_path("v2aq_manual_digitization_tasks.md"), lines)
    return rows


def run_ground_truth_boundary_audit(args=None):
    geom = {r["candidate_id"]: r for r in load_csv(dataset_path("v2aq_event_geometry_candidates.csv"))}
    joins = {r["candidate_id"]: r for r in load_csv(dataset_path("v2aq_crosswalk_geometry_join_candidates.csv"))}
    rows = []
    for d in derive_candidates():
        eg = geom.get(d["candidate_id"], {})
        jn = joins.get(d["candidate_id"], {})
        rows.append({
            "boundary_id": f"GTB_v2aq_{d['candidate_id']}",
            "candidate_id": d["candidate_id"],
            "event_geometry_candidate_status": clean(eg.get("geometry_status")),
            "patch_geometry_status": "AVAILABLE" if d["has_patch_geometry"] else "MISSING",
            "crosswalk_status": ("EXPLICIT" if d["has_explicit_crosswalk"]
                                 else "CANDIDATE" if d["has_crosswalk_candidate"] else "MISSING"),
            "patch_link_review_status": clean(jn.get("join_status")) or "NEEDS_EVENT_GEOMETRY",
            "ground_truth_blocked": "true",
            "label_blocked": "true",
            "protocol_b_blocked": "true",
            "why_not_ground_truth": ("Sem geometria de evento explicita validada + crosswalk Sentinel "
                                     "explicito + revisao externa; referencia de evento nao e ground truth."),
        })
    assert_no_operational_promotion(rows)
    assert_no_label_creation(rows)
    for r in rows:
        if not (r["ground_truth_blocked"] == "true" and r["label_blocked"] == "true"
                and r["protocol_b_blocked"] == "true"):
            raise ValueError("ground truth / label / protocol B must remain blocked.")
    write_csv(dataset_path("v2aq_ground_truth_boundary_audit.csv"), BOUNDARY_COLUMNS, rows)
    lines = [
        "# v2aq - ground truth boundary audit",
        "",
        "ground_truth_blocked/label_blocked/protocol_b_blocked=true para todos os candidatos.",
        "",
    ]
    lines.extend(write_markdown_table(
        ["candidate_id", "event_geometry_candidate_status", "crosswalk_status", "ground_truth_blocked", "protocol_b_blocked"],
        [(r["candidate_id"], r["event_geometry_candidate_status"], r["crosswalk_status"],
          r["ground_truth_blocked"], r["protocol_b_blocked"]) for r in rows]))
    write_markdown(doc_path("v2aq_ground_truth_boundary_audit.md"), lines)
    return rows


# --- guardrail regression --------------------------------------------------
def _regression_artifacts():
    artifacts = []
    if os.path.isdir(DATASET_DIR):
        for n in sorted(os.listdir(DATASET_DIR)):
            if n.endswith(".csv") and n.startswith("v2aq_"):
                artifacts.append((rel_dataset(n), dataset_path(n), "csv"))
    if os.path.isdir(DOCS_DIR):
        for n in sorted(os.listdir(DOCS_DIR)):
            if n.endswith(".md"):
                artifacts.append((rel_doc(n), doc_path(n), "text"))
    if os.path.isdir(GEOJSON_DIR):
        for n in sorted(os.listdir(GEOJSON_DIR)):
            if n.endswith(".geojson"):
                artifacts.append((rel_geojson(n), geojson_path(n), "text"))
    return artifacts


def _scan_csv(path):
    counts = {"forbidden_true_flag": 0, "forbidden_status": 0, "absolute_path": 0,
              "local_only": 0, "forbidden_kv": 0, "unsafe_language": 0}
    for row in load_csv(path):
        for key, value in row.items():
            key_l = clean(key).lower()
            value_s = clean(value)
            value_l = value_s.lower()
            if key_l in FORBIDDEN_TRUE_FIELDS and is_true(value_s):
                counts["forbidden_true_flag"] += 1
            if value_s in FORBIDDEN_STATUS_VALUES:
                counts["forbidden_status"] += 1
            if ABSOLUTE_PATH_RE.search(value_s):
                counts["absolute_path"] += 1
            if LOCAL_ONLY_MARKER in value_l:
                counts["local_only"] += 1
            squashed = re.sub(r"\s*=\s*", "=", value_l)
            for marker in FORBIDDEN_KV_MARKERS:
                if marker in squashed:
                    counts["forbidden_kv"] += 1
            for phrase in UNSAFE_LANGUAGE:
                if phrase in value_l and not _field_allows_unsafe(key, value_s):
                    counts["unsafe_language"] += 1
    return counts


def run_guardrail_regression(args=None):
    check_types = ["forbidden_true_flag", "forbidden_status", "absolute_path",
                   "non_versionable_path_marker", "forbidden_kv", "unsafe_language"]
    rows = []
    total_fail = 0
    for rel, path, kind in _regression_artifacts():
        counts = _scan_csv(path) if kind == "csv" else scan_text_violations(read_text(path))
        for check_type in check_types:
            key = "local_only" if check_type == "non_versionable_path_marker" else check_type
            count = counts.get(key, 0)
            status = "PASS" if count == 0 else "FAIL"
            if status == "FAIL":
                total_fail += 1
            rows.append({
                "regression_id": f"GR_v2aq_{len(rows):05d}",
                "artifact_path": rel,
                "check_type": check_type,
                "violation_count": str(count),
                "status": status,
                "severity": "none" if count == 0 else "blocking",
                "notes": "Fail-closed guardrail regression over v2aq outputs (incl. geojson).",
            })
    write_csv(dataset_path("v2aq_guardrail_regression.csv"), REGRESSION_COLUMNS, rows)
    if total_fail:
        fails = [(r["artifact_path"], r["check_type"]) for r in rows if r["status"] == "FAIL"]
        raise ValueError(f"v2aq guardrail regression failed: {fails[:5]}")
    return rows


# --- next action -----------------------------------------------------------
def run_next_action_ranker(args=None):
    joins = load_csv(dataset_path("v2aq_crosswalk_geometry_join_candidates.csv"))
    geojson = load_csv(dataset_path("v2aq_geojson_candidate_index.csv"))
    has_review_ready = any(r.get("patch_level_review_ready") == "true" for r in joins)
    has_null_geojson = any(r.get("geometry_present") == "false" for r in geojson)
    missing_crosswalk = any(r.get("has_crosswalk_candidate") == "false" for r in joins)
    if has_review_ready:
        top = "EXECUTE_PATCH_LINK_REVIEW_WITH_EXPLICIT_GEOMETRY"
    elif has_null_geojson:
        top = "DIGITIZE_EVENT_GEOMETRY_FOR_TOP_CANDIDATES"
    elif missing_crosswalk:
        top = "RESOLVE_SENTINEL_CROSSWALK_FOR_TOP_CANDIDATES"
    else:
        top = "OPEN_OFFICIAL_MAPS_AND_DIGITIZE"
    options = [
        (top, 100, "v2aq join/geojson readiness", "v2aq_crosswalk_geometry_join_candidates.csv"),
        ("OPEN_OFFICIAL_MAPS_AND_DIGITIZE", 85, "v2aq manual digitization tasks", "v2aq_manual_digitization_tasks.csv"),
        ("VERIFY_GEOMETRY_LICENSE", 70, "geometry provenance", "v2aq_event_geometry_candidates.csv"),
        ("MAINTAIN_EVENT_REFERENCE_ONLY", 55, "event reference baseline", "v2aq_ground_truth_boundary_audit.csv"),
        ("TRAINING_PROTOCOL_B_OVERLAY_LABEL_GT_PROMOTION", 0, "blocked by guardrails", "none"),
    ]
    rows, seen = [], set()
    rank = 1
    for action, score, required, artifact in sorted(options, key=lambda x: (-x[1], x[0])):
        if action in seen:
            continue
        seen.add(action)
        rows.append({
            "rank": str(rank),
            "next_action": action,
            "score": str(score),
            "allowed": "false" if score == 0 else "true",
            "blocked_operational_use": "true",
            "required_input": required,
            "recommended_artifact": artifact,
            "notes": ("No next action may recommend training, Protocol B, automatic overlay, "
                      "labels, operational ground truth, automatic geometry/date inference, or promotion."),
        })
        rank += 1
    write_csv(dataset_path("v2aq_next_actions_registry.csv"), NEXT_COLUMNS, rows)
    return rows


# --- completion report -----------------------------------------------------
def run_completion_report(args=None):
    priority = load_csv(dataset_path("v2aq_priority_candidates.csv"))
    anchors = load_csv(dataset_path("v2aq_spatial_anchor_strength.csv"))
    event_geom = load_csv(dataset_path("v2aq_event_geometry_candidates.csv"))
    geojson = load_csv(dataset_path("v2aq_geojson_candidate_index.csv"))
    match = load_csv(dataset_path("v2aq_patch_geometry_match_candidates.csv"))
    joins = load_csv(dataset_path("v2aq_crosswalk_geometry_join_candidates.csv"))
    packets = load_csv(dataset_path("v2aq_patch_link_review_packet.csv"))
    scores = load_csv(dataset_path("v2aq_patch_reference_candidate_scores.csv"))
    tasks = load_csv(dataset_path("v2aq_manual_digitization_tasks.csv"))
    boundary = load_csv(dataset_path("v2aq_ground_truth_boundary_audit.csv"))
    regression = load_csv(dataset_path("v2aq_guardrail_regression.csv"))
    next_rows = load_csv(dataset_path("v2aq_next_actions_registry.csv"))
    geom_real = sum(1 for r in geojson if r.get("geometry_present") == "true")
    geom_null = sum(1 for r in geojson if r.get("geometry_present") == "false")
    review_ready = sum(1 for r in joins if r.get("patch_level_review_ready") == "true")
    gt_blocked = sum(1 for r in boundary if r.get("ground_truth_blocked") == "true")
    regression_fail = sum(1 for r in regression if r.get("status") == "FAIL")
    rows = [
        {"completion_id": "CR_v2aq_000", "metric": "candidates_loaded", "value": str(len(priority)),
         "status": "RECORDED", "notes": "From v2ap/v2ao."},
        {"completion_id": "CR_v2aq_001", "metric": "anchor_strength_rows", "value": str(len(anchors)),
         "status": "RECORDED", "notes": "Anchor specificity classified."},
        {"completion_id": "CR_v2aq_002", "metric": "event_geometry_candidates", "value": str(len(event_geom)),
         "status": "RECORDED", "notes": "No geometry invented."},
        {"completion_id": "CR_v2aq_003", "metric": "geojson_created", "value": str(len(geojson)),
         "status": "RECORDED", "notes": f"real geometry={geom_real}, null={geom_null}."},
        {"completion_id": "CR_v2aq_004", "metric": "geojson_geometry_real", "value": str(geom_real),
         "status": "RECORDED", "notes": "Only from explicit source."},
        {"completion_id": "CR_v2aq_005", "metric": "geojson_geometry_null", "value": str(geom_null),
         "status": "RECORDED", "notes": "geometry: null; manual digitization."},
        {"completion_id": "CR_v2aq_006", "metric": "patch_geometry_match_candidates", "value": str(len(match)),
         "status": "RECORDED", "notes": "overlay_executed=false."},
        {"completion_id": "CR_v2aq_007", "metric": "crosswalk_geometry_joins", "value": str(len(joins)),
         "status": "RECORDED", "notes": f"review_ready={review_ready}."},
        {"completion_id": "CR_v2aq_008", "metric": "patch_link_review_packets", "value": str(len(packets)),
         "status": "RECORDED", "notes": "PENDING_PATCH_LINK_REVIEW."},
        {"completion_id": "CR_v2aq_009", "metric": "patch_reference_scores", "value": str(len(scores)),
         "status": "RECORDED", "notes": "ground truth/label blocked."},
        {"completion_id": "CR_v2aq_010", "metric": "manual_digitization_tasks", "value": str(len(tasks)),
         "status": "RECORDED", "notes": "All carry do_not_infer."},
        {"completion_id": "CR_v2aq_011", "metric": "ground_truth_blocked_all", "value": str(gt_blocked),
         "status": "PASS" if boundary and gt_blocked == len(boundary) else "FAIL", "notes": "All blocked."},
        {"completion_id": "CR_v2aq_012", "metric": "guardrail_regression_failures", "value": str(regression_fail),
         "status": "PASS" if regression_fail == 0 else "FAIL", "notes": "Fail-closed."},
        {"completion_id": "CR_v2aq_013", "metric": "next_action_rank_1",
         "value": next_rows[0]["next_action"] if next_rows else "", "status": "SAFE_NEXT_ACTION",
         "notes": "Geometry digitization / patch-link review path."},
        {"completion_id": "CR_v2aq_014", "metric": "final_decision",
         "value": "event_geometry_and_patch_link_readiness_built_no_operational_ground_truth",
         "status": "NO_OPERATIONAL_GROUND_TRUTH", "notes": "patch_truth_allowed=false; protocol_b blocked."},
    ]
    write_csv(dataset_path("v2aq_completion_report.csv"), COMPLETION_COLUMNS, rows)
    lines = [
        "# v2aq completion report",
        "",
        f"Candidates loaded: {len(priority)}.",
        f"Anchor strength rows: {len(anchors)}.",
        f"Event geometry candidates: {len(event_geom)}.",
        f"GeoJSON created: {len(geojson)} (real geometry: {geom_real}, null: {geom_null}).",
        f"Patch geometry match candidates: {len(match)} (overlay_executed=false).",
        f"Crosswalk geometry joins: {len(joins)} (review_ready: {review_ready}).",
        f"Patch-link review packets: {len(packets)}.",
        f"Patch reference scores: {len(scores)}.",
        f"Manual digitization tasks: {len(tasks)}.",
        f"Ground-truth blocked: {gt_blocked}/{len(boundary)}.",
        f"Guardrail regression failures: {regression_fail}.",
        f"Next action rank 1: {next_rows[0]['next_action'] if next_rows else ''}.",
        "Final decision: event geometry / patch-link readiness built; no operational ground truth.",
    ]
    write_markdown(doc_path("v2aq_completion_report.md"), lines)
    return rows


# --- orchestrator ----------------------------------------------------------
_ORCHESTRATION = [
    ("priority_candidate", "run_priority_candidate_builder",
     ["v2aq_priority_candidates.csv"], ["v2aq_priority_candidates.md"]),
    ("spatial_anchor_strength", "run_spatial_anchor_strength_builder",
     ["v2aq_spatial_anchor_strength.csv"], ["v2aq_spatial_anchor_strength.md"]),
    ("event_geometry_candidate", "run_event_geometry_candidate_builder",
     ["v2aq_event_geometry_candidates.csv"], ["v2aq_event_geometry_candidates.md"]),
    ("geojson_candidate_exporter", "run_geojson_candidate_exporter",
     ["v2aq_geojson_candidate_index.csv"], ["v2aq_geojson_candidate_index.md"]),
    ("patch_geometry_match", "run_patch_geometry_match_builder",
     ["v2aq_patch_geometry_match_candidates.csv"], ["v2aq_patch_geometry_match_candidates.md"]),
    ("crosswalk_geometry_join", "run_crosswalk_geometry_join_builder",
     ["v2aq_crosswalk_geometry_join_candidates.csv"], ["v2aq_crosswalk_geometry_join_candidates.md"]),
    ("patch_link_review_packet", "run_patch_link_review_packet_builder",
     ["v2aq_patch_link_review_packet.csv"], ["v2aq_patch_link_review_packet.md"]),
    ("patch_reference_candidate_scorer", "run_patch_reference_candidate_scorer",
     ["v2aq_patch_reference_candidate_scores.csv"], ["v2aq_patch_reference_candidate_scores.md"]),
    ("manual_digitization_tasks", "run_manual_digitization_task_builder",
     ["v2aq_manual_digitization_tasks.csv"], ["v2aq_manual_digitization_tasks.md"]),
    ("ground_truth_boundary_audit", "run_ground_truth_boundary_audit",
     ["v2aq_ground_truth_boundary_audit.csv"], ["v2aq_ground_truth_boundary_audit.md"]),
    ("guardrail_regression", "run_guardrail_regression",
     ["v2aq_guardrail_regression.csv"], []),
    ("next_action_ranker", "run_next_action_ranker",
     ["v2aq_next_actions_registry.csv"], []),
    ("completion_report", "run_completion_report",
     ["v2aq_completion_report.csv"], ["v2aq_completion_report.md"]),
]


def _manifest_row(order, name, status, ds_out, doc_out, notes):
    outputs = [rel_dataset(o) for o in ds_out] + [rel_doc(o) for o in doc_out]
    hashes = [sha256_file(dataset_path(o))[:16] for o in ds_out]
    hashes += [sha256_file(doc_path(o))[:16] for o in doc_out]
    return {
        "step_order": str(order), "step_name": name, "status": status,
        "outputs": "|".join(outputs), "output_hashes": "|".join(h for h in hashes if h),
        "notes": notes,
    }


def _write_manifest_md(rows):
    lines = ["# v2aq - orchestrator run manifest", "",
             f"Etapas executadas: {len(rows)}. Nenhuma operacao git foi executada.", ""]
    lines.extend(write_markdown_table(
        ["ordem", "etapa", "status", "outputs"],
        [(r["step_order"], r["step_name"], r["status"], r["outputs"]) for r in rows]))
    write_markdown(doc_path("v2aq_orchestrator_run_manifest.md"), lines)


def run_master_orchestrator(args=None):
    rows = []
    for order, (name, func_name, ds_out, doc_out) in enumerate(_ORCHESTRATION, 1):
        func = globals()[func_name]
        try:
            func(args)
        except Exception as exc:
            rows.append(_manifest_row(order, name, "FAIL", ds_out, doc_out,
                                      f"{type(exc).__name__}: {exc}"))
            write_csv(dataset_path("v2aq_orchestrator_run_manifest.csv"), MANIFEST_COLUMNS, rows)
            _write_manifest_md(rows)
            raise
        rows.append(_manifest_row(order, name, "OK", ds_out, doc_out, "Completed."))
    write_csv(dataset_path("v2aq_orchestrator_run_manifest.csv"), MANIFEST_COLUMNS, rows)
    _write_manifest_md(rows)
    return rows


def run_all(args=None):
    return run_master_orchestrator(args)
