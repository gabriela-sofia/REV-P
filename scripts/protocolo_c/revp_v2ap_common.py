#!/usr/bin/env python3
"""v2ap Patch-Level Geometry and Sentinel Crosswalk Acquisition Sprint.

Attacks the two blockers left after v2ao: patch-compatible spatial geometry and an
explicit Sentinel crosswalk. Builds, validates and audits a geometric/temporal
readiness layer for the strongest candidates -- WITHOUT executing operational
overlay, creating labels/masks/predictions, promoting ground truth, inferring
Sentinel dates by visual/DINO similarity, or inventing coordinates/geometry.

Only ``v2ap_*`` artifacts are written. Prior artifacts are read-only. Sentinel dates
are taken only from explicit filename/manifest/metadata fields. patch_truth_allowed
stays false in this stage.
"""

import argparse
import csv
import datetime as _dt
import hashlib
import json
import os
import re

PROTOCOL_VERSION = "v2ap"
REPO_ROOT = os.environ.get("REPO_ROOT", ".")
DATASET_ROOT = os.environ.get("DATASET_ROOT", os.path.join(REPO_ROOT, "datasets"))
DATASET_DIR = os.environ.get("DATASET_DIR", os.path.join(DATASET_ROOT, "protocolo_c"))
DOCS_DIR = os.environ.get(
    "DOCS_DIR", "docs/protocolo_c/v2ap_patch_geometry_sentinel_crosswalk")
CONFIG_DIR = os.environ.get("CONFIG_DIR", "configs/protocolo_c")

V2AO_INPUTS = {
    "form": "v2ao_human_review_form_filled.csv",
    "matrix": "v2ao_reference_candidate_matrix.csv",
    "levels": "v2ao_reference_level_classification.csv",
    "boundary": "v2ao_patch_truth_boundary_audit.csv",
    "trace": "v2ao_human_review_trace.csv",
    "blocker": "v2ao_ground_truth_promotion_blocker_audit.csv",
    "ranking": "v2ao_final_candidate_ranking.csv",
}
V2AN_OPTIONAL = {
    "spatial": "v2an_spatial_anchor_registry.csv",
    "crosswalk": "v2an_temporal_sentinel_crosswalk_audit.csv",
    "patch": "v2an_patch_link_readiness_audit.csv",
}
OBSERVED_REGISTRY = "observed_event_reference_candidate_registry.csv"

REGION_BY_PREFIX = {"REC": "Recife", "PET": "Petropolis", "CTB": "Curitiba"}

# --- repo scan configuration ----------------------------------------------
SCAN_SUBDIRS = ("datasets", "outputs", "data", "docs", "scripts")
SCAN_EXCLUDE_DIRS = {".git", "local_runs", "__pycache__", "node_modules", ".venv",
                     "venv", ".pytest_cache", "embeddings"}
SCAN_TEXT_EXTS = {".csv", ".json", ".yaml", ".yml", ".md", ".txt"}
HEAVY_EXTS = {".tif", ".tiff", ".geotiff", ".npz", ".npy", ".zip", ".tar", ".gz"}
SENTINEL_NAME_RE = re.compile(r"(sentinel|_s2_|_s1_|scene|tile|asset|crosswalk|manifest)", re.I)
PATCH_NAME_RE = re.compile(r"(patch|geometry|bbox|bounds|registry)", re.I)
DINO_RE = re.compile(r"dino", re.I)
SENTINEL_DATE_COLS = ["sentinel_scene_date", "scene_date", "asset_date", "anchor_date",
                      "acquisition_date", "s2_date", "date"]
REGION_COLS = ["region", "municipality", "city", "uf"]
PATCH_ID_COLS = ["patch_id", "reference_patch_id", "scene_id_sanitized", "asset_id",
                 "tile", "scene_id"]
GEOMETRY_COLS = ["geometry", "geom", "wkt", "geojson", "latitude", "longitude",
                 "anchor_latitude", "anchor_longitude", "bbox", "bounds", "centroid", "crs"]
DATE_RE = re.compile(r"(20\d{2})[-_/]?(0[1-9]|1[0-2])[-_/]?(0[1-9]|[12]\d|3[01])")

# --- guardrail vocabulary --------------------------------------------------
FORBIDDEN_TRUE_FIELDS = {
    "ground_truth_created", "ground_reference_created", "label_created",
    "operational_ground_truth", "training_ready", "training_use_allowed",
    "overlay_ready", "overlay_executed", "prediction_ready", "promotion_allowed",
    "can_create_ground_truth", "can_create_label", "raw_data_versioned",
    "raw_data_downloaded", "sentinel_date_inferred", "crosswalk_inferred",
    "protocol_b_open", "protocol_b_reopened", "operational_use_allowed",
    "patch_truth_allowed", "flood_mask_created", "coordinate_invented",
    "geometry_invented",
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
    "overlay=true", "prediction=true", "protocol_b_open=true",
    "protocol_b_reopen=true", "sentinel_date_inferred=true", "crosswalk_inferred=true",
    "operational_validation=true", "promotion_allowed=true",
    "can_create_ground_truth=true", "can_create_label=true", "raw_data_versioned=true",
    "patch_truth_allowed=true",
]
UNSAFE_LANGUAGE = [
    "ground truth validado", "classe positiva", "classe negativa", "label operacional",
    "deteccao de enchente", "deteccao de inundacao", "predicao de inundacao",
    "mascara de inundacao", "modelo preditivo", "validacao operacional",
    "treinamento supervisionado pronto", "similaridade visual confirma",
    "dino confirma evento", "overlay operacional executado",
]
SAFE_UNSAFE_FIELDS = {
    "forbidden_use", "do_not_infer", "blocking_reason", "why_still_blocked",
    "why_patch_truth_blocked", "dominant_blocker", "notes", "selection_reason",
    "recommended_source", "query_terms", "expected_artifact", "missing_geometry_type",
    "missing_crosswalk_component", "best_existing_anchor", "recommended_manifest_to_check",
    "recommended_script_or_registry", "crosswalk_evidence_type", "geometry_source",
    "geometry_status", "geometry_readiness_status", "patch_event_link_status",
    "patch_reference_status", "crosswalk_candidate_status", "patch_truth_allowed",
    "safe_use", "safe_to_use_as_crosswalk_evidence", "safe_for_patch_link",
    "required_next_action", "required_next_step", "recommended_next_step",
    "date_detection_method", "asset_type", "window_policy", "readiness_band",
    "event_reference_level", "event_reference_status", "patch_level_reference_candidate",
    "human_review_decision", "confidence_band", "reference_level", "strongest_anchor_type",
    "event_geometry_status", "patch_geometry_status", "sentinel_crosswalk_status",
    "path", "source_file", "source_manifest", "region", "candidate_asset_or_patch",
    "reason", "purpose", "status",
}
SAFE_CONTEXT_MARKERS = [
    "nao pode dizer", "nao usar", "nao afirmar", "nao ha", "nao deve", "nao temos",
    "nao realiza", "nao detecta", "nao cria", "nao existe", "nao produz", "nao inferir",
    "nao significa", "nao implica", "nao foi", "nao e ", "nao ", "do_not", "do not",
    "proibid", "forbidden", "limitation", "limitacao", "blocker", "blocked", "bloque",
    "does not", "not ", "no ", "sem ", "evitar", "ausencia", "pendente", "candidato",
    "review-only", "needs_", "missing", "not_established", "not_created", "insufficient",
    "anchor_only", "do_not_infer", "external_validation", "filename_or_manifest",
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


def rel_dataset(name):
    return f"datasets/protocolo_c/{name}"


def rel_doc(name):
    return f"docs/protocolo_c/v2ap_patch_geometry_sentinel_crosswalk/{name}"


def repo_relative_path(path):
    raw = str(path)
    if ABSOLUTE_PATH_RE.search(raw):
        rel = os.path.relpath(raw, REPO_ROOT)
        raw = rel
    raw = raw.replace("\\", "/")
    if ABSOLUTE_PATH_RE.search(raw):
        raise ValueError(f"Refusing absolute path: {path}")
    return raw


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


def normalize_date(value):
    raw = clean(value)
    if not raw:
        return ""
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%Y%m%d"):
        try:
            return _dt.datetime.strptime(raw, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    m = DATE_RE.search(raw)
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else ""


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


def normalize_patch_id(value):
    return re.sub(r"\s+", "", clean(value)).upper()


def normalize_candidate_id(value):
    return clean(value).upper()


def extract_date_from_text_safe(text):
    """Explicit date from filename/manifest text only -- never visual inference."""
    m = DATE_RE.search(clean(text))
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else ""


def date_window_overlap(start1, end1, start2, end2):
    try:
        a1 = _dt.datetime.strptime(normalize_date(start1), "%Y-%m-%d")
        a2 = _dt.datetime.strptime(normalize_date(end1) or normalize_date(start1), "%Y-%m-%d")
        b1 = _dt.datetime.strptime(normalize_date(start2), "%Y-%m-%d")
        b2 = _dt.datetime.strptime(normalize_date(end2) or normalize_date(start2), "%Y-%m-%d")
    except ValueError:
        return False
    return a1 <= b2 and b1 <= a2


# --- io helpers ------------------------------------------------------------
def load_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def assert_output_is_v2ap(path):
    base = os.path.basename(str(path))
    if LOCAL_ONLY_MARKER in str(path).lower():
        raise ValueError(f"Refusing local_only output path: {path}")
    if not base.startswith("v2ap_"):
        raise ValueError(f"Refusing to write non-v2ap output: {path}")
    return True


def write_csv(path, columns, rows):
    assert_output_is_v2ap(path)
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
    assert_output_is_v2ap(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_json(path, payload):
    assert_output_is_v2ap(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


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


def load_v2ao_candidate_stack():
    missing = [name for name in V2AO_INPUTS.values()
               if not os.path.exists(dataset_path(name))]
    if missing:
        raise FileNotFoundError("v2ap requires v2ao outputs; missing: " + ",".join(missing))
    stack = {key: load_csv(dataset_path(name)) for key, name in V2AO_INPUTS.items()}
    for key, name in V2AN_OPTIONAL.items():
        stack[key] = load_csv(dataset_path(name))
    stack["observed"] = load_csv(root_dataset_path(OBSERVED_REGISTRY))
    assert_min_schema(stack["levels"], ["candidate_id", "reference_level"], V2AO_INPUTS["levels"])
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
            if key_l in {"can_create_ground_truth", "ground_truth_created"} and is_true(value_s):
                raise ValueError(f"{key_l}=true forbidden at row {idx}.")
    return True


def assert_no_fake_sentinel_crosswalk(rows):
    for idx, row in enumerate(rows):
        for key, value in row.items():
            key_l = clean(key).lower()
            if key_l in {"sentinel_date_inferred", "crosswalk_inferred"} and is_true(value):
                raise ValueError(f"{key_l}=true forbidden at row {idx}.")
            if key_l == "can_be_used_as_explicit_crosswalk" and is_true(value):
                evidence = clean(row.get("crosswalk_evidence_type")).lower()
                if evidence in {"visual_similarity", "dino_similarity", "manual_guess",
                                "temporal_guess", ""}:
                    raise ValueError(
                        f"explicit crosswalk without manifest/filename/metadata evidence at row {idx}.")
    return True


def assert_no_fake_geometry(rows):
    for idx, row in enumerate(rows):
        for key, value in row.items():
            key_l = clean(key).lower()
            if key_l in {"coordinate_invented", "geometry_invented"} and is_true(value):
                raise ValueError(f"invented geometry/coordinate at row {idx}.")
            if key_l in {"has_event_coordinates", "has_event_geometry"} and is_true(value):
                src = clean(row.get("geometry_source") or row.get("blocking_reason")).lower()
                if "invent" in src or "guess" in src:
                    raise ValueError(f"geometry from invention/guess at row {idx}.")
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


# --- repo scanning ---------------------------------------------------------
def _iter_repo_files():
    for sub in SCAN_SUBDIRS:
        base = os.path.join(REPO_ROOT, sub)
        if not os.path.isdir(base):
            continue
        for root, dirs, files in os.walk(base):
            dirs[:] = sorted(d for d in dirs if d not in SCAN_EXCLUDE_DIRS)
            for fname in sorted(files):
                ext = os.path.splitext(fname)[1].lower()
                if ext in HEAVY_EXTS:
                    continue
                # Never inventory this stage's own outputs (avoids self-reference and
                # keeps the scan deterministic across reruns).
                if fname.startswith("v2ap_"):
                    continue
                yield os.path.join(root, fname), fname, ext


def _csv_header(path):
    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            return [clean(c).lower() for c in next(reader, [])]
    except Exception:
        return []


def _first_present(header, candidates):
    for c in candidates:
        if c in header:
            return c
    return ""


def scan_repo_for_sentinel_assets(row_cap=120):
    """Inventory Sentinel/asset references by path/manifest only (no raw data opened)."""
    assets = []
    seen = set()
    for full, fname, ext in _iter_repo_files():
        if not SENTINEL_NAME_RE.search(fname):
            continue
        rel = repo_relative_path(os.path.relpath(full, REPO_ROOT))
        is_dino = bool(DINO_RE.search(rel))
        asset_type = ("sentinel_asset" if re.search(r"sentinel|_s2_|_s1_|scene|tile", fname, re.I)
                      else "crosswalk_registry" if "crosswalk" in fname.lower()
                      else "manifest" if "manifest" in fname.lower() else "asset_reference")
        emitted_rows = False
        if ext == ".csv":
            header = _csv_header(full)
            date_col = _first_present(header, SENTINEL_DATE_COLS)
            region_col = _first_present(header, REGION_COLS)
            patch_col = _first_present(header, PATCH_ID_COLS)
            if date_col:
                for r in load_csv(full)[:row_cap]:
                    asset_date = normalize_date(r.get(date_col))
                    if not asset_date:
                        continue
                    region = normalize_region(r.get(region_col)) if region_col else ""
                    patch_id = normalize_patch_id(r.get(patch_col)) if patch_col else ""
                    key = (rel, region, asset_date, patch_id)
                    if key in seen:
                        continue
                    seen.add(key)
                    assets.append(_asset_record(len(assets), rel, region, patch_id, asset_date,
                                                "manifest_field_explicit", asset_type, rel, is_dino))
                    emitted_rows = True
        if not emitted_rows:
            fdate = extract_date_from_text_safe(fname)
            region = normalize_region(fname)
            key = (rel, region, fdate, "")
            if key in seen:
                continue
            seen.add(key)
            method = "filename_date_explicit" if fdate else "no_explicit_date"
            assets.append(_asset_record(len(assets), rel, region, "", fdate, method,
                                        asset_type, rel, is_dino))
    return assets


def _asset_record(idx, rel, region, patch_id, date, method, asset_type, manifest, is_dino):
    safe = (method in {"manifest_field_explicit", "filename_date_explicit", "metadata_field_explicit"}
            and bool(date) and not is_dino)
    return {
        "asset_id": f"SA_v2ap_{idx:05d}",
        "path": rel,
        "region_detected": region or "Unspecified",
        "patch_id_detected": patch_id,
        "date_detected": date,
        "date_detection_method": method,
        "asset_type": asset_type,
        "source_manifest": manifest,
        "sha256_if_small": "",
        "safe_to_use_as_crosswalk_evidence": normalize_bool(safe),
        "notes": ("DINO source: not usable as crosswalk/date evidence." if is_dino
                  else "Explicit date from filename/manifest only; no visual inference."),
    }


def scan_repo_for_patch_registries():
    rows = []
    for full, fname, ext in _iter_repo_files():
        if ext != ".csv" or not PATCH_NAME_RE.search(fname):
            continue
        header = _csv_header(full)
        patch_col = _first_present(header, PATCH_ID_COLS)
        if not patch_col:
            continue
        rel = repo_relative_path(os.path.relpath(full, REPO_ROOT))
        has_geom = any(c in header for c in ("geometry", "geom", "wkt", "geojson"))
        has_coord = any(c in header for c in ("latitude", "longitude", "anchor_latitude", "anchor_longitude"))
        has_bbox = any(c in header for c in ("bbox", "bounds"))
        has_centroid = "centroid" in header
        has_crs = "crs" in header
        region_col = _first_present(header, REGION_COLS)
        geometry_present = has_geom or has_coord or has_bbox or has_centroid
        rows.append({
            "registry_item_id": f"PR_v2ap_{len(rows):04d}",
            "source_file": rel,
            "patch_id": patch_col,
            "region": region_col or "multiple_or_none",
            "has_geometry": normalize_bool(has_geom or has_coord),
            "has_bbox": normalize_bool(has_bbox),
            "has_centroid": normalize_bool(has_centroid),
            "has_crs": normalize_bool(has_crs),
            "geometry_source": ("registry_column_present" if geometry_present
                                else "no_geometry_column"),
            "geometry_status": ("HAS_GEOMETRY_COLUMNS" if geometry_present
                                else "NO_GEOMETRY_COLUMNS_BLOCKER"),
            "safe_for_patch_link": normalize_bool(geometry_present and bool(region_col)),
            "notes": "Inventory by header columns only; no geometry invented, no raw data opened.",
        })
    return rows


# --- record constructors ---------------------------------------------------
def build_geometry_readiness_row(candidate_id, region, anchor_count, strongest_anchor,
                                 has_event_geometry, has_patch_geometry, has_patch_bbox,
                                 status, blocking_reason):
    return {
        "geometry_readiness_id": f"GR_v2ap_{candidate_id}",
        "candidate_id": candidate_id,
        "region": region,
        "spatial_anchor_count": str(anchor_count),
        "strongest_anchor_type": strongest_anchor,
        "has_event_geometry": normalize_bool(has_event_geometry),
        "has_event_coordinates": "false",
        "has_patch_geometry": normalize_bool(has_patch_geometry),
        "has_patch_bbox": normalize_bool(has_patch_bbox),
        "manual_geometry_collection_needed": normalize_bool(not has_event_geometry),
        "geometry_readiness_status": status,
        "blocking_reason": blocking_reason,
    }


def build_crosswalk_candidate_row(idx, candidate_id, region, asset, event_start, event_end,
                                  within_window, region_match, patch_match, evidence_type,
                                  status, explicit, blocking_reason):
    return {
        "crosswalk_candidate_id": f"CC_v2ap_{idx:05d}",
        "candidate_id": candidate_id,
        "region": region,
        "asset_id": asset.get("asset_id", ""),
        "patch_id_detected": asset.get("patch_id_detected", ""),
        "asset_date": asset.get("date_detected", ""),
        "event_date_start": event_start,
        "event_date_end": event_end,
        "within_temporal_window": normalize_bool(within_window),
        "region_match": normalize_bool(region_match),
        "patch_id_match": normalize_bool(patch_match),
        "crosswalk_evidence_type": evidence_type,
        "crosswalk_candidate_status": status,
        "can_be_used_as_explicit_crosswalk": normalize_bool(explicit),
        "blocking_reason": blocking_reason,
    }


def build_patch_reference_readiness_row(candidate_id, region, level, geometry_score,
                                        crosswalk_score, patch_registry_score, band,
                                        dominant_blocker, status):
    overall = geometry_score + crosswalk_score + patch_registry_score
    return {
        "score_id": f"PRS_v2ap_{candidate_id}",
        "candidate_id": candidate_id,
        "region": region,
        "event_reference_level": level,
        "geometry_score": str(geometry_score),
        "crosswalk_score": str(crosswalk_score),
        "patch_registry_score": str(patch_registry_score),
        "overall_patch_reference_score": str(overall),
        "readiness_band": band,
        "dominant_blocker": dominant_blocker,
        "patch_reference_status": status,
        "can_create_ground_truth": "false",
        "can_create_label": "false",
        "protocol_b_status": "BLOCKED",
    }


# --- derivation ------------------------------------------------------------
_LEVEL_RANK = {
    "C0_REJECTED_OR_INSUFFICIENT": 0, "C1_CONTEXTUAL_OBSERVED_EVENT": 1,
    "C2_DOCUMENTED_OBSERVED_EVENT": 2, "C3_STRONG_REFERENCE_CANDIDATE": 3,
    "C4_READY_FOR_EXTERNAL_VALIDATION_REVIEW": 4,
}


def derive_candidates(stack):
    levels = {r["candidate_id"]: clean(r.get("reference_level")) for r in stack["levels"]}
    matrix = {r["candidate_id"]: r for r in stack["matrix"]}
    ranking = {r["candidate_id"]: r for r in stack["ranking"]}
    form = {r["candidate_id"]: r for r in stack["form"]}
    observed = {clean(r.get("observed_event_id")): r for r in stack["observed"]}
    anchor_count, anchor_types = {}, {}
    for r in stack.get("spatial") or []:
        cid = r["candidate_id"]
        anchor_count[cid] = anchor_count.get(cid, 0) + 1
        anchor_types.setdefault(cid, []).append(clean(r.get("anchor_type")))
    out = []
    for cid, level in levels.items():
        mx = matrix.get(cid, {})
        obs = observed.get(cid, {})
        frm = form.get(cid, {})
        region = normalize_region(mx.get("region"), cid)
        out.append({
            "candidate_id": cid,
            "region": region,
            "event_name": clean(mx.get("event_name")),
            "reference_level": level,
            "level_rank": _LEVEL_RANK.get(level, 0),
            "human_review_decision": clean(mx.get("human_review_decision")),
            "confidence_band": clean(mx.get("confidence_band")),
            "spatial_strength": clean(mx.get("spatial_strength")),
            "source_strength": clean(mx.get("source_strength")),
            "readiness_score": int(clean(ranking.get(cid, {}).get("readiness_score")) or 0),
            "dominant_blocker": clean(mx.get("remaining_blockers")),
            "event_date_start": normalize_date(obs.get("date_start")),
            "event_date_end": normalize_date(obs.get("date_end")) or normalize_date(obs.get("date_start")),
            "anchor_count": anchor_count.get(cid, 0),
            "anchor_types": anchor_types.get(cid, []),
            "geometry_or_map_available": is_true(frm.get("geometry_or_map_available")),
            "spatial_anchor_confirmed": is_true(frm.get("spatial_anchor_confirmed")),
        })
    out.sort(key=lambda d: (-d["level_rank"], -d["readiness_score"], d["candidate_id"]))
    return out


def _region_has_patch_geometry(patch_rows, region):
    for r in patch_rows:
        if is_true(r.get("has_geometry")) or is_true(r.get("has_bbox")):
            return True
    return False


# --- column schemas --------------------------------------------------------
SELECTION_COLUMNS = [
    "selection_id", "candidate_id", "region", "reference_level", "human_review_decision",
    "confidence_band", "readiness_score", "included_in_v2ap", "selection_reason",
    "dominant_blocker", "forbidden_use",
]
ASSET_COLUMNS = [
    "asset_id", "path", "region_detected", "patch_id_detected", "date_detected",
    "date_detection_method", "asset_type", "source_manifest", "sha256_if_small",
    "safe_to_use_as_crosswalk_evidence", "notes",
]
PATCH_REGISTRY_COLUMNS = [
    "registry_item_id", "source_file", "patch_id", "region", "has_geometry", "has_bbox",
    "has_centroid", "has_crs", "geometry_source", "geometry_status", "safe_for_patch_link",
    "notes",
]
GEOMETRY_COLUMNS_OUT = [
    "geometry_readiness_id", "candidate_id", "region", "spatial_anchor_count",
    "strongest_anchor_type", "has_event_geometry", "has_event_coordinates",
    "has_patch_geometry", "has_patch_bbox", "manual_geometry_collection_needed",
    "geometry_readiness_status", "blocking_reason",
]
TEMPORAL_COLUMNS = [
    "temporal_window_id", "candidate_id", "region", "event_date_start", "event_date_end",
    "event_window_days", "acceptable_sentinel_window_start", "acceptable_sentinel_window_end",
    "window_policy", "notes",
]
CROSSWALK_COLUMNS = [
    "crosswalk_candidate_id", "candidate_id", "region", "asset_id", "patch_id_detected",
    "asset_date", "event_date_start", "event_date_end", "within_temporal_window",
    "region_match", "patch_id_match", "crosswalk_evidence_type", "crosswalk_candidate_status",
    "can_be_used_as_explicit_crosswalk", "blocking_reason",
]
LINK_COLUMNS = [
    "link_readiness_id", "candidate_id", "region", "has_event_anchor", "has_event_geometry",
    "has_patch_geometry", "has_sentinel_crosswalk_candidate", "has_explicit_sentinel_crosswalk",
    "patch_event_link_status", "patch_level_reference_candidate", "patch_truth_allowed",
    "required_next_action",
]
GEOM_PACKET_COLUMNS = [
    "collection_id", "candidate_id", "region", "missing_geometry_type", "best_existing_anchor",
    "recommended_source", "query_terms", "expected_artifact", "collection_priority", "do_not_infer",
]
CROSS_PACKET_COLUMNS = [
    "collection_id", "candidate_id", "region", "missing_crosswalk_component",
    "candidate_asset_or_patch", "recommended_manifest_to_check", "recommended_script_or_registry",
    "collection_priority", "do_not_infer",
]
SCORE_COLUMNS = [
    "score_id", "candidate_id", "region", "event_reference_level", "geometry_score",
    "crosswalk_score", "patch_registry_score", "overall_patch_reference_score", "readiness_band",
    "dominant_blocker", "patch_reference_status", "can_create_ground_truth", "can_create_label",
    "protocol_b_status",
]
BOUNDARY_COLUMNS = [
    "boundary_update_id", "candidate_id", "event_reference_status", "patch_reference_status",
    "event_geometry_status", "patch_geometry_status", "sentinel_crosswalk_status",
    "patch_truth_allowed", "why_still_blocked", "safe_use", "forbidden_use",
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
def run_candidate_selection_builder(args=None):
    stack = load_v2ao_candidate_stack()
    rows = []
    for d in derive_candidates(stack):
        high = d["level_rank"] >= _LEVEL_RANK["C3_STRONG_REFERENCE_CANDIDATE"]
        reason = ("HIGH_PRIORITY_GEOMETRY_AND_CROSSWALK" if high
                  else "LOW_PRIORITY_GEOMETRY_COLLECTION")
        rows.append({
            "selection_id": f"SEL_v2ap_{d['candidate_id']}",
            "candidate_id": d["candidate_id"],
            "region": d["region"],
            "reference_level": d["reference_level"],
            "human_review_decision": d["human_review_decision"],
            "confidence_band": d["confidence_band"],
            "readiness_score": str(d["readiness_score"]),
            "included_in_v2ap": "true",
            "selection_reason": reason,
            "dominant_blocker": short_fragment(d["dominant_blocker"], 120),
            "forbidden_use": FORBIDDEN_USE,
        })
    assert_no_operational_promotion(rows)
    write_csv(dataset_path("v2ap_candidate_selection.csv"), SELECTION_COLUMNS, rows)
    lines = [
        "# v2ap - selecao de candidatos para aprofundamento geometrico/temporal",
        "",
        "C4/C3 sao alta prioridade; C2/C1 entram como LOW_PRIORITY_GEOMETRY_COLLECTION.",
        "Nenhum candidato vira ground truth nesta etapa.",
        "",
    ]
    lines.extend(write_markdown_table(
        ["candidate_id", "reference_level", "decision", "readiness", "selection_reason"],
        [(r["candidate_id"], r["reference_level"], r["human_review_decision"],
          r["readiness_score"], r["selection_reason"]) for r in rows]))
    write_markdown(doc_path("v2ap_candidate_selection.md"), lines)
    return rows


def run_sentinel_asset_inventory_builder(args=None):
    load_v2ao_candidate_stack()
    rows = scan_repo_for_sentinel_assets()
    if not rows:
        rows.append(_asset_record(0, "no_sentinel_asset_found", "", "", "", "no_explicit_date",
                                  "asset_reference", "", False))
    assert_no_operational_promotion(rows)
    assert_no_absolute_paths_in_content(rows)
    assert_no_local_only(rows)
    write_csv(dataset_path("v2ap_sentinel_asset_inventory.csv"), ASSET_COLUMNS, rows)
    usable = sum(1 for r in rows if r["safe_to_use_as_crosswalk_evidence"] == "true")
    lines = [
        "# v2ap - inventario de assets Sentinel",
        "",
        f"Assets/refs inventariados: {len(rows)}; com data explicita usavel como evidencia: {usable}.",
        "Datas vem apenas de filename/manifest/metadata explicitos; nenhuma inferencia visual ou DINO.",
        "Nenhum dado bruto pesado foi aberto.",
        "",
    ]
    lines.extend(write_markdown_table(
        ["asset_id", "path", "region", "date_detected", "method", "safe_evidence"],
        [(r["asset_id"], short_fragment(r["path"], 60), r["region_detected"], r["date_detected"],
          r["date_detection_method"], r["safe_to_use_as_crosswalk_evidence"]) for r in rows[:60]]))
    write_markdown(doc_path("v2ap_sentinel_asset_inventory.md"), lines)
    return rows


def run_patch_registry_inventory_builder(args=None):
    load_v2ao_candidate_stack()
    rows = scan_repo_for_patch_registries()
    if not rows:
        rows.append({
            "registry_item_id": "PR_v2ap_0000", "source_file": "no_patch_registry_found",
            "patch_id": "none", "region": "none", "has_geometry": "false", "has_bbox": "false",
            "has_centroid": "false", "has_crs": "false", "geometry_source": "no_geometry_column",
            "geometry_status": "NO_GEOMETRY_COLUMNS_BLOCKER", "safe_for_patch_link": "false",
            "notes": "No patch registry with geometry columns found.",
        })
    assert_no_operational_promotion(rows)
    assert_no_absolute_paths_in_content(rows)
    write_csv(dataset_path("v2ap_patch_registry_inventory.csv"), PATCH_REGISTRY_COLUMNS, rows)
    with_geom = sum(1 for r in rows if r["has_geometry"] == "true" or r["has_bbox"] == "true")
    lines = [
        "# v2ap - inventario de patch registries",
        "",
        f"Registries de patch inventariados: {len(rows)}; com colunas de geometria: {with_geom}.",
        "Inventario por colunas de header apenas; nenhuma geometria inventada.",
        "",
    ]
    lines.extend(write_markdown_table(
        ["registry_item_id", "source_file", "has_geometry", "has_bbox", "geometry_status"],
        [(r["registry_item_id"], short_fragment(r["source_file"], 60), r["has_geometry"],
          r["has_bbox"], r["geometry_status"]) for r in rows[:60]]))
    write_markdown(doc_path("v2ap_patch_registry_inventory.md"), lines)
    return rows


def run_spatial_geometry_readiness_builder(args=None):
    stack = load_v2ao_candidate_stack()
    patch_rows = load_csv(dataset_path("v2ap_patch_registry_inventory.csv"))
    rows = []
    for d in derive_candidates(stack):
        has_patch_geom = _region_has_patch_geometry(patch_rows, d["region"])
        has_patch_bbox = any(is_true(r.get("has_bbox")) for r in patch_rows)
        strongest = (d["anchor_types"][0] if d["anchor_types"] else
                     ("map_or_technical_area" if d["geometry_or_map_available"] else "municipio"))
        has_event_geom = d["geometry_or_map_available"]
        if has_event_geom and has_patch_geom:
            status = "EVENT_AND_PATCH_GEOMETRY_READY"
        elif has_event_geom:
            status = "EVENT_GEOMETRY_READY"
        elif has_patch_geom:
            status = "PATCH_GEOMETRY_READY"
        elif d["anchor_count"] > 0 and d["spatial_anchor_confirmed"]:
            status = "ANCHOR_ONLY_NEEDS_GEOMETRY"
        else:
            status = "INSUFFICIENT_SPATIAL_EVIDENCE"
        blocking = ("Geometria de evento ausente; coletar geometria oficial sem inventar."
                    if not has_event_geom else "Pronto para revisao de geometria de evento.")
        rows.append(build_geometry_readiness_row(
            d["candidate_id"], d["region"], d["anchor_count"], strongest,
            has_event_geom, has_patch_geom, has_patch_bbox, status, blocking))
    assert_no_operational_promotion(rows)
    assert_no_fake_geometry(rows)
    write_csv(dataset_path("v2ap_spatial_geometry_readiness.csv"), GEOMETRY_COLUMNS_OUT, rows)
    lines = [
        "# v2ap - readiness de geometria espacial",
        "",
        "Nenhuma coordenada/geometria e inventada. Sem geometria de evento, o candidato",
        "permanece ANCHOR_ONLY_NEEDS_GEOMETRY ou INSUFFICIENT_SPATIAL_EVIDENCE.",
        "",
    ]
    lines.extend(write_markdown_table(
        ["candidate_id", "anchors", "has_event_geometry", "has_patch_geometry", "geometry_readiness_status"],
        [(r["candidate_id"], r["spatial_anchor_count"], r["has_event_geometry"],
          r["has_patch_geometry"], r["geometry_readiness_status"]) for r in rows]))
    write_markdown(doc_path("v2ap_spatial_geometry_readiness.md"), lines)
    return rows


def temporal_margin_days():
    return 8


def run_temporal_window_builder(args=None):
    stack = load_v2ao_candidate_stack()
    margin = temporal_margin_days()
    rows = []
    for d in derive_candidates(stack):
        ds, de = d["event_date_start"], d["event_date_end"]
        window_days = ""
        win_start, win_end = "", ""
        try:
            a = _dt.datetime.strptime(ds, "%Y-%m-%d")
            b = _dt.datetime.strptime(de or ds, "%Y-%m-%d")
            window_days = str((b - a).days + 1)
            win_start = (a - _dt.timedelta(days=margin)).strftime("%Y-%m-%d")
            win_end = (b + _dt.timedelta(days=margin)).strftime("%Y-%m-%d")
        except ValueError:
            pass
        rows.append({
            "temporal_window_id": f"TW_v2ap_{d['candidate_id']}",
            "candidate_id": d["candidate_id"],
            "region": d["region"],
            "event_date_start": ds,
            "event_date_end": de,
            "event_window_days": window_days,
            "acceptable_sentinel_window_start": win_start,
            "acceptable_sentinel_window_end": win_end,
            "window_policy": f"explicit_event_window_plus_margin_days={margin} (policy, not truth)",
            "notes": "Janela de politica para triagem; nenhum asset Sentinel selecionado aqui.",
        })
    assert_no_operational_promotion(rows)
    write_csv(dataset_path("v2ap_event_sentinel_temporal_window.csv"), TEMPORAL_COLUMNS, rows)
    lines = [
        "# v2ap - janela temporal evento-Sentinel",
        "",
        f"Janela = evento +/- {margin} dias (politica, nao verdade). Nenhuma data Sentinel inferida.",
        "",
    ]
    lines.extend(write_markdown_table(
        ["candidate_id", "event_start", "event_end", "window_start", "window_end"],
        [(r["candidate_id"], r["event_date_start"], r["event_date_end"],
          r["acceptable_sentinel_window_start"], r["acceptable_sentinel_window_end"]) for r in rows]))
    write_markdown(doc_path("v2ap_event_sentinel_temporal_window.md"), lines)
    return rows


def run_sentinel_crosswalk_candidate_builder(args=None):
    stack = load_v2ao_candidate_stack()
    assets = load_csv(dataset_path("v2ap_sentinel_asset_inventory.csv"))
    windows = {r["candidate_id"]: r for r in load_csv(dataset_path("v2ap_event_sentinel_temporal_window.csv"))}
    usable_assets = [a for a in assets if is_true(a.get("safe_to_use_as_crosswalk_evidence"))]
    rows = []
    for d in derive_candidates(stack):
        win = windows.get(d["candidate_id"], {})
        ws, we = win.get("acceptable_sentinel_window_start"), win.get("acceptable_sentinel_window_end")
        matched = False
        for a in usable_assets:
            region_match = normalize_region(a.get("region_detected")) == d["region"]
            if not region_match:
                continue
            within = date_window_overlap(a.get("date_detected"), a.get("date_detected"), ws, we)
            patch_match = bool(clean(a.get("patch_id_detected")))
            explicit = region_match and within and bool(clean(a.get("date_detected"))) \
                and a.get("date_detection_method") in {"manifest_field_explicit",
                                                       "filename_date_explicit", "metadata_field_explicit"}
            status = ("EXPLICIT_CROSSWALK_CANDIDATE" if explicit
                      else "TEMPORAL_REGION_MATCH_WEAK" if within else "NO_TEMPORAL_MATCH")
            blocking = ("" if explicit else
                        "Sem asset/patch explicito na janela; nao inferir por similaridade visual ou DINO.")
            if within:
                rows.append(build_crosswalk_candidate_row(
                    len(rows), d["candidate_id"], d["region"], a, d["event_date_start"],
                    d["event_date_end"], within, region_match, patch_match,
                    a.get("date_detection_method"), status, explicit, blocking))
                matched = True
        if not matched:
            rows.append(build_crosswalk_candidate_row(
                len(rows), d["candidate_id"], d["region"], {}, d["event_date_start"],
                d["event_date_end"], False, False, False, "none",
                "NO_SENTINEL_ASSET_IN_WINDOW", False,
                "Nenhum asset Sentinel com data explicita na janela; crosswalk nao inferido."))
    assert_no_operational_promotion(rows)
    assert_no_fake_sentinel_crosswalk(rows)
    write_csv(dataset_path("v2ap_sentinel_crosswalk_candidates.csv"), CROSSWALK_COLUMNS, rows)
    explicit_n = sum(1 for r in rows if r["can_be_used_as_explicit_crosswalk"] == "true")
    lines = [
        "# v2ap - candidatos de crosswalk Sentinel",
        "",
        f"Linhas: {len(rows)}; crosswalks explicitos candidatos: {explicit_n}.",
        "Crosswalk explicito exige regiao + data explicita + asset/patch + fonte manifest/filename/metadata.",
        "Nenhum crosswalk e inferido por DINO ou similaridade visual; nenhum aplicado a ground truth.",
        "",
    ]
    lines.extend(write_markdown_table(
        ["candidate_id", "asset_date", "within_window", "region_match", "explicit", "status"],
        [(r["candidate_id"], r["asset_date"], r["within_temporal_window"], r["region_match"],
          r["can_be_used_as_explicit_crosswalk"], r["crosswalk_candidate_status"]) for r in rows[:60]]))
    write_markdown(doc_path("v2ap_sentinel_crosswalk_candidates.md"), lines)
    return rows


def run_patch_event_link_readiness_builder(args=None):
    stack = load_v2ao_candidate_stack()
    geometry = {r["candidate_id"]: r for r in load_csv(dataset_path("v2ap_spatial_geometry_readiness.csv"))}
    crosswalks = load_csv(dataset_path("v2ap_sentinel_crosswalk_candidates.csv"))
    cross_by_cid = {}
    for r in crosswalks:
        cross_by_cid.setdefault(r["candidate_id"], []).append(r)
    rows = []
    for d in derive_candidates(stack):
        geo = geometry.get(d["candidate_id"], {})
        cc = cross_by_cid.get(d["candidate_id"], [])
        has_event_anchor = d["anchor_count"] > 0
        has_event_geom = is_true(geo.get("has_event_geometry"))
        has_patch_geom = is_true(geo.get("has_patch_geometry"))
        has_cc = any(r.get("within_temporal_window") == "true" for r in cc)
        has_explicit = any(r.get("can_be_used_as_explicit_crosswalk") == "true" for r in cc)
        patch_level = has_event_geom and has_patch_geom and has_explicit
        if patch_level:
            status = "PATCH_REFERENCE_CANDIDATE_READY_FOR_EXTERNAL_VALIDATION"
            action = "EXTERNAL_VALIDATE_PATCH_REFERENCE"
        elif not has_event_geom:
            status = "EVENT_REFERENCE_ONLY_NEEDS_EVENT_GEOMETRY"
            action = "COLLECT_EVENT_GEOMETRY"
        elif not has_explicit:
            status = "NEEDS_EXPLICIT_SENTINEL_CROSSWALK"
            action = "RESOLVE_SENTINEL_CROSSWALK"
        else:
            status = "NEEDS_PATCH_GEOMETRY"
            action = "CHECK_PATCH_GEOMETRY_REGISTRIES"
        rows.append({
            "link_readiness_id": f"LR_v2ap_{d['candidate_id']}",
            "candidate_id": d["candidate_id"],
            "region": d["region"],
            "has_event_anchor": normalize_bool(has_event_anchor),
            "has_event_geometry": normalize_bool(has_event_geom),
            "has_patch_geometry": normalize_bool(has_patch_geom),
            "has_sentinel_crosswalk_candidate": normalize_bool(has_cc),
            "has_explicit_sentinel_crosswalk": normalize_bool(has_explicit),
            "patch_event_link_status": status,
            "patch_level_reference_candidate": normalize_bool(patch_level),
            "patch_truth_allowed": "false",
            "required_next_action": action,
        })
    assert_no_operational_promotion(rows)
    for r in rows:
        if is_true(r["patch_truth_allowed"]):
            raise ValueError("patch_truth_allowed must stay false in v2ap.")
    write_csv(dataset_path("v2ap_patch_event_link_readiness.csv"), LINK_COLUMNS, rows)
    lines = [
        "# v2ap - readiness do link evento-patch",
        "",
        "patch_truth_allowed=false para todos. patch_level_reference_candidate=true exige",
        "geometria de evento + geometria de patch + crosswalk Sentinel explicito.",
        "",
    ]
    lines.extend(write_markdown_table(
        ["candidate_id", "event_geometry", "patch_geometry", "explicit_crosswalk", "status"],
        [(r["candidate_id"], r["has_event_geometry"], r["has_patch_geometry"],
          r["has_explicit_sentinel_crosswalk"], r["patch_event_link_status"]) for r in rows]))
    write_markdown(doc_path("v2ap_patch_event_link_readiness.md"), lines)
    return rows


def run_geometry_collection_packet_builder(args=None):
    stack = load_v2ao_candidate_stack()
    geometry = {r["candidate_id"]: r for r in load_csv(dataset_path("v2ap_spatial_geometry_readiness.csv"))}
    rows = []
    for d in derive_candidates(stack):
        geo = geometry.get(d["candidate_id"], {})
        if is_true(geo.get("has_event_geometry")):
            continue
        priority = ("HIGH" if d["level_rank"] >= _LEVEL_RANK["C4_READY_FOR_EXTERNAL_VALIDATION_REVIEW"]
                    else "HIGH" if d["level_rank"] >= _LEVEL_RANK["C3_STRONG_REFERENCE_CANDIDATE"]
                    else "MEDIUM" if d["anchor_count"] > 0 else "LOW")
        rows.append({
            "collection_id": f"GC_v2ap_{d['candidate_id']}",
            "candidate_id": d["candidate_id"],
            "region": d["region"],
            "missing_geometry_type": "event_polygon_or_affected_area_geometry",
            "best_existing_anchor": clean(geo.get("strongest_anchor_type")) or "municipio",
            "recommended_source": "orgao oficial (Defesa Civil/DRM/Prefeitura) ou laudo tecnico publico",
            "query_terms": f"{d['region']} {d['event_date_start']} area afetada poligono mapa oficial",
            "expected_artifact": "GeoJSON/shapefile oficial ou mapa georreferenciado (revisao manual)",
            "collection_priority": priority,
            "do_not_infer": "Nao inventar coordenada/geometria; nao usar DINO; nao usar similaridade visual.",
        })
    if not rows:
        rows.append({
            "collection_id": "GC_v2ap_NONE", "candidate_id": "all_have_event_geometry",
            "region": "", "missing_geometry_type": "none", "best_existing_anchor": "",
            "recommended_source": "", "query_terms": "", "expected_artifact": "",
            "collection_priority": "LOW", "do_not_infer": "Nao inventar geometria.",
        })
    assert_no_operational_promotion(rows)
    write_csv(dataset_path("v2ap_geometry_collection_packet.csv"), GEOM_PACKET_COLUMNS, rows)
    lines = [
        "# v2ap - pacote de coleta de geometria",
        "",
        "Lista o que coletar manual/externamente para fechar geometria. Nao inferir nada.",
        "Prioridade: PET_2022_02_15 (C4), depois C3 fortes, depois ancoras especificas.",
        "",
    ]
    lines.extend(write_markdown_table(
        ["candidate_id", "missing_geometry_type", "collection_priority", "best_existing_anchor"],
        [(r["candidate_id"], r["missing_geometry_type"], r["collection_priority"],
          r["best_existing_anchor"]) for r in rows]))
    write_markdown(doc_path("v2ap_geometry_collection_packet.md"), lines)
    return rows


def run_crosswalk_collection_packet_builder(args=None):
    stack = load_v2ao_candidate_stack()
    link = {r["candidate_id"]: r for r in load_csv(dataset_path("v2ap_patch_event_link_readiness.csv"))}
    rows = []
    for d in derive_candidates(stack):
        lr = link.get(d["candidate_id"], {})
        if is_true(lr.get("has_explicit_sentinel_crosswalk")):
            continue
        missing = []
        if not is_true(lr.get("has_sentinel_crosswalk_candidate")):
            missing.append("asset_date")
        if not clean(lr.get("has_patch_geometry")) == "true":
            missing.append("patch_geometry")
        if not is_true(lr.get("has_event_geometry")):
            missing.append("event_geometry")
        missing.append("manifest_link")
        priority = ("HIGH" if d["level_rank"] >= _LEVEL_RANK["C3_STRONG_REFERENCE_CANDIDATE"] else "MEDIUM")
        rows.append({
            "collection_id": f"XC_v2ap_{d['candidate_id']}",
            "candidate_id": d["candidate_id"],
            "region": d["region"],
            "missing_crosswalk_component": "|".join(missing),
            "candidate_asset_or_patch": "verificar registries de patch/sentinel por regiao e data",
            "recommended_manifest_to_check": "datasets/protocolo_c/v1us_patch_registry_resolution.csv; datasets/official_anchor_sentinel_patch_registry.csv",
            "recommended_script_or_registry": "datasets/event_sentinel_temporal_window_registry.csv",
            "collection_priority": priority,
            "do_not_infer": "Nao inferir data Sentinel; nao usar DINO para linkar data; sem similaridade visual.",
        })
    if not rows:
        rows.append({
            "collection_id": "XC_v2ap_NONE", "candidate_id": "all_have_explicit_crosswalk",
            "region": "", "missing_crosswalk_component": "none", "candidate_asset_or_patch": "",
            "recommended_manifest_to_check": "", "recommended_script_or_registry": "",
            "collection_priority": "LOW", "do_not_infer": "Nao inferir data Sentinel.",
        })
    assert_no_operational_promotion(rows)
    assert_no_absolute_paths_in_content(rows)
    write_csv(dataset_path("v2ap_crosswalk_collection_packet.csv"), CROSS_PACKET_COLUMNS, rows)
    lines = [
        "# v2ap - pacote de coleta de crosswalk Sentinel",
        "",
        "Lista o que falta para crosswalk Sentinel explicito. Nenhuma data e inferida.",
        "",
    ]
    lines.extend(write_markdown_table(
        ["candidate_id", "missing_crosswalk_component", "collection_priority"],
        [(r["candidate_id"], r["missing_crosswalk_component"], r["collection_priority"]) for r in rows]))
    write_markdown(doc_path("v2ap_crosswalk_collection_packet.md"), lines)
    return rows


def run_patch_reference_readiness_scorer(args=None):
    stack = load_v2ao_candidate_stack()
    geometry = {r["candidate_id"]: r for r in load_csv(dataset_path("v2ap_spatial_geometry_readiness.csv"))}
    link = {r["candidate_id"]: r for r in load_csv(dataset_path("v2ap_patch_event_link_readiness.csv"))}
    patch_rows = load_csv(dataset_path("v2ap_patch_registry_inventory.csv"))
    rows = []
    for d in derive_candidates(stack):
        geo = geometry.get(d["candidate_id"], {})
        lr = link.get(d["candidate_id"], {})
        geometry_score = 40 if is_true(geo.get("has_event_geometry")) else (
            15 if d["anchor_count"] > 0 and d["spatial_anchor_confirmed"] else 0)
        if is_true(lr.get("has_explicit_sentinel_crosswalk")):
            crosswalk_score = 40
        elif is_true(lr.get("has_sentinel_crosswalk_candidate")):
            crosswalk_score = 15
        else:
            crosswalk_score = 0
        patch_registry_score = 20 if _region_has_patch_geometry(patch_rows, d["region"]) else 0
        overall = geometry_score + crosswalk_score + patch_registry_score
        band = "HIGH" if overall >= 70 else "MEDIUM" if overall >= 40 else "LOW"
        if not is_true(geo.get("has_event_geometry")):
            blocker = "missing_event_geometry"
        elif not is_true(lr.get("has_explicit_sentinel_crosswalk")):
            blocker = "missing_explicit_sentinel_crosswalk"
        else:
            blocker = "needs_external_validation"
        if is_true(lr.get("patch_level_reference_candidate")):
            status = "PATCH_REFERENCE_CANDIDATE_READY_FOR_EXTERNAL_VALIDATION"
        elif overall >= 40:
            status = "PARTIAL_PATCH_READINESS_NEEDS_GEOMETRY_OR_CROSSWALK"
        else:
            status = "EVENT_REFERENCE_ONLY"
        rows.append(build_patch_reference_readiness_row(
            d["candidate_id"], d["region"], d["reference_level"], geometry_score,
            crosswalk_score, patch_registry_score, band, blocker, status))
    assert_no_operational_promotion(rows)
    assert_no_fake_ground_truth(rows)
    write_csv(dataset_path("v2ap_patch_reference_readiness_scores.csv"), SCORE_COLUMNS, rows)
    lines = [
        "# v2ap - scores de readiness patch-level",
        "",
        "can_create_ground_truth/label=false; protocol_b=BLOCKED para todos.",
        "",
    ]
    lines.extend(write_markdown_table(
        ["candidate_id", "geometry", "crosswalk", "patch_registry", "overall", "band", "status"],
        [(r["candidate_id"], r["geometry_score"], r["crosswalk_score"], r["patch_registry_score"],
          r["overall_patch_reference_score"], r["readiness_band"], r["patch_reference_status"]) for r in rows]))
    write_markdown(doc_path("v2ap_patch_reference_readiness_scores.md"), lines)
    return rows


def run_patch_truth_boundary_update_builder(args=None):
    stack = load_v2ao_candidate_stack()
    geometry = {r["candidate_id"]: r for r in load_csv(dataset_path("v2ap_spatial_geometry_readiness.csv"))}
    link = {r["candidate_id"]: r for r in load_csv(dataset_path("v2ap_patch_event_link_readiness.csv"))}
    scores = {r["candidate_id"]: r for r in load_csv(dataset_path("v2ap_patch_reference_readiness_scores.csv"))}
    rows = []
    for d in derive_candidates(stack):
        geo = geometry.get(d["candidate_id"], {})
        lr = link.get(d["candidate_id"], {})
        sc = scores.get(d["candidate_id"], {})
        event_geom = "READY" if is_true(geo.get("has_event_geometry")) else "MISSING"
        patch_geom = "READY" if is_true(geo.get("has_patch_geometry")) else "MISSING"
        cross = "EXPLICIT" if is_true(lr.get("has_explicit_sentinel_crosswalk")) else "MISSING"
        all_explicit = event_geom == "READY" and patch_geom == "READY" and cross == "EXPLICIT"
        patch_ref_status = ("READY_FOR_EXTERNAL_VALIDATION" if all_explicit
                            else clean(sc.get("patch_reference_status")) or "EVENT_REFERENCE_ONLY")
        rows.append({
            "boundary_update_id": f"BU_v2ap_{d['candidate_id']}",
            "candidate_id": d["candidate_id"],
            "event_reference_status": d["reference_level"],
            "patch_reference_status": patch_ref_status,
            "event_geometry_status": event_geom,
            "patch_geometry_status": patch_geom,
            "sentinel_crosswalk_status": cross,
            "patch_truth_allowed": "false",
            "why_still_blocked": ("Falta geometria de evento e/ou crosswalk Sentinel explicito; "
                                  "mesmo com tudo explicito, vira READY_FOR_EXTERNAL_VALIDATION, nao ground truth."),
            "safe_use": "Referencia observacional candidata para validacao externa futura.",
            "forbidden_use": FORBIDDEN_USE,
        })
    assert_no_operational_promotion(rows)
    for r in rows:
        if is_true(r["patch_truth_allowed"]):
            raise ValueError("patch_truth_allowed must stay false in v2ap.")
    write_csv(dataset_path("v2ap_patch_truth_boundary_update.csv"), BOUNDARY_COLUMNS, rows)
    lines = [
        "# v2ap - atualizacao do patch truth boundary",
        "",
        "patch_truth_allowed=false nesta etapa. Mesmo com geometria+crosswalk explicitos,",
        "o maximo e READY_FOR_EXTERNAL_VALIDATION, nunca ground truth operacional.",
        "",
    ]
    lines.extend(write_markdown_table(
        ["candidate_id", "event_geometry", "patch_geometry", "crosswalk", "patch_truth_allowed", "patch_reference_status"],
        [(r["candidate_id"], r["event_geometry_status"], r["patch_geometry_status"],
          r["sentinel_crosswalk_status"], r["patch_truth_allowed"], r["patch_reference_status"]) for r in rows]))
    write_markdown(doc_path("v2ap_patch_truth_boundary_update.md"), lines)
    return rows


# --- guardrail regression --------------------------------------------------
def _regression_artifacts():
    artifacts = []
    if os.path.isdir(DATASET_DIR):
        for n in sorted(os.listdir(DATASET_DIR)):
            if n.endswith(".csv") and n.startswith("v2ap_"):
                artifacts.append((rel_dataset(n), dataset_path(n), "csv"))
    if os.path.isdir(DOCS_DIR):
        for n in sorted(os.listdir(DOCS_DIR)):
            if n.endswith(".md"):
                artifacts.append((rel_doc(n), doc_path(n), "text"))
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
                "regression_id": f"GR_v2ap_{len(rows):05d}",
                "artifact_path": rel,
                "check_type": check_type,
                "violation_count": str(count),
                "status": status,
                "severity": "none" if count == 0 else "blocking",
                "notes": "Fail-closed guardrail regression over v2ap outputs.",
            })
    write_csv(dataset_path("v2ap_guardrail_regression.csv"), REGRESSION_COLUMNS, rows)
    if total_fail:
        fails = [(r["artifact_path"], r["check_type"]) for r in rows if r["status"] == "FAIL"]
        raise ValueError(f"v2ap guardrail regression failed: {fails[:5]}")
    return rows


# --- next action -----------------------------------------------------------
def run_next_action_ranker(args=None):
    link = load_csv(dataset_path("v2ap_patch_event_link_readiness.csv"))
    geometry = load_csv(dataset_path("v2ap_spatial_geometry_readiness.csv"))
    has_patch_ref = any(r.get("patch_level_reference_candidate") == "true" for r in link)
    missing_geometry = any(r.get("has_event_geometry") == "false" for r in geometry)
    missing_crosswalk = any(r.get("has_explicit_sentinel_crosswalk") == "false" for r in link)
    if has_patch_ref:
        top = "EXTERNAL_VALIDATE_PATCH_REFERENCE_CANDIDATES"
    elif missing_geometry:
        top = "COLLECT_EVENT_GEOMETRY_FOR_TOP_CANDIDATES"
    elif missing_crosswalk:
        top = "RESOLVE_SENTINEL_PATCH_DATE_CROSSWALK"
    else:
        top = "CHECK_PATCH_GEOMETRY_REGISTRIES"
    options = [
        (top, 100, "v2ap link/geometry readiness", "v2ap_patch_event_link_readiness.csv"),
        ("CHECK_PATCH_GEOMETRY_REGISTRIES", 85, "v2ap patch registry inventory", "v2ap_patch_registry_inventory.csv"),
        ("BUILD_MANUAL_GEOJSON_PLACEHOLDER_FOR_REVIEW", 75, "geometry collection packet", "v2ap_geometry_collection_packet.csv"),
        ("VERIFY_SENTINEL_MANIFESTS", 65, "sentinel asset inventory", "v2ap_sentinel_asset_inventory.csv"),
        ("MAINTAIN_EVENT_REFERENCE_ONLY", 55, "event reference baseline", "v2ap_patch_truth_boundary_update.csv"),
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
                      "labels, operational ground truth, automatic date inference, or promotion."),
        })
        rank += 1
    write_csv(dataset_path("v2ap_next_actions_registry.csv"), NEXT_COLUMNS, rows)
    return rows


# --- completion report -----------------------------------------------------
def run_completion_report(args=None):
    selection = load_csv(dataset_path("v2ap_candidate_selection.csv"))
    assets = load_csv(dataset_path("v2ap_sentinel_asset_inventory.csv"))
    patch_reg = load_csv(dataset_path("v2ap_patch_registry_inventory.csv"))
    geometry = load_csv(dataset_path("v2ap_spatial_geometry_readiness.csv"))
    windows = load_csv(dataset_path("v2ap_event_sentinel_temporal_window.csv"))
    crosswalks = load_csv(dataset_path("v2ap_sentinel_crosswalk_candidates.csv"))
    link = load_csv(dataset_path("v2ap_patch_event_link_readiness.csv"))
    scores = load_csv(dataset_path("v2ap_patch_reference_readiness_scores.csv"))
    boundary = load_csv(dataset_path("v2ap_patch_truth_boundary_update.csv"))
    regression = load_csv(dataset_path("v2ap_guardrail_regression.csv"))
    next_rows = load_csv(dataset_path("v2ap_next_actions_registry.csv"))
    usable_assets = sum(1 for r in assets if r.get("safe_to_use_as_crosswalk_evidence") == "true")
    explicit_cross = sum(1 for r in crosswalks if r.get("can_be_used_as_explicit_crosswalk") == "true")
    patch_ref = sum(1 for r in link if r.get("patch_level_reference_candidate") == "true")
    missing_geometry = sum(1 for r in geometry if r.get("has_event_geometry") == "false")
    patch_allowed = sum(1 for r in boundary if r.get("patch_truth_allowed") == "true")
    regression_fail = sum(1 for r in regression if r.get("status") == "FAIL")
    rows = [
        {"completion_id": "CR_v2ap_000", "metric": "candidates_loaded", "value": str(len(selection)),
         "status": "PASS" if len(selection) == 9 else "RECORDED", "notes": "From v2ao."},
        {"completion_id": "CR_v2ap_001", "metric": "sentinel_assets_inventoried", "value": str(len(assets)),
         "status": "RECORDED", "notes": "No raw data opened."},
        {"completion_id": "CR_v2ap_002", "metric": "sentinel_assets_usable_as_evidence", "value": str(usable_assets),
         "status": "RECORDED", "notes": "Explicit filename/manifest dates only."},
        {"completion_id": "CR_v2ap_003", "metric": "patch_registries_found", "value": str(len(patch_reg)),
         "status": "RECORDED", "notes": "Header-column inventory."},
        {"completion_id": "CR_v2ap_004", "metric": "temporal_windows", "value": str(len(windows)),
         "status": "RECORDED", "notes": "Policy windows; no asset selected."},
        {"completion_id": "CR_v2ap_005", "metric": "crosswalk_candidate_rows", "value": str(len(crosswalks)),
         "status": "RECORDED", "notes": "No DINO/visual inference."},
        {"completion_id": "CR_v2ap_006", "metric": "explicit_crosswalk_candidates", "value": str(explicit_cross),
         "status": "RECORDED", "notes": "Region+explicit date+asset/patch+manifest evidence."},
        {"completion_id": "CR_v2ap_007", "metric": "patch_level_reference_candidates", "value": str(patch_ref),
         "status": "RECORDED", "notes": "Needs geometry+patch+explicit crosswalk."},
        {"completion_id": "CR_v2ap_008", "metric": "candidates_missing_event_geometry", "value": str(missing_geometry),
         "status": "RECORDED", "notes": "No geometry invented."},
        {"completion_id": "CR_v2ap_009", "metric": "patch_truth_allowed", "value": str(patch_allowed),
         "status": "GUARDRAIL_OK" if patch_allowed == 0 else "FAIL", "notes": "Must remain 0."},
        {"completion_id": "CR_v2ap_010", "metric": "readiness_scores", "value": str(len(scores)),
         "status": "RECORDED", "notes": "Patch-level readiness."},
        {"completion_id": "CR_v2ap_011", "metric": "guardrail_regression_failures", "value": str(regression_fail),
         "status": "PASS" if regression_fail == 0 else "FAIL", "notes": "Fail-closed."},
        {"completion_id": "CR_v2ap_012", "metric": "next_action_rank_1",
         "value": next_rows[0]["next_action"] if next_rows else "", "status": "SAFE_NEXT_ACTION",
         "notes": "Geometry/crosswalk acquisition path."},
        {"completion_id": "CR_v2ap_013", "metric": "final_decision",
         "value": "geometry_crosswalk_readiness_built_no_operational_ground_truth",
         "status": "NO_OPERATIONAL_GROUND_TRUTH", "notes": "patch_truth_allowed=false; protocol_b blocked."},
    ]
    write_csv(dataset_path("v2ap_completion_report.csv"), COMPLETION_COLUMNS, rows)
    lines = [
        "# v2ap completion report",
        "",
        f"Candidates loaded: {len(selection)}.",
        f"Sentinel assets inventoried: {len(assets)} (usable as evidence: {usable_assets}).",
        f"Patch registries found: {len(patch_reg)}.",
        f"Temporal windows: {len(windows)}.",
        f"Crosswalk candidate rows: {len(crosswalks)} (explicit: {explicit_cross}).",
        f"Patch-level reference candidates: {patch_ref}.",
        f"Candidates missing event geometry: {missing_geometry}.",
        f"patch_truth_allowed: {patch_allowed} (must be 0).",
        f"Guardrail regression failures: {regression_fail}.",
        f"Next action rank 1: {next_rows[0]['next_action'] if next_rows else ''}.",
        "Final decision: geometry/crosswalk readiness built; no operational ground truth, no Protocol B.",
    ]
    write_markdown(doc_path("v2ap_completion_report.md"), lines)
    return rows


# --- orchestrator ----------------------------------------------------------
_ORCHESTRATION = [
    ("candidate_selection", "run_candidate_selection_builder",
     ["v2ap_candidate_selection.csv"], ["v2ap_candidate_selection.md"]),
    ("sentinel_asset_inventory", "run_sentinel_asset_inventory_builder",
     ["v2ap_sentinel_asset_inventory.csv"], ["v2ap_sentinel_asset_inventory.md"]),
    ("patch_registry_inventory", "run_patch_registry_inventory_builder",
     ["v2ap_patch_registry_inventory.csv"], ["v2ap_patch_registry_inventory.md"]),
    ("spatial_geometry_readiness", "run_spatial_geometry_readiness_builder",
     ["v2ap_spatial_geometry_readiness.csv"], ["v2ap_spatial_geometry_readiness.md"]),
    ("temporal_window", "run_temporal_window_builder",
     ["v2ap_event_sentinel_temporal_window.csv"], ["v2ap_event_sentinel_temporal_window.md"]),
    ("sentinel_crosswalk_candidates", "run_sentinel_crosswalk_candidate_builder",
     ["v2ap_sentinel_crosswalk_candidates.csv"], ["v2ap_sentinel_crosswalk_candidates.md"]),
    ("patch_event_link_readiness", "run_patch_event_link_readiness_builder",
     ["v2ap_patch_event_link_readiness.csv"], ["v2ap_patch_event_link_readiness.md"]),
    ("geometry_collection_packet", "run_geometry_collection_packet_builder",
     ["v2ap_geometry_collection_packet.csv"], ["v2ap_geometry_collection_packet.md"]),
    ("crosswalk_collection_packet", "run_crosswalk_collection_packet_builder",
     ["v2ap_crosswalk_collection_packet.csv"], ["v2ap_crosswalk_collection_packet.md"]),
    ("patch_reference_readiness_scorer", "run_patch_reference_readiness_scorer",
     ["v2ap_patch_reference_readiness_scores.csv"], ["v2ap_patch_reference_readiness_scores.md"]),
    ("patch_truth_boundary_update", "run_patch_truth_boundary_update_builder",
     ["v2ap_patch_truth_boundary_update.csv"], ["v2ap_patch_truth_boundary_update.md"]),
    ("guardrail_regression", "run_guardrail_regression",
     ["v2ap_guardrail_regression.csv"], []),
    ("next_action_ranker", "run_next_action_ranker",
     ["v2ap_next_actions_registry.csv"], []),
    ("completion_report", "run_completion_report",
     ["v2ap_completion_report.csv"], ["v2ap_completion_report.md"]),
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
    lines = ["# v2ap - orchestrator run manifest", "",
             f"Etapas executadas: {len(rows)}. Nenhuma operacao git foi executada.", ""]
    lines.extend(write_markdown_table(
        ["ordem", "etapa", "status", "outputs"],
        [(r["step_order"], r["step_name"], r["status"], r["outputs"]) for r in rows]))
    write_markdown(doc_path("v2ap_orchestrator_run_manifest.md"), lines)


def run_master_orchestrator(args=None):
    rows = []
    for order, (name, func_name, ds_out, doc_out) in enumerate(_ORCHESTRATION, 1):
        func = globals()[func_name]
        try:
            func(args)
        except Exception as exc:
            rows.append(_manifest_row(order, name, "FAIL", ds_out, doc_out,
                                      f"{type(exc).__name__}: {exc}"))
            write_csv(dataset_path("v2ap_orchestrator_run_manifest.csv"), MANIFEST_COLUMNS, rows)
            _write_manifest_md(rows)
            raise
        rows.append(_manifest_row(order, name, "OK", ds_out, doc_out, "Completed."))
    write_csv(dataset_path("v2ap_orchestrator_run_manifest.csv"), MANIFEST_COLUMNS, rows)
    _write_manifest_md(rows)
    return rows


def run_all(args=None):
    return run_master_orchestrator(args)
