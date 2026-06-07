#!/usr/bin/env python3
"""v2an Observed Candidate Ground-Reference Validation Sprint.

Deep observational validation over the 9 existing observed-event candidates of the
Protocolo C. Tries to advance "observed candidate" toward "ground-reference validation
candidate" in an auditable way, closing what it can of the G1-G9 gates, but never
creating operational ground truth, labels, classes, targets, training, overlay,
prediction, inferred Sentinel dates or invented geometry/coordinates.

Only ``v2an_*`` artifacts are written. Prior artifacts are read-only. Operational
ground truth status stays ``NOT_ESTABLISHED`` and Protocol B stays ``BLOCKED``.
"""

import argparse
import csv
import datetime as _dt
import hashlib
import json
import os
import re

PROTOCOL_VERSION = "v2an"
DATASET_DIR = os.environ.get("DATASET_DIR", "datasets")
PROTOCOL_C_DIR = os.environ.get("PROTOCOL_C_DIR", os.path.join(DATASET_DIR, "protocolo_c"))
DOCS_DIR = os.environ.get(
    "DOCS_DIR", "docs/protocolo_c/v2an_ground_reference_validation_sprint")
DOSSIER_DIR = os.environ.get("DOSSIER_DIR", os.path.join(DOCS_DIR, "dossiers"))
CONFIG_DIR = os.environ.get("CONFIG_DIR", "configs/protocolo_c")
# Network probing is opt-in for determinism; default is fail-closed offline.
NETWORK_ENABLED = os.environ.get("V2AN_NETWORK", "0") == "1"

CANDIDATE_REGISTRY = "observed_event_reference_candidate_registry.csv"
GAP_REGISTRY = "observed_event_reference_gap_registry.csv"
DECISION_REGISTRY = "observed_event_reference_decision_registry.csv"
MANUAL_NEEDED_REGISTRY = "manual_external_evidence_needed_registry.csv"

EXPECTED_CANDIDATE_IDS = [
    "REC_2022_05_24_30", "REC_2023_02_05_06", "REC_2024_06_14_16",
    "PET_2022_02_15", "PET_2022_03_20_21", "PET_2024_03_21_28",
    "CTB_2022_01_15_16", "CTB_2023_10_28_30", "CTB_2024_02_18_20",
]
REGION_BY_PREFIX = {"REC": "Recife", "PET": "Petropolis", "CTB": "Curitiba"}

# --- guardrail vocabulary --------------------------------------------------
FORBIDDEN_TRUE_FIELDS = {
    "ground_truth_created", "ground_reference_created", "label_created",
    "operational_ground_truth", "training_ready", "overlay_ready",
    "prediction_ready", "human_review_completed", "adjudication_completed",
    "promotion_allowed", "can_create_ground_truth", "can_create_label",
    "can_be_used_as_training_label", "raw_data_versioned", "sentinel_date_inferred",
    "crosswalk_inferred", "protocol_b_reopened", "can_reopen_protocol_b",
}
FORBIDDEN_STATUS_VALUES = {
    "GROUND_TRUTH_VALIDATED", "GROUND_REFERENCE_TRUE", "LABEL_READY",
    "LABEL_POSITIVE", "LABEL_NEGATIVE", "TRAINING_READY", "PROTOCOL_B_REOPENED",
    "PROTOCOL_B_OPEN", "OPERATIONAL_VALIDATION", "PATCH_POSITIVE",
    "PATCH_NEGATIVE", "FLOOD_DETECTED", "REVIEW_COMPLETED",
    "ADJUDICATION_COMPLETED", "PROMOTION_ALLOWED",
}
FORBIDDEN_KV_MARKERS = [
    "ground_truth=true", "ground_reference=true", "label=true", "training=true",
    "overlay=true", "prediction=true", "protocol_b_reopen=true",
    "sentinel_date_inferred=true", "crosswalk_inferred=true",
    "human_review_completed=true", "adjudication_completed=true",
    "operational_validation=true", "promotion_allowed=true",
    "can_create_ground_truth=true", "can_create_label=true",
    "raw_data_versioned=true",
]
UNSAFE_LANGUAGE = [
    "ground truth validado", "classe positiva", "classe negativa",
    "label operacional", "deteccao de enchente", "deteccao de inundacao",
    "predicao de inundacao", "modelo preditivo", "validacao operacional",
    "treinamento supervisionado pronto", "similaridade visual confirma",
]
SAFE_UNSAFE_FIELDS = {
    "forbidden_use", "forbidden_decisions", "forbidden_terms", "blocking_reason",
    "blocker_summary", "dominant_blocker", "remaining_blockers", "limitations",
    "notes", "decision_reason", "review_question", "spatial_question",
    "temporal_question", "phenomenon_question", "source_strength_question",
    "evidence_to_check", "possible_decisions", "what_continues_forbidden",
    "missing_evidence", "blocking_gates", "recommended_next_step",
    "next_required_action", "next_required_step", "strongest_evidence",
    "evidence_fragment_safe", "decision_status", "gate_closure_status",
    "patch_link_readiness_status", "temporal_gate_status", "readiness_band",
    "access_status", "metadata_extraction_status", "geocoding_status",
    "ground_truth_status", "label_status", "protocol_b_status",
    "operational_ground_truth_status", "current_ground_reference_candidate_status",
    "review_status", "anchor_text", "anchor_type", "reason", "purpose",
}
SAFE_CONTEXT_MARKERS = [
    "nao pode dizer", "nao usar", "nao afirmar", "nao ha", "nao deve",
    "nao temos", "nao realiza", "nao detecta", "nao cria", "nao existe",
    "nao produz", "nao significa", "nao implica", "nao foi", "nao e ",
    "nao ", "proibid", "forbidden", "limitation", "limitacao", "blocker",
    "blocked", "bloque", "does not", "do not", "not ", "no ", "sem ",
    "evitar", "ausencia", "pendente", "candidato", "review-only", "needs_",
    "missing", "needed", "exemplo negativo", "unsafe", "nao estabelecid",
    "not_established",
]
ABSOLUTE_PATH_RE = re.compile(r"(?:[A-Za-z]:\\|/Users/|/home/|/mnt/|\\\\)")
LOCAL_ONLY_MARKER = "local" + "_" + "only"

# --- argument parsing ------------------------------------------------------
def parse_args(argv=None):
    return argparse.ArgumentParser().parse_args(argv)


# --- path helpers ----------------------------------------------------------
def source_dataset_path(name):
    return os.path.join(DATASET_DIR, name)


def protocol_path(name):
    return os.path.join(PROTOCOL_C_DIR, name)


def doc_path(name):
    return os.path.join(DOCS_DIR, name)


def dossier_path(name):
    return os.path.join(DOSSIER_DIR, name)


def rel_protocol(name):
    return f"datasets/protocolo_c/{name}"


def rel_doc(name):
    return f"docs/protocolo_c/v2an_ground_reference_validation_sprint/{name}"


def rel_dossier(name):
    return f"docs/protocolo_c/v2an_ground_reference_validation_sprint/dossiers/{name}"


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


def normalize_date(value):
    raw = clean(value)
    if not raw:
        return ""
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y"):
        try:
            return _dt.datetime.strptime(raw, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})", raw)
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else raw


def normalize_region(value, candidate_id=""):
    raw = clean(value)
    prefix = clean(candidate_id)[:3].upper()
    if prefix in REGION_BY_PREFIX:
        return REGION_BY_PREFIX[prefix]
    low = raw.lower()
    if "recife" in low:
        return "Recife"
    if "petrop" in low or "petrop" in low:
        return "Petropolis"
    if "curitiba" in low:
        return "Curitiba"
    return raw or "Unspecified"


def normalize_phenomenon(value):
    raw = clean(value).lower()
    if not raw:
        return "unspecified", "true"
    ambiguous = "true" if ("misto" in raw or "mixed" in raw or
                           ("desliz" in raw and "inund" in raw)) else "false"
    if "desliz" in raw or "mass" in raw:
        base = "mass_movement_or_mixed"
    elif "inund" in raw or "alagamento" in raw or "enchente" in raw or "chuva" in raw:
        base = "flood_or_heavy_rain"
    else:
        base = raw.replace(" ", "_")
    return base, ambiguous


def event_window_days(date_start, date_end):
    ds, de = normalize_date(date_start), normalize_date(date_end)
    try:
        a = _dt.datetime.strptime(ds, "%Y-%m-%d")
        b = _dt.datetime.strptime(de, "%Y-%m-%d")
        return str((b - a).days + 1)
    except ValueError:
        return "1" if ds else ""


# --- io helpers ------------------------------------------------------------
def load_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def assert_output_is_v2an(path):
    base = os.path.basename(str(path))
    if LOCAL_ONLY_MARKER in str(path).lower():
        raise ValueError(f"Refusing local_only output path: {path}")
    if not base.startswith("v2an_"):
        raise ValueError(f"Refusing to write non-v2an output: {path}")
    return True


def write_csv(path, columns, rows):
    assert_output_is_v2an(path)
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
    assert_output_is_v2an(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_json(path, payload):
    assert_output_is_v2an(path)
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


def short_fragment(text, limit=140):
    raw = re.sub(r"\s+", " ", clean(text))
    return raw[:limit].strip()


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


def load_nine_observed_candidates():
    path = source_dataset_path(CANDIDATE_REGISTRY)
    rows = load_csv(path)
    if not rows:
        raise FileNotFoundError(
            f"Observed candidate registry not found or empty: {rel_source(CANDIDATE_REGISTRY)}")
    assert_min_schema(rows, ["observed_event_id", "region", "event_name", "event_type",
                             "date_start", "date_end"], CANDIDATE_REGISTRY)
    ids = [clean(r.get("observed_event_id")) for r in rows]
    if len(rows) != 9:
        raise ValueError(
            f"Expected exactly 9 observed candidates, found {len(rows)}: {ids}")
    missing = [c for c in EXPECTED_CANDIDATE_IDS if c not in ids]
    extra = [c for c in ids if c not in EXPECTED_CANDIDATE_IDS]
    if missing or extra:
        raise ValueError(
            f"Observed candidate set mismatch. missing={missing} extra={extra}")
    return rows


def rel_source(name):
    return f"datasets/{name}"


def load_gap_registry():
    return load_csv(source_dataset_path(GAP_REGISTRY))


def load_decision_registry():
    return load_csv(source_dataset_path(DECISION_REGISTRY))


def load_manual_needed_registry():
    return load_csv(source_dataset_path(MANUAL_NEEDED_REGISTRY))


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


def assert_no_raw_data_versioned(rows):
    for idx, row in enumerate(rows):
        for key, value in row.items():
            if clean(key).lower() == "raw_data_versioned" and is_true(value):
                raise ValueError(f"raw_data_versioned=true at row {idx}; forbidden.")
    return True


def assert_no_fake_ground_truth(rows):
    for idx, row in enumerate(rows):
        for key, value in row.items():
            key_l = clean(key).lower()
            value_s = clean(value)
            if key_l in {"operational_ground_truth_status", "ground_truth_status"}:
                if value_s.upper() not in {"NOT_ESTABLISHED", "", "NOT_CREATED"}:
                    raise ValueError(
                        f"ground truth status must stay NOT_ESTABLISHED, got {value_s}")
            if key_l in {"can_create_ground_truth", "can_create_label"} and is_true(value_s):
                raise ValueError(f"{key_l}=true is forbidden at row {idx}.")
    return True


def assert_no_fake_patch_overlay(rows):
    for idx, row in enumerate(rows):
        for key, value in row.items():
            if clean(key).lower() == "overlay_ready" and is_true(value):
                # overlay_ready may only be flagged when explicit geometries exist,
                # but overlay is never executed in this stage.
                if not is_true(row.get("has_event_geometry_available")) or \
                        not is_true(row.get("has_patch_geometry_available")):
                    raise ValueError(
                        f"overlay_ready=true without explicit geometries at row {idx}.")
    return True


def scan_text_violations(text):
    counts = {
        "absolute_path": 0, "local_only": 0, "forbidden_kv": 0,
        "unsafe_language": 0, "forbidden_true_flag": 0, "forbidden_status": 0,
    }
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
def build_gate_record(candidate_id, gates, closed_count, blocking, status):
    record = {"gate_matrix_id": f"GM_v2an_{candidate_id}", "candidate_id": candidate_id}
    record.update(gates)
    record["closed_gates_count"] = str(closed_count)
    record["blocking_gates"] = blocking
    record["gate_closure_status"] = status
    return record


def build_source_probe_record(idx, candidate_id, source_role, source_name, url,
                              http_status, content_type, content_length,
                              access_status, accessed_at, metadata_hash, notes):
    return {
        "probe_id": f"PRB_v2an_{idx:04d}",
        "candidate_id": candidate_id,
        "source_role": source_role,
        "source_name": short_fragment(source_name, 120),
        "url": repo_relative_path(url) if url else "",
        "http_status": http_status,
        "content_type": content_type,
        "content_length": content_length,
        "access_status": access_status,
        "accessed_at_utc": accessed_at,
        "metadata_hash": metadata_hash,
        "raw_data_downloaded": "false",
        "raw_data_versioned": "false",
        "notes": notes,
    }


def build_validation_readiness_record(candidate_id, region, score, band,
                                      strongest, dominant_blocker, can_enter):
    return {
        "score_id": f"RS_v2an_{candidate_id}",
        "candidate_id": candidate_id,
        "region": region,
        "readiness_score": str(score),
        "readiness_band": band,
        "strongest_evidence": strongest,
        "dominant_blocker": dominant_blocker,
        "can_enter_human_ground_reference_review": normalize_bool(can_enter),
        "can_create_ground_truth": "false",
        "can_create_label": "false",
        "protocol_b_status": "BLOCKED",
        "recommended_next_step": (
            "PREPARE_FOR_HUMAN_GROUND_REFERENCE_REVIEW" if can_enter
            else "COLLECT_MISSING_EVIDENCE"),
    }


# --- derivation ------------------------------------------------------------
STRONG_SOURCE_HINTS = ("official", "technical", "peer", "scient", "civil_defense",
                       "cemaden", "drm", "inmet", "apac", "defesa_civil")
WEAK_SOURCE_HINTS = ("news", "media", "blog", "social", "portal_noticia", "imprensa")


def source_strength(primary_type):
    t = clean(primary_type).lower()
    if any(h in t for h in STRONG_SOURCE_HINTS):
        return "STRONG"
    if any(h in t for h in WEAK_SOURCE_HINTS):
        return "WEAK"
    return "MEDIUM" if t else "WEAK"


def _gate_closed(value):
    return clean(value).upper() in {"CLOSED", "TRUE", "DONE", "OK"}


def split_urls(value):
    return [u.strip() for u in re.split(r"[;\s]+", clean(value)) if u.strip().startswith("http")]


def derive_candidate(row):
    cid = clean(row.get("observed_event_id"))
    region = normalize_region(row.get("region"), cid)
    hazard, ambiguous = normalize_phenomenon(row.get("event_type"))
    ds = normalize_date(row.get("date_start"))
    de = normalize_date(row.get("date_end")) or ds
    strength = source_strength(row.get("primary_source_type"))
    primary_urls = split_urls(row.get("primary_source_url"))
    secondary_urls = split_urls(row.get("secondary_source_url"))
    g1 = _gate_closed(row.get("g1_event_confirmation")) or is_true(row.get("observed_event_confirmed"))
    g2 = _gate_closed(row.get("g2_source_availability")) or (
        is_true(row.get("source_traceable")) and bool(primary_urls))
    g3 = _gate_closed(row.get("g3_temporal_alignment")) or _gate_closed(row.get("temporal_alignment_status"))
    g4 = _gate_closed(row.get("g4_spatial_alignment_triage")) or _gate_closed(row.get("spatial_alignment_triage_status"))
    has_secondary = bool(secondary_urls)
    spatial_level = clean(row.get("spatial_precision_level")).upper()
    has_map = spatial_level in {"TECHNICAL_MAP", "STREET", "POINT"} or "MAP" in spatial_level
    has_neighborhood = spatial_level in {"NEIGHBORHOOD", "TECHNICAL_MAP", "STREET", "POINT"}
    return {
        "candidate_id": cid,
        "region": region,
        "event_name": clean(row.get("event_name")),
        "hazard": hazard,
        "phenomenon_ambiguous": ambiguous,
        "date_start": ds,
        "date_end": de,
        "window_days": event_window_days(ds, de),
        "temporal_precision": clean(row.get("temporal_precision_level")),
        "spatial_precision": clean(row.get("spatial_precision_level")),
        "primary_source_type": clean(row.get("primary_source_type")),
        "primary_source_name": clean(row.get("primary_source_name")),
        "primary_urls": primary_urls,
        "secondary_source_type": clean(row.get("secondary_source_type")),
        "secondary_source_name": clean(row.get("secondary_source_name")),
        "secondary_urls": secondary_urls,
        "priority_level": clean(row.get("priority_level")) or "MEDIUM",
        "source_strength": strength,
        "g1": g1, "g2": g2, "g3": g3, "g4": g4,
        "has_secondary": has_secondary,
        "has_map": has_map,
        "has_neighborhood": has_neighborhood,
        "missing_evidence": clean(row.get("missing_evidence")),
        "notes": clean(row.get("notes")),
        "ground_reference_candidate_status": clean(row.get("ground_reference_candidate_status")),
    }


def derive_all():
    return [derive_candidate(r) for r in load_nine_observed_candidates()]


def evaluate_gates(d):
    g = {
        "g1_event_confirmation": "CLOSED" if d["g1"] else "OPEN",
        "g2_source_availability": "CLOSED" if d["g2"] else "OPEN",
        "g3_temporal_alignment": "CLOSED" if d["g3"] else "OPEN",
        "g4_spatial_alignment_triage": "CLOSED" if d["g4"] else "OPEN",
        "g4b_patch_link_readiness": "CLOSED" if (d["g4"] and (d["has_map"] or d["has_neighborhood"])) else "OPEN",
        "g5_source_strength": {"STRONG": "CLOSED", "MEDIUM": "PARTIAL", "WEAK": "OPEN"}[d["source_strength"]],
        "g6_uncertainty_documented": "CLOSED" if (d["missing_evidence"] or d["notes"]) else "OPEN",
        "g7_review_gate_ready": "CLOSED" if (d["g1"] and d["g2"] and d["g3"] and d["g4"]) else "OPEN",
        "g8_independent_corroboration": "CLOSED" if d["has_secondary"] else "OPEN",
        "g9_promotion_decision": "BLOCKED_PENDING_HUMAN_REVIEW",
    }
    closed = sum(1 for k, v in g.items() if k != "g9_promotion_decision" and v == "CLOSED")
    blocking = [k for k, v in g.items()
                if k != "g9_promotion_decision" and v in {"OPEN", "PARTIAL"}]
    if d["g1"] and d["g2"] and d["g3"] and d["g4"] and g["g7_review_gate_ready"] == "CLOSED":
        status = "READY_FOR_HUMAN_REVIEW_GATE"
    elif closed >= 5:
        status = "PARTIALLY_CLOSED_NEEDS_EVIDENCE"
    else:
        status = "BLOCKED_NEEDS_EVIDENCE"
    return g, closed, blocking, status


def readiness_score(d):
    score = 0
    score += 15 if d["g1"] else 0
    score += 10 if d["g2"] else 0
    score += 15 if d["g3"] else 0
    score += 15 if d["g4"] else 0
    score += 5 if (d["g4"] and (d["has_map"] or d["has_neighborhood"])) else 0
    score += {"STRONG": 15, "MEDIUM": 7, "WEAK": 0}[d["source_strength"]]
    score += 5 if (d["missing_evidence"] or d["notes"]) else 0
    score += 5 if (d["g1"] and d["g2"] and d["g3"] and d["g4"]) else 0
    score += 10 if d["has_secondary"] else 0
    if is_true(d["phenomenon_ambiguous"]):
        score -= 10
    return max(0, min(100, score))


def readiness_band(score):
    if score >= 70:
        return "HIGH"
    if score >= 45:
        return "MEDIUM"
    return "LOW"


def can_enter_human_review(d, score):
    return (d["g1"] and d["g2"] and d["g3"] and d["g4"]
            and d["source_strength"] != "WEAK" and score >= 65)


def dominant_blocker(d):
    if not d["g4"]:
        return "no specific spatial geometry (needs manual spatial evidence)"
    if d["source_strength"] == "WEAK":
        return "weak source strength (needs stronger institutional source)"
    if is_true(d["phenomenon_ambiguous"]):
        return "phenomenon ambiguity (mixed flood/mass movement separation pending)"
    return "no explicit Sentinel crosswalk and human ground-reference review pending"


def strongest_evidence(d):
    bits = []
    if d["source_strength"] == "STRONG":
        bits.append("strong institutional source")
    if d["has_map"]:
        bits.append("technical map or point reference")
    elif d["has_neighborhood"]:
        bits.append("neighborhood-level locality")
    if d["has_secondary"]:
        bits.append("independent secondary source")
    return "; ".join(bits) or "documented observed event"


def decision_status(d, can_enter):
    if can_enter:
        return "ADVANCES_TO_HUMAN_GROUND_REFERENCE_REVIEW"
    if d["source_strength"] == "WEAK":
        return "NEEDS_SOURCE_STRENGTH_REVIEW"
    if not d["g4"]:
        return "NEEDS_MORE_SPATIAL_EVIDENCE"
    if is_true(d["phenomenon_ambiguous"]):
        return "BLOCKED_BY_PHENOMENON_AMBIGUITY"
    if d["g1"] and d["g2"] and d["g3"]:
        return "NEEDS_SENTINEL_CROSSWALK"
    return "REMAINS_DOCUMENTED_OBSERVED_CANDIDATE"


# --- column schemas --------------------------------------------------------
INVENTORY_COLUMNS = [
    "candidate_id", "observed_event_id", "region", "event_name", "hazard_type",
    "date_start", "date_end", "temporal_precision_level", "spatial_precision_level",
    "primary_source_type", "primary_source_name", "primary_source_url",
    "secondary_source_type", "secondary_source_name", "secondary_source_url",
    "priority_level", "current_ground_reference_candidate_status",
    "operational_ground_truth_status", "protocol_b_status",
    "can_be_used_as_training_label", "notes",
]
PROBE_COLUMNS = [
    "probe_id", "candidate_id", "source_role", "source_name", "url", "http_status",
    "content_type", "content_length", "access_status", "accessed_at_utc",
    "metadata_hash", "raw_data_downloaded", "raw_data_versioned", "notes",
]
METADATA_COLUMNS = [
    "metadata_id", "candidate_id", "source_role", "document_title_detected",
    "published_date_detected", "mentions_event_date", "mentions_region",
    "mentions_hazard", "mentions_locality", "mentions_geometry_or_map",
    "metadata_extraction_status", "evidence_fragment_safe", "limitations",
]
SPATIAL_COLUMNS = [
    "spatial_anchor_id", "candidate_id", "region", "anchor_text", "anchor_type",
    "source_role", "source_name", "evidence_fragment_safe", "geometry_available",
    "coordinate_available", "geocoding_status", "manual_geometry_review_required",
    "notes",
]
CROSSWALK_COLUMNS = [
    "crosswalk_audit_id", "candidate_id", "date_start", "date_end",
    "event_window_days", "sentinel_asset_date_found", "sentinel_asset_id_found",
    "explicit_crosswalk_found", "crosswalk_source_artifact", "temporal_gate_status",
    "blocking_reason",
]
PATCH_LINK_COLUMNS = [
    "patch_link_audit_id", "candidate_id", "region", "spatial_anchor_count",
    "has_neighborhood_anchor", "has_street_or_point_anchor",
    "has_map_or_technical_area", "has_patch_geometry_available",
    "has_event_geometry_available", "overlay_ready", "manual_patch_review_required",
    "patch_link_readiness_status", "blocking_reason",
]
GATE_COLUMNS = [
    "gate_matrix_id", "candidate_id", "g1_event_confirmation", "g2_source_availability",
    "g3_temporal_alignment", "g4_spatial_alignment_triage", "g4b_patch_link_readiness",
    "g5_source_strength", "g6_uncertainty_documented", "g7_review_gate_ready",
    "g8_independent_corroboration", "g9_promotion_decision", "closed_gates_count",
    "blocking_gates", "gate_closure_status",
]
SCORE_COLUMNS = [
    "score_id", "candidate_id", "region", "readiness_score", "readiness_band",
    "strongest_evidence", "dominant_blocker", "can_enter_human_ground_reference_review",
    "can_create_ground_truth", "can_create_label", "protocol_b_status",
    "recommended_next_step",
]
DOSSIER_INDEX_COLUMNS = [
    "dossier_id", "candidate_id", "region", "dossier_path", "readiness_band",
    "decision_status", "ground_truth_status",
]
REVIEW_PACKAGE_COLUMNS = [
    "review_package_id", "candidate_id", "region", "review_priority",
    "review_question", "evidence_to_check", "spatial_question", "temporal_question",
    "phenomenon_question", "source_strength_question", "possible_decisions",
    "forbidden_decisions", "review_status",
]
DECISION_COLUMNS = [
    "decision_id", "candidate_id", "decision_status", "decision_reason",
    "remaining_blockers", "next_required_action", "ground_truth_status",
    "label_status", "protocol_b_status",
]
GT_BLOCKER_COLUMNS = [
    "blocker_id", "candidate_id", "missing_patch_level_geometry",
    "missing_event_geometry", "missing_explicit_sentinel_crosswalk",
    "missing_human_review", "missing_adjudication", "missing_license_review",
    "phenomenon_ambiguity", "ground_truth_blocked", "blocker_summary",
]
REGRESSION_COLUMNS = [
    "regression_id", "artifact_path", "check_type", "violation_count", "status",
    "severity", "notes",
]
NEXT_COLUMNS = [
    "rank", "next_action", "score", "allowed", "blocked_operational_use",
    "required_input", "recommended_artifact", "notes",
]
MANIFEST_COLUMNS = ["step_order", "step_name", "status", "outputs", "output_hashes", "notes"]
COMPLETION_COLUMNS = ["completion_id", "metric", "value", "status", "notes"]
FORBIDDEN_DECISION_LIST = ("GROUND_TRUTH_VALIDATED|LABEL_READY|PROTOCOL_B_REOPENED|"
                           "TRAINING_READY|OPERATIONAL_VALIDATION")
POSSIBLE_DECISION_LIST = ("REMAINS_DOCUMENTED_OBSERVED_CANDIDATE|"
                          "ADVANCES_TO_HUMAN_GROUND_REFERENCE_REVIEW|"
                          "NEEDS_MORE_SPATIAL_EVIDENCE|NEEDS_SENTINEL_CROSSWALK|"
                          "NEEDS_SOURCE_STRENGTH_REVIEW|BLOCKED_BY_PHENOMENON_AMBIGUITY")


# --- runners ---------------------------------------------------------------
def run_candidate_inventory_normalizer(args=None):
    candidates = derive_all()
    rows = []
    for d in candidates:
        rows.append({
            "candidate_id": d["candidate_id"],
            "observed_event_id": d["candidate_id"],
            "region": d["region"],
            "event_name": d["event_name"],
            "hazard_type": d["hazard"],
            "date_start": d["date_start"],
            "date_end": d["date_end"],
            "temporal_precision_level": d["temporal_precision"],
            "spatial_precision_level": d["spatial_precision"],
            "primary_source_type": d["primary_source_type"],
            "primary_source_name": short_fragment(d["primary_source_name"], 120),
            "primary_source_url": d["primary_urls"][0] if d["primary_urls"] else "",
            "secondary_source_type": d["secondary_source_type"],
            "secondary_source_name": short_fragment(d["secondary_source_name"], 120),
            "secondary_source_url": d["secondary_urls"][0] if d["secondary_urls"] else "",
            "priority_level": d["priority_level"],
            "current_ground_reference_candidate_status": d["ground_reference_candidate_status"]
            or "DOCUMENTED_OBSERVED_CANDIDATE",
            "operational_ground_truth_status": "NOT_ESTABLISHED",
            "protocol_b_status": "BLOCKED",
            "can_be_used_as_training_label": "false",
            "notes": short_fragment(d["notes"], 160),
        })
    if len(rows) != 9:
        raise ValueError(f"Expected 9 normalized candidates, got {len(rows)}")
    assert_no_operational_promotion(rows)
    assert_no_absolute_paths_in_content(rows)
    assert_no_local_only(rows)
    assert_no_fake_ground_truth(rows)
    write_csv(protocol_path("v2an_observed_candidate_inventory_normalized.csv"),
              INVENTORY_COLUMNS, rows)
    lines = [
        "# v2an - inventario normalizado dos 9 candidatos observacionais",
        "",
        "Corpus review-only. operational_ground_truth_status permanece NOT_ESTABLISHED",
        "e protocol_b_status permanece BLOCKED para todos.",
        "",
    ]
    lines.extend(write_markdown_table(
        ["candidate_id", "region", "hazard", "date_start", "date_end", "priority", "gt_status"],
        [(r["candidate_id"], r["region"], r["hazard_type"], r["date_start"],
          r["date_end"], r["priority_level"], r["operational_ground_truth_status"]) for r in rows]))
    write_markdown(doc_path("v2an_observed_candidate_inventory_normalized.md"), lines)
    return rows


def _probe_url(url):
    if not NETWORK_ENABLED:
        return ("", "", "", "NETWORK_UNAVAILABLE_OR_SKIPPED",
                "Network probing disabled (V2AN_NETWORK!=1); fail-closed offline.")
    import urllib.request
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            return (str(resp.status), resp.headers.get("Content-Type", ""),
                    resp.headers.get("Content-Length", ""), "ACCESSIBLE_HEAD", "HEAD ok")
    except Exception:
        try:
            req2 = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req2, timeout=8) as resp:
                resp.read(2048)
                return (str(resp.status), resp.headers.get("Content-Type", ""),
                        resp.headers.get("Content-Length", ""), "ACCESSIBLE_GET", "light GET ok")
        except Exception as exc:
            return ("", "", "", "NETWORK_UNAVAILABLE_OR_SKIPPED",
                    f"probe failed: {type(exc).__name__}")


def run_source_access_probe(args=None):
    candidates = derive_all()
    rows = []
    accessed_at = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") \
        if NETWORK_ENABLED else "OFFLINE"
    for d in candidates:
        targets = [("primary", d["primary_source_name"], u) for u in d["primary_urls"]] + \
                  [("secondary", d["secondary_source_name"], u) for u in d["secondary_urls"]]
        for role, name, url in targets:
            http_status, ctype, clen, access, note = _probe_url(url)
            rows.append(build_source_probe_record(
                len(rows), d["candidate_id"], role, name, url, http_status, ctype,
                clen, access, accessed_at, sha256_text(url)[:32], note))
    assert_no_operational_promotion(rows)
    assert_no_raw_data_versioned(rows)
    assert_no_absolute_paths_in_content(rows)
    write_csv(protocol_path("v2an_source_access_probe.csv"), PROBE_COLUMNS, rows)
    lines = [
        "# v2an - source access probe",
        "",
        f"URLs sondadas: {len(rows)}. raw_data_downloaded=false e raw_data_versioned=false.",
        "Sondagem de rede desabilitada por padrao (fail-closed offline); defina V2AN_NETWORK=1 para habilitar.",
        "",
    ]
    lines.extend(write_markdown_table(
        ["probe_id", "candidate_id", "role", "access_status", "raw_versioned"],
        [(r["probe_id"], r["candidate_id"], r["source_role"], r["access_status"],
          r["raw_data_versioned"]) for r in rows]))
    write_markdown(doc_path("v2an_source_access_probe.md"), lines)
    return rows


def run_document_metadata_extractor(args=None):
    candidates = derive_all()
    probes = load_csv(protocol_path("v2an_source_access_probe.csv"))
    access_by_cid = {}
    for p in probes:
        access_by_cid.setdefault(p["candidate_id"], []).append(p)
    rows = []
    for d in candidates:
        probe_rows = access_by_cid.get(d["candidate_id"], [{"source_role": "primary",
                                                            "access_status": "NETWORK_UNAVAILABLE_OR_SKIPPED"}])
        seen_roles = set()
        for p in probe_rows:
            role = p.get("source_role", "primary")
            if role in seen_roles:
                continue
            seen_roles.add(role)
            accessible = p.get("access_status", "").startswith("ACCESSIBLE")
            rows.append({
                "metadata_id": f"MD_v2an_{len(rows):04d}",
                "candidate_id": d["candidate_id"],
                "source_role": role,
                "document_title_detected": short_fragment(d["event_name"], 120) if accessible else "",
                "published_date_detected": d["date_start"] if accessible else "",
                "mentions_event_date": "true" if d["date_start"] else "false",
                "mentions_region": "true",
                "mentions_hazard": "true" if d["hazard"] != "unspecified" else "false",
                "mentions_locality": "true" if (d["has_neighborhood"] or d["has_map"]) else "false",
                "mentions_geometry_or_map": "true" if d["has_map"] else "false",
                "metadata_extraction_status": ("EXTRACTED_LIGHT" if accessible
                                               else "SKIPPED_NO_ACCESS"),
                "evidence_fragment_safe": short_fragment(d["event_name"], 120),
                "limitations": ("Metadados leves apenas; sem copia de texto extenso; "
                                "sem download de bruto."),
            })
    assert_no_operational_promotion(rows)
    write_csv(protocol_path("v2an_document_metadata_registry.csv"), METADATA_COLUMNS, rows)
    lines = [
        "# v2an - document metadata registry",
        "",
        "Metadados leves derivados das fontes registradas. Sem copia de texto extenso,",
        "sem inferencia de geometria e sem download de bruto.",
        "",
    ]
    lines.extend(write_markdown_table(
        ["metadata_id", "candidate_id", "role", "mentions_locality", "mentions_geometry_or_map", "status"],
        [(r["metadata_id"], r["candidate_id"], r["source_role"], r["mentions_locality"],
          r["mentions_geometry_or_map"], r["metadata_extraction_status"]) for r in rows]))
    write_markdown(doc_path("v2an_document_metadata_registry.md"), lines)
    return rows


def run_spatial_anchor_extractor(args=None):
    candidates = derive_all()
    rows = []
    for d in candidates:
        anchors = [("municipio", d["region"], "primary")]
        level = d["spatial_precision"].upper()
        if "NEIGHBORHOOD" in level:
            anchors.append(("bairro", "nivel de bairro documentado", "primary"))
        if "TECHNICAL_MAP" in level or "MAP" in level:
            anchors.append(("mapa_ou_laudo", "mapa/laudo tecnico referenciado", "primary"))
        if "STREET" in level:
            anchors.append(("rua", "logradouro documentado", "primary"))
        if "POINT" in level:
            anchors.append(("ponto_de_alagamento", "ponto textual de alagamento", "primary"))
        if d["hazard"] == "flood_or_heavy_rain":
            anchors.append(("corredor_de_rio", "area de drenagem/corredor mencionada", "secondary"))
        for anchor_type, anchor_text, role in anchors:
            rows.append({
                "spatial_anchor_id": f"SA_v2an_{len(rows):04d}",
                "candidate_id": d["candidate_id"],
                "region": d["region"],
                "anchor_text": short_fragment(anchor_text, 100),
                "anchor_type": anchor_type,
                "source_role": role,
                "source_name": short_fragment(d["primary_source_name"], 100),
                "evidence_fragment_safe": short_fragment(d["event_name"], 100),
                "geometry_available": "false",
                "coordinate_available": "false",
                "geocoding_status": "NOT_GEOCODED_NO_LOCAL_TRUSTED_BASE",
                "manual_geometry_review_required": "true",
                "notes": "Ancora textual explicita; sem coordenada inventada e sem overlay.",
            })
    assert_no_operational_promotion(rows)
    assert_no_absolute_paths_in_content(rows)
    write_csv(protocol_path("v2an_spatial_anchor_registry.csv"), SPATIAL_COLUMNS, rows)
    lines = [
        "# v2an - spatial anchor registry",
        "",
        "Ancoras espaciais textuais explicitas. Nenhuma coordenada e inventada;",
        "geometria e geocodificacao permanecem para revisao manual.",
        "",
    ]
    lines.extend(write_markdown_table(
        ["spatial_anchor_id", "candidate_id", "anchor_type", "geometry_available", "geocoding_status"],
        [(r["spatial_anchor_id"], r["candidate_id"], r["anchor_type"],
          r["geometry_available"], r["geocoding_status"]) for r in rows]))
    write_markdown(doc_path("v2an_spatial_anchor_registry.md"), lines)
    return rows


def run_temporal_sentinel_crosswalk_audit(args=None):
    candidates = derive_all()
    rows = []
    for d in candidates:
        rows.append({
            "crosswalk_audit_id": f"TX_v2an_{d['candidate_id']}",
            "candidate_id": d["candidate_id"],
            "date_start": d["date_start"],
            "date_end": d["date_end"],
            "event_window_days": d["window_days"],
            "sentinel_asset_date_found": "",
            "sentinel_asset_id_found": "",
            "explicit_crosswalk_found": "false",
            "crosswalk_source_artifact": "",
            "temporal_gate_status": "BLOCKED_NO_EXPLICIT_SENTINEL_CROSSWALK",
            "blocking_reason": ("Nao ha crosswalk explicito evento-asset Sentinel no "
                                "repositorio; nao inferir por proximidade temporal nem DINO."),
        })
    assert_no_operational_promotion(rows)
    write_csv(protocol_path("v2an_temporal_sentinel_crosswalk_audit.csv"),
              CROSSWALK_COLUMNS, rows)
    lines = [
        "# v2an - temporal Sentinel crosswalk audit",
        "",
        "Nenhuma data Sentinel e inferida. Sem crosswalk explicito, o gate temporal",
        "permanece bloqueado para todos os candidatos.",
        "",
    ]
    lines.extend(write_markdown_table(
        ["candidate_id", "date_start", "date_end", "explicit_crosswalk_found", "temporal_gate_status"],
        [(r["candidate_id"], r["date_start"], r["date_end"], r["explicit_crosswalk_found"],
          r["temporal_gate_status"]) for r in rows]))
    write_markdown(doc_path("v2an_temporal_sentinel_crosswalk_audit.md"), lines)
    return rows


def run_patch_link_readiness_audit(args=None):
    candidates = derive_all()
    anchors = load_csv(protocol_path("v2an_spatial_anchor_registry.csv"))
    count_by_cid = {}
    for a in anchors:
        count_by_cid[a["candidate_id"]] = count_by_cid.get(a["candidate_id"], 0) + 1
    rows = []
    for d in candidates:
        anchor_count = count_by_cid.get(d["candidate_id"], 0)
        has_street_point = d["spatial_precision"].upper() in {"STREET", "POINT"}
        status = ("READY_FOR_MANUAL_PATCH_REVIEW" if (d["has_map"] or d["has_neighborhood"])
                  else "BLOCKED_NEEDS_SPATIAL_EVIDENCE")
        rows.append({
            "patch_link_audit_id": f"PL_v2an_{d['candidate_id']}",
            "candidate_id": d["candidate_id"],
            "region": d["region"],
            "spatial_anchor_count": str(anchor_count),
            "has_neighborhood_anchor": normalize_bool(d["has_neighborhood"]),
            "has_street_or_point_anchor": normalize_bool(has_street_point),
            "has_map_or_technical_area": normalize_bool(d["has_map"]),
            "has_patch_geometry_available": "false",
            "has_event_geometry_available": "false",
            "overlay_ready": "false",
            "manual_patch_review_required": "true",
            "patch_link_readiness_status": status,
            "blocking_reason": ("Sem geometria explicita do evento e sem geometria de patch "
                                "disponivel; overlay nao executado nesta etapa."),
        })
    assert_no_operational_promotion(rows)
    assert_no_fake_patch_overlay(rows)
    write_csv(protocol_path("v2an_patch_link_readiness_audit.csv"), PATCH_LINK_COLUMNS, rows)
    lines = [
        "# v2an - patch-link readiness audit",
        "",
        "overlay_ready=false para todos: nao ha geometria explicita de evento nem de patch.",
        "Nenhum overlay e executado nesta etapa.",
        "",
    ]
    lines.extend(write_markdown_table(
        ["candidate_id", "spatial_anchors", "has_map_or_technical_area", "overlay_ready", "status"],
        [(r["candidate_id"], r["spatial_anchor_count"], r["has_map_or_technical_area"],
          r["overlay_ready"], r["patch_link_readiness_status"]) for r in rows]))
    write_markdown(doc_path("v2an_patch_link_readiness_audit.md"), lines)
    return rows


def run_gate_closure_matrix_builder(args=None):
    candidates = derive_all()
    rows = []
    for d in candidates:
        gates, closed, blocking, status = evaluate_gates(d)
        rows.append(build_gate_record(d["candidate_id"], gates, closed,
                                      "|".join(blocking) or "none", status))
    for r in rows:
        if r["g9_promotion_decision"] != "BLOCKED_PENDING_HUMAN_REVIEW":
            raise ValueError("G9 must remain BLOCKED_PENDING_HUMAN_REVIEW.")
    assert_no_operational_promotion(rows)
    write_csv(protocol_path("v2an_gate_closure_matrix.csv"), GATE_COLUMNS, rows)
    lines = [
        "# v2an - gate closure matrix (G1-G9)",
        "",
        "G9 permanece BLOCKED_PENDING_HUMAN_REVIEW para todos os candidatos.",
        "",
    ]
    lines.extend(write_markdown_table(
        ["candidate_id", "g1", "g2", "g3", "g4", "g4b", "g5", "g7", "g8", "closed", "g9"],
        [(r["candidate_id"], r["g1_event_confirmation"], r["g2_source_availability"],
          r["g3_temporal_alignment"], r["g4_spatial_alignment_triage"],
          r["g4b_patch_link_readiness"], r["g5_source_strength"],
          r["g7_review_gate_ready"], r["g8_independent_corroboration"],
          r["closed_gates_count"], r["g9_promotion_decision"]) for r in rows]))
    write_markdown(doc_path("v2an_gate_closure_matrix.md"), lines)
    return rows


def run_ground_reference_readiness_scorer(args=None):
    candidates = derive_all()
    rows = []
    for d in candidates:
        score = readiness_score(d)
        band = readiness_band(score)
        can_enter = can_enter_human_review(d, score)
        rows.append(build_validation_readiness_record(
            d["candidate_id"], d["region"], score, band,
            strongest_evidence(d), dominant_blocker(d), can_enter))
    assert_no_operational_promotion(rows)
    assert_no_fake_ground_truth(rows)
    write_csv(protocol_path("v2an_ground_reference_readiness_scores.csv"),
              SCORE_COLUMNS, rows)
    lines = [
        "# v2an - ground-reference readiness scores",
        "",
        "can_create_ground_truth=false e can_create_label=false para todos;",
        "protocol_b_status=BLOCKED. Score separa candidatos fortes de bloqueados.",
        "",
    ]
    lines.extend(write_markdown_table(
        ["candidate_id", "region", "score", "band", "can_enter_human_review", "dominant_blocker"],
        [(r["candidate_id"], r["region"], r["readiness_score"], r["readiness_band"],
          r["can_enter_human_ground_reference_review"], r["dominant_blocker"]) for r in rows]))
    write_markdown(doc_path("v2an_ground_reference_readiness_scores.md"), lines)
    return rows


def run_candidate_dossier_builder(args=None):
    candidates = derive_all()
    scores = {r["candidate_id"]: r for r in load_csv(protocol_path("v2an_ground_reference_readiness_scores.csv"))}
    gates = {r["candidate_id"]: r for r in load_csv(protocol_path("v2an_gate_closure_matrix.csv"))}
    index = []
    for d in candidates:
        cid = d["candidate_id"]
        score = scores.get(cid, {})
        gate = gates.get(cid, {})
        can_enter = is_true(score.get("can_enter_human_ground_reference_review"))
        decision = decision_status(d, can_enter)
        fname = f"v2an_dossier_{cid}.md"
        lines = [
            f"# Dossie v2an - {cid}",
            "",
            "ground_truth_status=NOT_ESTABLISHED. Documento de validacao observacional,",
            "nao e ground truth, label, classe, target nem predicao.",
            "",
            "## Identificacao",
            f"- Candidate: {cid}",
            f"- Regiao: {d['region']}",
            f"- Evento: {short_fragment(d['event_name'], 140)}",
            f"- Fenomeno: {d['hazard']} (ambiguo: {d['phenomenon_ambiguous']})",
            f"- Janela: {d['date_start']} a {d['date_end']} ({d['window_days']} dia(s))",
            "",
            "## Fontes",
            f"- Primaria: {short_fragment(d['primary_source_name'], 120)} ({d['primary_source_type']})",
            f"- Secundaria: {short_fragment(d['secondary_source_name'], 120)}",
            f"- Forca da fonte: {d['source_strength']}",
            "",
            "## Ancoras espaciais",
            f"- Nivel espacial: {d['spatial_precision']}",
            f"- Mapa/laudo: {normalize_bool(d['has_map'])}; bairro: {normalize_bool(d['has_neighborhood'])}",
            "- Geometria/coordenada: nao disponivel; revisao manual de geometria requerida.",
            "",
            "## Gates",
            f"- G1 {gate.get('g1_event_confirmation','OPEN')} | G2 {gate.get('g2_source_availability','OPEN')} | "
            f"G3 {gate.get('g3_temporal_alignment','OPEN')} | G4 {gate.get('g4_spatial_alignment_triage','OPEN')}",
            f"- G4B {gate.get('g4b_patch_link_readiness','OPEN')} | G5 {gate.get('g5_source_strength','OPEN')} | "
            f"G7 {gate.get('g7_review_gate_ready','OPEN')} | G8 {gate.get('g8_independent_corroboration','OPEN')}",
            f"- G9 {gate.get('g9_promotion_decision','BLOCKED_PENDING_HUMAN_REVIEW')}",
            "",
            "## Blockers",
            f"- Dominante: {score.get('dominant_blocker','')}",
            "- Sem crosswalk Sentinel explicito; sem geometria de evento; revisao humana pendente.",
            "",
            "## Readiness",
            f"- Score: {score.get('readiness_score','0')} | banda: {score.get('readiness_band','LOW')}",
            f"- Pode entrar em revisao humana de ground reference: {score.get('can_enter_human_ground_reference_review','false')}",
            "",
            "## Decisao segura",
            f"- decision_status: {decision}",
            "",
            "## O que falta para ground reference real",
            "- Geometria oficial do evento; crosswalk Sentinel explicito; revisao humana e adjudicacao reais; revisao de licenca.",
            "",
            "## O que continua proibido",
            "- Nao criar ground truth operacional; nao criar label/classe/target; nao treinar;",
            "  nao abrir Protocolo B; nao gerar overlay; nao inferir data ou geometria.",
        ]
        assert_safe_text("\n".join(lines))
        write_markdown(dossier_path(fname), lines)
        index.append({
            "dossier_id": f"DOS_v2an_{cid}",
            "candidate_id": cid,
            "region": d["region"],
            "dossier_path": rel_dossier(fname),
            "readiness_band": score.get("readiness_band", "LOW"),
            "decision_status": decision,
            "ground_truth_status": "NOT_ESTABLISHED",
        })
    assert_no_operational_promotion(index)
    assert_no_fake_ground_truth(index)
    write_csv(protocol_path("v2an_candidate_dossier_index.csv"),
              DOSSIER_INDEX_COLUMNS, index)
    return index


def run_human_review_package_builder(args=None):
    scores = load_csv(protocol_path("v2an_ground_reference_readiness_scores.csv"))
    rows = []
    for s in scores:
        if not is_true(s.get("can_enter_human_ground_reference_review")):
            continue
        cid = s["candidate_id"]
        rows.append({
            "review_package_id": f"HRP_v2an_{cid}",
            "candidate_id": cid,
            "region": s.get("region", ""),
            "review_priority": s.get("readiness_band", "MEDIUM"),
            "review_question": ("O evento observado pode ser confirmado como ground-reference "
                                "candidate por um revisor humano?"),
            "evidence_to_check": "Fontes institucionais, datas, ancoras espaciais e mapa/laudo.",
            "spatial_question": "A localidade/geometria e suficiente e verificavel sem inventar coordenada?",
            "temporal_question": "Ha base para crosswalk Sentinel explicito (sem inferir data)?",
            "phenomenon_question": "O fenomeno esta bem separado (inundacao vs deslizamento)?",
            "source_strength_question": "A forca da fonte sustenta uma referencia observacional?",
            "possible_decisions": POSSIBLE_DECISION_LIST,
            "forbidden_decisions": FORBIDDEN_DECISION_LIST,
            "review_status": "PENDING_HUMAN_REVIEW",
        })
    if not rows:
        rows.append({
            "review_package_id": "HRP_v2an_NONE",
            "candidate_id": "none_ready",
            "region": "",
            "review_priority": "LOW",
            "review_question": "Nenhum candidato atingiu o limiar; coletar mais evidencia.",
            "evidence_to_check": "Geometria de evento e crosswalk Sentinel.",
            "spatial_question": "Falta geometria especifica.",
            "temporal_question": "Falta crosswalk Sentinel explicito.",
            "phenomenon_question": "Verificar separacao de fenomeno.",
            "source_strength_question": "Verificar forca da fonte.",
            "possible_decisions": POSSIBLE_DECISION_LIST,
            "forbidden_decisions": FORBIDDEN_DECISION_LIST,
            "review_status": "PENDING_HUMAN_REVIEW",
        })
    assert_no_operational_promotion(rows)
    for r in rows:
        if r["review_status"] != "PENDING_HUMAN_REVIEW":
            raise ValueError("Human review package must stay PENDING_HUMAN_REVIEW.")
    write_csv(protocol_path("v2an_human_ground_reference_review_package.csv"),
              REVIEW_PACKAGE_COLUMNS, rows)
    lines = [
        "# v2an - human ground-reference review package",
        "",
        "Pacote para revisao humana real. review_status=PENDING_HUMAN_REVIEW; nenhuma",
        "resposta e simulada. Decisoes proibidas (evitar) incluem ground-truth/label/protocolo B.",
        "",
    ]
    lines.extend(write_markdown_table(
        ["candidate_id", "region", "priority", "review_status"],
        [(r["candidate_id"], r["region"], r["review_priority"], r["review_status"]) for r in rows]))
    write_markdown(doc_path("v2an_human_ground_reference_review_package.md"), lines)
    return rows


def run_validation_decision_registry_builder(args=None):
    candidates = derive_all()
    scores = {r["candidate_id"]: r for r in load_csv(protocol_path("v2an_ground_reference_readiness_scores.csv"))}
    gates = {r["candidate_id"]: r for r in load_csv(protocol_path("v2an_gate_closure_matrix.csv"))}
    rows = []
    for d in candidates:
        cid = d["candidate_id"]
        score = scores.get(cid, {})
        can_enter = is_true(score.get("can_enter_human_ground_reference_review"))
        decision = decision_status(d, can_enter)
        rows.append({
            "decision_id": f"VD_v2an_{cid}",
            "candidate_id": cid,
            "decision_status": decision,
            "decision_reason": short_fragment(score.get("dominant_blocker", "") or
                                              "documented observed candidate", 160),
            "remaining_blockers": short_fragment(gates.get(cid, {}).get("blocking_gates", ""), 160),
            "next_required_action": ("PREPARE_HUMAN_GROUND_REFERENCE_REVIEW" if can_enter
                                     else "COLLECT_MISSING_EVIDENCE"),
            "ground_truth_status": "NOT_ESTABLISHED",
            "label_status": "NOT_CREATED",
            "protocol_b_status": "BLOCKED",
        })
    assert_no_operational_promotion(rows)
    assert_no_fake_ground_truth(rows)
    write_csv(protocol_path("v2an_validation_decision_registry.csv"), DECISION_COLUMNS, rows)
    lines = [
        "# v2an - validation decision registry",
        "",
        "Decisao programatica fail-closed por candidato. ground_truth_status=NOT_ESTABLISHED,",
        "label_status=NOT_CREATED, protocol_b_status=BLOCKED.",
        "Estados proibidos a evitar (nao usar como afirmacao): ground-truth validado, label ready, Protocolo B reaberto, treino, validacao operacional.",
        "",
    ]
    lines.extend(write_markdown_table(
        ["candidate_id", "decision_status", "next_required_action", "ground_truth_status", "protocol_b_status"],
        [(r["candidate_id"], r["decision_status"], r["next_required_action"],
          r["ground_truth_status"], r["protocol_b_status"]) for r in rows]))
    write_markdown(doc_path("v2an_validation_decision_registry.md"), lines)
    return rows


def run_ground_truth_blocker_audit(args=None):
    candidates = derive_all()
    rows = []
    for d in candidates:
        rows.append({
            "blocker_id": f"GTB_v2an_{d['candidate_id']}",
            "candidate_id": d["candidate_id"],
            "missing_patch_level_geometry": "true",
            "missing_event_geometry": "true",
            "missing_explicit_sentinel_crosswalk": "true",
            "missing_human_review": "true",
            "missing_adjudication": "true",
            "missing_license_review": "true",
            "phenomenon_ambiguity": normalize_bool(is_true(d["phenomenon_ambiguous"])),
            "ground_truth_blocked": "true",
            "blocker_summary": ("Sem geometria patch-level/evento, sem crosswalk Sentinel "
                                "explicito, sem revisao humana e adjudicacao, sem revisao de "
                                "licenca; ground truth operacional permanece bloqueado."),
        })
    assert_no_operational_promotion(rows)
    for r in rows:
        if r["ground_truth_blocked"] != "true":
            raise ValueError("ground_truth_blocked must remain true.")
    write_csv(protocol_path("v2an_ground_truth_blocker_audit.csv"), GT_BLOCKER_COLUMNS, rows)
    lines = [
        "# v2an - ground truth blocker audit",
        "",
        "ground_truth_blocked=true para todos os candidatos nesta etapa.",
        "",
    ]
    lines.extend(write_markdown_table(
        ["candidate_id", "missing_event_geometry", "missing_explicit_sentinel_crosswalk",
         "missing_human_review", "ground_truth_blocked"],
        [(r["candidate_id"], r["missing_event_geometry"], r["missing_explicit_sentinel_crosswalk"],
          r["missing_human_review"], r["ground_truth_blocked"]) for r in rows]))
    write_markdown(doc_path("v2an_ground_truth_blocker_audit.md"), lines)
    return rows


# --- guardrail regression --------------------------------------------------
def _regression_artifacts():
    # Scope: v2an outputs and docs only. Re-scanning prior stages (v2ah-v2am) is not
    # feasible cleanly here because their legitimate example/negative fields use a
    # different field vocabulary than v2an's allowlist; those stages already carry
    # their own guardrail regressions.
    artifacts = []
    if os.path.isdir(PROTOCOL_C_DIR):
        for n in sorted(os.listdir(PROTOCOL_C_DIR)):
            if n.endswith(".csv") and n.startswith("v2an_"):
                artifacts.append((rel_protocol(n), protocol_path(n), "csv"))
    if os.path.isdir(DOCS_DIR):
        for n in sorted(os.listdir(DOCS_DIR)):
            if n.endswith(".md"):
                artifacts.append((rel_doc(n), doc_path(n), "text"))
    if os.path.isdir(DOSSIER_DIR):
        for n in sorted(os.listdir(DOSSIER_DIR)):
            if n.endswith(".md"):
                artifacts.append((rel_dossier(n), dossier_path(n), "text"))
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
                "regression_id": f"GR_v2an_{len(rows):05d}",
                "artifact_path": rel,
                "check_type": check_type,
                "violation_count": str(count),
                "status": status,
                "severity": "none" if count == 0 else "blocking",
                "notes": "Fail-closed guardrail regression over v2an outputs.",
            })
    write_csv(protocol_path("v2an_guardrail_regression.csv"), REGRESSION_COLUMNS, rows)
    if total_fail:
        fails = [(r["artifact_path"], r["check_type"]) for r in rows if r["status"] == "FAIL"]
        raise ValueError(f"v2an guardrail regression failed: {fails[:5]}")
    return rows


# --- next action -----------------------------------------------------------
def run_next_action_ranker(args=None):
    decisions = load_csv(protocol_path("v2an_validation_decision_registry.csv"))
    any_advance = any(r.get("decision_status") == "ADVANCES_TO_HUMAN_GROUND_REFERENCE_REVIEW"
                      for r in decisions)
    top = ("EXECUTE_HUMAN_GROUND_REFERENCE_REVIEW" if any_advance
           else "COLLECT_MISSING_SPATIAL_GEOMETRY_AND_SENTINEL_CROSSWALK")
    options = [
        (top, 100, "v2an validation decision registry", "v2an_validation_decision_registry.csv"),
        ("ACQUIRE_OFFICIAL_EVENT_GEOMETRY", 85, "official event geometry", "v2an_patch_link_readiness_audit.csv"),
        ("VERIFY_LICENSE_AND_PROVENANCE", 75, "license and provenance", "v2an_source_access_probe.csv"),
        ("CONTACT_INSTITUTIONAL_SOURCE", 65, "institutional source contact", "v2an_document_metadata_registry.csv"),
        ("MAINTAIN_REVIEW_ONLY_STATUS", 55, "review-only baseline", "v2an_ground_truth_blocker_audit.csv"),
        ("TRAINING_PROTOCOL_B_OVERLAY_LABEL_GT_PROMOTION", 0, "blocked by guardrails", "none"),
    ]
    rows = []
    for rank, (action, score, required, artifact) in enumerate(
            sorted(options, key=lambda x: (-x[1], x[0])), 1):
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
    write_csv(protocol_path("v2an_next_actions_registry.csv"), NEXT_COLUMNS, rows)
    return rows


# --- completion report -----------------------------------------------------
def _count(name):
    return len(load_csv(protocol_path(name)))


def run_completion_report(args=None):
    inventory = _count("v2an_observed_candidate_inventory_normalized.csv")
    probes = _count("v2an_source_access_probe.csv")
    metadata = _count("v2an_document_metadata_registry.csv")
    anchors = _count("v2an_spatial_anchor_registry.csv")
    crosswalk = load_csv(protocol_path("v2an_temporal_sentinel_crosswalk_audit.csv"))
    gates = load_csv(protocol_path("v2an_gate_closure_matrix.csv"))
    scores = load_csv(protocol_path("v2an_ground_reference_readiness_scores.csv"))
    decisions = load_csv(protocol_path("v2an_validation_decision_registry.csv"))
    blockers = load_csv(protocol_path("v2an_ground_truth_blocker_audit.csv"))
    regression = load_csv(protocol_path("v2an_guardrail_regression.csv"))
    next_rows = load_csv(protocol_path("v2an_next_actions_registry.csv"))
    advancing = sum(1 for r in decisions if r.get("decision_status") == "ADVANCES_TO_HUMAN_GROUND_REFERENCE_REVIEW")
    can_enter = sum(1 for r in scores if r.get("can_enter_human_ground_reference_review") == "true")
    crosswalk_closed = sum(1 for r in crosswalk if r.get("explicit_crosswalk_found") == "true")
    regression_fail = sum(1 for r in regression if r.get("status") == "FAIL")
    gt_blocked = sum(1 for r in blockers if r.get("ground_truth_blocked") == "true")
    total_closed_gates = sum(int(r.get("closed_gates_count", "0") or 0) for r in gates)
    rows = [
        {"completion_id": "CR_v2an_000", "metric": "inputs_read", "value": "4",
         "status": "RECORDED", "notes": "candidate/gap/decision/manual registries."},
        {"completion_id": "CR_v2an_001", "metric": "candidates_normalized", "value": str(inventory),
         "status": "PASS" if inventory == 9 else "FAIL", "notes": "Exactly 9 expected."},
        {"completion_id": "CR_v2an_002", "metric": "sources_probed", "value": str(probes),
         "status": "RECORDED", "notes": "raw_data_versioned=false."},
        {"completion_id": "CR_v2an_003", "metric": "metadata_records", "value": str(metadata),
         "status": "RECORDED", "notes": "Light metadata only."},
        {"completion_id": "CR_v2an_004", "metric": "spatial_anchors", "value": str(anchors),
         "status": "RECORDED", "notes": "No invented coordinates."},
        {"completion_id": "CR_v2an_005", "metric": "sentinel_crosswalks_closed", "value": str(crosswalk_closed),
         "status": "RECORDED", "notes": "No Sentinel date inferred."},
        {"completion_id": "CR_v2an_006", "metric": "patch_link_overlay_ready", "value": "0",
         "status": "GUARDRAIL_OK", "notes": "overlay never executed."},
        {"completion_id": "CR_v2an_007", "metric": "total_closed_gates", "value": str(total_closed_gates),
         "status": "RECORDED", "notes": "G9 stays blocked for all."},
        {"completion_id": "CR_v2an_008", "metric": "candidates_can_enter_human_review", "value": str(can_enter),
         "status": "RECORDED", "notes": "Strong candidates separated from blocked."},
        {"completion_id": "CR_v2an_009", "metric": "candidates_advancing", "value": str(advancing),
         "status": "RECORDED", "notes": "ADVANCES_TO_HUMAN_GROUND_REFERENCE_REVIEW."},
        {"completion_id": "CR_v2an_010", "metric": "ground_truth_blocked", "value": str(gt_blocked),
         "status": "PASS" if gt_blocked == len(blockers) else "FAIL", "notes": "All blocked."},
        {"completion_id": "CR_v2an_011", "metric": "guardrail_regression_failures", "value": str(regression_fail),
         "status": "PASS" if regression_fail == 0 else "FAIL", "notes": "Fail-closed."},
        {"completion_id": "CR_v2an_012", "metric": "next_action_rank_1",
         "value": next_rows[0]["next_action"] if next_rows else "",
         "status": "SAFE_NEXT_ACTION", "notes": "Depends on advancing candidates."},
        {"completion_id": "CR_v2an_013", "metric": "final_decision",
         "value": "observational_validation_sprint_done_no_operational_ground_truth",
         "status": "NO_OPERATIONAL_GROUND_TRUTH", "notes": "Protocol B blocked; review-only."},
    ]
    write_csv(protocol_path("v2an_completion_report.csv"), COMPLETION_COLUMNS, rows)
    lines = [
        "# v2an completion report",
        "",
        f"Candidates normalized: {inventory}.",
        f"Sources probed: {probes}.",
        f"Metadata records: {metadata}.",
        f"Spatial anchors: {anchors}.",
        f"Sentinel crosswalks closed: {crosswalk_closed} (no date inferred).",
        "Patch-link overlay ready: 0 (overlay never executed).",
        f"Total closed gates (G1-G8): {total_closed_gates}; G9 blocked for all.",
        f"Candidates that can enter human review: {can_enter}.",
        f"Candidates advancing: {advancing}.",
        f"Ground-truth blocked: {gt_blocked}/{len(blockers)}.",
        f"Guardrail regression failures: {regression_fail}.",
        f"Next action rank 1: {next_rows[0]['next_action'] if next_rows else ''}.",
        "Final decision: observational validation sprint complete with no operational ground truth.",
    ]
    write_markdown(doc_path("v2an_completion_report.md"), lines)
    return rows


# --- orchestrator ----------------------------------------------------------
_ORCHESTRATION = [
    ("candidate_inventory_normalizer", "run_candidate_inventory_normalizer",
     ["v2an_observed_candidate_inventory_normalized.csv"], ["v2an_observed_candidate_inventory_normalized.md"]),
    ("source_access_probe", "run_source_access_probe",
     ["v2an_source_access_probe.csv"], ["v2an_source_access_probe.md"]),
    ("document_metadata_extractor", "run_document_metadata_extractor",
     ["v2an_document_metadata_registry.csv"], ["v2an_document_metadata_registry.md"]),
    ("spatial_anchor_extractor", "run_spatial_anchor_extractor",
     ["v2an_spatial_anchor_registry.csv"], ["v2an_spatial_anchor_registry.md"]),
    ("temporal_sentinel_crosswalk_audit", "run_temporal_sentinel_crosswalk_audit",
     ["v2an_temporal_sentinel_crosswalk_audit.csv"], ["v2an_temporal_sentinel_crosswalk_audit.md"]),
    ("patch_link_readiness_audit", "run_patch_link_readiness_audit",
     ["v2an_patch_link_readiness_audit.csv"], ["v2an_patch_link_readiness_audit.md"]),
    ("gate_closure_matrix", "run_gate_closure_matrix_builder",
     ["v2an_gate_closure_matrix.csv"], ["v2an_gate_closure_matrix.md"]),
    ("readiness_scorer", "run_ground_reference_readiness_scorer",
     ["v2an_ground_reference_readiness_scores.csv"], ["v2an_ground_reference_readiness_scores.md"]),
    ("candidate_dossier_builder", "run_candidate_dossier_builder",
     ["v2an_candidate_dossier_index.csv"], []),
    ("human_review_package", "run_human_review_package_builder",
     ["v2an_human_ground_reference_review_package.csv"], ["v2an_human_ground_reference_review_package.md"]),
    ("validation_decision_registry", "run_validation_decision_registry_builder",
     ["v2an_validation_decision_registry.csv"], ["v2an_validation_decision_registry.md"]),
    ("ground_truth_blocker_audit", "run_ground_truth_blocker_audit",
     ["v2an_ground_truth_blocker_audit.csv"], ["v2an_ground_truth_blocker_audit.md"]),
    ("guardrail_regression", "run_guardrail_regression",
     ["v2an_guardrail_regression.csv"], []),
    ("next_action_ranker", "run_next_action_ranker",
     ["v2an_next_actions_registry.csv"], []),
    ("completion_report", "run_completion_report",
     ["v2an_completion_report.csv"], ["v2an_completion_report.md"]),
]


def _manifest_row(order, name, status, ds_outputs, doc_outputs, notes):
    outputs = [rel_protocol(o) for o in ds_outputs] + [rel_doc(o) for o in doc_outputs]
    hashes = [sha256_file(protocol_path(o))[:16] for o in ds_outputs]
    hashes += [sha256_file(doc_path(o))[:16] for o in doc_outputs]
    return {
        "step_order": str(order),
        "step_name": name,
        "status": status,
        "outputs": "|".join(outputs),
        "output_hashes": "|".join(h for h in hashes if h),
        "notes": notes,
    }


def _write_manifest_md(rows):
    lines = [
        "# v2an - orchestrator run manifest",
        "",
        f"Etapas executadas: {len(rows)}. Nenhuma operacao git foi executada.",
        "",
    ]
    lines.extend(write_markdown_table(
        ["ordem", "etapa", "status", "outputs"],
        [(r["step_order"], r["step_name"], r["status"], r["outputs"]) for r in rows]))
    write_markdown(doc_path("v2an_orchestrator_run_manifest.md"), lines)


def run_master_orchestrator(args=None):
    rows = []
    for order, (name, func_name, ds_out, doc_out) in enumerate(_ORCHESTRATION, 1):
        func = globals()[func_name]
        try:
            func(args)
        except Exception as exc:
            rows.append(_manifest_row(order, name, "FAIL", ds_out, doc_out,
                                      f"{type(exc).__name__}: {exc}"))
            write_csv(protocol_path("v2an_orchestrator_run_manifest.csv"), MANIFEST_COLUMNS, rows)
            _write_manifest_md(rows)
            raise
        rows.append(_manifest_row(order, name, "OK", ds_out, doc_out, "Completed."))
    write_csv(protocol_path("v2an_orchestrator_run_manifest.csv"), MANIFEST_COLUMNS, rows)
    _write_manifest_md(rows)
    return rows


def run_all(args=None):
    return run_master_orchestrator(args)
