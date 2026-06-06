#!/usr/bin/env python3
"""v2am TCC Appendix & Evidence Atlas Export builder.

Aggregates the read-only artifacts of v2ah, v2ai, v2aj, v2ak and v2al into a safe,
traceable and navigable Evidence Atlas / appendix / defense package for the Protocolo C.

This stage only writes ``v2am_*`` artifacts. It never overwrites the main manuscript,
never modifies prior outputs, and never creates operational ground truth, ground
reference, labels, classes, targets, training, overlay, prediction, inferred Sentinel
dates, inferred crosswalks, fake human review or fake adjudication.
"""

import argparse
import csv
import hashlib
import json
import os
import re

PROTOCOL_VERSION = "v2am"
DATASET_DIR = os.environ.get("DATASET_DIR", "datasets/protocolo_c")
DOCS_DIR = os.environ.get("DOCS_DIR", "docs/tcc_exports")
ATLAS_DIR = os.environ.get(
    "ATLAS_DIR", "docs/tcc_exports/v2am_appendix_evidence_atlas")
V2AL_INTEGRATION_DIR = os.environ.get(
    "V2AL_INTEGRATION_DIR", "docs/tcc_exports/v2al_manuscript_integration")
CONFIG_DIR = os.environ.get("CONFIG_DIR", "configs/protocolo_c")

STAGE_PREFIXES = ("v2ah", "v2ai", "v2aj", "v2ak", "v2al")

# --- minimal required artifacts per stage (fail-closed readiness) ----------
MIN_REQUIRED_DATASETS = {
    "v2ah": ["v2ah_candidate_reference_review_queue.csv",
             "v2ah_ground_truth_search_stop_gate.csv"],
    "v2ai": ["v2ai_review_assignment_registry.csv",
             "v2ai_adjudication_queue.csv",
             "v2ai_safe_promotion_blockers.csv"],
    "v2aj": ["v2aj_tcc_protocol_c_claims_matrix.csv",
             "v2aj_methodological_limitations_export.csv",
             "v2aj_results_tables_export_registry.csv"],
    "v2ak": ["v2ak_safe_language_glossary.csv",
             "v2ak_claim_usage_audit.csv"],
    "v2al": ["v2al_table_caption_export.csv",
             "v2al_section_insertion_matrix.csv"],
}
MIN_REQUIRED_DOCS = {
    "v2ak": ["protocolo_c_v2ak_metodologia_draft.md"],
}
MIN_SCHEMAS = {
    "v2ah_candidate_reference_review_queue.csv":
        ["review_queue_id", "package_id", "region", "candidate_status"],
    "v2ah_ground_truth_search_stop_gate.csv":
        ["stop_gate_id", "ground_truth_search_status"],
    "v2ai_review_assignment_registry.csv":
        ["assignment_id", "package_id", "assignment_status"],
    "v2ai_adjudication_queue.csv":
        ["adjudication_id", "package_id", "adjudication_status"],
    "v2ai_safe_promotion_blockers.csv":
        ["package_id", "promotion_status", "promotion_allowed"],
    "v2aj_tcc_protocol_c_claims_matrix.csv":
        ["claim_id", "claim_allowed", "safe_wording", "unsafe_wording"],
    "v2aj_methodological_limitations_export.csv":
        ["limitation_id", "limitation_name", "safe_tcc_wording"],
    "v2aj_results_tables_export_registry.csv":
        ["table_id", "suggested_title", "safe_caption"],
    "v2ak_safe_language_glossary.csv": ["term", "status"],
    "v2al_table_caption_export.csv":
        ["caption_id", "source_table_id", "safe_caption"],
    "v2al_section_insertion_matrix.csv":
        ["insertion_id", "v2ak_source_draft", "target_tcc_section"],
}

# --- guardrail vocabulary --------------------------------------------------
FORBIDDEN_TRUE_FIELDS = {
    "ground_truth_created", "ground_reference_created", "label_created",
    "training_ready", "overlay_ready", "prediction_ready",
    "operational_claims", "human_review_completed",
    "adjudication_completed", "promotion_allowed", "promotion_created",
    "auto_insert", "safe_to_autowrite",
}
FORBIDDEN_STATUS_VALUES = {
    "GROUND_TRUTH_VALIDATED", "GROUND_REFERENCE_TRUE", "LABEL_POSITIVE",
    "LABEL_NEGATIVE", "TRAINING_READY", "PROTOCOL_B_OPEN",
    "OPERATIONAL_VALIDATION", "PATCH_POSITIVE", "PATCH_NEGATIVE",
    "FLOOD_DETECTED", "REVIEW_COMPLETED", "ADJUDICATION_COMPLETED",
    "PROMOTION_ALLOWED", "PROMOTED",
}
FORBIDDEN_KV_MARKERS = [
    "ground_truth=true", "ground_reference=true", "label=true",
    "training=true", "overlay=true", "prediction=true",
    "protocol_b_reopen=true", "sentinel_date_inferred=true",
    "crosswalk_inferred=true", "human_review_completed=true",
    "adjudication_completed=true", "operational_validation=true",
    "promotion_allowed=true", "promotion_created=true",
]
UNSAFE_LANGUAGE = [
    "ground truth validado",
    "classe positiva",
    "classe negativa",
    "label operacional",
    "deteccao de enchente",
    "deteccao de inundacao",
    "predicao de inundacao",
    "modelo preditivo",
    "validacao operacional",
    "treinamento supervisionado pronto",
]
SAFE_UNSAFE_FIELDS = {
    "unsafe_wording", "unsafe_tcc_wording", "unsafe_caption", "claim_text",
    "claim_fragment", "claim_status", "claim_category", "status", "term",
    "reason", "safe_alternative", "example_sentence", "violation_reason",
    "forbidden_caption", "safe_caption", "safe_title", "forbidden_interpretation",
    "allowed_interpretation", "safe_summary", "risk_if_inserted_without_review",
    "notes", "manual_action", "required_human_check", "question", "content",
    "decision_point", "sensitive_point", "recommended_action", "forbidden_use",
    "allowed_use", "safe_use", "forbidden_terms", "safe_terms",
    "what_it_does_not_imply", "what_it_does_not_mean", "banca_risk_if_wrong",
    "risk_if_answered_wrong", "why_it_matters", "short_answer",
    "technical_answer", "evidence_to_show", "summary", "safe_explanation",
    "safe_wording", "recommended_fix", "label", "purpose", "role_in_appendix",
}
SAFE_CONTEXT_MARKERS = [
    "nao pode dizer", "nao usar", "nao afirmar", "nao ha", "nao deve",
    "nao temos", "nao realiza", "nao detecta", "nao cria", "nao existe",
    "nao produz", "nao significa", "nao implica", "nao e ", "nao foi",
    "nao ", "proibido", "prohibited", "forbidden", "limitation", "limitacao",
    "does not", "do not", "not ", "no ", "sem ", "evitar", "ausencia",
    "trocar", "substituir", "exemplo negativo", "unsafe", "bloque", "blocked",
    "pendente", "candidato", "review-only", "review only",
]
ABSOLUTE_PATH_RE = re.compile(r"(?:[A-Za-z]:\\|/Users/|/home/|/mnt/|\\\\)")
LOCAL_ONLY_MARKER = "local" + "_" + "only"

# --- column schemas --------------------------------------------------------
ARTIFACT_INDEX_COLUMNS = [
    "artifact_id", "stage", "artifact_type", "path", "exists", "non_empty",
    "sha256", "row_count", "column_count", "schema_status", "role_in_appendix",
    "safe_use", "forbidden_use",
]
ATLAS_COLUMNS = [
    "atlas_item_id", "axis", "summary", "source_artifacts", "evidence_status",
    "allowed_interpretation", "forbidden_interpretation",
    "recommended_appendix_location", "banca_question_answered",
]
DAG_NODE_COLUMNS = [
    "node_id", "stage", "label", "artifact_path", "node_type",
    "claim_safety_status",
]
DAG_EDGE_COLUMNS = [
    "edge_id", "source_node", "target_node", "relationship",
    "guardrail_preserved", "promotion_created",
]
CATALOG_COLUMNS = [
    "catalog_id", "item_type", "suggested_title", "source_artifacts",
    "recommended_location", "safe_caption", "forbidden_caption",
    "allowed_interpretation", "forbidden_interpretation", "priority",
    "manual_review_required",
]
CLAIMS_GUARDRAILS_COLUMNS = [
    "registry_id", "claim_or_guardrail", "status", "safe_wording",
    "unsafe_wording", "source_artifact", "why_it_matters",
    "banca_risk_if_wrong",
]
REVIEW_QUEUE_COLUMNS = [
    "review_item_id", "package_group", "candidate_count", "assignment_count",
    "review_status", "adjudication_status", "promotion_status", "allowed_use",
    "forbidden_use",
]
LIMITATIONS_COLUMNS = [
    "limitation_item_id", "limitation_name", "safe_explanation",
    "what_it_does_not_imply", "mitigation", "future_work", "source_artifact",
    "recommended_section",
]
DEFENSE_COLUMNS = [
    "question_id", "question", "short_answer", "technical_answer",
    "evidence_to_show", "risk_if_answered_wrong", "safe_terms",
    "forbidden_terms",
]
FINAL_AUDIT_COLUMNS = [
    "audit_id", "file", "claim_fragment", "claim_category", "allowed",
    "context_is_safe", "violation", "violation_reason", "recommended_fix",
]
APPENDIX_INDEX_COLUMNS = [
    "index_item_id", "appendix_file", "title", "purpose", "source_artifacts",
    "recommended_reader", "review_priority",
]
MANIFEST_COLUMNS = [
    "step_order", "step_name", "status", "outputs", "output_hashes", "notes",
]
REGRESSION_COLUMNS = [
    "regression_id", "artifact_path", "check_type", "violation_count",
    "status", "severity", "notes",
]
NEXT_COLUMNS = [
    "rank", "next_action", "score", "allowed", "blocked_operational_use",
    "required_input", "recommended_script_or_artifact", "notes",
]
COMPLETION_COLUMNS = ["completion_id", "metric", "value", "status", "notes"]


# --- argument parsing ------------------------------------------------------
def parse_args(argv=None):
    return argparse.ArgumentParser().parse_args(argv)


# --- path helpers ----------------------------------------------------------
def dataset_path(name):
    return os.path.join(DATASET_DIR, name)


def doc_path(name):
    return os.path.join(DOCS_DIR, name)


def atlas_path(name):
    return os.path.join(ATLAS_DIR, name)


def integration_path(name):
    return os.path.join(V2AL_INTEGRATION_DIR, name)


def rel_dataset(name):
    return f"datasets/protocolo_c/{name}"


def rel_doc(name):
    return f"docs/tcc_exports/{name}"


def rel_atlas(name):
    return f"docs/tcc_exports/v2am_appendix_evidence_atlas/{name}"


def rel_integration(name):
    return f"docs/tcc_exports/v2al_manuscript_integration/{name}"


def repo_relative_path(path):
    """Normalise to a repo-relative, forward-slash path; never absolute."""
    raw = str(path).replace("\\", "/")
    if ABSOLUTE_PATH_RE.search(str(path)):
        raise ValueError(f"Refusing absolute path: {path}")
    return raw


# --- value helpers ---------------------------------------------------------
def clean(value):
    return str(value or "").strip()


def is_true(value):
    """Fail-closed boolean: only the exact token ``true`` counts as true."""
    return clean(value).lower() == "true"


def normalize_bool(value):
    return "true" if is_true(value) else "false"


def normalize_status(value, allowed, default="UNKNOWN_FAIL_CLOSED"):
    v = clean(value).upper()
    return v if v in {a.upper() for a in allowed} else default


def safe_slug(text):
    return re.sub(r"[^a-z0-9]+", "-", clean(text).lower()).strip("-") or "item"


# --- io helpers ------------------------------------------------------------
def load_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def csv_shape(path):
    if not os.path.exists(path):
        return 0, 0
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        return 0, 0
    return len(rows) - 1, len(rows[0])


def write_csv(path, columns, rows):
    assert_no_manuscript_overwrite(path)
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
    assert_no_manuscript_overwrite(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_json(path, payload):
    assert_no_manuscript_overwrite(path)
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
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_markdown_table(headers, rows):
    """Return a deterministic GitHub-flavoured markdown table as a list of lines."""
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        cells = [clean(c).replace("|", "\\|").replace("\n", " ") for c in row]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


# --- schema / readiness ----------------------------------------------------
def assert_min_schema(rows, required, artifact):
    if not rows:
        raise FileNotFoundError(f"Required artifact is missing or empty: {artifact}")
    missing = [c for c in required if c not in rows[0]]
    if missing:
        raise ValueError(f"{artifact} missing required columns: {','.join(missing)}")
    return True


def assert_stage_artifacts_ready():
    missing = []
    for names in MIN_REQUIRED_DATASETS.values():
        for name in names:
            if not os.path.exists(dataset_path(name)):
                missing.append(rel_dataset(name))
    for names in MIN_REQUIRED_DOCS.values():
        for name in names:
            if not os.path.exists(doc_path(name)):
                missing.append(rel_doc(name))
    if missing:
        raise FileNotFoundError(
            "v2am requires v2ah-v2al; missing minimal artifacts: " + ",".join(missing))
    for name, cols in MIN_SCHEMAS.items():
        p = dataset_path(name)
        if os.path.exists(p):
            assert_min_schema(load_csv(p), cols, name)
    return True


def load_stage_inventory():
    """Discover read-only stage artifacts (datasets, docs, configs) repo-relative."""
    inventory = []
    if os.path.isdir(DATASET_DIR):
        for n in sorted(os.listdir(DATASET_DIR)):
            if n.endswith(".csv") and n.split("_", 1)[0] in STAGE_PREFIXES:
                inventory.append((_stage_of(n), "dataset_csv", rel_dataset(n), dataset_path(n)))
    if os.path.isdir(DOCS_DIR):
        for n in sorted(os.listdir(DOCS_DIR)):
            if n.endswith((".md", ".tex")) and any(p in n for p in STAGE_PREFIXES):
                inventory.append((_stage_of(n), "doc_text", rel_doc(n), doc_path(n)))
    if os.path.isdir(V2AL_INTEGRATION_DIR):
        for n in sorted(os.listdir(V2AL_INTEGRATION_DIR)):
            if n.endswith((".md", ".tex")):
                inventory.append(("v2al", "doc_text", rel_integration(n), integration_path(n)))
    if os.path.isdir(CONFIG_DIR):
        for n in sorted(os.listdir(CONFIG_DIR)):
            if n.endswith(".yaml") and n.split("_", 1)[0] in STAGE_PREFIXES:
                inventory.append((_stage_of(n), "config_yaml", f"configs/protocolo_c/{n}", os.path.join(CONFIG_DIR, n)))
    return inventory


def _stage_of(name):
    head = name.split("_", 1)[0]
    if head in STAGE_PREFIXES:
        return head
    for p in STAGE_PREFIXES:
        if p in name:
            return p
    return "unknown"


# --- guardrail assertions --------------------------------------------------
def _field_allows_unsafe(key, value):
    key_l = clean(key).lower()
    value_l = clean(value).lower()
    return key_l in SAFE_UNSAFE_FIELDS or any(m in value_l for m in SAFE_CONTEXT_MARKERS)


def assert_no_absolute_paths_in_content(rows_or_text):
    items = rows_or_text if isinstance(rows_or_text, list) else [{"_": rows_or_text}]
    for idx, row in enumerate(items):
        values = row.values() if isinstance(row, dict) else [row]
        for value in values:
            if ABSOLUTE_PATH_RE.search(clean(value)):
                raise ValueError(f"Absolute path in content at row {idx}: {value}")
    return True


def assert_no_local_only(rows_or_text):
    items = rows_or_text if isinstance(rows_or_text, list) else [{"_": rows_or_text}]
    for idx, row in enumerate(items):
        values = row.values() if isinstance(row, dict) else [row]
        for value in values:
            if LOCAL_ONLY_MARKER in clean(value).lower():
                raise ValueError(f"local_only marker in content at row {idx}: {value}")
    return True


def assert_no_operational_claim(rows):
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
        raise ValueError(f"Operational claim violation: {sample}")
    return True


def assert_no_fake_review(rows):
    violations = []
    for idx, row in enumerate(rows):
        for key, value in row.items():
            key_l = clean(key).lower()
            value_s = clean(value)
            if key_l in {"human_review_completed", "adjudication_completed"} and is_true(value_s):
                violations.append((idx, key))
            if value_s in {"REVIEW_COMPLETED", "ADJUDICATION_COMPLETED", "HUMAN_REVIEW_DONE"}:
                violations.append((idx, key))
    if violations:
        sample = "; ".join(f"row={r[0]} field={r[1]}" for r in violations[:5])
        raise ValueError(f"Fake review detected: {sample}")
    return True


def scan_text_violations(text):
    counts = {
        "absolute_path": 0, "local_only": 0, "forbidden_kv": 0,
        "unsafe_language": 0, "forbidden_true_flag": 0, "forbidden_status": 0,
    }
    for line in text.splitlines():
        line_l = line.lower()
        if ABSOLUTE_PATH_RE.search(line):
            counts["absolute_path"] += 1
        if LOCAL_ONLY_MARKER in line_l:
            counts["local_only"] += 1
        squashed = re.sub(r"\s*=\s*", "=", line_l)
        for marker in FORBIDDEN_KV_MARKERS:
            if marker in squashed:
                counts["forbidden_kv"] += 1
        # Match status enums case-sensitively (uppercase) so that legitimate
        # lowercase fields like "promotion_allowed=false" are not flagged.
        if any(s in line for s in FORBIDDEN_STATUS_VALUES):
            counts["forbidden_status"] += 1
        if ("operational_claims: true" in line_l or
                "ground_truth_created: true" in line_l or
                "promotion_created: true" in line_l):
            counts["forbidden_true_flag"] += 1
        safe_context = any(m in line_l for m in SAFE_CONTEXT_MARKERS)
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


def assert_no_manuscript_overwrite(path, manuscript_paths=None):
    """Only ``v2am_`` basenames may be written; protects the main manuscript."""
    raw = str(path)
    base = os.path.basename(raw)
    if LOCAL_ONLY_MARKER in raw.lower():
        raise ValueError(f"Refusing local_only output path: {raw}")
    if manuscript_paths:
        norm = raw.replace("\\", "/")
        for mp in manuscript_paths:
            if norm.endswith(str(mp).replace("\\", "/")):
                raise ValueError(f"Refusing to overwrite manuscript candidate: {raw}")
    if not base.startswith("v2am_"):
        raise ValueError(
            f"Refusing auto-write outside v2am outputs (would touch manuscript): {raw}")
    return True


# --- record constructors ---------------------------------------------------
def build_artifact_record(idx, stage, artifact_type, rel_path, full_path,
                          role, safe_use, forbidden_use):
    exists = os.path.exists(full_path)
    rel_path = repo_relative_path(rel_path)
    row_count, column_count = (0, 0)
    sha = ""
    non_empty = False
    schema_status = "absent"
    if exists:
        sha = sha256_file(full_path)
        if artifact_type == "dataset_csv":
            row_count, column_count = csv_shape(full_path)
            non_empty = row_count > 0
            base = os.path.basename(full_path)
            if base in MIN_SCHEMAS:
                rows = load_csv(full_path)
                missing = [c for c in MIN_SCHEMAS[base] if not rows or c not in rows[0]]
                schema_status = "schema_ok" if not missing else "schema_missing_cols"
            else:
                schema_status = "present_unverified"
        else:
            text = read_text(full_path)
            row_count = len(text.splitlines())
            non_empty = bool(text.strip())
            schema_status = "present_unverified"
    return {
        "artifact_id": f"ART_v2am_{idx:04d}",
        "stage": stage,
        "artifact_type": artifact_type,
        "path": rel_path,
        "exists": normalize_bool(exists),
        "non_empty": normalize_bool(non_empty),
        "sha256": sha,
        "row_count": str(row_count),
        "column_count": str(column_count),
        "schema_status": schema_status,
        "role_in_appendix": role,
        "safe_use": safe_use,
        "forbidden_use": forbidden_use,
    }


def build_claim_record(idx, claim_or_guardrail, status, safe_wording,
                       unsafe_wording, source_artifact, why_it_matters,
                       banca_risk):
    return {
        "registry_id": f"CG_v2am_{idx:04d}",
        "claim_or_guardrail": claim_or_guardrail,
        "status": status,
        "safe_wording": safe_wording,
        "unsafe_wording": unsafe_wording,
        "source_artifact": source_artifact,
        "why_it_matters": why_it_matters,
        "banca_risk_if_wrong": banca_risk,
    }


def build_dag_edge(idx, source_node, target_node, relationship):
    return {
        "edge_id": f"EDG_v2am_{idx:03d}",
        "source_node": source_node,
        "target_node": target_node,
        "relationship": relationship,
        "guardrail_preserved": "true",
        "promotion_created": "false",
    }


# --- shared counts ---------------------------------------------------------
def stage_counts():
    return {
        "candidates": len(load_csv(dataset_path("v2ah_candidate_reference_review_queue.csv"))),
        "assignments": len(load_csv(dataset_path("v2ai_review_assignment_registry.csv"))),
        "adjudications": len(load_csv(dataset_path("v2ai_adjudication_queue.csv"))),
        "blockers": len(load_csv(dataset_path("v2ai_safe_promotion_blockers.csv"))),
        "limitations": len(load_csv(dataset_path("v2aj_methodological_limitations_export.csv"))),
        "claims": len(load_csv(dataset_path("v2aj_tcc_protocol_c_claims_matrix.csv"))),
    }


# --- inventory -------------------------------------------------------------
_ROLE_BY_TYPE = {
    "dataset_csv": "appendix_data_table",
    "doc_text": "appendix_narrative_or_section",
    "config_yaml": "appendix_policy_reference",
}


def run_artifact_inventory_builder(args=None):
    assert_stage_artifacts_ready()
    rows = []
    for stage, art_type, rel_path, full_path in load_stage_inventory():
        role = _ROLE_BY_TYPE.get(art_type, "appendix_reference")
        record = build_artifact_record(
            len(rows), stage, art_type, rel_path, full_path, role,
            "Use as read-only appendix evidence and traceability source.",
            "Do not use as ground truth, label, training, overlay or prediction.")
        rows.append(record)
    assert_no_operational_claim(rows)
    assert_no_absolute_paths_in_content(rows)
    write_csv(dataset_path("v2am_appendix_artifact_index.csv"),
              ARTIFACT_INDEX_COLUMNS, rows)
    lines = [
        "# Protocolo C v2am - indice de artefatos do apendice",
        "",
        f"Inventariados {len(rows)} artefatos read-only de v2ah a v2al.",
        "Todos os caminhos sao repo-relativos; nenhum dado bruto e versionado.",
        "",
    ]
    table = [(r["artifact_id"], r["stage"], r["artifact_type"], r["path"],
              r["exists"], r["row_count"], r["schema_status"]) for r in rows]
    lines.extend(write_markdown_table(
        ["artifact_id", "stage", "type", "path", "exists", "rows", "schema"], table))
    write_markdown(atlas_path("v2am_appendix_artifact_index.md"), lines)
    return rows


# --- evidence atlas --------------------------------------------------------
def run_evidence_atlas_registry_builder(args=None):
    assert_stage_artifacts_ready()
    c = stage_counts()
    axes = [
        ("candidatos review-only",
         f"{c['candidates']} pacotes preservados como candidatos revisaveis, sem promocao operacional.",
         "datasets/protocolo_c/v2ah_candidate_reference_review_queue.csv",
         "candidates_only_pending_review",
         "Descrever corpus de candidatos para revisao futura.",
         "Nao ler como deteccao, classe, label ou referencia validada.",
         "apendice: fila de revisao",
         "O que significam os 172 candidatos?"),
        ("blockers de promocao",
         f"{c['blockers']} pacotes com promotion_allowed=false e blockers explicitos.",
         "datasets/protocolo_c/v2ai_safe_promotion_blockers.csv",
         "promotion_blocked",
         "Explicar por que nenhum candidato e promovido.",
         "Nao ler como problema resolvido nem validacao pendente trivial.",
         "apendice: claims e guardrails",
         "Entao como validam o projeto?"),
        ("revisao humana pendente",
         f"{c['assignments']} slots de revisao humana preparados, ainda pendentes.",
         "datasets/protocolo_c/v2ai_review_assignment_registry.csv",
         "human_review_pending",
         "Mostrar estrutura de revisao futura.",
         "Nao ler como revisao concluida nem identidade real de revisor.",
         "apendice: fila de revisao",
         "A revisao humana foi feita?"),
        ("adjudicacao pendente",
         f"{c['adjudications']} itens aguardando adjudicacao apos revisao futura.",
         "datasets/protocolo_c/v2ai_adjudication_queue.csv",
         "adjudication_pending",
         "Mostrar plano de adjudicacao.",
         "Nao ler como consenso atingido.",
         "apendice: fila de revisao",
         "Houve adjudicacao?"),
        ("claims permitidos/proibidos",
         f"Matriz com {c['claims']} claims separando linguagem segura e proibida.",
         "datasets/protocolo_c/v2aj_tcc_protocol_c_claims_matrix.csv",
         "claims_separated",
         "Usar como guardrail de escrita.",
         "Nao reintroduzir claim proibido como afirmacao.",
         "apendice: claims e guardrails",
         "Que afirmacoes voces podem fazer?"),
        ("limitacoes metodologicas",
         f"{c['limitations']} limitacoes documentadas como controle metodologico.",
         "datasets/protocolo_c/v2aj_methodological_limitations_export.csv",
         "limitations_documented",
         "Apresentar limitacoes como delimitacao controlada.",
         "Nao ler como falha descontrolada do projeto.",
         "apendice: limitacoes",
         "A ausencia de ground truth invalida o trabalho?"),
        ("integracao segura no manuscrito",
         "Bundles Markdown/LaTeX e matriz de insercao preparados para revisao manual.",
         "datasets/protocolo_c/v2al_section_insertion_matrix.csv",
         "integration_prepared_manual_review",
         "Inserir secoes apos revisao humana.",
         "Nao inserir automaticamente nem promover candidato.",
         "apendice: indice e catalogo",
         "Como o Protocolo C entra no texto?"),
        ("captions e tabelas",
         "Legendas seguras de governanca/revisao para tabelas do TCC.",
         "datasets/protocolo_c/v2al_table_caption_export.csv",
         "captions_safe",
         "Usar legendas de governanca e revisao.",
         "Nao usar caption de acuracia ou validacao operacional.",
         "apendice: catalogo de tabelas e figuras",
         "As tabelas mostram desempenho?"),
        ("guardrails",
         "Regressoes de guardrail fail-closed mantidas em todas as etapas.",
         "datasets/protocolo_c/v2ak_safe_language_glossary.csv",
         "guardrails_active",
         "Mostrar que linguagem e governanca foram auditadas.",
         "Nao ler como validacao de desempenho.",
         "apendice: claims e guardrails",
         "Como garantem que nao houve overclaim?"),
    ]
    rows = []
    for axis, summary, src, status, allowed, forbidden, location, banca in axes:
        rows.append({
            "atlas_item_id": f"ATL_v2am_{len(rows):03d}",
            "axis": axis,
            "summary": summary,
            "source_artifacts": src,
            "evidence_status": status,
            "allowed_interpretation": allowed,
            "forbidden_interpretation": forbidden,
            "recommended_appendix_location": location,
            "banca_question_answered": banca,
        })
    assert_no_operational_claim(rows)
    assert_no_fake_review(rows)
    write_csv(dataset_path("v2am_evidence_atlas_registry.csv"), ATLAS_COLUMNS, rows)
    lines = [
        "# Protocolo C v2am - Evidence Atlas",
        "",
        "Atlas tecnico de evidencia por eixo cientifico. Documento de apoio e auditoria,",
        "nao e o capitulo final do TCC. Corpus review-only, sem ground truth operacional.",
        "",
    ]
    for r in rows:
        lines.append(f"## {r['axis']}")
        lines.append(r["summary"])
        lines.append("")
        lines.append(f"- Fonte: {r['source_artifacts']}")
        lines.append(f"- Status: {r['evidence_status']}")
        lines.append(f"- Interpretacao permitida: {r['allowed_interpretation']}")
        lines.append(f"- Interpretacao proibida: {r['forbidden_interpretation']}")
        lines.append(f"- Local sugerido: {r['recommended_appendix_location']}")
        lines.append(f"- Pergunta de banca: {r['banca_question_answered']}")
        lines.append("")
    text = "\n".join(lines)
    assert_safe_text(text)
    write_markdown(atlas_path("v2am_protocol_c_evidence_atlas.md"), lines)
    return rows


# --- traceability DAG ------------------------------------------------------
_DAG_NODES = [
    ("N_v2ah_stop_gate", "v2ah", "v2ah stop gate",
     "datasets/protocolo_c/v2ah_ground_truth_search_stop_gate.csv", "gate"),
    ("N_v2ah_review_queue", "v2ah", "v2ah review queue",
     "datasets/protocolo_c/v2ah_candidate_reference_review_queue.csv", "queue"),
    ("N_v2ai_assignments", "v2ai", "v2ai assignments",
     "datasets/protocolo_c/v2ai_review_assignment_registry.csv", "queue"),
    ("N_v2ai_adjudication", "v2ai", "v2ai adjudication queue",
     "datasets/protocolo_c/v2ai_adjudication_queue.csv", "queue"),
    ("N_v2aj_claims", "v2aj", "v2aj claims matrix",
     "datasets/protocolo_c/v2aj_tcc_protocol_c_claims_matrix.csv", "matrix"),
    ("N_v2aj_summary", "v2aj", "v2aj evidence summary",
     "datasets/protocolo_c/v2aj_tcc_evidence_summary_table.csv", "summary"),
    ("N_v2ak_drafts", "v2ak", "v2ak drafts",
     "docs/tcc_exports/protocolo_c_v2ak_metodologia_draft.md", "drafts"),
    ("N_v2al_bundles", "v2al", "v2al bundles",
     "docs/tcc_exports/v2al_manuscript_integration/v2al_metodologia_section_candidate.md", "bundles"),
    ("N_v2am_atlas", "v2am", "v2am atlas",
     "docs/tcc_exports/v2am_appendix_evidence_atlas/v2am_protocol_c_evidence_atlas.md", "atlas"),
]
_DAG_EDGES = [
    ("N_v2ah_stop_gate", "N_v2ah_review_queue", "stop_gate_bounds_queue"),
    ("N_v2ah_review_queue", "N_v2ai_assignments", "v2ah_to_v2ai"),
    ("N_v2ai_assignments", "N_v2ai_adjudication", "assignments_feed_adjudication"),
    ("N_v2ai_adjudication", "N_v2aj_claims", "v2ai_to_v2aj"),
    ("N_v2aj_claims", "N_v2aj_summary", "claims_inform_summary"),
    ("N_v2aj_summary", "N_v2ak_drafts", "v2aj_to_v2ak"),
    ("N_v2ak_drafts", "N_v2al_bundles", "v2ak_to_v2al"),
    ("N_v2al_bundles", "N_v2am_atlas", "v2al_to_v2am"),
]


def run_traceability_dag_builder(args=None):
    assert_stage_artifacts_ready()
    nodes = []
    for node_id, stage, label, path, node_type in _DAG_NODES:
        nodes.append({
            "node_id": node_id,
            "stage": stage,
            "label": label,
            "artifact_path": repo_relative_path(path),
            "node_type": node_type,
            "claim_safety_status": "review_only_no_promotion",
        })
    edges = []
    for src, tgt, rel in _DAG_EDGES:
        edges.append(build_dag_edge(len(edges), src, tgt, rel))
    assert_no_operational_claim(nodes)
    assert_no_operational_claim(edges)
    if any(e["promotion_created"] != "false" for e in edges):
        raise ValueError("DAG edge created promotion; forbidden.")
    write_csv(dataset_path("v2am_traceability_dag_nodes.csv"), DAG_NODE_COLUMNS, nodes)
    write_csv(dataset_path("v2am_traceability_dag_edges.csv"), DAG_EDGE_COLUMNS, edges)
    md = [
        "# Protocolo C v2am - DAG de rastreabilidade",
        "",
        "Grafo de rastreabilidade entre etapas e artefatos. Cada aresta preserva",
        "guardrails e nao cria promocao operacional.",
        "",
        "## Nodes",
    ]
    md.extend(write_markdown_table(
        ["node_id", "stage", "label", "artifact_path", "claim_safety_status"],
        [(n["node_id"], n["stage"], n["label"], n["artifact_path"],
          n["claim_safety_status"]) for n in nodes]))
    md.extend(["", "## Edges"])
    md.extend(write_markdown_table(
        ["edge_id", "source", "target", "relationship", "guardrail_preserved", "promotion_created"],
        [(e["edge_id"], e["source_node"], e["target_node"], e["relationship"],
          e["guardrail_preserved"], e["promotion_created"]) for e in edges]))
    write_markdown(atlas_path("v2am_traceability_dag.md"), md)
    mmd = ["flowchart TD"]
    for n in nodes:
        mmd.append(f'    {n["node_id"]}["{n["label"]} ({n["stage"]})"]')
    for e in edges:
        mmd.append(f'    {e["source_node"]} -->|{e["relationship"]}| {e["target_node"]}')
    write_markdown(atlas_path("v2am_traceability_dag.mmd"), mmd)
    return nodes, edges


# --- tables & figures catalog ---------------------------------------------
def run_tables_figures_catalog_builder(args=None):
    assert_stage_artifacts_ready()
    items = [
        ("table", "Candidatos review-only do Protocolo C",
         "datasets/protocolo_c/v2ah_candidate_reference_review_queue.csv",
         "resultados_ou_apendice",
         "Pacotes mantidos como candidatos revisaveis (governanca e revisao).",
         "tabela de deteccao de enchente",
         "Descrever estado dos candidatos.",
         "Nao inferir acuracia nem validacao operacional.", "high"),
        ("table", "Blockers de promocao por pacote",
         "datasets/protocolo_c/v2ai_safe_promotion_blockers.csv",
         "apendice",
         "Blockers que mantem promotion_allowed=false.",
         "tabela de validacao operacional concluida",
         "Explicar por que nao ha promocao.",
         "Nao afirmar que blockers foram resolvidos.", "high"),
        ("table", "Matriz de claims permitidos e proibidos",
         "datasets/protocolo_c/v2aj_tcc_protocol_c_claims_matrix.csv",
         "apendice",
         "Separacao explicita de linguagem segura e proibida.",
         "lista livre de claims",
         "Usar como guardrail de escrita.",
         "Nao usar wording proibido como afirmacao.", "high"),
        ("table", "Fila de revisao humana pendente",
         "datasets/protocolo_c/v2ai_review_assignment_registry.csv",
         "apendice",
         "Slots de revisao humana preparados e pendentes.",
         "tabela de revisoes concluidas",
         "Mostrar estrutura de revisao futura.",
         "Nao afirmar revisao concluida.", "medium"),
        ("table", "Limitacoes metodologicas",
         "datasets/protocolo_c/v2aj_methodological_limitations_export.csv",
         "limitacoes",
         "Limitacoes como controle metodologico.",
         "tabela de falhas do modelo",
         "Apresentar delimitacao controlada.",
         "Nao ler como falha descontrolada.", "medium"),
        ("figure", "DAG de rastreabilidade do Protocolo C",
         "docs/tcc_exports/v2am_appendix_evidence_atlas/v2am_traceability_dag.mmd",
         "apendice",
         "Grafo de rastreabilidade entre etapas v2ah-v2am.",
         "diagrama de pipeline preditivo",
         "Mostrar linhagem e guardrails.",
         "Nao ler como fluxo de inferencia operacional.", "medium"),
        ("figure", "Atlas de evidencia por eixo",
         "docs/tcc_exports/v2am_appendix_evidence_atlas/v2am_protocol_c_evidence_atlas.md",
         "apendice",
         "Resumo de evidencia review-only por eixo cientifico.",
         "mapa de deteccao de inundacao",
         "Orientar leitura da banca.",
         "Nao ler como evidencia de evento observado.", "medium"),
    ]
    rows = []
    for item_type, title, src, loc, safe_cap, forb_cap, allowed, forbidden, prio in items:
        rows.append({
            "catalog_id": f"CAT_v2am_{len(rows):03d}",
            "item_type": item_type,
            "suggested_title": title,
            "source_artifacts": src,
            "recommended_location": loc,
            "safe_caption": safe_cap + " Dados de governanca e revisao, sem acuracia e sem validacao para uso operacional.",
            "forbidden_caption": forb_cap,
            "allowed_interpretation": allowed,
            "forbidden_interpretation": forbidden,
            "priority": prio,
            "manual_review_required": "true",
        })
    assert_no_operational_claim(rows)
    write_csv(dataset_path("v2am_tables_figures_catalog.csv"), CATALOG_COLUMNS, rows)
    lines = [
        "# Protocolo C v2am - catalogo de tabelas e figuras",
        "",
        "Catalogo de tabelas e figuras sugeridas para corpo/apendice. Todas exigem",
        "revisao manual. Nenhuma traz metrica de acuracia, nem validacao para uso",
        "operacional, nem treinamento supervisionado.",
        "",
    ]
    lines.extend(write_markdown_table(
        ["catalog_id", "tipo", "titulo", "local", "prioridade", "fonte"],
        [(r["catalog_id"], r["item_type"], r["suggested_title"],
          r["recommended_location"], r["priority"], r["source_artifacts"]) for r in rows]))
    write_markdown(atlas_path("v2am_tables_and_figures_catalog.md"), lines)
    return rows


# --- claims & guardrails appendix ------------------------------------------
def run_claims_guardrails_appendix_builder(args=None):
    assert_stage_artifacts_ready()
    claims = load_csv(dataset_path("v2aj_tcc_protocol_c_claims_matrix.csv"))
    rows = []
    for claim in claims:
        allowed = is_true(claim.get("claim_allowed"))
        rows.append(build_claim_record(
            len(rows),
            "claim",
            "allowed" if allowed else "prohibited",
            clean(claim.get("safe_wording")),
            clean(claim.get("unsafe_wording")),
            "datasets/protocolo_c/v2aj_tcc_protocol_c_claims_matrix.csv",
            clean(claim.get("reason")) or "Mantem linguagem controlada.",
            "Overclaim ou promocao indevida de candidato." if not allowed
            else "Subdeclarar evidencia contextual valida."))
    guardrails = [
        ("guardrail: sem ground truth operacional patch-level",
         "Declarar ausencia de referencia operacional patch-level.",
         "ground truth validado",
         "datasets/protocolo_c/v2ah_ground_truth_search_stop_gate.csv",
         "Define o limite de inferencia defensavel.",
         "Banca pode acusar overclaim se ground truth for afirmado."),
        ("guardrail: promotion_allowed=false",
         "Manter candidatos sem promocao operacional.",
         "classe positiva",
         "datasets/protocolo_c/v2ai_safe_promotion_blockers.csv",
         "Impede transformar candidato em verdade.",
         "Promover candidato sem evidencia seria insustentavel."),
        ("guardrail: revisao humana pendente",
         "Tratar revisao como trabalho futuro, nao concluido.",
         "deteccao de enchente",
         "datasets/protocolo_c/v2ai_review_assignment_registry.csv",
         "Evita simular revisao.",
         "Afirmar revisao concluida seria falso."),
        ("guardrail: sem treino/overlay/predicao",
         "Manter pipeline read-only e estrutural.",
         "treinamento supervisionado pronto",
         "datasets/protocolo_c/v2ak_safe_language_glossary.csv",
         "Mantem escopo Sentinel-first review-only.",
         "Afirmar modelo preditivo seria overclaim."),
    ]
    for label, safe_w, unsafe_w, src, why, risk in guardrails:
        rows.append(build_claim_record(
            len(rows), label, "guardrail", safe_w, unsafe_w, src, why, risk))
    assert_no_operational_claim(rows)
    write_csv(dataset_path("v2am_claims_guardrails_registry.csv"),
              CLAIMS_GUARDRAILS_COLUMNS, rows)
    lines = [
        "# Protocolo C v2am - claims e guardrails (apendice)",
        "",
        "Consolidacao de claims permitidos/proibidos e guardrails para a banca.",
        "Termos proibidos aparecem apenas como exemplos negativos / unsafe wording,",
        "nunca como afirmacao positiva do que o projeto faz.",
        "",
        "## Linguagem segura para banca",
        "- Falar em evidencia contextual, candidatos revisaveis e estado review-only.",
        "- Declarar explicitamente a ausencia de ground truth operacional patch-level.",
        "- Tratar revisao e adjudicacao como pendentes.",
        "",
        "## Registro",
    ]
    lines.extend(write_markdown_table(
        ["registry_id", "tipo", "status", "linguagem_segura", "exemplo_proibido_unsafe", "risco_banca"],
        [(r["registry_id"], r["claim_or_guardrail"], r["status"], r["safe_wording"],
          f"evitar (unsafe): {r['unsafe_wording']}", r["banca_risk_if_wrong"]) for r in rows]))
    write_markdown(atlas_path("v2am_claims_and_guardrails_appendix.md"), lines)
    return rows


# --- review queue appendix -------------------------------------------------
def run_review_queue_appendix_builder(args=None):
    assert_stage_artifacts_ready()
    queue = load_csv(dataset_path("v2ah_candidate_reference_review_queue.csv"))
    assignments = load_csv(dataset_path("v2ai_review_assignment_registry.csv"))
    groups = {}
    for r in queue:
        g = clean(r.get("region")) or "UNSPECIFIED"
        groups.setdefault(g, {"candidates": 0, "assignments": 0})
        groups[g]["candidates"] += 1
    for a in assignments:
        g = clean(a.get("region")) or "UNSPECIFIED"
        groups.setdefault(g, {"candidates": 0, "assignments": 0})
        groups[g]["assignments"] += 1
    rows = []
    for group in sorted(groups):
        data = groups[group]
        rows.append({
            "review_item_id": f"RQ_v2am_{len(rows):03d}",
            "package_group": group,
            "candidate_count": str(data["candidates"]),
            "assignment_count": str(data["assignments"]),
            "review_status": "PENDING_HUMAN_REVIEW",
            "adjudication_status": "PENDING_ADJUDICATION",
            "promotion_status": "PROMOTION_BLOCKED",
            "allowed_use": "review_queue_and_tcc_context_only",
            "forbidden_use": "ground_reference|label|training|overlay|prediction|promotion",
        })
    assert_no_operational_claim(rows)
    assert_no_fake_review(rows)
    write_csv(dataset_path("v2am_review_queue_appendix_registry.csv"),
              REVIEW_QUEUE_COLUMNS, rows)
    c = stage_counts()
    lines = [
        "# Protocolo C v2am - fila de revisao (apendice)",
        "",
        "Resumo da fila de revisao por grupo. Estado de governanca, nao verdade operacional.",
        "",
        f"- Todos os {c['candidates']} pacotes permanecem candidatos revisaveis.",
        "- Revisao humana esta pendente.",
        "- Adjudicacao esta pendente.",
        "- Promocao esta bloqueada (promotion_allowed=false).",
        "",
    ]
    lines.extend(write_markdown_table(
        ["grupo", "candidatos", "slots_revisao", "revisao", "adjudicacao", "promocao"],
        [(r["package_group"], r["candidate_count"], r["assignment_count"],
          r["review_status"], r["adjudication_status"], r["promotion_status"]) for r in rows]))
    write_markdown(atlas_path("v2am_review_queue_appendix.md"), lines)
    return rows


# --- limitations appendix --------------------------------------------------
def run_limitations_appendix_builder(args=None):
    assert_stage_artifacts_ready()
    limitations = load_csv(dataset_path("v2aj_methodological_limitations_export.csv"))
    rows = []
    for lim in limitations:
        rows.append({
            "limitation_item_id": f"LIM_v2am_{len(rows):03d}",
            "limitation_name": clean(lim.get("limitation_name")),
            "safe_explanation": clean(lim.get("safe_tcc_wording")) or clean(lim.get("what_it_means")),
            "what_it_does_not_imply": clean(lim.get("what_it_does_not_mean")) or "Nao implica falha do pipeline.",
            "mitigation": clean(lim.get("mitigation_already_done")) or "Stop gate e blockers documentados.",
            "future_work": clean(lim.get("future_work")) or "Usar nova fonte qualificada e revisao humana.",
            "source_artifact": "datasets/protocolo_c/v2aj_methodological_limitations_export.csv",
            "recommended_section": "limitacoes_e_trabalhos_futuros",
        })
    if not rows:
        rows.append({
            "limitation_item_id": "LIM_v2am_000",
            "limitation_name": "no_operational_patch_ground_truth",
            "safe_explanation": "Nao ha referencia operacional patch-level reivindicada.",
            "what_it_does_not_imply": "Nao implica que o pipeline falhou.",
            "mitigation": "Stop gate e blockers documentados.",
            "future_work": "Usar nova fonte qualificada e revisao humana.",
            "source_artifact": "datasets/protocolo_c/v2aj_methodological_limitations_export.csv",
            "recommended_section": "limitacoes_e_trabalhos_futuros",
        })
    assert_no_operational_claim(rows)
    write_csv(dataset_path("v2am_limitations_appendix_registry.csv"),
              LIMITATIONS_COLUMNS, rows)
    lines = [
        "# Protocolo C v2am - limitacoes (apendice)",
        "",
        "Limitacoes formuladas como controle metodologico, nao falha descontrolada.",
        "",
    ]
    for r in rows:
        lines.append(f"## {r['limitation_name']}")
        lines.append(f"- Explicacao segura: {r['safe_explanation']}")
        lines.append(f"- Nao implica: {r['what_it_does_not_imply']}")
        lines.append(f"- Mitigacao: {r['mitigation']}")
        lines.append(f"- Trabalho futuro: {r['future_work']}")
        lines.append(f"- Secao recomendada: {r['recommended_section']}")
        lines.append("")
    text = "\n".join(lines)
    assert_safe_text(text)
    write_markdown(atlas_path("v2am_limitations_appendix.md"), lines)
    return rows


# --- defense question bank -------------------------------------------------
def run_defense_question_bank_builder(args=None):
    assert_stage_artifacts_ready()
    c = stage_counts()
    bank = [
        ("Voces tem ground truth?",
         "Nao temos ground truth operacional patch-level.",
         "Nao ha referencia operacional patch-level; a busca foi parada em GROUND_TRUTH_SEARCH_STOPPED_UNTIL_NEW_QUALIFIED_SOURCE e os pacotes permanecem candidatos revisaveis.",
         "datasets/protocolo_c/v2ah_ground_truth_search_stop_gate.csv",
         "Afirmar ground truth inexistente seria overclaim grave.",
         "ausencia de ground truth operacional; candidato revisavel",
         "ground truth validado"),
        ("Entao como validam o projeto?",
         "Nao ha validacao operacional; o trabalho e review-only e auditavel.",
         "A contribuicao e metodologica: organizacao de candidatos, blockers, claims e guardrails; a validacao de desempenho nao e reivindicada.",
         "datasets/protocolo_c/v2ai_safe_promotion_blockers.csv",
         "Confundir governanca com validacao operacional seria erro.",
         "review-only; governanca metodologica",
         "validacao operacional"),
        ("O que significam os 172 candidatos?",
         f"Sao {c['candidates']} pacotes candidatos revisaveis, sem promocao.",
         "Cada pacote e um candidato de evidencia contextual que permanece bloqueado para promocao ate revisao humana e nova fonte qualificada.",
         "datasets/protocolo_c/v2ah_candidate_reference_review_queue.csv",
         "Tratar candidatos como verdade seria insustentavel.",
         "candidatos revisaveis; review-only",
         "classe positiva"),
        ("DINOv2 detecta enchente?",
         "Nao. DINOv2 nao realiza deteccao de enchente; e suporte estrutural de triagem.",
         "DINOv2 fornece embeddings estruturais para triagem; nao produz deteccao nem classe de inundacao observada.",
         "datasets/protocolo_c/v2ak_safe_language_glossary.csv",
         "Afirmar deteccao seria overclaim.",
         "suporte estrutural; review-only",
         "deteccao de enchente"),
        ("GIS virou label?",
         "Nao. GIS e contexto territorial, nao cria label nem classe.",
         "O GIS fornece contexto territorial externo; nao foi convertido em label, classe ou target.",
         "datasets/protocolo_c/v2aj_tcc_protocol_c_claims_matrix.csv",
         "Afirmar label seria falso.",
         "contexto territorial; sem label",
         "label operacional"),
        ("Por que nao treinaram o modelo?",
         "Porque nao ha referencia qualificada; treino esta bloqueado.",
         "Sem ground truth operacional e com blockers ativos, treino supervisionado nao e justificavel; o escopo permanece estrutural e review-only.",
         "datasets/protocolo_c/v2ai_safe_promotion_blockers.csv",
         "Afirmar treino pronto seria overclaim.",
         "treino bloqueado; review-only",
         "treinamento supervisionado pronto"),
        ("O que falta para o Protocolo B?",
         "Falta nova fonte qualificada e revisao humana concluida.",
         "O Protocolo B permanece fechado ate evidencia observacional qualificada, revisao humana e adjudicacao reais; nada disso foi simulado.",
         "datasets/protocolo_c/v2ah_ground_truth_search_stop_gate.csv",
         "Reabrir Protocolo B sem evidencia seria erro.",
         "Protocolo B bloqueado; pendente",
         "ground truth validado"),
        ("A ausencia de ground truth invalida o trabalho?",
         "Nao. A ausencia e uma limitacao controlada e documentada.",
         "A contribuicao metodologica (governanca, blockers, claims, rastreabilidade) e valida independentemente; a limitacao delimita o que pode ser afirmado.",
         "datasets/protocolo_c/v2aj_methodological_limitations_export.csv",
         "Tratar limitacao como falha total seria injusto e impreciso.",
         "limitacao controlada; governanca",
         "validacao operacional"),
        ("O que exatamente v2ah/v2ai/v2aj/v2ak/v2al provaram?",
         "Provaram organizacao review-only auditavel, nao desempenho operacional.",
         "v2ah parou a busca e consolidou candidatos; v2ai preparou revisao/adjudicacao com blockers; v2aj separou claims e limitacoes; v2ak gerou drafts seguros; v2al preparou integracao manual. Nenhuma etapa criou ground truth, label ou predicao.",
         "datasets/protocolo_c/v2am_traceability_dag_edges.csv",
         "Atribuir prova de desempenho seria overclaim.",
         "rastreabilidade; review-only",
         "modelo preditivo"),
    ]
    rows = []
    for q, short_a, tech_a, evidence, risk, safe_t, forb_t in bank:
        rows.append({
            "question_id": f"Q_v2am_{len(rows):02d}",
            "question": q,
            "short_answer": short_a,
            "technical_answer": tech_a,
            "evidence_to_show": evidence,
            "risk_if_answered_wrong": risk,
            "safe_terms": safe_t,
            "forbidden_terms": forb_t,
        })
    assert_no_operational_claim(rows)
    assert_no_fake_review(rows)
    write_csv(dataset_path("v2am_defense_question_bank.csv"), DEFENSE_COLUMNS, rows)
    lines = [
        "# Protocolo C v2am - banco de perguntas de banca",
        "",
        "Perguntas e respostas seguras para a defesa. As respostas negam explicitamente",
        "qualquer afirmacao operacional; termos proibidos aparecem apenas como termo a evitar.",
        "",
    ]
    for r in rows:
        lines.append(f"## {r['question']}")
        lines.append(f"- Resposta curta: {r['short_answer']}")
        lines.append(f"- Resposta tecnica: {r['technical_answer']}")
        lines.append(f"- Evidencia a mostrar: {r['evidence_to_show']}")
        lines.append(f"- Termos seguros: {r['safe_terms']}")
        lines.append(f"- Termo a evitar (unsafe): {r['forbidden_terms']}")
        lines.append("")
    text = "\n".join(lines)
    assert_safe_text(text)
    write_markdown(atlas_path("v2am_defense_question_bank.md"), lines)
    return rows


# --- final claim consistency audit -----------------------------------------
def _audit_target_docs():
    targets = []
    if os.path.isdir(DOCS_DIR):
        for n in sorted(os.listdir(DOCS_DIR)):
            if n.endswith((".md", ".tex")) and ("v2ak" in n or "v2al" in n):
                targets.append((rel_doc(n), doc_path(n)))
    if os.path.isdir(V2AL_INTEGRATION_DIR):
        for n in sorted(os.listdir(V2AL_INTEGRATION_DIR)):
            if n.endswith((".md", ".tex")):
                targets.append((rel_integration(n), integration_path(n)))
    if os.path.isdir(ATLAS_DIR):
        for n in sorted(os.listdir(ATLAS_DIR)):
            # Skip the audit's own rendering, which deliberately lists forbidden
            # fragments and would otherwise self-flag and grow on re-runs.
            if n == "v2am_final_claim_consistency_audit.md":
                continue
            if n.endswith((".md", ".mmd")):
                targets.append((rel_atlas(n), atlas_path(n)))
    return targets


def run_final_claim_consistency_audit(args=None):
    assert_stage_artifacts_ready()
    rows = []
    for rel, path in _audit_target_docs():
        for line in read_text(path).splitlines():
            line_l = line.lower()
            safe_context = any(m in line_l for m in SAFE_CONTEXT_MARKERS)
            for phrase in UNSAFE_LANGUAGE:
                if phrase in line_l:
                    violation = not safe_context
                    rows.append({
                        "audit_id": f"FCA_v2am_{len(rows):05d}",
                        "file": rel,
                        "claim_fragment": phrase,
                        "claim_category": "forbidden_language",
                        "allowed": "false",
                        "context_is_safe": normalize_bool(safe_context),
                        "violation": normalize_bool(violation),
                        "violation_reason": "" if not violation else "Forbidden phrase as positive assertion.",
                        "recommended_fix": "" if not violation else "Reformular com negacao explicita ou marcar como unsafe wording.",
                    })
    if not rows:
        rows.append({
            "audit_id": "FCA_v2am_00000",
            "file": "all_v2ak_v2al_v2am_docs",
            "claim_fragment": "none",
            "claim_category": "clean",
            "allowed": "true",
            "context_is_safe": "true",
            "violation": "false",
            "violation_reason": "",
            "recommended_fix": "",
        })
    violations = [r for r in rows if r["violation"] == "true"]
    write_csv(dataset_path("v2am_final_claim_consistency_audit.csv"),
              FINAL_AUDIT_COLUMNS, rows)
    lines = [
        "# Protocolo C v2am - auditoria final de consistencia de claims",
        "",
        f"Linhas auditadas com termo sensivel: {len(rows)}. Violacoes: {len(violations)}.",
        "Termos proibidos sao permitidos apenas como unsafe wording, exemplo negativo,",
        "pergunta de banca, resposta que nega, limitacao ou guardrail.",
        "",
    ]
    lines.extend(write_markdown_table(
        ["audit_id", "file", "fragmento", "contexto_seguro", "violacao"],
        [(r["audit_id"], r["file"], f"evitar (unsafe): {r['claim_fragment']}",
          r["context_is_safe"], r["violation"]) for r in rows]))
    write_markdown(atlas_path("v2am_final_claim_consistency_audit.md"), lines)
    if violations:
        sample = [(v["file"], v["claim_fragment"]) for v in violations[:5]]
        raise ValueError(f"Final claim consistency audit failed: {sample}")
    return rows


# --- appendix index --------------------------------------------------------
_APPENDIX_ORDER = [
    ("v2am_appendix_index.md", "Appendix index",
     "Indice navegavel do apendice.", "all v2am", "orientador_e_banca", "1"),
    ("v2am_protocol_c_evidence_atlas.md", "Evidence atlas",
     "Atlas de evidencia por eixo.", "v2ah-v2al", "banca", "2"),
    ("v2am_traceability_dag.md", "Traceability DAG",
     "Grafo de rastreabilidade.", "v2ah-v2am", "banca", "3"),
    ("v2am_claims_and_guardrails_appendix.md", "Claims and guardrails",
     "Claims permitidos/proibidos e guardrails.", "v2aj/v2ak", "banca", "4"),
    ("v2am_review_queue_appendix.md", "Review queue",
     "Fila de revisao pendente.", "v2ah/v2ai", "orientador", "5"),
    ("v2am_limitations_appendix.md", "Limitations",
     "Limitacoes como controle metodologico.", "v2aj/v2ak", "banca", "6"),
    ("v2am_tables_and_figures_catalog.md", "Tables and figures catalog",
     "Catalogo de tabelas e figuras.", "v2aj/v2al", "orientador", "7"),
    ("v2am_defense_question_bank.md", "Defense question bank",
     "Perguntas e respostas de banca.", "v2ah-v2al", "candidato", "8"),
    ("v2am_final_claim_consistency_audit.md", "Final claim consistency audit",
     "Auditoria final de claims.", "v2ak/v2al/v2am", "orientador_e_banca", "9"),
]


def run_appendix_index_builder(args=None):
    assert_stage_artifacts_ready()
    rows = []
    for fname, title, purpose, src, reader, prio in _APPENDIX_ORDER:
        rows.append({
            "index_item_id": f"IDX_v2am_{len(rows):02d}",
            "appendix_file": rel_atlas(fname),
            "title": title,
            "purpose": purpose,
            "source_artifacts": src,
            "recommended_reader": reader,
            "review_priority": prio,
        })
    assert_no_operational_claim(rows)
    write_csv(dataset_path("v2am_appendix_index_registry.csv"),
              APPENDIX_INDEX_COLUMNS, rows)
    lines = [
        "# Protocolo C v2am - indice do apendice (Evidence Atlas)",
        "",
        "Apendice de evidencia, defesa e auditoria do Protocolo C. Corpus review-only,",
        "sem ground truth operacional. Ordem sugerida de leitura:",
        "",
    ]
    for r in rows:
        lines.append(f"{r['review_priority']}. [{r['title']}]({os.path.basename(r['appendix_file'])}) "
                     f"- {r['purpose']} (leitor: {r['recommended_reader']})")
    lines.append("")
    write_markdown(atlas_path("v2am_appendix_index.md"), lines)
    return rows


# --- next action ranker ----------------------------------------------------
def run_next_action_ranker(args=None):
    assert_stage_artifacts_ready()
    options = [
        ("MANUAL_APPENDIX_REVIEW_AND_ORIENTATION_MEETING", 100,
         "v2am appendix evidence atlas",
         "docs/tcc_exports/v2am_appendix_evidence_atlas/"),
        ("INTEGRATE_APPROVED_SECTIONS_IN_TCC", 90,
         "approved v2al/v2am sections after human review",
         "v2al_section_insertion_matrix.csv"),
        ("FINAL_TCC_CLAIM_AUDIT", 80,
         "v2am final claim consistency audit",
         "v2am_final_claim_consistency_audit.csv"),
        ("HUMAN_REVIEW_EXECUTION", 70,
         "v2ai assignments and v2aj guide",
         "v2ai_review_assignment_registry.csv"),
        ("WAIT_FOR_NEW_QUALIFIED_SOURCE", 60,
         "v2ah stop gate",
         "v2ah_ground_truth_search_stop_gate.csv"),
        ("TRAINING_PROTOCOL_B_OVERLAY_LABEL_GT_PROMOTION", 0,
         "blocked by guardrails", "none"),
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
            "recommended_script_or_artifact": artifact,
            "notes": ("No next action may recommend training, Protocol B, overlay, "
                      "labels, ground truth, automatic date inference, or operational "
                      "promotion."),
        })
    write_csv(dataset_path("v2am_next_actions_registry.csv"), NEXT_COLUMNS, rows)
    return rows


# --- guardrail regression --------------------------------------------------
def _regression_artifacts():
    artifacts = []
    if os.path.isdir(DATASET_DIR):
        for n in sorted(os.listdir(DATASET_DIR)):
            if n.endswith(".csv") and (n.startswith("v2am_") or n.split("_", 1)[0] in STAGE_PREFIXES):
                artifacts.append((rel_dataset(n), dataset_path(n), "csv"))
    if os.path.isdir(ATLAS_DIR):
        for n in sorted(os.listdir(ATLAS_DIR)):
            if n.endswith((".md", ".mmd")):
                artifacts.append((rel_atlas(n), atlas_path(n), "text"))
    if os.path.isdir(DOCS_DIR):
        for n in sorted(os.listdir(DOCS_DIR)):
            if n.endswith((".md", ".tex")) and ("v2ak" in n or "v2al" in n):
                artifacts.append((rel_doc(n), doc_path(n), "text"))
    if os.path.isdir(V2AL_INTEGRATION_DIR):
        for n in sorted(os.listdir(V2AL_INTEGRATION_DIR)):
            if n.endswith((".md", ".tex")):
                artifacts.append((rel_integration(n), integration_path(n), "text"))
    return artifacts


def _scan_csv_for_regression(path):
    counts = {
        "forbidden_true_flag": 0, "forbidden_status": 0, "absolute_path": 0,
        "local_only": 0, "forbidden_kv": 0, "unsafe_language": 0,
    }
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
    assert_stage_artifacts_ready()
    check_types = ["forbidden_true_flag", "forbidden_status", "absolute_path",
                   "non_versionable_path_marker", "forbidden_kv", "unsafe_language"]
    rows = []
    total_fail = 0
    for rel, path, kind in _regression_artifacts():
        counts = (_scan_csv_for_regression(path) if kind == "csv"
                  else scan_text_violations(read_text(path)))
        for check_type in check_types:
            key = "local_only" if check_type == "non_versionable_path_marker" else check_type
            count = counts.get(key, 0)
            status = "PASS" if count == 0 else "FAIL"
            if status == "FAIL":
                total_fail += 1
            rows.append({
                "regression_id": f"GR_v2am_{len(rows):05d}",
                "artifact_path": rel,
                "check_type": check_type,
                "violation_count": str(count),
                "status": status,
                "severity": "none" if count == 0 else "blocking",
                "notes": "Fail-closed guardrail regression over v2am/v2ak/v2al/stage outputs.",
            })
    write_csv(dataset_path("v2am_guardrail_regression.csv"), REGRESSION_COLUMNS, rows)
    if total_fail:
        fails = [(r["artifact_path"], r["check_type"]) for r in rows if r["status"] == "FAIL"]
        raise ValueError(f"v2am guardrail regression failed: {fails[:5]}")
    return rows


# --- orchestrator ----------------------------------------------------------
_ORCHESTRATION = [
    ("artifact_inventory", "run_artifact_inventory_builder",
     ["v2am_appendix_artifact_index.csv"], ["v2am_appendix_artifact_index.md"]),
    ("evidence_atlas", "run_evidence_atlas_registry_builder",
     ["v2am_evidence_atlas_registry.csv"], ["v2am_protocol_c_evidence_atlas.md"]),
    ("traceability_dag", "run_traceability_dag_builder",
     ["v2am_traceability_dag_nodes.csv", "v2am_traceability_dag_edges.csv"],
     ["v2am_traceability_dag.md", "v2am_traceability_dag.mmd"]),
    ("tables_figures_catalog", "run_tables_figures_catalog_builder",
     ["v2am_tables_figures_catalog.csv"], ["v2am_tables_and_figures_catalog.md"]),
    ("claims_guardrails_appendix", "run_claims_guardrails_appendix_builder",
     ["v2am_claims_guardrails_registry.csv"], ["v2am_claims_and_guardrails_appendix.md"]),
    ("review_queue_appendix", "run_review_queue_appendix_builder",
     ["v2am_review_queue_appendix_registry.csv"], ["v2am_review_queue_appendix.md"]),
    ("limitations_appendix", "run_limitations_appendix_builder",
     ["v2am_limitations_appendix_registry.csv"], ["v2am_limitations_appendix.md"]),
    ("defense_question_bank", "run_defense_question_bank_builder",
     ["v2am_defense_question_bank.csv"], ["v2am_defense_question_bank.md"]),
    ("final_claim_consistency_audit", "run_final_claim_consistency_audit",
     ["v2am_final_claim_consistency_audit.csv"], ["v2am_final_claim_consistency_audit.md"]),
    ("appendix_index", "run_appendix_index_builder",
     ["v2am_appendix_index_registry.csv"], ["v2am_appendix_index.md"]),
    ("next_action_ranker", "run_next_action_ranker",
     ["v2am_next_actions_registry.csv"], []),
    ("completion_report", "run_completion_report",
     ["v2am_completion_report.csv"], ["v2am_completion_report.md"]),
]


def run_master_orchestrator(args=None):
    assert_stage_artifacts_ready()
    rows = []
    for order, (name, func_name, ds_outputs, doc_outputs) in enumerate(_ORCHESTRATION, 1):
        func = globals()[func_name]
        status = "OK"
        notes = "Completed."
        try:
            func(args)
        except Exception as exc:  # fail-fast at first critical error
            status = "FAIL"
            notes = f"{type(exc).__name__}: {exc}"
            rows.append(_manifest_row(order, name, status, ds_outputs, doc_outputs, notes))
            write_csv(dataset_path("v2am_orchestrator_run_manifest.csv"),
                      MANIFEST_COLUMNS, rows)
            _write_manifest_md(rows)
            raise
        rows.append(_manifest_row(order, name, status, ds_outputs, doc_outputs, notes))
    write_csv(dataset_path("v2am_orchestrator_run_manifest.csv"),
              MANIFEST_COLUMNS, rows)
    _write_manifest_md(rows)
    return rows


def _manifest_row(order, name, status, ds_outputs, doc_outputs, notes):
    outputs = [rel_dataset(o) for o in ds_outputs] + [rel_atlas(o) for o in doc_outputs]
    hashes = []
    for o in ds_outputs:
        hashes.append(sha256_file(dataset_path(o))[:16])
    for o in doc_outputs:
        hashes.append(sha256_file(atlas_path(o))[:16])
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
        "# Protocolo C v2am - manifesto de execucao do orchestrator",
        "",
        f"Etapas executadas: {len(rows)}.",
        "Nenhuma operacao git foi executada.",
        "",
    ]
    lines.extend(write_markdown_table(
        ["ordem", "etapa", "status", "outputs"],
        [(r["step_order"], r["step_name"], r["status"], r["outputs"]) for r in rows]))
    write_markdown(atlas_path("v2am_orchestrator_run_manifest.md"), lines)


# --- completion report -----------------------------------------------------
def _count(name):
    return len(load_csv(dataset_path(name)))


def run_completion_report(args=None):
    assert_stage_artifacts_ready()
    inventory = _count("v2am_appendix_artifact_index.csv")
    atlas = _count("v2am_evidence_atlas_registry.csv")
    nodes = _count("v2am_traceability_dag_nodes.csv")
    edges = _count("v2am_traceability_dag_edges.csv")
    catalog = _count("v2am_tables_figures_catalog.csv")
    claims_g = _count("v2am_claims_guardrails_registry.csv")
    review_q = _count("v2am_review_queue_appendix_registry.csv")
    limitations = _count("v2am_limitations_appendix_registry.csv")
    defense = _count("v2am_defense_question_bank.csv")
    final_audit = load_csv(dataset_path("v2am_final_claim_consistency_audit.csv"))
    regression = load_csv(dataset_path("v2am_guardrail_regression.csv"))
    next_rows = load_csv(dataset_path("v2am_next_actions_registry.csv"))
    audit_violations = sum(1 for r in final_audit if r.get("violation") == "true")
    regression_fail = sum(1 for r in regression if r.get("status") == "FAIL")
    rows = [
        {"completion_id": "CR_v2am_000", "metric": "inputs_read",
         "value": str(len(load_stage_inventory())), "status": "RECORDED",
         "notes": "Read-only stage artifacts discovered for v2ah-v2al."},
        {"completion_id": "CR_v2am_001", "metric": "artifacts_inventoried",
         "value": str(inventory), "status": "RECORDED", "notes": "v2am_appendix_artifact_index.csv"},
        {"completion_id": "CR_v2am_002", "metric": "atlas_items",
         "value": str(atlas), "status": "RECORDED", "notes": "Evidence atlas axes."},
        {"completion_id": "CR_v2am_003", "metric": "dag_nodes",
         "value": str(nodes), "status": "RECORDED", "notes": "Traceability nodes."},
        {"completion_id": "CR_v2am_004", "metric": "dag_edges",
         "value": str(edges), "status": "RECORDED", "notes": "Traceability edges; promotion_created=false."},
        {"completion_id": "CR_v2am_005", "metric": "tables_figures_catalog",
         "value": str(catalog), "status": "RECORDED", "notes": "No accuracy/validation/training items."},
        {"completion_id": "CR_v2am_006", "metric": "claims_guardrails",
         "value": str(claims_g), "status": "RECORDED", "notes": "Claims and guardrails consolidated."},
        {"completion_id": "CR_v2am_007", "metric": "review_queue_items",
         "value": str(review_q), "status": "RECORDED", "notes": "All pending/blocked."},
        {"completion_id": "CR_v2am_008", "metric": "limitations",
         "value": str(limitations), "status": "RECORDED", "notes": "Methodological controls."},
        {"completion_id": "CR_v2am_009", "metric": "defense_questions",
         "value": str(defense), "status": "RECORDED", "notes": "Safe defense answers."},
        {"completion_id": "CR_v2am_010", "metric": "final_audit_violations",
         "value": str(audit_violations),
         "status": "PASS" if audit_violations == 0 else "FAIL",
         "notes": "Final claim consistency audit."},
        {"completion_id": "CR_v2am_011", "metric": "guardrail_regression_failures",
         "value": str(regression_fail),
         "status": "PASS" if regression_fail == 0 else "FAIL",
         "notes": "Fail-closed guardrail regression."},
        {"completion_id": "CR_v2am_012", "metric": "main_manuscript_overwritten",
         "value": "false", "status": "GUARDRAIL_OK", "notes": "Outputs are v2am_ only."},
        {"completion_id": "CR_v2am_013", "metric": "next_action_rank_1",
         "value": next_rows[0]["next_action"] if next_rows else "",
         "status": "SAFE_NEXT_ACTION", "notes": "Manual appendix review and orientation meeting."},
        {"completion_id": "CR_v2am_014", "metric": "final_decision",
         "value": "evidence_atlas_and_appendix_package_ready_no_operational_promotion",
         "status": "NO_OPERATIONAL_PROMOTION",
         "notes": "No science altered; v2ah-v2al inputs read-only."},
    ]
    write_csv(dataset_path("v2am_completion_report.csv"), COMPLETION_COLUMNS, rows)
    lines = [
        "# Protocolo C v2am completion report",
        "",
        f"Inputs read: {len(load_stage_inventory())}.",
        f"Artifacts inventoried: {inventory}.",
        f"Atlas items: {atlas}.",
        f"DAG nodes/edges: {nodes}/{edges}.",
        f"Tables/figures catalogued: {catalog}.",
        f"Claims/guardrails consolidated: {claims_g}.",
        f"Review queue items: {review_q}.",
        f"Limitations exported: {limitations}.",
        f"Defense questions: {defense}.",
        f"Final audit violations: {audit_violations}.",
        f"Guardrail regression failures: {regression_fail}.",
        "Main manuscript overwritten: false.",
        f"Next action rank 1: {next_rows[0]['next_action'] if next_rows else ''}.",
        ("Final decision: evidence atlas and appendix package ready with no operational "
         "promotion and no manuscript edit."),
    ]
    write_markdown(atlas_path("v2am_completion_report.md"), lines)
    return rows


def run_all(args=None):
    return run_master_orchestrator(args)
