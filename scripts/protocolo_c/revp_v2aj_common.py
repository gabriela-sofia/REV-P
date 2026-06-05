#!/usr/bin/env python3
"""v2aj Safe TCC evidence export and review guide.

Builds safe scientific communication artifacts from v2ah/v2ai without changing
prior outputs and without creating operational truth, labels, training targets,
overlay, prediction, inferred dates, inferred crosswalks, or fake review.
"""

import argparse
import csv
import hashlib
import os
import re
from collections import Counter

PROTOCOL_VERSION = "v2aj"
DATASET_DIR = os.environ.get("DATASET_DIR", "datasets/protocolo_c")
DOCS_DIR = "docs/metodologia_cientifica"
CONFIG_DIR = "configs/protocolo_c"

REQUIRED_INPUTS = [
    "v2ah_ground_truth_search_stop_gate.csv",
    "v2ah_candidate_reference_review_queue.csv",
    "v2ah_safe_tcc_export_registry.csv",
    "v2ah_guardrail_regression.csv",
    "v2ah_completion_report.csv",
    "v2ai_review_assignment_registry.csv",
    "v2ai_reviewer_decision_template.csv",
    "v2ai_adjudication_queue.csv",
    "v2ai_uncertainty_registry.csv",
    "v2ai_safe_promotion_blockers.csv",
    "v2ai_guardrail_regression.csv",
    "v2ai_next_actions_registry.csv",
    "v2ai_completion_report.csv",
]

V2AJ_DATASETS = [
    "v2aj_tcc_protocol_c_claims_matrix.csv",
    "v2aj_tcc_evidence_summary_table.csv",
    "v2aj_review_guide_registry.csv",
    "v2aj_orientation_meeting_packet.csv",
    "v2aj_methodological_limitations_export.csv",
    "v2aj_results_tables_export_registry.csv",
    "v2aj_guardrail_regression.csv",
    "v2aj_next_actions_registry.csv",
    "v2aj_completion_report.csv",
]
V2AJ_DOCS = [
    "protocolo_c_v2aj_orientation_meeting_packet.md",
    "protocolo_c_v2aj_methodological_limitations_export.md",
    "protocolo_c_v2aj_safe_tcc_export.md",
    "protocolo_c_v2aj_completion_report.md",
]

FORBIDDEN_TRUE_FIELDS = {
    "ground_truth", "ground_reference", "label", "training", "overlay",
    "prediction", "protocol_b_reopen", "sentinel_date_inferred",
    "crosswalk_inferred", "human_review_completed", "adjudication_completed",
    "operational_validation", "can_create_ground_reference",
    "can_create_training_label", "promotion_allowed",
}
FORBIDDEN_STATUS_VALUES = {
    "GROUND_TRUTH_VALIDATED", "GROUND_REFERENCE_TRUE", "LABEL_POSITIVE",
    "LABEL_NEGATIVE", "TRAINING_READY", "PROTOCOL_B_OPEN",
    "OPERATIONAL_VALIDATION", "GROUND_REFERENCE", "GROUND_TRUTH",
    "TRAINING_LABEL", "PATCH_POSITIVE", "PATCH_NEGATIVE",
    "FLOOD_DETECTED", "OPERATIONAL_VALIDATED",
}
UNSAFE_LANGUAGE = [
    "ground truth validado",
    "deteccao de enchente",
    "classe positiva",
    "classe negativa",
    "label operacional",
    "treinamento supervisionado",
    "predicao de inundacao",
    "validacao operacional",
    "validacao de inundacao observada",
]
SAFE_UNSAFE_FIELDS = {
    "unsafe_wording", "claim_forbidden", "interpretation_forbidden",
    "unsafe_caption", "forbidden_interpretation", "unsafe_tcc_wording",
    "disallowed_basis", "forbidden_use", "prohibited_use",
}
ABSOLUTE_PATH_RE = re.compile(r"(?:[A-Za-z]:\\|/Users/|/home/|/mnt/|\\\\)")

CLAIMS_COLUMNS = [
    "claim_id", "section_target", "claim_type", "claim_allowed",
    "safe_wording", "unsafe_wording", "reason", "source_artifact",
    "required_disclaimer", "guardrail_category",
]
SUMMARY_COLUMNS = [
    "summary_id", "metric_name", "metric_value", "interpretation_safe",
    "interpretation_forbidden", "source_artifact", "tcc_section",
]
GUIDE_COLUMNS = [
    "guide_item_id", "review_stage", "review_question",
    "allowed_answer_values", "decision_effect", "required_evidence",
    "disallowed_basis", "notes",
]
PACKET_COLUMNS = [
    "packet_item_id", "topic", "short_summary", "evidence_source",
    "decision_needed", "risk_if_misstated", "recommended_wording",
]
LIMITATION_COLUMNS = [
    "limitation_id", "limitation_name", "what_it_means",
    "what_it_does_not_mean", "safe_tcc_wording", "unsafe_tcc_wording",
    "mitigation_already_done", "future_work",
]
TABLE_COLUMNS = [
    "table_id", "suggested_title", "source_artifacts", "tcc_section",
    "safe_caption", "unsafe_caption", "allowed_interpretation",
    "forbidden_interpretation", "include_in_main_text", "include_in_appendix",
]
GUARDRAIL_COLUMNS = [
    "guardrail_check_id", "artifact_path", "check_type", "violation_count",
    "status", "severity", "notes",
]
NEXT_COLUMNS = [
    "rank", "next_action", "score", "allowed", "blocked_operational_use",
    "required_input", "recommended_script_or_artifact", "notes",
]
COMPLETION_COLUMNS = [
    "completion_id", "metric", "value", "status", "notes",
]


def parse_args(argv=None):
    return argparse.ArgumentParser().parse_args(argv)


def dataset_path(name):
    return os.path.join(DATASET_DIR, name)


def doc_path(name):
    return os.path.join(DOCS_DIR, name)


def config_path(name):
    return os.path.join(CONFIG_DIR, name)


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


def read_text(path):
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write_markdown(path, lines):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def clean(value):
    return str(value or "").strip()


def is_true(value):
    return clean(value).lower() == "true"


def bool_closed(value):
    return "true" if is_true(value) else "false"


def rel_dataset(name):
    return f"datasets/protocolo_c/{name}"


def rel_doc(name):
    return f"docs/metodologia_cientifica/{name}"


def assert_min_schema(rows, required, artifact):
    if not rows:
        raise FileNotFoundError(f"Required artifact is missing or empty: {artifact}")
    missing = [c for c in required if c not in rows[0]]
    if missing:
        raise ValueError(f"{artifact} missing required columns: {','.join(missing)}")
    return True


def assert_v2ah_v2ai_ready():
    missing = [name for name in REQUIRED_INPUTS if not os.path.exists(dataset_path(name))]
    if missing:
        raise FileNotFoundError("v2ah and v2ai are required before v2aj; missing: " + ",".join(missing))
    assert_min_schema(load_csv(dataset_path("v2ah_candidate_reference_review_queue.csv")), ["package_id"], "v2ah_candidate_reference_review_queue.csv")
    assert_min_schema(load_csv(dataset_path("v2ai_review_assignment_registry.csv")), ["assignment_id"], "v2ai_review_assignment_registry.csv")
    return True


def safe_claim_id(text):
    return "CLM_" + hashlib.sha256(clean(text).encode("utf-8")).hexdigest()[:12]


def build_input_manifest():
    assert_v2ah_v2ai_ready()
    rows = []
    for name in REQUIRED_INPUTS:
        path = dataset_path(name)
        rows.append({
            "artifact_path": rel_dataset(name),
            "sha256_prefix": sha256_file(path)[:16],
            "row_count": str(len(load_csv(path))),
        })
    return rows


def metric_from_completion(prefix, metric):
    rows = load_csv(dataset_path(f"{prefix}_completion_report.csv"))
    for row in rows:
        if row.get("metric") == metric:
            return row.get("value", "")
    return ""


def artifact_count(name):
    return len(load_csv(dataset_path(name)))


def write_report_table(title, rows, columns):
    lines = [f"## {title}", "", "| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(clean(row.get(c)).replace("|", "/") for c in columns) + " |")
    return lines


def _field_allows_unsafe(key, value):
    key_l = clean(key).lower()
    value_l = clean(value).lower()
    return key_l in SAFE_UNSAFE_FIELDS or "unsafe" in key_l or "forbidden" in key_l or "prohibited" in key_l or "nao pode dizer" in value_l


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
            if "local" + "_" + "only" in value_s and any(token in key_l for token in ("path", "source", "artifact")):
                violations.append((idx, key, "non_versionable_path_marker"))
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


def _safe_claim_rows():
    specs = [
        ("metodologia", "allowed", "camada revisavel de candidatos", "ground truth validado", "v2ah_candidate_reference_review_queue.csv", "review_only"),
        ("resultados", "allowed", "evidencia contextual", "deteccao de enchente", "v2aj_tcc_evidence_summary_table.csv", "no_detection"),
        ("discussao", "allowed", "suporte territorial externo", "classe positiva", "v2ah_safe_tcc_export_registry.csv", "no_label"),
        ("limitacoes", "allowed", "uso review-only", "label operacional", "v2ai_safe_promotion_blockers.csv", "review_only"),
        ("limitacoes", "allowed", "sem ground truth operacional patch-level", "ground truth validado", "v2ah_ground_truth_search_stop_gate.csv", "no_ground_truth"),
        ("metodologia", "allowed", "sem label binario", "classe positiva", "v2ai_review_outcome_registry.csv", "no_label"),
        ("resultados", "allowed", "sem validacao para uso operacional", "validacao de inundacao observada", "v2ai_completion_report.csv", "no_validation"),
        ("resultados", "allowed", "sem predicao", "modelo preditivo", "v2ai_next_actions_registry.csv", "no_prediction"),
        ("trabalhos_futuros", "allowed", "pacotes aguardando revisao humana", "treinamento supervisionado pronto", "v2ai_adjudication_queue.csv", "pending_review"),
        ("apresentacao_oral", "forbidden", "nao afirmar resultado operacional", "Protocolo B aberto", "v2aj_guardrail_regression.csv", "no_protocol_b"),
    ]
    rows = []
    for section, ctype, safe, unsafe, source, category in specs:
        rows.append({
            "claim_id": safe_claim_id(section + safe + unsafe),
            "section_target": section,
            "claim_type": ctype,
            "claim_allowed": "true" if ctype == "allowed" else "false",
            "safe_wording": safe,
            "unsafe_wording": unsafe,
            "reason": "Evidence supports controlled methodological wording only.",
            "source_artifact": rel_dataset(source),
            "required_disclaimer": "review-only; no operational ground truth, label, training, overlay, or prediction",
            "guardrail_category": category,
        })
    return rows


def run_tcc_claims_matrix_builder(args=None):
    assert_v2ah_v2ai_ready()
    rows = _safe_claim_rows()
    assert_no_operational_claim(rows)
    write_csv(dataset_path("v2aj_tcc_protocol_c_claims_matrix.csv"), CLAIMS_COLUMNS, rows)
    return rows


def run_evidence_summary_table_builder(args=None):
    assert_v2ah_v2ai_ready()
    metrics = [
        ("total_candidates", artifact_count("v2ah_candidate_reference_review_queue.csv"), "172 packages are review-only candidates.", "Do not report them as labels.", "v2ah_candidate_reference_review_queue.csv", "results"),
        ("total_assignments", artifact_count("v2ai_review_assignment_registry.csv"), "Two reviewer slots were prepared per candidate.", "Do not imply review was completed.", "v2ai_review_assignment_registry.csv", "methods"),
        ("pending_templates", artifact_count("v2ai_reviewer_decision_template.csv"), "Templates remain pending human review.", "Do not treat templates as decisions.", "v2ai_reviewer_decision_template.csv", "methods"),
        ("pending_adjudication_packages", artifact_count("v2ai_adjudication_queue.csv"), "Packages wait for future adjudication.", "Do not claim adjudication was executed.", "v2ai_adjudication_queue.csv", "limitations"),
        ("promotion_blocked_packages", sum(1 for r in load_csv(dataset_path("v2ai_safe_promotion_blockers.csv")) if r.get("promotion_allowed") == "false"), "Promotion is blocked for every package.", "Do not claim operational promotion.", "v2ai_safe_promotion_blockers.csv", "results"),
        ("guardrail_checks_v2ah", artifact_count("v2ah_guardrail_regression.csv"), "v2ah guardrails are available as QA evidence.", "Do not report QA as validation of events.", "v2ah_guardrail_regression.csv", "methods"),
        ("guardrail_checks_v2ai", artifact_count("v2ai_guardrail_regression.csv"), "v2ai guardrails are available as QA evidence.", "Do not report QA as human review.", "v2ai_guardrail_regression.csv", "methods"),
        ("stop_gate_status", metric_from_completion("v2ah", "stop_gate"), "Ground-reference search is stopped until new qualified source.", "Do not say the search produced ground truth.", "v2ah_completion_report.csv", "limitations"),
        ("next_action_rank_1", metric_from_completion("v2ai", "next_action_rank_1"), "Next action remains safe communication or review execution.", "Do not recommend operational modelling.", "v2ai_completion_report.csv", "future_work"),
    ]
    rows = []
    for idx, (name, value, safe, forbidden, source, section) in enumerate(metrics):
        rows.append({
            "summary_id": f"SUM_v2aj_{idx:03d}",
            "metric_name": name,
            "metric_value": str(value),
            "interpretation_safe": safe,
            "interpretation_forbidden": forbidden,
            "source_artifact": rel_dataset(source),
            "tcc_section": section,
        })
    assert_no_operational_claim(rows)
    write_csv(dataset_path("v2aj_tcc_evidence_summary_table.csv"), SUMMARY_COLUMNS, rows)
    return rows


def run_review_guide_registry_builder(args=None):
    assert_v2ah_v2ai_ready()
    specs = [
        ("phenomenon", "Is the phenomenon compatible with the event narrative?", "compatible|incompatible|uncertain", "contextual classification only", "documented source evidence", "DINO or visual similarity alone"),
        ("spatial", "Is there explicit spatial coherence?", "yes|no|uncertain", "may reduce uncertainty only", "coordinate, polygon, or source-localized evidence", "GIS context alone"),
        ("temporal", "Is the temporal window compatible?", "yes|no|uncertain", "may reduce uncertainty only", "explicit date or acquisition lineage", "filename or region-only date"),
        ("source", "Is source confidence sufficient?", "high|medium|low|uncertain", "supports review routing", "institutional source or documented artifact", "isolated media without provenance"),
        ("sufficiency", "Is evidence sufficient for anything beyond context?", "no|uncertain", "keeps package blocked", "multi-source evidence package", "quickview alone"),
        ("uncertainty", "Which uncertainty dominates?", "spatial|temporal|source|phenomenon|crosswalk", "sets adjudication focus", "v2ai uncertainty registry", "assumption or reviewer guess"),
        ("adjudication", "Does this need adjudication after two reviews?", "yes|no|unknown_until_review", "routes future adjudication", "two completed independent reviews", "single reviewer only"),
        ("promotion_block", "Can this be promoted operationally?", "no", "promotion remains blocked", "external operational evidence plus review", "DINO, visual similarity, quickview, media, or contextual GIS alone"),
    ]
    rows = []
    for idx, spec in enumerate(specs):
        rows.append({
            "guide_item_id": f"GUIDE_v2aj_{idx:03d}",
            "review_stage": spec[0],
            "review_question": spec[1],
            "allowed_answer_values": spec[2],
            "decision_effect": spec[3],
            "required_evidence": spec[4],
            "disallowed_basis": spec[5],
            "notes": "Review guide is future-oriented and cannot create an operational reference by itself.",
        })
    assert_no_operational_claim(rows)
    write_csv(dataset_path("v2aj_review_guide_registry.csv"), GUIDE_COLUMNS, rows)
    return rows


def run_orientation_meeting_packet_builder(args=None):
    assert_v2ah_v2ai_ready()
    rows = [
        ("state", "Protocol C is review-only with 172 candidates.", "v2ah_completion_report.csv|v2ai_completion_report.csv", "Confirm wording for methodology.", "Overstating as operational result.", "Use camada revisavel de candidatos."),
        ("closed", "Stop gate and promotion blockers are closed.", "v2ah_ground_truth_search_stop_gate.csv|v2ai_safe_promotion_blockers.csv", "Confirm stop-gate framing.", "Presenting blockers as failure.", "Frame as controlled methodological limitation."),
        ("blocked", "Ground reference, labels, training, overlay and prediction remain blocked.", "v2ai_guardrail_regression.csv", "Confirm limitation language.", "Suggesting validation or detection.", "State no validation for use was produced."),
        ("review", "Human review package exists, but no review was executed.", "v2ai_review_assignment_registry.csv", "Define future review protocol.", "Implying reviewer agreement.", "Say pending human review."),
        ("questions", "Ask which tables belong in main text versus appendix.", "v2aj_results_tables_export_registry.csv", "Advisor selection needed.", "Crowding TCC with governance tables.", "Use selected summary tables in text and detailed matrices in appendix."),
    ]
    out = []
    for idx, item in enumerate(rows):
        out.append({
            "packet_item_id": f"PKT_v2aj_{idx:03d}",
            "topic": item[0],
            "short_summary": item[1],
            "evidence_source": item[2],
            "decision_needed": item[3],
            "risk_if_misstated": item[4],
            "recommended_wording": item[5],
        })
    assert_no_operational_claim(out)
    write_csv(dataset_path("v2aj_orientation_meeting_packet.csv"), PACKET_COLUMNS, out)
    write_markdown(doc_path("protocolo_c_v2aj_orientation_meeting_packet.md"), [
        "# Protocolo C v2aj orientation meeting packet",
        "",
        "## Current state",
        "Protocol C has 172 review-only candidate packages and no operational patch-level reference.",
        "",
        "## Closed items",
        "The stop gate is closed until a new qualified source exists. Promotion blockers remain active.",
        "",
        "## Blocked items",
        "Ground reference, labels, training, overlay, prediction, date inference, and Protocol B remain blocked.",
        "",
        "## Questions for advisor",
        "- Which summary tables should be in the main text?",
        "- Which matrices should move to appendix?",
        "- Is the review-only terminology acceptable for the methodology section?",
    ])
    return out


def run_methodological_limitations_export_builder(args=None):
    assert_v2ah_v2ai_ready()
    specs = [
        ("no_operational_patch_ground_truth", "No operational patch-level reference exists.", "The pipeline failed.", "Controlled limitation: no operational patch-level reference is claimed.", "ground truth validado", "Stop gate and blockers documented.", "Use new qualified source and review."),
        ("no_explicit_anchor_sentinel_date_crosswalk", "Sentinel date linkage lacks explicit crosswalk.", "Dates can be inferred by region.", "Dates remain unlinkable without explicit crosswalk.", "data inferida", "v2ag and v2ah documented block.", "Find versionable crosswalk source."),
        ("no_observed_geometry", "Observed geometry is absent.", "No spatial work was done.", "Geometry absence is tracked as blocker.", "validacao operacional", "Geometry blocker matrix exists.", "Acquire institutional geometry."),
        ("no_occurrence_coordinates", "Occurrence coordinates are absent.", "Context coordinates are event coordinates.", "Contextual coordinates do not create patch truth.", "classe positiva", "Coordinate blockers retained.", "Acquire verified coordinates."),
        ("heterogeneous_external_evidence", "Sources vary by region.", "Evidence is inconsistent or invalid.", "Regional heterogeneity is disclosed and routed.", "label operacional", "Regional registries preserve status.", "Normalize future source intake."),
        ("pending_human_review", "Human review has not been executed.", "Review was simulated.", "Review package is prepared but pending.", "revisao concluida", "v2ai assignments/templates exist.", "Execute real review."),
        ("pending_adjudication", "Adjudication has not been executed.", "Consensus exists.", "Adjudication queue waits for completed reviews.", "adjudicacao concluida", "v2ai adjudication queue exists.", "Run adjudication after reviews."),
        ("dino_support_only", "DINO is structural support only.", "DINO validates events.", "DINO can support review routing only.", "deteccao de enchente", "DINO guardrails retained.", "Keep DINO support-only."),
        ("gis_context_only", "GIS is contextual evidence only.", "GIS creates labels.", "GIS context informs review without labels.", "treinamento supervisionado", "Safe-use registries retained.", "Use GIS with explicit source evidence."),
    ]
    rows = []
    for idx, item in enumerate(specs):
        rows.append({
            "limitation_id": f"LIM_v2aj_{idx:03d}",
            "limitation_name": item[0],
            "what_it_means": item[1],
            "what_it_does_not_mean": item[2],
            "safe_tcc_wording": item[3],
            "unsafe_tcc_wording": item[4],
            "mitigation_already_done": item[5],
            "future_work": item[6],
        })
    assert_no_operational_claim(rows)
    write_csv(dataset_path("v2aj_methodological_limitations_export.csv"), LIMITATION_COLUMNS, rows)
    write_markdown(doc_path("protocolo_c_v2aj_methodological_limitations_export.md"), [
        "# Protocolo C v2aj methodological limitations export",
        "",
        "The absence of an operational patch-level reference is a controlled methodological limitation.",
        "It means the current package supports review-only interpretation, not operational validation.",
        "The limitation is mitigated by explicit stop gates, promotion blockers, pending review templates, and guardrail regressions.",
        "Future work requires qualified source evidence, real human review, and adjudication.",
    ])
    return rows


def run_results_tables_export_builder(args=None):
    assert_v2ah_v2ai_ready()
    specs = [
        ("candidate_state", "Review-only candidate package state", "v2ah_candidate_reference_review_queue.csv", "results", "Candidates retained as review-only packages.", "operational results table", "Describe review queue status.", "Do not infer accuracy or validation.", "true", "true"),
        ("promotion_blockers", "Promotion blockers by package", "v2ai_safe_promotion_blockers.csv", "limitations", "All packages remain blocked from promotion.", "validated event blockers resolved", "Explain why no promotion occurs.", "Do not claim blockers are solved.", "true", "true"),
        ("claims_matrix", "Safe and unsafe claim matrix", "v2aj_tcc_protocol_c_claims_matrix.csv", "methods", "Claims are explicitly separated.", "free-form claims", "Use as writing guardrail.", "Do not bypass unsafe wording.", "true", "true"),
        ("review_queue", "Human review assignment package", "v2ai_review_assignment_registry.csv", "future_work", "Reviewer slots prepared for future review.", "completed review results", "Explain future review structure.", "Do not imply review completed.", "false", "true"),
        ("uncertainties", "Uncertainty registry", "v2ai_uncertainty_registry.csv", "discussion", "Uncertainties remain unresolved.", "uncertainties solved", "Discuss dominant blockers.", "Do not auto-resolve uncertainty.", "true", "true"),
        ("review_guide", "Structured review guide", "v2aj_review_guide_registry.csv", "appendix", "Guide defines future review questions.", "review outcome table", "Place detailed guide in appendix.", "Do not treat as decision.", "false", "true"),
        ("next_actions", "Safe next actions", "v2aj_next_actions_registry.csv", "future_work", "Next actions remain non-operational.", "model training plan", "Report safe next steps.", "Do not recommend training or overlay.", "true", "true"),
    ]
    rows = []
    for idx, s in enumerate(specs):
        rows.append({
            "table_id": f"TAB_v2aj_{idx:03d}",
            "suggested_title": s[1],
            "source_artifacts": rel_dataset(s[2]),
            "tcc_section": s[3],
            "safe_caption": s[4],
            "unsafe_caption": s[5],
            "allowed_interpretation": s[6],
            "forbidden_interpretation": s[7],
            "include_in_main_text": s[8],
            "include_in_appendix": s[9],
        })
    assert_no_operational_claim(rows)
    write_csv(dataset_path("v2aj_results_tables_export_registry.csv"), TABLE_COLUMNS, rows)
    return rows


def run_safe_markdown_report_builder(args=None):
    assert_v2ah_v2ai_ready()
    candidates = artifact_count("v2ah_candidate_reference_review_queue.csv")
    assignments = artifact_count("v2ai_review_assignment_registry.csv")
    templates = artifact_count("v2ai_reviewer_decision_template.csv")
    lines = [
        "# Protocolo C v2aj safe TCC export",
        "",
        "## Estado consolidado do Protocolo C",
        f"O estado consolidado contem {candidates} candidatos em uso review-only.",
        "",
        "## O que a v2ah fechou",
        "A v2ah fechou o stop gate ate nova fonte qualificada e reteve os candidatos como camada revisavel.",
        "",
        "## O que a v2ai estruturou",
        f"A v2ai estruturou {assignments} slots e {templates} templates pendentes, sem executar revisao humana.",
        "",
        "## O que pode ser dito no TCC",
        "Pode-se afirmar que ha uma camada revisavel de candidatos, evidencia contextual e blockers auditados.",
        "",
        "## O que nao pode ser dito",
        "Nao pode dizer: ground truth validado, deteccao de enchente, classe positiva, label operacional, treinamento supervisionado, predicao de inundacao ou validacao operacional.",
        "",
        "## Como explicar os 172 candidatos",
        "Os 172 itens sao pacotes revisaveis, nao labels, nao targets e nao referencias operacionais.",
        "",
        "## Como explicar revisao humana pendente",
        "A revisao humana foi estruturada como fila e template, mas permanece pendente.",
        "",
        "## Como explicar ausencia de ground truth operacional",
        "A ausencia e uma limitacao metodologica controlada por stop gate, blockers e guardrails.",
        "",
        "## Proximos passos seguros",
        "Escrita segura do Protocolo C, reuniao de orientacao, revisao humana real futura ou espera por nova fonte qualificada.",
    ]
    write_markdown(doc_path("protocolo_c_v2aj_safe_tcc_export.md"), lines)
    return [{"doc_path": rel_doc("protocolo_c_v2aj_safe_tcc_export.md"), "status": "WRITTEN_SAFE_EXPORT"}]


def _scan_csv(path):
    checks = Counter()
    rows = load_csv(path)
    for row in rows:
        for key, value in row.items():
            key_l = clean(key).lower()
            value_s = clean(value)
            value_l = value_s.lower()
            if key_l in FORBIDDEN_TRUE_FIELDS and is_true(value_s):
                checks["forbidden_true_flag"] += 1
            if value_s in FORBIDDEN_STATUS_VALUES:
                checks["forbidden_promotion_status"] += 1
            if ABSOLUTE_PATH_RE.search(value_s):
                checks["absolute_path"] += 1
            if "local" + "_" + "only" in value_s and any(token in key_l for token in ("path", "source", "artifact")):
                checks["non_versionable_path_marker"] += 1
            for phrase in UNSAFE_LANGUAGE:
                if phrase in value_l and not _field_allows_unsafe(key, value_s):
                    checks["unsafe_language"] += 1
    return checks


def _scan_markdown(path):
    checks = Counter()
    for line in read_text(path).splitlines():
        line_l = line.lower()
        allowed_context = any(marker in line_l for marker in ("nao pode dizer", "unsafe_wording", "forbidden", "prohibited"))
        if ABSOLUTE_PATH_RE.search(line):
            checks["absolute_path"] += 1
        if "local" + "_" + "only" in line:
            checks["non_versionable_path_marker"] += 1
        for phrase in UNSAFE_LANGUAGE:
            if phrase in line_l and not allowed_context:
                checks["unsafe_language"] += 1
    return checks


def run_guardrail_regression(args=None):
    names = []
    if os.path.exists(DATASET_DIR):
        names.extend(n for n in os.listdir(DATASET_DIR) if n.endswith(".csv") and (n.startswith("v2aj_") or n.startswith("v2ah_") or n.startswith("v2ai_")))
    docs = []
    if os.path.exists(DOCS_DIR):
        docs.extend(n for n in os.listdir(DOCS_DIR) if n.endswith(".md") and "v2aj" in n)
    check_types = ["forbidden_true_flag", "forbidden_promotion_status", "absolute_path", "non_versionable_path_marker", "unsafe_language"]
    rows = []
    for name in sorted(set(names)):
        checks = _scan_csv(dataset_path(name))
        for ctype in check_types:
            count = checks.get(ctype, 0)
            rows.append({
                "guardrail_check_id": f"GR_v2aj_{len(rows):05d}",
                "artifact_path": rel_dataset(name),
                "check_type": ctype,
                "violation_count": str(count),
                "status": "PASS" if count == 0 else "FAIL",
                "severity": "none" if count == 0 else "blocking",
                "notes": "Fail-closed scan over v2aj and recent review exports.",
            })
    for name in sorted(set(docs)):
        checks = _scan_markdown(doc_path(name))
        for ctype in check_types:
            count = checks.get(ctype, 0)
            rows.append({
                "guardrail_check_id": f"GR_v2aj_{len(rows):05d}",
                "artifact_path": rel_doc(name),
                "check_type": ctype,
                "violation_count": str(count),
                "status": "PASS" if count == 0 else "FAIL",
                "severity": "none" if count == 0 else "blocking",
                "notes": "Fail-closed scan over v2aj markdown.",
            })
    write_csv(dataset_path("v2aj_guardrail_regression.csv"), GUARDRAIL_COLUMNS, rows)
    return rows


def run_next_action_ranker(args=None):
    options = [
        ("SAFE_TCC_PROTOCOL_C_WRITEUP", 100, "v2aj_safe_tcc_export.md and claims matrix", "docs/metodologia_cientifica/protocolo_c_v2aj_safe_tcc_export.md"),
        ("ORIENTATION_MEETING_REVIEW", 90, "orientation packet", "v2aj_orientation_meeting_packet.csv"),
        ("HUMAN_REVIEW_EXECUTION", 80, "v2ai assignments and review guide", "v2ai_review_assignment_registry.csv"),
        ("WAIT_FOR_NEW_QUALIFIED_SOURCE", 70, "stop gate and blockers", "v2ah_ground_truth_search_stop_gate.csv"),
        ("APPENDIX_TABLES_EXPORT", 65, "results table export registry", "v2aj_results_tables_export_registry.csv"),
        ("TRAINING_OVERLAY_LABEL_GROUND_TRUTH", 0, "blocked by guardrails", "none"),
    ]
    rows = []
    for rank, (action, score, required, artifact) in enumerate(sorted(options, key=lambda x: (-x[1], x[0])), 1):
        rows.append({
            "rank": str(rank),
            "next_action": action,
            "score": str(score),
            "allowed": "false" if score == 0 else "true",
            "blocked_operational_use": "true",
            "required_input": required,
            "recommended_script_or_artifact": artifact,
            "notes": "No next action may recommend training, Protocol B, overlay, labels, ground truth, automatic date inference, or operational promotion.",
        })
    write_csv(dataset_path("v2aj_next_actions_registry.csv"), NEXT_COLUMNS, rows)
    return rows


def guardrail_failures():
    return sum(1 for r in load_csv(dataset_path("v2aj_guardrail_regression.csv")) if r.get("status") == "FAIL")


def run_completion_report(args=None):
    assert_v2ah_v2ai_ready()
    generated = [n for n in V2AJ_DATASETS if os.path.exists(dataset_path(n))]
    docs = [n for n in V2AJ_DOCS if os.path.exists(doc_path(n))]
    claims = load_csv(dataset_path("v2aj_tcc_protocol_c_claims_matrix.csv"))
    rows = [
        {"completion_id": "CR_v2aj_000", "metric": "inputs_read", "value": str(len(build_input_manifest())), "status": "RECORDED", "notes": "|".join(r["artifact_path"] for r in build_input_manifest())},
        {"completion_id": "CR_v2aj_001", "metric": "outputs_created", "value": str(len(generated)), "status": "RECORDED", "notes": "|".join(rel_dataset(n) for n in generated)},
        {"completion_id": "CR_v2aj_002", "metric": "documents_created", "value": str(len(docs)), "status": "RECORDED", "notes": "|".join(rel_doc(n) for n in docs)},
        {"completion_id": "CR_v2aj_003", "metric": "claims_allowed", "value": str(sum(1 for r in claims if r.get("claim_allowed") == "true")), "status": "SAFE_CLAIMS", "notes": "Allowed claims remain review-only."},
        {"completion_id": "CR_v2aj_004", "metric": "claims_forbidden", "value": str(sum(1 for r in claims if r.get("claim_allowed") == "false")), "status": "FORBIDDEN_CLAIMS", "notes": "Forbidden claims are explicitly separated."},
        {"completion_id": "CR_v2aj_005", "metric": "evidence_metrics", "value": str(artifact_count("v2aj_tcc_evidence_summary_table.csv")), "status": "RECORDED", "notes": "Summary metrics exported."},
        {"completion_id": "CR_v2aj_006", "metric": "review_guide_items", "value": str(artifact_count("v2aj_review_guide_registry.csv")), "status": "RECORDED", "notes": "Future review guide only."},
        {"completion_id": "CR_v2aj_007", "metric": "orientation_packet_items", "value": str(artifact_count("v2aj_orientation_meeting_packet.csv")), "status": "RECORDED", "notes": "Briefing for advisor meeting."},
        {"completion_id": "CR_v2aj_008", "metric": "limitations_exported", "value": str(artifact_count("v2aj_methodological_limitations_export.csv")), "status": "RECORDED", "notes": "Controlled limitations exported."},
        {"completion_id": "CR_v2aj_009", "metric": "tables_recommended", "value": str(artifact_count("v2aj_results_tables_export_registry.csv")), "status": "RECORDED", "notes": "Tables mapped for main text or appendix."},
        {"completion_id": "CR_v2aj_010", "metric": "guardrail_failures", "value": str(guardrail_failures()), "status": "PASS" if guardrail_failures() == 0 else "FAIL", "notes": "v2aj guardrail regression."},
        {"completion_id": "CR_v2aj_011", "metric": "next_action_rank_1", "value": load_csv(dataset_path("v2aj_next_actions_registry.csv"))[0]["next_action"], "status": "SAFE_NEXT_ACTION", "notes": "No operational promotion."},
        {"completion_id": "CR_v2aj_012", "metric": "decision_final", "value": "safe_tcc_export_ready_no_operational_promotion", "status": "NO_OPERATIONAL_PROMOTION", "notes": "No labels, training, overlay, prediction, ground truth, or fake review."},
    ]
    write_csv(dataset_path("v2aj_completion_report.csv"), COMPLETION_COLUMNS, rows)
    write_markdown(doc_path("protocolo_c_v2aj_completion_report.md"), [
        "# Protocolo C v2aj completion report",
        "",
        f"Inputs read: {len(build_input_manifest())}.",
        f"Outputs created: {len(generated)}.",
        f"Documents created: {len(docs)}.",
        f"Allowed claims: {sum(1 for r in claims if r.get('claim_allowed') == 'true')}.",
        f"Forbidden claims: {sum(1 for r in claims if r.get('claim_allowed') == 'false')}.",
        f"Evidence metrics: {artifact_count('v2aj_tcc_evidence_summary_table.csv')}.",
        f"Review guide items: {artifact_count('v2aj_review_guide_registry.csv')}.",
        f"Orientation packet items: {artifact_count('v2aj_orientation_meeting_packet.csv')}.",
        f"Limitations exported: {artifact_count('v2aj_methodological_limitations_export.csv')}.",
        f"Tables recommended: {artifact_count('v2aj_results_tables_export_registry.csv')}.",
        f"Guardrail failures: {guardrail_failures()}.",
        "Final decision: safe TCC export ready with no operational promotion.",
    ])
    return rows


def run_all(args=None):
    run_tcc_claims_matrix_builder(args)
    run_evidence_summary_table_builder(args)
    run_review_guide_registry_builder(args)
    run_orientation_meeting_packet_builder(args)
    run_methodological_limitations_export_builder(args)
    run_results_tables_export_builder(args)
    run_safe_markdown_report_builder(args)
    run_guardrail_regression(args)
    run_next_action_ranker(args)
    return run_completion_report(args)
