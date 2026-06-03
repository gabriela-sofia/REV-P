"""
v1ho — Dossiês de evidência por evento candidato.
Audita documento metodológico, templates, schemas e três registros.
"""
import csv
import os
import pytest

BASE = os.path.join(os.path.dirname(__file__), "..")
DATASETS = os.path.join(BASE, "datasets")
SCHEMAS = os.path.join(DATASETS, "schemas")
DOCS = os.path.join(BASE, "docs", "metodologia_cientifica")
TEMPLATES = os.path.join(BASE, "docs", "templates")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_csv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_text(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------------------
# Existence checks
# ---------------------------------------------------------------------------

def test_v1ho_methodology_doc_exists():
    assert os.path.exists(os.path.join(DOCS, "protocolo_c_dossies_eventos_candidatos.md"))


def test_v1ho_template_dossier_exists():
    assert os.path.exists(os.path.join(TEMPLATES, "protocolo_c_dossie_evento_candidato.md"))


def test_v1ho_template_log_busca_exists():
    assert os.path.exists(os.path.join(TEMPLATES, "protocolo_c_log_busca_evento.md"))


def test_v1ho_dossier_schema_exists():
    assert os.path.exists(os.path.join(SCHEMAS, "event_evidence_dossier_schema.csv"))


def test_v1ho_requirements_schema_exists():
    assert os.path.exists(os.path.join(SCHEMAS, "event_evidence_requirements_schema.csv"))


def test_v1ho_decision_schema_exists():
    assert os.path.exists(os.path.join(SCHEMAS, "event_dossier_decision_schema.csv"))


def test_v1ho_dossier_registry_exists():
    assert os.path.exists(os.path.join(DATASETS, "event_evidence_dossier_registry.csv"))


def test_v1ho_requirements_registry_exists():
    assert os.path.exists(os.path.join(DATASETS, "event_evidence_requirements_registry.csv"))


def test_v1ho_decision_registry_exists():
    assert os.path.exists(os.path.join(DATASETS, "event_dossier_decision_registry.csv"))


# ---------------------------------------------------------------------------
# Schema fields presence
# ---------------------------------------------------------------------------

def test_v1ho_dossier_schema_required_fields():
    rows = read_csv(os.path.join(SCHEMAS, "event_evidence_dossier_schema.csv"))
    names = {r["field_name"] for r in rows}
    required = {
        "dossier_id", "screening_id", "region", "candidate_event_name",
        "event_status", "dossier_status",
        "minimum_temporal_evidence_status", "minimum_spatial_evidence_status",
        "minimum_source_traceability_status",
        "can_advance_to_source_review", "can_advance_to_review_gate",
        "can_support_ground_reference_candidate",
        "operational_ground_truth_status", "protocol_b_status", "multimodal_status",
        "allowed_claim", "forbidden_claim",
    }
    missing = required - names
    assert not missing, f"Campos ausentes no schema de dossiê: {missing}"


def test_v1ho_requirements_schema_required_fields():
    rows = read_csv(os.path.join(SCHEMAS, "event_evidence_requirements_schema.csv"))
    names = {r["field_name"] for r in rows}
    required = {
        "requirement_id", "dossier_id", "region", "candidate_event_name",
        "requirement_type", "required_evidence", "target_gate",
        "required_source_strength", "current_status", "blocking_if_missing",
        "review_needed", "allowed_if_satisfied", "forbidden_if_missing",
    }
    missing = required - names
    assert not missing, f"Campos ausentes no schema de requisitos: {missing}"


def test_v1ho_decision_schema_required_fields():
    rows = read_csv(os.path.join(SCHEMAS, "event_dossier_decision_schema.csv"))
    names = {r["field_name"] for r in rows}
    required = {
        "decision_id", "dossier_id", "screening_id", "region",
        "dossier_status", "decision_status", "decision_reason",
        "allowed_next_step", "forbidden_next_step",
        "can_create_patch_event_link", "can_start_review_gate",
        "can_reassess_protocol_b", "can_start_multimodal",
        "allowed_claim", "forbidden_claim",
    }
    missing = required - names
    assert not missing, f"Campos ausentes no schema de decisão: {missing}"


# ---------------------------------------------------------------------------
# Dossier registry — controlled values
# ---------------------------------------------------------------------------

VALID_DOSSIER_STATUS = {
    "DOSSIER_NOT_STARTED", "DOSSIER_OPEN", "DOSSIER_PARTIAL",
    "DOSSIER_READY_FOR_SOURCE_REVIEW", "DOSSIER_READY_FOR_REVIEW_GATE",
    "DOSSIER_BLOCKED", "METHOD_REFERENCE_ONLY",
}
VALID_EVIDENCE_STATUS = {
    "AVAILABLE", "PARTIAL", "MISSING", "NOT_ASSESSED", "METHOD_REFERENCE_ONLY",
}
VALID_TRACEABILITY_STATUS = {
    "TRACEABLE", "PARTIAL", "MISSING", "NOT_ASSESSED", "METHOD_REFERENCE_ONLY",
}
VALID_OGT_STATUS = {
    "NOT_ESTABLISHED", "BLOCKED_PENDING_EVIDENCE",
    "FUTURE_ELIGIBILITY_ONLY", "METHOD_REFERENCE_ONLY",
}
VALID_PROTOCOL_B = {"BLOCKED", "FUTURE_REASSESSMENT_ONLY", "METHOD_REFERENCE_ONLY"}
VALID_MULTIMODAL = {"HOLD", "FUTURE_READINESS_ONLY", "METHOD_REFERENCE_ONLY"}


def test_v1ho_dossier_status_valid():
    rows = read_csv(os.path.join(DATASETS, "event_evidence_dossier_registry.csv"))
    for row in rows:
        assert row["dossier_status"] in VALID_DOSSIER_STATUS, \
            f"{row['dossier_id']}: dossier_status inválido '{row['dossier_status']}'"


def test_v1ho_temporal_evidence_status_valid():
    rows = read_csv(os.path.join(DATASETS, "event_evidence_dossier_registry.csv"))
    for row in rows:
        assert row["minimum_temporal_evidence_status"] in VALID_EVIDENCE_STATUS, \
            f"{row['dossier_id']}: minimum_temporal_evidence_status inválido"


def test_v1ho_spatial_evidence_status_valid():
    rows = read_csv(os.path.join(DATASETS, "event_evidence_dossier_registry.csv"))
    for row in rows:
        assert row["minimum_spatial_evidence_status"] in VALID_EVIDENCE_STATUS, \
            f"{row['dossier_id']}: minimum_spatial_evidence_status inválido"


def test_v1ho_ogt_status_valid():
    rows = read_csv(os.path.join(DATASETS, "event_evidence_dossier_registry.csv"))
    for row in rows:
        assert row["operational_ground_truth_status"] in VALID_OGT_STATUS, \
            f"{row['dossier_id']}: operational_ground_truth_status inválido"


def test_v1ho_protocol_b_status_valid():
    rows = read_csv(os.path.join(DATASETS, "event_evidence_dossier_registry.csv"))
    for row in rows:
        assert row["protocol_b_status"] in VALID_PROTOCOL_B, \
            f"{row['dossier_id']}: protocol_b_status inválido"


def test_v1ho_multimodal_status_valid():
    rows = read_csv(os.path.join(DATASETS, "event_evidence_dossier_registry.csv"))
    for row in rows:
        assert row["multimodal_status"] in VALID_MULTIMODAL, \
            f"{row['dossier_id']}: multimodal_status inválido"


# ---------------------------------------------------------------------------
# Dossier registry — guardrails obrigatórios
# ---------------------------------------------------------------------------

def test_v1ho_dossier_can_support_ground_reference_always_false():
    rows = read_csv(os.path.join(DATASETS, "event_evidence_dossier_registry.csv"))
    for row in rows:
        assert row["can_support_ground_reference_candidate"] == "false", \
            f"{row['dossier_id']}: can_support_ground_reference_candidate deve ser 'false'"


def test_v1ho_dossier_can_advance_to_review_gate_false():
    rows = read_csv(os.path.join(DATASETS, "event_evidence_dossier_registry.csv"))
    for row in rows:
        assert row["can_advance_to_review_gate"] == "false", \
            f"{row['dossier_id']}: can_advance_to_review_gate deve ser 'false' nesta etapa"


def test_v1ho_dossier_ogt_not_established():
    rows = read_csv(os.path.join(DATASETS, "event_evidence_dossier_registry.csv"))
    for row in rows:
        assert row["operational_ground_truth_status"] == "NOT_ESTABLISHED", \
            f"{row['dossier_id']}: operational_ground_truth_status deve ser NOT_ESTABLISHED"


def test_v1ho_dossier_protocol_b_blocked():
    rows = read_csv(os.path.join(DATASETS, "event_evidence_dossier_registry.csv"))
    for row in rows:
        assert row["protocol_b_status"] == "BLOCKED", \
            f"{row['dossier_id']}: protocol_b_status deve ser BLOCKED"


def test_v1ho_dossier_multimodal_hold():
    rows = read_csv(os.path.join(DATASETS, "event_evidence_dossier_registry.csv"))
    for row in rows:
        assert row["multimodal_status"] == "HOLD", \
            f"{row['dossier_id']}: multimodal_status deve ser HOLD"


# ---------------------------------------------------------------------------
# Dossier registry — referências cruzadas com screening
# ---------------------------------------------------------------------------

def test_v1ho_dossier_screening_refs_exist():
    screening = read_csv(os.path.join(DATASETS, "event_candidate_screening_registry.csv"))
    valid_ids = {r["screening_id"] for r in screening}
    dossiers = read_csv(os.path.join(DATASETS, "event_evidence_dossier_registry.csv"))
    for row in dossiers:
        assert row["screening_id"] in valid_ids, \
            f"{row['dossier_id']}: screening_id '{row['screening_id']}' não existe no registry"


def test_v1ho_dossier_all_regions_covered():
    rows = read_csv(os.path.join(DATASETS, "event_evidence_dossier_registry.csv"))
    regions = {r["region"] for r in rows}
    for region in ("RECIFE", "PETROPOLIS", "CURITIBA"):
        assert region in regions, f"Região {region} ausente nos dossiês"


def test_v1ho_event_search_target_not_confirmed():
    """Eventos EVENT_SEARCH_TARGET não podem ter dossier_status DOSSIER_READY_FOR_REVIEW_GATE."""
    dossiers = read_csv(os.path.join(DATASETS, "event_evidence_dossier_registry.csv"))
    for row in dossiers:
        if row["event_status"] == "EVENT_SEARCH_TARGET":
            assert row["dossier_status"] != "DOSSIER_READY_FOR_REVIEW_GATE", \
                f"{row['dossier_id']}: EVENT_SEARCH_TARGET não pode ter dossiê READY_FOR_REVIEW_GATE"


def test_v1ho_no_confirmed_by_source_events_in_dossier():
    rows = read_csv(os.path.join(DATASETS, "event_evidence_dossier_registry.csv"))
    confirmed = [r["dossier_id"] for r in rows if r["event_status"] == "CONFIRMED_BY_SOURCE"]
    assert not confirmed, f"Dossiês com CONFIRMED_BY_SOURCE indevido: {confirmed}"


# ---------------------------------------------------------------------------
# Requirements registry — controlled values
# ---------------------------------------------------------------------------

VALID_REQ_TYPE = {
    "EVENT_CONFIRMATION", "TEMPORAL_EVIDENCE", "SPATIAL_EVIDENCE",
    "SOURCE_TRACEABILITY", "SOURCE_STRENGTH", "UNCERTAINTY_DOCUMENTATION",
    "LICENSE_PROVENANCE", "PATCH_RELATION", "REVIEW_GATE",
    "INDEPENDENT_CORROBORATION", "PROMOTION_DECISION",
}
VALID_REQ_STATUS = {
    "SATISFIED", "PARTIAL", "MISSING", "NOT_ASSESSED", "METHOD_REFERENCE_ONLY",
}
VALID_SOURCE_STRENGTH = {
    "DIRECT_OBSERVATION", "OFFICIAL_DOCUMENTATION", "EXPERT_ANNOTATION",
    "OPERATIONAL_ALGORITHMIC", "MODELLED_CONTEXT", "STRUCTURAL_CONTEXT",
    "METHOD_REFERENCE_ONLY",
}


def test_v1ho_requirements_type_valid():
    rows = read_csv(os.path.join(DATASETS, "event_evidence_requirements_registry.csv"))
    for row in rows:
        assert row["requirement_type"] in VALID_REQ_TYPE, \
            f"{row['requirement_id']}: requirement_type inválido '{row['requirement_type']}'"


def test_v1ho_requirements_status_valid():
    rows = read_csv(os.path.join(DATASETS, "event_evidence_requirements_registry.csv"))
    for row in rows:
        assert row["current_status"] in VALID_REQ_STATUS, \
            f"{row['requirement_id']}: current_status inválido '{row['current_status']}'"


def test_v1ho_requirements_source_strength_valid():
    rows = read_csv(os.path.join(DATASETS, "event_evidence_requirements_registry.csv"))
    for row in rows:
        assert row["required_source_strength"] in VALID_SOURCE_STRENGTH, \
            f"{row['requirement_id']}: required_source_strength inválido"


def test_v1ho_requirements_dossier_refs_exist():
    dossiers = read_csv(os.path.join(DATASETS, "event_evidence_dossier_registry.csv"))
    valid_ids = {r["dossier_id"] for r in dossiers}
    reqs = read_csv(os.path.join(DATASETS, "event_evidence_requirements_registry.csv"))
    for row in reqs:
        assert row["dossier_id"] in valid_ids, \
            f"{row['requirement_id']}: dossier_id '{row['dossier_id']}' não existe"


def test_v1ho_critical_requirements_blocking():
    """Requisitos críticos (G1, G4, G7, G9) devem ter blocking_if_missing=true."""
    rows = read_csv(os.path.join(DATASETS, "event_evidence_requirements_registry.csv"))
    critical = {"EVENT_CONFIRMATION", "SPATIAL_EVIDENCE", "REVIEW_GATE", "PROMOTION_DECISION"}
    for row in rows:
        if row["requirement_type"] in critical:
            assert row["blocking_if_missing"] == "true", \
                f"{row['requirement_id']}: requisito crítico deve ter blocking_if_missing=true"


def test_v1ho_no_satisfied_requirements_without_evidence():
    """Nenhum requisito deve estar SATISFIED nesta etapa — nenhuma fonte foi adquirida."""
    rows = read_csv(os.path.join(DATASETS, "event_evidence_requirements_registry.csv"))
    satisfied = [r["requirement_id"] for r in rows if r["current_status"] == "SATISFIED"]
    assert not satisfied, \
        f"Requisitos marcados SATISFIED sem evidência local: {satisfied}"


def test_v1ho_dino_cannot_satisfy_event_confirmation():
    """STRUCTURAL_CONTEXT não pode ser required_source_strength para EVENT_CONFIRMATION."""
    rows = read_csv(os.path.join(DATASETS, "event_evidence_requirements_registry.csv"))
    for row in rows:
        if row["requirement_type"] == "EVENT_CONFIRMATION":
            assert row["required_source_strength"] != "STRUCTURAL_CONTEXT", \
                f"{row['requirement_id']}: EVENT_CONFIRMATION não pode ter STRUCTURAL_CONTEXT como fonte"


def test_v1ho_dino_cannot_satisfy_temporal_evidence():
    """STRUCTURAL_CONTEXT não pode ser required_source_strength para TEMPORAL_EVIDENCE."""
    rows = read_csv(os.path.join(DATASETS, "event_evidence_requirements_registry.csv"))
    for row in rows:
        if row["requirement_type"] == "TEMPORAL_EVIDENCE":
            assert row["required_source_strength"] != "STRUCTURAL_CONTEXT", \
                f"{row['requirement_id']}: TEMPORAL_EVIDENCE não pode ter STRUCTURAL_CONTEXT como fonte"


def test_v1ho_dino_cannot_satisfy_spatial_evidence():
    """STRUCTURAL_CONTEXT não pode ser required_source_strength para SPATIAL_EVIDENCE."""
    rows = read_csv(os.path.join(DATASETS, "event_evidence_requirements_registry.csv"))
    for row in rows:
        if row["requirement_type"] == "SPATIAL_EVIDENCE":
            assert row["required_source_strength"] != "STRUCTURAL_CONTEXT", \
                f"{row['requirement_id']}: SPATIAL_EVIDENCE não pode ter STRUCTURAL_CONTEXT como fonte"


def test_v1ho_requirements_all_regions_covered():
    rows = read_csv(os.path.join(DATASETS, "event_evidence_requirements_registry.csv"))
    regions = {r["region"] for r in rows}
    for region in ("RECIFE", "PETROPOLIS", "CURITIBA"):
        assert region in regions, f"Região {region} ausente nos requisitos"


def test_v1ho_each_dossier_has_requirements():
    dossiers = read_csv(os.path.join(DATASETS, "event_evidence_dossier_registry.csv"))
    reqs = read_csv(os.path.join(DATASETS, "event_evidence_requirements_registry.csv"))
    req_dossiers = {r["dossier_id"] for r in reqs}
    for dos in dossiers:
        assert dos["dossier_id"] in req_dossiers, \
            f"Dossiê {dos['dossier_id']} não tem requisitos registrados"


# ---------------------------------------------------------------------------
# Decision registry — controlled values e guardrails
# ---------------------------------------------------------------------------

VALID_DECISION_STATUS = {
    "CONTINUE_SOURCE_SEARCH", "REQUEST_FORMAL_SOURCE", "WAIT_FOR_ACQUISITION",
    "READY_FOR_SOURCE_REVIEW", "READY_FOR_REVIEW_GATE",
    "BLOCK_EVENT_FOR_NOW", "METHOD_REFERENCE_ONLY",
}


def test_v1ho_decision_status_valid():
    rows = read_csv(os.path.join(DATASETS, "event_dossier_decision_registry.csv"))
    for row in rows:
        assert row["decision_status"] in VALID_DECISION_STATUS, \
            f"{row['decision_id']}: decision_status inválido '{row['decision_status']}'"


def test_v1ho_decision_can_reassess_protocol_b_always_false():
    rows = read_csv(os.path.join(DATASETS, "event_dossier_decision_registry.csv"))
    for row in rows:
        assert row["can_reassess_protocol_b"] == "false", \
            f"{row['decision_id']}: can_reassess_protocol_b deve ser 'false'"


def test_v1ho_decision_can_start_multimodal_always_false():
    rows = read_csv(os.path.join(DATASETS, "event_dossier_decision_registry.csv"))
    for row in rows:
        assert row["can_start_multimodal"] == "false", \
            f"{row['decision_id']}: can_start_multimodal deve ser 'false'"


def test_v1ho_decision_can_start_review_gate_false():
    rows = read_csv(os.path.join(DATASETS, "event_dossier_decision_registry.csv"))
    for row in rows:
        assert row["can_start_review_gate"] == "false", \
            f"{row['decision_id']}: can_start_review_gate deve ser 'false' nesta etapa"


def test_v1ho_decision_dossier_refs_exist():
    dossiers = read_csv(os.path.join(DATASETS, "event_evidence_dossier_registry.csv"))
    valid_dos = {r["dossier_id"] for r in dossiers}
    decisions = read_csv(os.path.join(DATASETS, "event_dossier_decision_registry.csv"))
    for row in decisions:
        assert row["dossier_id"] in valid_dos, \
            f"{row['decision_id']}: dossier_id '{row['dossier_id']}' não existe"


def test_v1ho_decision_screening_refs_exist():
    screening = read_csv(os.path.join(DATASETS, "event_candidate_screening_registry.csv"))
    valid_scr = {r["screening_id"] for r in screening}
    decisions = read_csv(os.path.join(DATASETS, "event_dossier_decision_registry.csv"))
    for row in decisions:
        assert row["screening_id"] in valid_scr, \
            f"{row['decision_id']}: screening_id '{row['screening_id']}' não existe"


def test_v1ho_one_decision_per_dossier():
    dossiers = read_csv(os.path.join(DATASETS, "event_evidence_dossier_registry.csv"))
    decisions = read_csv(os.path.join(DATASETS, "event_dossier_decision_registry.csv"))
    dec_dossiers = {r["dossier_id"] for r in decisions}
    for dos in dossiers:
        assert dos["dossier_id"] in dec_dossiers, \
            f"Dossiê {dos['dossier_id']} sem decisão registrada"


def test_v1ho_decision_forbidden_claim_blocks_ground_truth():
    rows = read_csv(os.path.join(DATASETS, "event_dossier_decision_registry.csv"))
    for row in rows:
        fc = row.get("forbidden_claim", "").upper()
        assert "OPERATIONAL_GROUND_TRUTH" in fc or "GROUND_TRUTH_OPERACIONAL" in fc, \
            f"{row['decision_id']}: forbidden_claim deve bloquear ground truth operacional"


def test_v1ho_decision_forbidden_claim_blocks_training():
    rows = read_csv(os.path.join(DATASETS, "event_dossier_decision_registry.csv"))
    for row in rows:
        fc = row.get("forbidden_claim", "").upper()
        assert "TRAINING_LABEL" in fc or "SUPERVISED_TRAINING" in fc, \
            f"{row['decision_id']}: forbidden_claim deve bloquear training label ou supervised training"


def test_v1ho_decision_forbidden_next_step_blocks_training():
    rows = read_csv(os.path.join(DATASETS, "event_dossier_decision_registry.csv"))
    for row in rows:
        fns = row.get("forbidden_next_step", "").lower()
        assert "treino" in fns or "training" in fns or "label" in fns, \
            f"{row['decision_id']}: forbidden_next_step deve bloquear treino/training"


def test_v1ho_decision_forbidden_next_step_blocks_multimodal():
    rows = read_csv(os.path.join(DATASETS, "event_dossier_decision_registry.csv"))
    for row in rows:
        fns = row.get("forbidden_next_step", "").lower()
        assert "multimodal" in fns, \
            f"{row['decision_id']}: forbidden_next_step deve bloquear multimodal"


# ---------------------------------------------------------------------------
# Documento metodológico
# ---------------------------------------------------------------------------

def test_v1ho_doc_affirms_no_data_download():
    text = read_text(os.path.join(DOCS, "protocolo_c_dossies_eventos_candidatos.md"))
    markers = ["Não baixa dados", "não baixa dados", "não baixar dado", "metadata-only"]
    assert any(m in text for m in markers), \
        "Documento não afirma que nenhum dado é baixado"


def test_v1ho_doc_affirms_multimodal_hold():
    text = read_text(os.path.join(DOCS, "protocolo_c_dossies_eventos_candidatos.md"))
    assert "multimodal" in text.lower() and "hold" in text.lower(), \
        "Documento não afirma que multimodal está em hold"


def test_v1ho_doc_affirms_dino_review_only():
    text = read_text(os.path.join(DOCS, "protocolo_c_dossies_eventos_candidatos.md"))
    assert "DINOv2" in text or "DINO" in text, "Documento não menciona DINO"
    dino_markers = ["review-only", "não fecha", "não satisfaz", "não constituem evidência"]
    assert any(m in text for m in dino_markers), \
        "Documento não afirma que DINO é review-only"


def test_v1ho_doc_defines_dossier_states():
    text = read_text(os.path.join(DOCS, "protocolo_c_dossies_eventos_candidatos.md"))
    for state in ("DOSSIER_OPEN", "DOSSIER_PARTIAL", "DOSSIER_BLOCKED", "DOSSIER_READY_FOR_SOURCE_REVIEW"):
        assert state in text, f"Documento não define estado '{state}'"


def test_v1ho_doc_no_operational_ground_truth_claim():
    text = read_text(os.path.join(DOCS, "protocolo_c_dossies_eventos_candidatos.md"))
    forbidden = [
        "ground truth operacional declarado",
        "referência operacional estabelecida",
        "operational ground truth established",
    ]
    for term in forbidden:
        assert term.lower() not in text.lower(), \
            f"Documento contém claim proibido: '{term}'"


def test_v1ho_doc_no_supervised_training_claim():
    text = read_text(os.path.join(DOCS, "protocolo_c_dossies_eventos_candidatos.md"))
    forbidden = ["treinamento supervisionado", "supervised training output", "flood label criado"]
    for term in forbidden:
        assert term.lower() not in text.lower(), \
            f"Documento contém term proibido: '{term}'"


def test_v1ho_doc_mentions_minimum_evidence():
    text = read_text(os.path.join(DOCS, "protocolo_c_dossies_eventos_candidatos.md"))
    assert "evidência mínima" in text.lower() or "mínimo" in text.lower(), \
        "Documento não menciona evidência mínima"


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

def test_v1ho_template_dossier_has_placeholders():
    text = read_text(os.path.join(TEMPLATES, "protocolo_c_dossie_evento_candidato.md"))
    for ph in ("[DOSSIER_ID]", "[REGIAO]", "[FONTES_ALVO]", "[CLAIM_PERMITIDO]", "[CLAIM_PROIBIDO]"):
        assert ph in text, f"Template dossiê não contém placeholder '{ph}'"


def test_v1ho_template_dossier_not_ground_truth():
    text = read_text(os.path.join(TEMPLATES, "protocolo_c_dossie_evento_candidato.md"))
    assert "não é ground truth" in text.lower() or "não constitui declaração" in text.lower(), \
        "Template dossiê não afirma que não é ground truth"


def test_v1ho_template_log_busca_has_fields():
    text = read_text(os.path.join(TEMPLATES, "protocolo_c_log_busca_evento.md"))
    for field in ("[LOG_ID]", "[DOSSIER_ID]", "[FONTE_CONSULTADA]", "[DECISAO]"):
        assert field in text, f"Template log de busca não contém campo '{field}'"


# ---------------------------------------------------------------------------
# Segurança — sem paths privados
# ---------------------------------------------------------------------------

def test_v1ho_no_private_paths_in_registries():
    files = [
        os.path.join(DATASETS, "event_evidence_dossier_registry.csv"),
        os.path.join(DATASETS, "event_evidence_requirements_registry.csv"),
        os.path.join(DATASETS, "event_dossier_decision_registry.csv"),
    ]
    private = ["C:\\Users\\", "/home/", "local_runs/", ".npz", ".tif"]
    for path in files:
        content = read_text(path)
        for marker in private:
            assert marker not in content, \
                f"Path privado '{marker}' em {os.path.basename(path)}"
