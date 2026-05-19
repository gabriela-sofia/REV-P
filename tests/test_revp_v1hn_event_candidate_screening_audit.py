"""
v1hn — Triagem de eventos candidatos reais por região.
Audita os três registros de triagem e seus schemas.
"""
import csv
import os
import pytest

BASE = os.path.join(os.path.dirname(__file__), "..")
DATASETS = os.path.join(BASE, "datasets")
SCHEMAS = os.path.join(DATASETS, "schemas")
DOCS = os.path.join(BASE, "docs", "metodologia_cientifica")


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

def test_v1hn_methodology_doc_exists():
    path = os.path.join(DOCS, "protocolo_c_triagem_eventos_candidatos.md")
    assert os.path.exists(path), "Documento metodológico v1hn não encontrado"


def test_v1hn_screening_registry_exists():
    assert os.path.exists(os.path.join(DATASETS, "event_candidate_screening_registry.csv"))


def test_v1hn_screening_schema_exists():
    assert os.path.exists(os.path.join(SCHEMAS, "event_candidate_screening_schema.csv"))


def test_v1hn_backlog_registry_exists():
    assert os.path.exists(os.path.join(DATASETS, "event_source_search_backlog.csv"))


def test_v1hn_backlog_schema_exists():
    assert os.path.exists(os.path.join(SCHEMAS, "event_source_search_backlog_schema.csv"))


def test_v1hn_scope_registry_exists():
    assert os.path.exists(os.path.join(DATASETS, "event_patch_screening_scope.csv"))


def test_v1hn_scope_schema_exists():
    assert os.path.exists(os.path.join(SCHEMAS, "event_patch_screening_scope_schema.csv"))


# ---------------------------------------------------------------------------
# Schema field presence
# ---------------------------------------------------------------------------

def test_v1hn_screening_schema_required_fields():
    rows = read_csv(os.path.join(SCHEMAS, "event_candidate_screening_schema.csv"))
    field_names = {r["field_name"] for r in rows}
    required = {
        "screening_id", "region", "candidate_status", "search_priority",
        "gates_not_addressable_by_dino", "operational_ground_truth_status",
        "promotion_allowed", "protocol_b_status", "multimodal_status",
        "dino_gate_closure_allowed",
    }
    missing = required - field_names
    assert not missing, f"Campos obrigatórios ausentes no schema: {missing}"


def test_v1hn_backlog_schema_required_fields():
    rows = read_csv(os.path.join(SCHEMAS, "event_source_search_backlog_schema.csv"))
    field_names = {r["field_name"] for r in rows}
    required = {
        "backlog_id", "screening_id", "region", "source_name",
        "source_family", "access_mode", "search_status",
    }
    missing = required - field_names
    assert not missing, f"Campos obrigatórios ausentes no schema do backlog: {missing}"


def test_v1hn_scope_schema_required_fields():
    rows = read_csv(os.path.join(SCHEMAS, "event_patch_screening_scope_schema.csv"))
    field_names = {r["field_name"] for r in rows}
    required = {
        "scope_id", "screening_id", "patch_id", "region",
        "in_search_perimeter", "spatial_overlap_assessed",
        "temporal_alignment_assessed", "dino_used_as_support_only",
        "promotion_allowed",
    }
    missing = required - field_names
    assert not missing, f"Campos obrigatórios ausentes no schema de escopo: {missing}"


# ---------------------------------------------------------------------------
# Screening registry — controlled values
# ---------------------------------------------------------------------------

VALID_REGIONS = {"RECIFE", "PETROPOLIS", "CURITIBA", "METHOD_REFERENCE"}
VALID_CANDIDATE_STATUS = {
    "EVENT_SEARCH_TARGET", "PENDING_SOURCE_REVIEW",
    "CONFIRMED_BY_SOURCE", "BLOCKED", "METHOD_REFERENCE_ONLY",
}
VALID_SEARCH_PRIORITY = {"HIGH", "MEDIUM", "LOW", "METHOD_REFERENCE_ONLY"}
VALID_OGT_STATUS = {"NOT_ESTABLISHED", "BLOCKED_PENDING_EVIDENCE"}


def test_v1hn_screening_regions_valid():
    rows = read_csv(os.path.join(DATASETS, "event_candidate_screening_registry.csv"))
    for row in rows:
        assert row["region"] in VALID_REGIONS, \
            f"{row['screening_id']}: region inválida '{row['region']}'"


def test_v1hn_screening_candidate_status_valid():
    rows = read_csv(os.path.join(DATASETS, "event_candidate_screening_registry.csv"))
    for row in rows:
        assert row["candidate_status"] in VALID_CANDIDATE_STATUS, \
            f"{row['screening_id']}: candidate_status inválido '{row['candidate_status']}'"


def test_v1hn_screening_search_priority_valid():
    rows = read_csv(os.path.join(DATASETS, "event_candidate_screening_registry.csv"))
    for row in rows:
        assert row["search_priority"] in VALID_SEARCH_PRIORITY, \
            f"{row['screening_id']}: search_priority inválida '{row['search_priority']}'"


def test_v1hn_screening_ogt_status_valid():
    rows = read_csv(os.path.join(DATASETS, "event_candidate_screening_registry.csv"))
    for row in rows:
        assert row["operational_ground_truth_status"] in VALID_OGT_STATUS, \
            f"{row['screening_id']}: operational_ground_truth_status inválido"


# ---------------------------------------------------------------------------
# Screening registry — guardrails obrigatórios
# ---------------------------------------------------------------------------

def test_v1hn_screening_promotion_always_false():
    rows = read_csv(os.path.join(DATASETS, "event_candidate_screening_registry.csv"))
    for row in rows:
        assert row["promotion_allowed"] == "false", \
            f"{row['screening_id']}: promotion_allowed deve ser 'false'"


def test_v1hn_screening_protocol_b_blocked():
    rows = read_csv(os.path.join(DATASETS, "event_candidate_screening_registry.csv"))
    for row in rows:
        assert row["protocol_b_status"] == "BLOCKED", \
            f"{row['screening_id']}: protocol_b_status deve ser 'BLOCKED'"


def test_v1hn_screening_multimodal_hold():
    rows = read_csv(os.path.join(DATASETS, "event_candidate_screening_registry.csv"))
    for row in rows:
        assert row["multimodal_status"] == "HOLD", \
            f"{row['screening_id']}: multimodal_status deve ser 'HOLD'"


def test_v1hn_screening_dino_gate_closure_always_false():
    rows = read_csv(os.path.join(DATASETS, "event_candidate_screening_registry.csv"))
    for row in rows:
        assert row["dino_gate_closure_allowed"] == "false", \
            f"{row['screening_id']}: dino_gate_closure_allowed deve ser 'false'"


def test_v1hn_screening_ogt_not_established():
    rows = read_csv(os.path.join(DATASETS, "event_candidate_screening_registry.csv"))
    for row in rows:
        assert row["operational_ground_truth_status"] == "NOT_ESTABLISHED", \
            f"{row['screening_id']}: operational_ground_truth_status deve ser NOT_ESTABLISHED"


# ---------------------------------------------------------------------------
# Screening registry — regiões e eventos presentes
# ---------------------------------------------------------------------------

def test_v1hn_screening_all_regions_present():
    rows = read_csv(os.path.join(DATASETS, "event_candidate_screening_registry.csv"))
    regions = {r["region"] for r in rows}
    for region in ("RECIFE", "PETROPOLIS", "CURITIBA"):
        assert region in regions, f"Região {region} ausente no screening registry"


def test_v1hn_screening_recife_events_present():
    rows = read_csv(os.path.join(DATASETS, "event_candidate_screening_registry.csv"))
    recife_ids = {r["screening_id"] for r in rows if r["region"] == "RECIFE"}
    assert len(recife_ids) >= 2, "Esperados pelo menos 2 eventos candidatos para Recife"


def test_v1hn_screening_petropolis_event_present():
    rows = read_csv(os.path.join(DATASETS, "event_candidate_screening_registry.csv"))
    pet_ids = {r["screening_id"] for r in rows if r["region"] == "PETROPOLIS"}
    assert len(pet_ids) >= 1, "Esperado pelo menos 1 evento candidato para Petrópolis"


def test_v1hn_screening_curitiba_events_present():
    rows = read_csv(os.path.join(DATASETS, "event_candidate_screening_registry.csv"))
    cur_ids = {r["screening_id"] for r in rows if r["region"] == "CURITIBA"}
    assert len(cur_ids) >= 2, "Esperados pelo menos 2 eventos candidatos para Curitiba"


def test_v1hn_petropolis_event_is_pending_source_review():
    rows = read_csv(os.path.join(DATASETS, "event_candidate_screening_registry.csv"))
    pet_rows = [r for r in rows if r["region"] == "PETROPOLIS"]
    assert pet_rows, "Nenhum evento de Petrópolis encontrado"
    for row in pet_rows:
        assert row["candidate_status"] in ("PENDING_SOURCE_REVIEW", "EVENT_SEARCH_TARGET"), \
            f"{row['screening_id']}: status inválido para Petrópolis"


def test_v1hn_no_confirmed_by_source_events():
    """Nenhum evento deve estar CONFIRMED_BY_SOURCE nesta etapa."""
    rows = read_csv(os.path.join(DATASETS, "event_candidate_screening_registry.csv"))
    confirmed = [r["screening_id"] for r in rows if r["candidate_status"] == "CONFIRMED_BY_SOURCE"]
    assert not confirmed, f"Eventos marcados CONFIRMED_BY_SOURCE indevidamente: {confirmed}"


# ---------------------------------------------------------------------------
# Backlog — controlled values
# ---------------------------------------------------------------------------

VALID_SOURCE_FAMILY = {
    "CIVIL_DEFENSE_RECORD", "OFFICIAL_OBSERVED_FLOOD_MAP",
    "SENTINEL_EVENT_IMAGE", "OPERATIONAL_FLOOD_PRODUCT",
    "MUNICIPAL_GIS_LAYER", "DINO_STRUCTURAL_SUPPORT", "METHOD_REFERENCE",
}
VALID_ACCESS_MODE = {
    "FORMAL_REQUEST", "PUBLIC_PORTAL_REVIEW",
    "PUBLIC_DOWNLOAD", "METHOD_REFERENCE_ONLY",
}
VALID_SEARCH_STATUS = {
    "NOT_STARTED", "IN_PROGRESS", "SOURCE_FOUND",
    "SOURCE_NOT_FOUND", "BLOCKED",
}


def test_v1hn_backlog_source_family_valid():
    rows = read_csv(os.path.join(DATASETS, "event_source_search_backlog.csv"))
    for row in rows:
        assert row["source_family"] in VALID_SOURCE_FAMILY, \
            f"{row['backlog_id']}: source_family inválida '{row['source_family']}'"


def test_v1hn_backlog_access_mode_valid():
    rows = read_csv(os.path.join(DATASETS, "event_source_search_backlog.csv"))
    for row in rows:
        assert row["access_mode"] in VALID_ACCESS_MODE, \
            f"{row['backlog_id']}: access_mode inválido '{row['access_mode']}'"


def test_v1hn_backlog_search_status_valid():
    rows = read_csv(os.path.join(DATASETS, "event_source_search_backlog.csv"))
    for row in rows:
        assert row["search_status"] in VALID_SEARCH_STATUS, \
            f"{row['backlog_id']}: search_status inválido '{row['search_status']}'"


def test_v1hn_backlog_all_not_started():
    """Nenhuma busca executada nesta etapa — todas devem ser NOT_STARTED."""
    rows = read_csv(os.path.join(DATASETS, "event_source_search_backlog.csv"))
    for row in rows:
        assert row["search_status"] == "NOT_STARTED", \
            f"{row['backlog_id']}: search_status deve ser NOT_STARTED nesta etapa"


def test_v1hn_backlog_screening_refs_exist():
    """Todos os screening_id referenciados no backlog devem existir no registry."""
    screening_rows = read_csv(os.path.join(DATASETS, "event_candidate_screening_registry.csv"))
    valid_ids = {r["screening_id"] for r in screening_rows}
    backlog_rows = read_csv(os.path.join(DATASETS, "event_source_search_backlog.csv"))
    for row in backlog_rows:
        assert row["screening_id"] in valid_ids, \
            f"{row['backlog_id']}: screening_id '{row['screening_id']}' não existe no registry"


def test_v1hn_backlog_all_regions_covered():
    rows = read_csv(os.path.join(DATASETS, "event_source_search_backlog.csv"))
    regions = {r["region"] for r in rows}
    for region in ("RECIFE", "PETROPOLIS", "CURITIBA"):
        assert region in regions, f"Região {region} ausente no backlog"


# ---------------------------------------------------------------------------
# Scope — controlled values e guardrails
# ---------------------------------------------------------------------------

def test_v1hn_scope_spatial_overlap_never_assessed():
    """spatial_overlap_assessed deve ser 'false' — nenhum dado bruto foi lido."""
    rows = read_csv(os.path.join(DATASETS, "event_patch_screening_scope.csv"))
    for row in rows:
        assert row["spatial_overlap_assessed"] == "false", \
            f"{row['scope_id']}: spatial_overlap_assessed deve ser 'false'"


def test_v1hn_scope_temporal_alignment_never_assessed():
    """temporal_alignment_assessed deve ser 'false' — metadata-only."""
    rows = read_csv(os.path.join(DATASETS, "event_patch_screening_scope.csv"))
    for row in rows:
        assert row["temporal_alignment_assessed"] == "false", \
            f"{row['scope_id']}: temporal_alignment_assessed deve ser 'false'"


def test_v1hn_scope_dino_support_only():
    rows = read_csv(os.path.join(DATASETS, "event_patch_screening_scope.csv"))
    for row in rows:
        assert row["dino_used_as_support_only"] == "true", \
            f"{row['scope_id']}: dino_used_as_support_only deve ser 'true'"


def test_v1hn_scope_promotion_always_false():
    rows = read_csv(os.path.join(DATASETS, "event_patch_screening_scope.csv"))
    for row in rows:
        assert row["promotion_allowed"] == "false", \
            f"{row['scope_id']}: promotion_allowed deve ser 'false'"


def test_v1hn_scope_screening_refs_exist():
    """Todos os screening_id referenciados no scope devem existir no registry."""
    screening_rows = read_csv(os.path.join(DATASETS, "event_candidate_screening_registry.csv"))
    valid_ids = {r["screening_id"] for r in screening_rows}
    scope_rows = read_csv(os.path.join(DATASETS, "event_patch_screening_scope.csv"))
    for row in scope_rows:
        assert row["screening_id"] in valid_ids, \
            f"{row['scope_id']}: screening_id '{row['screening_id']}' não existe no registry"


def test_v1hn_scope_all_regions_covered():
    rows = read_csv(os.path.join(DATASETS, "event_patch_screening_scope.csv"))
    regions = {r["region"] for r in rows}
    for region in ("RECIFE", "PETROPOLIS", "CURITIBA"):
        assert region in regions, f"Região {region} ausente no scope registry"


def test_v1hn_scope_has_entries_per_event():
    """Cada evento do screening registry deve ter pelo menos um patch no scope."""
    screening_rows = read_csv(os.path.join(DATASETS, "event_candidate_screening_registry.csv"))
    scope_rows = read_csv(os.path.join(DATASETS, "event_patch_screening_scope.csv"))
    scope_events = {r["screening_id"] for r in scope_rows}
    for row in screening_rows:
        assert row["screening_id"] in scope_events, \
            f"Evento {row['screening_id']} não tem patches no scope registry"


# ---------------------------------------------------------------------------
# Documento metodológico
# ---------------------------------------------------------------------------

def test_v1hn_doc_affirms_no_data_download():
    text = read_text(os.path.join(DOCS, "protocolo_c_triagem_eventos_candidatos.md"))
    markers = ["Nenhum dado bruto é baixado", "não baixa raster", "metadata-only"]
    assert any(m in text for m in markers), \
        "Documento não afirma explicitamente que nenhum dado é baixado"


def test_v1hn_doc_affirms_multimodal_hold():
    text = read_text(os.path.join(DOCS, "protocolo_c_triagem_eventos_candidatos.md"))
    assert "multimodal" in text.lower() and "hold" in text.lower(), \
        "Documento não afirma explicitamente que multimodal está em hold"


def test_v1hn_doc_affirms_dino_review_only():
    text = read_text(os.path.join(DOCS, "protocolo_c_triagem_eventos_candidatos.md"))
    assert "DINOv2" in text or "DINO" in text, "Documento não menciona DINO"
    dino_review_markers = [
        "review-only", "não pode fechar", "não fecha", "não constituem evidência"
    ]
    assert any(m in text for m in dino_review_markers), \
        "Documento não afirma explicitamente que DINO é review-only"


def test_v1hn_doc_no_operational_ground_truth_claim():
    text = read_text(os.path.join(DOCS, "protocolo_c_triagem_eventos_candidatos.md"))
    forbidden = [
        "ground truth operacional declarado",
        "referência operacional estabelecida",
        "operational ground truth established",
    ]
    for term in forbidden:
        assert term.lower() not in text.lower(), \
            f"Documento contém claim proibido: '{term}'"


def test_v1hn_doc_no_supervised_training_claim():
    text = read_text(os.path.join(DOCS, "protocolo_c_triagem_eventos_candidatos.md"))
    forbidden = [
        "treinamento supervisionado",
        "supervised training",
        "label de inundação",
        "flood label",
    ]
    for term in forbidden:
        assert term.lower() not in text.lower(), \
            f"Documento contém term proibido fora de contexto de bloqueio: '{term}'"


# ---------------------------------------------------------------------------
# Dangerous terms — nos registros CSV
# ---------------------------------------------------------------------------

def test_v1hn_no_confirmed_source_in_registries():
    """Nenhum registro deve ter candidate_status=CONFIRMED_BY_SOURCE nesta etapa."""
    rows = read_csv(os.path.join(DATASETS, "event_candidate_screening_registry.csv"))
    confirmed = [r for r in rows if r.get("candidate_status") == "CONFIRMED_BY_SOURCE"]
    assert not confirmed, \
        f"Registros com CONFIRMED_BY_SOURCE indevido: {[r['screening_id'] for r in confirmed]}"


def test_v1hn_no_private_paths_in_screening():
    text_files = [
        os.path.join(DATASETS, "event_candidate_screening_registry.csv"),
        os.path.join(DATASETS, "event_source_search_backlog.csv"),
        os.path.join(DATASETS, "event_patch_screening_scope.csv"),
    ]
    private_markers = ["C:\\Users\\", "/home/", "local_runs/", ".npz", ".tif"]
    for path in text_files:
        content = read_text(path)
        for marker in private_markers:
            assert marker not in content, \
                f"Path privado '{marker}' encontrado em {os.path.basename(path)}"
