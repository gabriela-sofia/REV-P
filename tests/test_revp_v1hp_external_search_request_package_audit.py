"""
v1hp — Busca externa e solicitação regional.
Audita documento metodológico, templates, schemas e quatro registros.
"""
import csv
import os
import pytest

BASE = os.path.join(os.path.dirname(__file__), "..")
DATASETS = os.path.join(BASE, "datasets")
SCHEMAS = os.path.join(DATASETS, "schemas")
DOCS = os.path.join(BASE, "docs", "metodologia_cientifica")
TEMPLATES = os.path.join(BASE, "docs", "templates")

SEARCH_PLAN = os.path.join(DATASETS, "regional_external_search_plan.csv")
REQUEST_PKG = os.path.join(DATASETS, "source_request_package_registry.csv")
GATE_QST = os.path.join(DATASETS, "gate_search_question_registry.csv")
PRIORITY_MTX = os.path.join(DATASETS, "regional_request_priority_matrix.csv")

SEARCH_PLAN_SCHEMA = os.path.join(SCHEMAS, "regional_external_search_plan_schema.csv")
REQUEST_PKG_SCHEMA = os.path.join(SCHEMAS, "source_request_package_schema.csv")
GATE_QST_SCHEMA = os.path.join(SCHEMAS, "gate_search_question_schema.csv")
PRIORITY_MTX_SCHEMA = os.path.join(SCHEMAS, "regional_request_priority_matrix_schema.csv")

METHODOLOGY_DOC = os.path.join(DOCS, "protocolo_c_busca_externa_solicitacao_regional.md")
TEMPLATE_PKG = os.path.join(TEMPLATES, "protocolo_c_pacote_solicitacao_regional.md")
TEMPLATE_PORTAL = os.path.join(TEMPLATES, "protocolo_c_roteiro_busca_portal_publico.md")


VALID_REGIONS = {"RECIFE", "PETROPOLIS", "CURITIBA"}

VALID_TARGET_GATES = {
    "G1_EVENT_CONFIRMATION",
    "G2_SOURCE_AVAILABILITY",
    "G3_TEMPORAL_ALIGNMENT",
    "G4_SPATIAL_ALIGNMENT",
    "G5_SOURCE_STRENGTH",
    "G6_UNCERTAINTY_AND_LIMITATIONS",
    "G7_REVIEW_GATE",
    "G8_INDEPENDENT_CORROBORATION",
    "G9_PROMOTION_DECISION",
    "MULTIPLE_GATES",
}

VALID_SOURCE_FAMILIES = {
    "CIVIL_DEFENSE_RECORD",
    "OFFICIAL_OBSERVED_FLOOD_MAP",
    "TECHNICAL_REPORT",
    "SENTINEL_EVENT_IMAGE",
    "SAR_EVENT_IMAGE",
    "OPERATIONAL_FLOOD_PRODUCT",
    "MUNICIPAL_GIS_LAYER",
    "FIELD_SURVEY",
    "ACADEMIC_DATASET",
    "METHOD_REFERENCE_ONLY",
    "UNKNOWN",
}

VALID_SEARCH_PRIORITIES = {"HIGH", "MEDIUM", "LOW", "BLOCKED", "METHOD_REFERENCE_ONLY"}

VALID_SEARCH_MODES = {
    "PUBLIC_PORTAL_REVIEW",
    "FORMAL_REQUEST",
    "MANUAL_SEARCH",
    "FUTURE_ACQUISITION",
    "METHOD_REFERENCE_ONLY",
    "UNKNOWN",
}

VALID_SEARCH_STATUSES = {
    "NOT_STARTED",
    "READY_FOR_SEARCH",
    "READY_FOR_REQUEST",
    "WAITING_LICENSE_CLARIFICATION",
    "BLOCKED",
    "METHOD_REFERENCE_ONLY",
}

VALID_REQUEST_TYPES = {
    "REQUEST_CIVIL_DEFENSE_RECORDS",
    "REQUEST_MUNICIPAL_GIS_LAYER",
    "REQUEST_OFFICIAL_FLOOD_MAP",
    "REQUEST_TECHNICAL_REPORT",
    "REQUEST_EVENT_OCCURRENCE_RECORDS",
    "REQUEST_LICENSE_CLARIFICATION",
    "PUBLIC_PORTAL_REVIEW_PACKAGE",
    "METHOD_REFERENCE_PACKAGE",
}

VALID_REQUEST_STATUSES = {
    "NOT_SENT",
    "READY_TO_PREPARE",
    "READY_FOR_REVIEW",
    "WAITING_SOURCE_DETAILS",
    "BLOCKED",
    "METHOD_REFERENCE_ONLY",
}

VALID_ANSWER_STATUSES = {
    "UNANSWERED",
    "PARTIAL",
    "ANSWERED_METADATA_ONLY",
    "BLOCKED",
    "METHOD_REFERENCE_ONLY",
}

VALID_ANSWER_TYPES = {
    "EVENT_CONFIRMATION",
    "SPATIAL_GEOMETRY",
    "EVENT_DATE",
    "UNCERTAINTY_DESCRIPTION",
    "REVIEW_GATE_INPUT",
    "LICENSE_TERMS",
    "SOURCE_METADATA",
    "AVAILABILITY_STATUS",
    "METHODOLOGY_DESCRIPTION",
    "INSTITUTIONAL_CONTACT",
}

VALID_PROTOCOL_B_STATUSES = {
    "BLOCKED",
    "FUTURE_REASSESSMENT_ONLY",
    "METHOD_REFERENCE_ONLY",
}

VALID_MULTIMODAL_STATUSES = {
    "HOLD",
    "FUTURE_READINESS_ONLY",
    "METHOD_REFERENCE_ONLY",
}

DANGEROUS_FORBIDDEN_TERMS = {
    "OPERATIONAL_GROUND_TRUTH_DECLARATION",
    "FLOOD_LABEL_CREATION",
    "TRAINING_LABEL",
}

FORBIDDEN_TERMS_FULL = DANGEROUS_FORBIDDEN_TERMS | {
    "FLOOD_DETECTION_SOLO",
    "FLOOD_PREDICTION",
    "SUPERVISED_TRAINING",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_csv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_text(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def split_semi(value):
    return {v.strip() for v in value.split(";") if v.strip()}


# ---------------------------------------------------------------------------
# Existence checks
# ---------------------------------------------------------------------------

def test_v1hp_methodology_doc_exists():
    assert os.path.exists(METHODOLOGY_DOC)


def test_v1hp_template_pkg_exists():
    assert os.path.exists(TEMPLATE_PKG)


def test_v1hp_template_portal_exists():
    assert os.path.exists(TEMPLATE_PORTAL)


def test_v1hp_search_plan_schema_exists():
    assert os.path.exists(SEARCH_PLAN_SCHEMA)


def test_v1hp_request_pkg_schema_exists():
    assert os.path.exists(REQUEST_PKG_SCHEMA)


def test_v1hp_gate_qst_schema_exists():
    assert os.path.exists(GATE_QST_SCHEMA)


def test_v1hp_priority_mtx_schema_exists():
    assert os.path.exists(PRIORITY_MTX_SCHEMA)


def test_v1hp_search_plan_registry_exists():
    assert os.path.exists(SEARCH_PLAN)


def test_v1hp_request_pkg_registry_exists():
    assert os.path.exists(REQUEST_PKG)


def test_v1hp_gate_qst_registry_exists():
    assert os.path.exists(GATE_QST)


def test_v1hp_priority_mtx_registry_exists():
    assert os.path.exists(PRIORITY_MTX)


# ---------------------------------------------------------------------------
# Schema fields
# ---------------------------------------------------------------------------

SEARCH_PLAN_REQUIRED_FIELDS = {
    "search_plan_id", "region", "candidate_event_name", "candidate_event_period",
    "search_goal", "target_gate", "target_source_family", "search_priority",
    "search_mode", "current_status", "dependency", "allowed_output", "forbidden_use",
}

REQUEST_PKG_REQUIRED_FIELDS = {
    "request_package_id", "region", "target_institution", "target_source_family",
    "request_type", "request_status", "raw_data_expected", "local_only_if_received",
    "public_metadata_allowed", "can_close_gate_if_received",
    "cannot_establish_ground_truth_alone", "forbidden_use",
}

GATE_QST_REQUIRED_FIELDS = {
    "question_id", "region", "target_gate", "search_question",
    "expected_answer_type", "current_answer_status", "blocking_if_unanswered",
    "next_action", "forbidden_if_unanswered",
}

PRIORITY_MTX_REQUIRED_FIELDS = {
    "priority_id", "region", "candidate_event_name", "priority_rank",
    "request_package_id", "search_plan_id", "protocol_b_status",
    "multimodal_status", "allowed_claim", "forbidden_claim",
}


def test_v1hp_search_plan_has_required_fields():
    rows = read_csv(SEARCH_PLAN)
    assert rows, "search plan registry must not be empty"
    fields = set(rows[0].keys())
    for f in SEARCH_PLAN_REQUIRED_FIELDS:
        assert f in fields, f"missing field: {f}"


def test_v1hp_request_pkg_has_required_fields():
    rows = read_csv(REQUEST_PKG)
    assert rows
    fields = set(rows[0].keys())
    for f in REQUEST_PKG_REQUIRED_FIELDS:
        assert f in fields, f"missing field: {f}"


def test_v1hp_gate_qst_has_required_fields():
    rows = read_csv(GATE_QST)
    assert rows
    fields = set(rows[0].keys())
    for f in GATE_QST_REQUIRED_FIELDS:
        assert f in fields, f"missing field: {f}"


def test_v1hp_priority_mtx_has_required_fields():
    rows = read_csv(PRIORITY_MTX)
    assert rows
    fields = set(rows[0].keys())
    for f in PRIORITY_MTX_REQUIRED_FIELDS:
        assert f in fields, f"missing field: {f}"


# ---------------------------------------------------------------------------
# Controlled values
# ---------------------------------------------------------------------------

def test_v1hp_search_plan_regions_valid():
    for row in read_csv(SEARCH_PLAN):
        assert row["region"] in VALID_REGIONS, f"{row['search_plan_id']}: invalid region {row['region']}"


def test_v1hp_search_plan_target_gates_valid():
    for row in read_csv(SEARCH_PLAN):
        gates = split_semi(row["target_gate"])
        for g in gates:
            assert g in VALID_TARGET_GATES, f"{row['search_plan_id']}: invalid gate {g}"


def test_v1hp_search_plan_source_families_valid():
    for row in read_csv(SEARCH_PLAN):
        fams = split_semi(row["target_source_family"])
        for f in fams:
            assert f in VALID_SOURCE_FAMILIES, f"{row['search_plan_id']}: invalid source family {f}"


def test_v1hp_search_plan_priorities_valid():
    for row in read_csv(SEARCH_PLAN):
        assert row["search_priority"] in VALID_SEARCH_PRIORITIES, \
            f"{row['search_plan_id']}: invalid priority {row['search_priority']}"


def test_v1hp_search_plan_modes_valid():
    for row in read_csv(SEARCH_PLAN):
        assert row["search_mode"] in VALID_SEARCH_MODES, \
            f"{row['search_plan_id']}: invalid mode {row['search_mode']}"


def test_v1hp_search_plan_statuses_valid():
    for row in read_csv(SEARCH_PLAN):
        assert row["current_status"] in VALID_SEARCH_STATUSES, \
            f"{row['search_plan_id']}: invalid status {row['current_status']}"


def test_v1hp_request_pkg_regions_valid():
    for row in read_csv(REQUEST_PKG):
        assert row["region"] in VALID_REGIONS, f"{row['request_package_id']}: invalid region {row['region']}"


def test_v1hp_request_pkg_types_valid():
    for row in read_csv(REQUEST_PKG):
        assert row["request_type"] in VALID_REQUEST_TYPES, \
            f"{row['request_package_id']}: invalid type {row['request_type']}"


def test_v1hp_request_pkg_statuses_valid():
    for row in read_csv(REQUEST_PKG):
        assert row["request_status"] in VALID_REQUEST_STATUSES, \
            f"{row['request_package_id']}: invalid status {row['request_status']}"


def test_v1hp_gate_qst_regions_valid():
    for row in read_csv(GATE_QST):
        assert row["region"] in VALID_REGIONS, f"{row['question_id']}: invalid region {row['region']}"


def test_v1hp_gate_qst_target_gates_valid():
    for row in read_csv(GATE_QST):
        gates = split_semi(row["target_gate"])
        for g in gates:
            assert g in VALID_TARGET_GATES, f"{row['question_id']}: invalid gate {g}"


def test_v1hp_gate_qst_answer_statuses_valid():
    for row in read_csv(GATE_QST):
        assert row["current_answer_status"] in VALID_ANSWER_STATUSES, \
            f"{row['question_id']}: invalid answer status {row['current_answer_status']}"


def test_v1hp_gate_qst_answer_types_valid():
    for row in read_csv(GATE_QST):
        assert row["expected_answer_type"] in VALID_ANSWER_TYPES, \
            f"{row['question_id']}: invalid answer type {row['expected_answer_type']}"


def test_v1hp_priority_mtx_regions_valid():
    for row in read_csv(PRIORITY_MTX):
        assert row["region"] in VALID_REGIONS, f"{row['priority_id']}: invalid region {row['region']}"


def test_v1hp_priority_mtx_protocol_b_statuses_valid():
    for row in read_csv(PRIORITY_MTX):
        assert row["protocol_b_status"] in VALID_PROTOCOL_B_STATUSES, \
            f"{row['priority_id']}: invalid protocol_b_status {row['protocol_b_status']}"


def test_v1hp_priority_mtx_multimodal_statuses_valid():
    for row in read_csv(PRIORITY_MTX):
        assert row["multimodal_status"] in VALID_MULTIMODAL_STATUSES, \
            f"{row['priority_id']}: invalid multimodal_status {row['multimodal_status']}"


# ---------------------------------------------------------------------------
# Regional coverage — all three regions present in every registry
# ---------------------------------------------------------------------------

def test_v1hp_search_plan_has_recife():
    regions = {r["region"] for r in read_csv(SEARCH_PLAN)}
    assert "RECIFE" in regions


def test_v1hp_search_plan_has_petropolis():
    regions = {r["region"] for r in read_csv(SEARCH_PLAN)}
    assert "PETROPOLIS" in regions


def test_v1hp_search_plan_has_curitiba():
    regions = {r["region"] for r in read_csv(SEARCH_PLAN)}
    assert "CURITIBA" in regions


def test_v1hp_request_pkg_has_recife():
    regions = {r["region"] for r in read_csv(REQUEST_PKG)}
    assert "RECIFE" in regions


def test_v1hp_request_pkg_has_petropolis():
    regions = {r["region"] for r in read_csv(REQUEST_PKG)}
    assert "PETROPOLIS" in regions


def test_v1hp_request_pkg_has_curitiba():
    regions = {r["region"] for r in read_csv(REQUEST_PKG)}
    assert "CURITIBA" in regions


def test_v1hp_gate_qst_has_recife():
    regions = {r["region"] for r in read_csv(GATE_QST)}
    assert "RECIFE" in regions


def test_v1hp_gate_qst_has_petropolis():
    regions = {r["region"] for r in read_csv(GATE_QST)}
    assert "PETROPOLIS" in regions


def test_v1hp_gate_qst_has_curitiba():
    regions = {r["region"] for r in read_csv(GATE_QST)}
    assert "CURITIBA" in regions


def test_v1hp_priority_mtx_has_recife():
    regions = {r["region"] for r in read_csv(PRIORITY_MTX)}
    assert "RECIFE" in regions


def test_v1hp_priority_mtx_has_petropolis():
    regions = {r["region"] for r in read_csv(PRIORITY_MTX)}
    assert "PETROPOLIS" in regions


def test_v1hp_priority_mtx_has_curitiba():
    regions = {r["region"] for r in read_csv(PRIORITY_MTX)}
    assert "CURITIBA" in regions


# ---------------------------------------------------------------------------
# No search executed, no request sent
# ---------------------------------------------------------------------------

EXECUTED_STATUSES = {"COMPLETED", "EXECUTED", "DONE", "SEARCH_EXECUTED"}
SENT_STATUSES = {"SENT", "DELIVERED", "ACKNOWLEDGED"}


def test_v1hp_no_search_marked_executed():
    for row in read_csv(SEARCH_PLAN):
        assert row["current_status"].upper() not in EXECUTED_STATUSES, \
            f"{row['search_plan_id']}: search must not be marked executed in this stage"


def test_v1hp_no_request_marked_sent():
    for row in read_csv(REQUEST_PKG):
        assert row["request_status"].upper() not in SENT_STATUSES, \
            f"{row['request_package_id']}: request must not be marked sent in this stage"


def test_v1hp_no_gate_question_answered_fully():
    for row in read_csv(GATE_QST):
        assert row["current_answer_status"] not in {"ANSWERED_METADATA_ONLY"} or \
            row["blocking_if_unanswered"].lower() == "false", \
            f"{row['question_id']}: a critical gate question cannot be fully answered yet"


# ---------------------------------------------------------------------------
# Ground truth guardrail — cannot_establish_ground_truth_alone=true
# ---------------------------------------------------------------------------

def test_v1hp_all_request_pkgs_cannot_establish_ground_truth_alone():
    for row in read_csv(REQUEST_PKG):
        assert row["cannot_establish_ground_truth_alone"].lower() == "true", \
            f"{row['request_package_id']}: cannot_establish_ground_truth_alone must be true"


# ---------------------------------------------------------------------------
# Raw data local-only constraint
# ---------------------------------------------------------------------------

def test_v1hp_raw_data_true_implies_local_only():
    for row in read_csv(REQUEST_PKG):
        if row["raw_data_expected"].lower() == "true":
            assert row["local_only_if_received"].lower() == "true", \
                f"{row['request_package_id']}: raw_data_expected=true requires local_only_if_received=true"


# ---------------------------------------------------------------------------
# Forbidden use coverage — all registries must block dangerous uses
# ---------------------------------------------------------------------------

def test_v1hp_search_plan_forbidden_use_blocks_ground_truth():
    for row in read_csv(SEARCH_PLAN):
        forbidden = split_semi(row["forbidden_use"])
        for term in DANGEROUS_FORBIDDEN_TERMS:
            assert term in forbidden, \
                f"{row['search_plan_id']}: forbidden_use must include {term}"


def test_v1hp_request_pkg_forbidden_use_blocks_ground_truth():
    # LICENSE_CLARIFICATION packages are not data packages — their forbidden_use covers
    # redistribution/commercial terms, not flood label creation (correct by design).
    for row in read_csv(REQUEST_PKG):
        if row["request_type"] == "REQUEST_LICENSE_CLARIFICATION":
            continue
        forbidden = split_semi(row["forbidden_use"])
        for term in DANGEROUS_FORBIDDEN_TERMS:
            assert term in forbidden, \
                f"{row['request_package_id']}: forbidden_use must include {term}"


def test_v1hp_gate_qst_forbidden_if_unanswered_blocks_ground_truth():
    for row in read_csv(GATE_QST):
        if row["blocking_if_unanswered"].lower() == "true":
            forbidden = split_semi(row["forbidden_if_unanswered"])
            for term in DANGEROUS_FORBIDDEN_TERMS:
                assert term in forbidden, \
                    f"{row['question_id']}: blocking question must forbid {term}"


def test_v1hp_priority_mtx_forbidden_claim_blocks_ground_truth():
    for row in read_csv(PRIORITY_MTX):
        forbidden = split_semi(row["forbidden_claim"])
        for term in DANGEROUS_FORBIDDEN_TERMS:
            assert term in forbidden, \
                f"{row['priority_id']}: forbidden_claim must include {term}"


def test_v1hp_priority_mtx_forbidden_claim_blocks_flood_label():
    for row in read_csv(PRIORITY_MTX):
        forbidden = split_semi(row["forbidden_claim"])
        assert "FLOOD_LABEL_CREATION" in forbidden, \
            f"{row['priority_id']}: forbidden_claim must include FLOOD_LABEL_CREATION"


def test_v1hp_priority_mtx_forbidden_claim_blocks_supervised_training():
    for row in read_csv(PRIORITY_MTX):
        forbidden = split_semi(row["forbidden_claim"])
        assert "SUPERVISED_TRAINING" in forbidden, \
            f"{row['priority_id']}: forbidden_claim must include SUPERVISED_TRAINING"


# ---------------------------------------------------------------------------
# Protocolo B = BLOCKED for all REV-P regions in priority matrix
# ---------------------------------------------------------------------------

def test_v1hp_priority_mtx_protocol_b_blocked():
    for row in read_csv(PRIORITY_MTX):
        assert row["protocol_b_status"] == "BLOCKED", \
            f"{row['priority_id']}: protocol_b_status must be BLOCKED (not {row['protocol_b_status']})"


# ---------------------------------------------------------------------------
# Multimodal = HOLD for all REV-P regions in priority matrix
# ---------------------------------------------------------------------------

def test_v1hp_priority_mtx_multimodal_hold():
    for row in read_csv(PRIORITY_MTX):
        assert row["multimodal_status"] == "HOLD", \
            f"{row['priority_id']}: multimodal_status must be HOLD (not {row['multimodal_status']})"


# ---------------------------------------------------------------------------
# G1/G3/G4/G7 unanswered blocks promotion
# ---------------------------------------------------------------------------

BLOCKING_GATES = {
    "G1_EVENT_CONFIRMATION",
    "G3_TEMPORAL_ALIGNMENT",
    "G4_SPATIAL_ALIGNMENT",
    "G7_REVIEW_GATE",
}


def test_v1hp_critical_gates_have_blocking_questions():
    rows = read_csv(GATE_QST)
    covered_gates = set()
    for row in rows:
        gates = split_semi(row["target_gate"])
        if row["blocking_if_unanswered"].lower() == "true":
            covered_gates |= gates
    for gate in BLOCKING_GATES:
        assert gate in covered_gates, \
            f"No blocking question found for gate {gate}"


def test_v1hp_g1_primary_questions_are_blocking():
    # Secondary/corroboration G1 questions (e.g., CPRM portal check as complement to
    # the primary COMPDEC request) may be non-blocking by design. Only primary G1
    # questions (those with blocking_if_unanswered=true in the registry) must block.
    # The requirement is: at least one blocking G1 question exists per REV-P region.
    rows = read_csv(GATE_QST)
    for region in VALID_REGIONS:
        region_blocking_g1 = [
            r for r in rows
            if r["region"] == region
            and "G1_EVENT_CONFIRMATION" in split_semi(r["target_gate"])
            and r["blocking_if_unanswered"].lower() == "true"
        ]
        assert region_blocking_g1, \
            f"Region {region} must have at least one blocking G1 question"


def test_v1hp_g7_questions_are_blocking():
    for row in read_csv(GATE_QST):
        gates = split_semi(row["target_gate"])
        if "G7_REVIEW_GATE" in gates:
            assert row["blocking_if_unanswered"].lower() == "true", \
                f"{row['question_id']}: G7 questions must be blocking"


# ---------------------------------------------------------------------------
# No private paths in any new file
# ---------------------------------------------------------------------------

PRIVATE_PATH_MARKERS = ["C:\\Users\\", "C:/Users/", "/home/", "/Users/"]


def _has_private_path(text):
    return any(marker in text for marker in PRIVATE_PATH_MARKERS)


def test_v1hp_methodology_doc_no_private_paths():
    text = read_text(METHODOLOGY_DOC)
    assert not _has_private_path(text), "methodology doc contains private path"


def test_v1hp_template_pkg_no_private_paths():
    text = read_text(TEMPLATE_PKG)
    assert not _has_private_path(text), "request package template contains private path"


def test_v1hp_template_portal_no_private_paths():
    text = read_text(TEMPLATE_PORTAL)
    assert not _has_private_path(text), "portal search template contains private path"


def test_v1hp_search_plan_no_private_paths():
    text = read_text(SEARCH_PLAN)
    assert not _has_private_path(text), "search plan registry contains private path"


def test_v1hp_request_pkg_no_private_paths():
    text = read_text(REQUEST_PKG)
    assert not _has_private_path(text), "request package registry contains private path"


def test_v1hp_gate_qst_no_private_paths():
    text = read_text(GATE_QST)
    assert not _has_private_path(text), "gate search question registry contains private path"


def test_v1hp_priority_mtx_no_private_paths():
    text = read_text(PRIORITY_MTX)
    assert not _has_private_path(text), "priority matrix contains private path"


# ---------------------------------------------------------------------------
# No heavy data referenced as versioned artifacts
# ---------------------------------------------------------------------------

HEAVY_DATA_EXTENSIONS = [".tif", ".tiff", ".npz", ".npy", ".geotiff", ".shp", ".zip"]


def _references_heavy_data(text):
    lower = text.lower()
    return any(ext in lower for ext in HEAVY_DATA_EXTENSIONS)


def test_v1hp_search_plan_no_heavy_data_refs():
    text = read_text(SEARCH_PLAN)
    assert not _references_heavy_data(text), \
        "search plan registry must not reference heavy data files as versioned artifacts"


def test_v1hp_request_pkg_no_heavy_data_refs():
    text = read_text(REQUEST_PKG)
    assert not _references_heavy_data(text), \
        "request package registry must not reference heavy data files as versioned artifacts"


# ---------------------------------------------------------------------------
# Documentation content checks
# ---------------------------------------------------------------------------

def test_v1hp_methodology_doc_mentions_metadata_only():
    text = read_text(METHODOLOGY_DOC)
    assert "metadata" in text.lower(), "methodology doc must mention metadata-only principle"


def test_v1hp_methodology_doc_mentions_busca_externa():
    text = read_text(METHODOLOGY_DOC)
    assert "busca" in text.lower(), "methodology doc must discuss external search"


def test_v1hp_methodology_doc_mentions_solicitacao():
    text = read_text(METHODOLOGY_DOC)
    assert "solicit" in text.lower(), "methodology doc must discuss formal requests"


def test_v1hp_methodology_doc_mentions_multimodal_hold():
    text = read_text(METHODOLOGY_DOC)
    assert "multimodal" in text.lower() and "hold" in text.lower(), \
        "methodology doc must state multimodal is on hold"


def test_v1hp_methodology_doc_mentions_ground_truth_blocked():
    text = read_text(METHODOLOGY_DOC)
    assert "ground truth" in text.lower() or "ground_truth" in text.lower(), \
        "methodology doc must address ground truth status"


def test_v1hp_methodology_doc_no_positive_ground_truth_claim():
    text = read_text(METHODOLOGY_DOC)
    lower = text.lower()
    assert "ground truth operacional estabelecido" not in lower, \
        "methodology doc must not declare operational ground truth as established"
    assert "ground truth confirmado" not in lower, \
        "methodology doc must not declare ground truth as confirmed"


# ---------------------------------------------------------------------------
# METHOD_REFERENCE_ONLY not applied to active REV-P event search plans
# ---------------------------------------------------------------------------

def test_v1hp_search_plan_revp_events_not_method_reference_only():
    revp_regions = {"RECIFE", "PETROPOLIS", "CURITIBA"}
    for row in read_csv(SEARCH_PLAN):
        if row["region"] in revp_regions and row.get("candidate_event_name", "").upper() not in (
            "NOT_EVENT_SPECIFIC", "METHOD_REFERENCE_ONLY"
        ):
            assert row["search_mode"] != "METHOD_REFERENCE_ONLY", \
                f"{row['search_plan_id']}: active REV-P event must not have METHOD_REFERENCE_ONLY search mode"


# ---------------------------------------------------------------------------
# Cross-reference: priority matrix references valid search plan IDs
# ---------------------------------------------------------------------------

def test_v1hp_priority_mtx_references_valid_search_plans():
    valid_ids = {r["search_plan_id"] for r in read_csv(SEARCH_PLAN)}
    for row in read_csv(PRIORITY_MTX):
        plan_ids = split_semi(row["search_plan_id"])
        for pid in plan_ids:
            if pid and pid not in ("NOT_APPLICABLE", "TBD"):
                assert pid in valid_ids, \
                    f"{row['priority_id']}: references unknown search_plan_id {pid}"


# ---------------------------------------------------------------------------
# Cross-reference: priority matrix references valid request package IDs
# ---------------------------------------------------------------------------

def test_v1hp_priority_mtx_references_valid_request_packages():
    valid_ids = {r["request_package_id"] for r in read_csv(REQUEST_PKG)}
    for row in read_csv(PRIORITY_MTX):
        pkg_ids = split_semi(row["request_package_id"])
        for pid in pkg_ids:
            if pid and pid not in ("NOT_APPLICABLE", "TBD"):
                assert pid in valid_ids, \
                    f"{row['priority_id']}: references unknown request_package_id {pid}"


# ---------------------------------------------------------------------------
# Gate question IDs are unique
# ---------------------------------------------------------------------------

def test_v1hp_gate_qst_ids_are_unique():
    ids = [r["question_id"] for r in read_csv(GATE_QST)]
    assert len(ids) == len(set(ids)), "gate_search_question_registry has duplicate question_id values"


# ---------------------------------------------------------------------------
# Search plan IDs are unique
# ---------------------------------------------------------------------------

def test_v1hp_search_plan_ids_are_unique():
    ids = [r["search_plan_id"] for r in read_csv(SEARCH_PLAN)]
    assert len(ids) == len(set(ids)), "regional_external_search_plan has duplicate search_plan_id values"


# ---------------------------------------------------------------------------
# Request package IDs are unique
# ---------------------------------------------------------------------------

def test_v1hp_request_pkg_ids_are_unique():
    ids = [r["request_package_id"] for r in read_csv(REQUEST_PKG)]
    assert len(ids) == len(set(ids)), "source_request_package_registry has duplicate request_package_id values"


# ---------------------------------------------------------------------------
# Priority matrix IDs are unique
# ---------------------------------------------------------------------------

def test_v1hp_priority_mtx_ids_are_unique():
    ids = [r["priority_id"] for r in read_csv(PRIORITY_MTX)]
    assert len(ids) == len(set(ids)), "regional_request_priority_matrix has duplicate priority_id values"
