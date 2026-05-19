#!/usr/bin/env python3
"""
test_revp_v1hm_evidence_acquisition_intake_audit.py

Audit tests for REV-P v1hm evidence acquisition, intake, provenance and licensing.
Validates that:
- Documents, templates, schemas, and registries exist
- Required fields are present and controlled values are valid
- License restrictions block operational ground truth
- Raw data publication is blocked when redistribution is not explicit
- use_for_operational_ground_truth_allowed is never TRUE
- DINO_STRUCTURAL_SUPPORT cannot close ground truth gates
- METHOD_REFERENCE_ONLY cannot be applied directly
- No supervised training is permitted
- No private paths are present
- Forbidden uses explicitly block dangerous claims
- Documentation states metadata-only, local-only, license/provenance and multimodal hold

Etapa: v1hm
Status: metadata-only, no acquisition executed, no data downloaded
"""

import csv
import re
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
DOCS_DIR = REPO_ROOT / "docs" / "metodologia_cientifica"
TEMPLATES_DIR = REPO_ROOT / "docs" / "templates"
DATASETS_DIR = REPO_ROOT / "datasets"
SCHEMAS_DIR = DATASETS_DIR / "schemas"


# ---------------------------------------------------------------------------
# Existence tests
# ---------------------------------------------------------------------------

def test_v1hm_operational_package_doc_exists():
    f = DOCS_DIR / "protocolo_c_pacote_operacional_aquisicao_evidencias.md"
    assert f.exists(), f"Document not found: {f}"


def test_v1hm_runbook_doc_exists():
    f = DOCS_DIR / "protocolo_c_runbook_aquisicao_evidencias.md"
    assert f.exists(), f"Runbook not found: {f}"


def test_v1hm_solicitacao_template_exists():
    f = TEMPLATES_DIR / "protocolo_c_solicitacao_fonte_observacional.md"
    assert f.exists(), f"Template not found: {f}"


def test_v1hm_checklist_template_exists():
    f = TEMPLATES_DIR / "protocolo_c_checklist_triagem_fonte.md"
    assert f.exists(), f"Checklist template not found: {f}"


def test_v1hm_tracker_schema_exists():
    f = SCHEMAS_DIR / "evidence_acquisition_tracker_schema.csv"
    assert f.exists(), f"Tracker schema not found: {f}"


def test_v1hm_intake_schema_exists():
    f = SCHEMAS_DIR / "evidence_source_intake_schema.csv"
    assert f.exists(), f"Intake schema not found: {f}"


def test_v1hm_provenance_schema_exists():
    f = SCHEMAS_DIR / "evidence_license_provenance_schema.csv"
    assert f.exists(), f"Provenance schema not found: {f}"


def test_v1hm_tracker_registry_exists():
    f = DATASETS_DIR / "evidence_acquisition_tracker.csv"
    assert f.exists(), f"Tracker registry not found: {f}"


def test_v1hm_intake_registry_exists():
    f = DATASETS_DIR / "evidence_source_intake_registry.csv"
    assert f.exists(), f"Intake registry not found: {f}"


def test_v1hm_provenance_registry_exists():
    f = DATASETS_DIR / "evidence_license_provenance_registry.csv"
    assert f.exists(), f"Provenance registry not found: {f}"


# ---------------------------------------------------------------------------
# Schema field tests
# ---------------------------------------------------------------------------

def _get_schema_fields(schema_path):
    with open(schema_path, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return [row['field_name'] for row in reader]


def test_v1hm_tracker_schema_required_fields():
    fields = _get_schema_fields(SCHEMAS_DIR / "evidence_acquisition_tracker_schema.csv")
    required = [
        'acquisition_id', 'region', 'target_source_name', 'source_family',
        'institution_or_provider', 'acquisition_status', 'access_mode',
        'request_required', 'request_status', 'source_reference',
        'expected_event_link', 'expected_gate_closure', 'expected_artifact_type',
        'raw_data_expected', 'local_only_required', 'public_metadata_allowed',
        'license_status', 'redistribution_status', 'privacy_or_sensitivity_status',
        'current_blocker', 'next_action', 'forbidden_use'
    ]
    for field in required:
        assert field in fields, f"Required field '{field}' missing from tracker schema"


def test_v1hm_intake_schema_required_fields():
    fields = _get_schema_fields(SCHEMAS_DIR / "evidence_source_intake_schema.csv")
    required = [
        'intake_id', 'acquisition_id', 'source_id', 'region', 'source_name',
        'source_family', 'source_type', 'provider', 'event_id', 'event_link_status',
        'acquisition_date', 'source_date_or_period', 'geometry_available',
        'crs_available', 'temporal_information_available', 'spatial_coverage_status',
        'uncertainty_available', 'license_status', 'redistribution_status',
        'local_asset_status', 'public_registry_safe', 'protocol_c_gates_supported',
        'intake_decision', 'blocked_reason', 'allowed_use', 'forbidden_use'
    ]
    for field in required:
        assert field in fields, f"Required field '{field}' missing from intake schema"


def test_v1hm_provenance_schema_required_fields():
    fields = _get_schema_fields(SCHEMAS_DIR / "evidence_license_provenance_schema.csv")
    required = [
        'provenance_id', 'source_id', 'acquisition_id', 'source_name', 'provider',
        'license_status', 'redistribution_status', 'citation_required',
        'citation_text_placeholder', 'public_metadata_allowed',
        'raw_data_publication_allowed', 'derived_data_publication_allowed',
        'local_only_required', 'sensitive_content_risk', 'provenance_completeness',
        'use_in_public_repo_allowed', 'use_for_reference_candidate_allowed',
        'use_for_operational_ground_truth_allowed', 'blocked_reason'
    ]
    for field in required:
        assert field in fields, f"Required field '{field}' missing from provenance schema"


# ---------------------------------------------------------------------------
# Controlled values tests
# ---------------------------------------------------------------------------

VALID_ACQUISITION_STATUS = {
    'NOT_STARTED', 'IDENTIFIED', 'REQUEST_REQUIRED', 'REQUESTED',
    'RECEIVED_METADATA_ONLY', 'RECEIVED_RAW_LOCAL_ONLY', 'REJECTED',
    'BLOCKED', 'METHOD_REFERENCE_ONLY'
}

VALID_SOURCE_FAMILIES = {
    'CIVIL_DEFENSE_RECORD', 'MUNICIPAL_GIS_LAYER', 'OFFICIAL_OBSERVED_FLOOD_MAP',
    'OPERATIONAL_FLOOD_PRODUCT', 'HIGH_RESOLUTION_IMAGE', 'EXPERT_ANNOTATION',
    'ACADEMIC_DATASET', 'HYDROGEOMORPHOLOGICAL_CONTEXT', 'SENTINEL_EVENT_IMAGE',
    'DINO_STRUCTURAL_SUPPORT', 'METHOD_REFERENCE'
}

VALID_LICENSE_STATUS = {
    'PUBLIC_REUSE_ALLOWED', 'PUBLIC_VIEW_ONLY', 'REQUEST_REQUIRED',
    'RESTRICTED', 'UNKNOWN', 'METHOD_REFERENCE_ONLY'
}

VALID_REDISTRIBUTION_STATUS = {
    'PUBLIC_METADATA_ONLY', 'PUBLIC_REUSABLE', 'LOCAL_ONLY_LICENSED',
    'REDISTRIBUTION_FORBIDDEN', 'UNKNOWN', 'METHOD_REFERENCE_ONLY'
}

VALID_INTAKE_DECISIONS = {
    'ACCEPT_METADATA_ONLY', 'ACCEPT_LOCAL_ONLY', 'REQUEST_MORE_INFORMATION',
    'BLOCK_USE', 'METHOD_REFERENCE_ONLY'
}

VALID_GROUND_TRUTH_ALLOWED = {'FALSE', 'FUTURE_REVIEW_REQUIRED'}


def test_v1hm_tracker_controlled_values():
    with open(DATASETS_DIR / "evidence_acquisition_tracker.csv", encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, start=2):
            status = row['acquisition_status'].strip()
            assert status in VALID_ACQUISITION_STATUS, f"Row {i}: invalid acquisition_status '{status}'"

            family = row['source_family'].strip()
            assert family in VALID_SOURCE_FAMILIES, f"Row {i}: invalid source_family '{family}'"

            license_s = row['license_status'].strip()
            assert license_s in VALID_LICENSE_STATUS, f"Row {i}: invalid license_status '{license_s}'"


def test_v1hm_intake_controlled_values():
    with open(DATASETS_DIR / "evidence_source_intake_registry.csv", encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, start=2):
            decision = row['intake_decision'].strip()
            assert decision in VALID_INTAKE_DECISIONS, f"Row {i}: invalid intake_decision '{decision}'"

            license_s = row['license_status'].strip()
            assert license_s in VALID_LICENSE_STATUS, f"Row {i}: invalid license_status '{license_s}'"


def test_v1hm_provenance_controlled_values():
    with open(DATASETS_DIR / "evidence_license_provenance_registry.csv", encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, start=2):
            gt_status = row['use_for_operational_ground_truth_allowed'].strip()
            assert gt_status in VALID_GROUND_TRUTH_ALLOWED, \
                f"Row {i}: invalid use_for_operational_ground_truth_allowed '{gt_status}'"


# ---------------------------------------------------------------------------
# License restriction guardrails
# ---------------------------------------------------------------------------

def test_v1hm_unknown_license_blocks_ground_truth():
    """Sources with UNKNOWN/RESTRICTED/REQUEST_REQUIRED license cannot allow ground truth."""
    with open(DATASETS_DIR / "evidence_license_provenance_registry.csv", encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            license_s = row['license_status'].strip()
            gt_allowed = row['use_for_operational_ground_truth_allowed'].strip()
            if license_s in {'UNKNOWN', 'RESTRICTED', 'REQUEST_REQUIRED'}:
                assert gt_allowed in VALID_GROUND_TRUTH_ALLOWED, \
                    f"Source {row['source_id']}: license '{license_s}' must block ground truth, got '{gt_allowed}'"


def test_v1hm_raw_data_publication_blocked_by_default():
    """Raw data publication must be false when redistribution is not explicitly public."""
    with open(DATASETS_DIR / "evidence_license_provenance_registry.csv", encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            redistrib = row['redistribution_status'].strip()
            raw_pub = row['raw_data_publication_allowed'].strip().lower()
            if redistrib not in {'PUBLIC_REUSABLE'}:
                assert raw_pub in {'false', 'no'}, \
                    f"Source {row['source_id']}: redistribution '{redistrib}' must have raw_data_publication_allowed=false"


def test_v1hm_operational_ground_truth_never_true():
    """use_for_operational_ground_truth_allowed must never be TRUE."""
    with open(DATASETS_DIR / "evidence_license_provenance_registry.csv", encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            gt_val = row['use_for_operational_ground_truth_allowed'].strip().upper()
            assert gt_val != 'TRUE', \
                f"Source {row['source_id']}: use_for_operational_ground_truth_allowed must not be TRUE"


# ---------------------------------------------------------------------------
# DINO and METHOD_REFERENCE guardrails
# ---------------------------------------------------------------------------

def test_v1hm_dino_cannot_close_ground_truth_gate():
    """DINO_STRUCTURAL_SUPPORT must not support ground truth gate closure."""
    with open(DATASETS_DIR / "evidence_acquisition_tracker.csv", encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['source_family'].strip() == 'DINO_STRUCTURAL_SUPPORT':
                forbidden = row['forbidden_use'].upper()
                assert 'OPERATIONAL_GROUND_TRUTH' in forbidden or 'GROUND_TRUTH' in forbidden, \
                    f"Acquisition {row['acquisition_id']}: DINO must explicitly forbid ground truth"


def test_v1hm_method_reference_not_applied_to_patches():
    """METHOD_REFERENCE_ONLY cannot have direct application to patches."""
    with open(DATASETS_DIR / "evidence_acquisition_tracker.csv", encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['acquisition_status'].strip() == 'METHOD_REFERENCE_ONLY':
                raw_expected = row['raw_data_expected'].strip().lower()
                assert raw_expected in {'false', 'no'}, \
                    f"Acquisition {row['acquisition_id']}: METHOD_REFERENCE_ONLY must not expect raw data"


# ---------------------------------------------------------------------------
# Supervised training and label guardrails
# ---------------------------------------------------------------------------

def test_v1hm_no_supervised_training_in_tracker():
    """No tracker entry must permit supervised training."""
    with open(DATASETS_DIR / "evidence_acquisition_tracker.csv", encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            forbidden = row['forbidden_use'].upper()
            assert 'SUPERVISED_TRAINING' in forbidden or 'FLOOD_LABEL' in forbidden, \
                f"Acquisition {row['acquisition_id']}: must explicitly forbid supervised training or flood labels"


def test_v1hm_no_supervised_training_in_intake():
    """No intake entry must permit supervised training."""
    with open(DATASETS_DIR / "evidence_source_intake_registry.csv", encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            forbidden = row['forbidden_use'].upper()
            assert 'SUPERVISED_TRAINING' in forbidden or 'FLOOD_LABEL' in forbidden, \
                f"Intake {row['intake_id']}: must explicitly forbid supervised training or flood labels"


# ---------------------------------------------------------------------------
# Private path and heavy file guardrails
# ---------------------------------------------------------------------------

PRIVATE_PATH_PATTERNS = [
    r'C:\\Users\\', r'C:/Users/', r'/home/', r'/Users/',
    r'\.npz', r'\.npy', r'\.tif\b', r'\.tiff\b', r'\.geotiff'
]


def _check_no_private_paths(filepath):
    with open(filepath, encoding='utf-8') as f:
        content = f.read()
    for pattern in PRIVATE_PATH_PATTERNS:
        matches = re.findall(pattern, content, re.IGNORECASE)
        assert not matches, f"File {filepath.name}: found private path pattern '{pattern}': {matches}"


def test_v1hm_no_private_paths_in_tracker():
    _check_no_private_paths(DATASETS_DIR / "evidence_acquisition_tracker.csv")


def test_v1hm_no_private_paths_in_intake():
    _check_no_private_paths(DATASETS_DIR / "evidence_source_intake_registry.csv")


def test_v1hm_no_private_paths_in_provenance():
    _check_no_private_paths(DATASETS_DIR / "evidence_license_provenance_registry.csv")


def test_v1hm_no_private_paths_in_operational_doc():
    _check_no_private_paths(DOCS_DIR / "protocolo_c_pacote_operacional_aquisicao_evidencias.md")


# ---------------------------------------------------------------------------
# Forbidden use completeness
# ---------------------------------------------------------------------------

def test_v1hm_forbidden_use_blocks_flood_detection_in_tracker():
    """Tracker entries must block flood detection claims."""
    with open(DATASETS_DIR / "evidence_acquisition_tracker.csv", encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['acquisition_status'].strip() == 'METHOD_REFERENCE_ONLY':
                continue  # method references have different forbidden_use format
            forbidden = row['forbidden_use'].upper()
            assert any(term in forbidden for term in ['DETECTION', 'PREDICTION', 'GROUND_TRUTH', 'LABEL']), \
                f"Acquisition {row['acquisition_id']}: forbidden_use must block detection/prediction/ground_truth/label"


def test_v1hm_forbidden_use_blocks_flood_prediction_in_intake():
    """Intake entries must block flood prediction claims."""
    with open(DATASETS_DIR / "evidence_source_intake_registry.csv", encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            forbidden = row['forbidden_use'].upper()
            assert any(term in forbidden for term in ['DETECTION', 'PREDICTION', 'GROUND_TRUTH', 'LABEL']), \
                f"Intake {row['intake_id']}: forbidden_use must block detection/prediction/ground_truth/label"


# ---------------------------------------------------------------------------
# Documentation content tests
# ---------------------------------------------------------------------------

def test_v1hm_operational_doc_mentions_metadata_first():
    with open(DOCS_DIR / "protocolo_c_pacote_operacional_aquisicao_evidencias.md", encoding='utf-8') as f:
        content = f.read().lower()
    assert 'metadata-first' in content or 'metadata first' in content, \
        "Operational document must mention metadata-first principle"


def test_v1hm_operational_doc_mentions_local_only():
    with open(DOCS_DIR / "protocolo_c_pacote_operacional_aquisicao_evidencias.md", encoding='utf-8') as f:
        content = f.read().lower()
    assert 'local-only' in content or 'local only' in content, \
        "Operational document must mention local-only for raw data"


def test_v1hm_operational_doc_mentions_license():
    with open(DOCS_DIR / "protocolo_c_pacote_operacional_aquisicao_evidencias.md", encoding='utf-8') as f:
        content = f.read().lower()
    assert 'licença' in content or 'license' in content, \
        "Operational document must mention license/provenance requirements"


def test_v1hm_operational_doc_mentions_multimodal_hold():
    with open(DOCS_DIR / "protocolo_c_pacote_operacional_aquisicao_evidencias.md", encoding='utf-8') as f:
        content = f.read().lower()
    assert 'multimodal' in content and ('hold' in content or 'espera' in content), \
        "Operational document must state multimodal is on hold"


def test_v1hm_runbook_mentions_no_download_this_stage():
    with open(DOCS_DIR / "protocolo_c_runbook_aquisicao_evidencias.md", encoding='utf-8') as f:
        content = f.read().lower()
    assert any(phrase in content for phrase in ['não baixar', 'not download', 'não executa', 'does not download', 'nenhum download']), \
        "Runbook must state that no download occurs in this stage"


def test_v1hm_template_contains_placeholders():
    with open(TEMPLATES_DIR / "protocolo_c_solicitacao_fonte_observacional.md", encoding='utf-8') as f:
        content = f.read()
    expected_placeholders = [
        '[INSTITUICAO]', '[REGIAO]', '[EVENTO_OU_PERIODO]',
        '[TIPO_DE_DADO_SOLICITADO]', '[JUSTIFICATIVA_ACADEMICA]',
        '[CONTATO_DA_PESQUISADORA]'
    ]
    for placeholder in expected_placeholders:
        assert placeholder in content, f"Template missing placeholder: {placeholder}"


def test_v1hm_checklist_contains_license_block():
    with open(TEMPLATES_DIR / "protocolo_c_checklist_triagem_fonte.md", encoding='utf-8') as f:
        content = f.read().lower()
    assert 'licença' in content or 'license' in content, \
        "Checklist must include license evaluation"
    assert 'bloqueio' in content or 'block' in content or 'blocked' in content, \
        "Checklist must include block criteria"


if __name__ == '__main__':
    import sys
    failed = []
    test_functions = [
        test_v1hm_operational_package_doc_exists,
        test_v1hm_runbook_doc_exists,
        test_v1hm_solicitacao_template_exists,
        test_v1hm_checklist_template_exists,
        test_v1hm_tracker_schema_exists,
        test_v1hm_intake_schema_exists,
        test_v1hm_provenance_schema_exists,
        test_v1hm_tracker_registry_exists,
        test_v1hm_intake_registry_exists,
        test_v1hm_provenance_registry_exists,
        test_v1hm_tracker_schema_required_fields,
        test_v1hm_intake_schema_required_fields,
        test_v1hm_provenance_schema_required_fields,
        test_v1hm_tracker_controlled_values,
        test_v1hm_intake_controlled_values,
        test_v1hm_provenance_controlled_values,
        test_v1hm_unknown_license_blocks_ground_truth,
        test_v1hm_raw_data_publication_blocked_by_default,
        test_v1hm_operational_ground_truth_never_true,
        test_v1hm_dino_cannot_close_ground_truth_gate,
        test_v1hm_method_reference_not_applied_to_patches,
        test_v1hm_no_supervised_training_in_tracker,
        test_v1hm_no_supervised_training_in_intake,
        test_v1hm_no_private_paths_in_tracker,
        test_v1hm_no_private_paths_in_intake,
        test_v1hm_no_private_paths_in_provenance,
        test_v1hm_no_private_paths_in_operational_doc,
        test_v1hm_forbidden_use_blocks_flood_detection_in_tracker,
        test_v1hm_forbidden_use_blocks_flood_prediction_in_intake,
        test_v1hm_operational_doc_mentions_metadata_first,
        test_v1hm_operational_doc_mentions_local_only,
        test_v1hm_operational_doc_mentions_license,
        test_v1hm_operational_doc_mentions_multimodal_hold,
        test_v1hm_runbook_mentions_no_download_this_stage,
        test_v1hm_template_contains_placeholders,
        test_v1hm_checklist_contains_license_block,
    ]
    for fn in test_functions:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {fn.__name__}: {e}")
            failed.append(fn.__name__)
    if failed:
        print(f"\n{len(failed)}/{len(test_functions)} tests failed")
        sys.exit(1)
    else:
        print(f"\nAll {len(test_functions)} tests passed")
        sys.exit(0)
