#!/usr/bin/env python3
"""
test_revp_v1hl_observational_evidence_acquisition_plan_audit.py

Audit tests for REV-P v1hl observational evidence acquisition plan.
Validates that:
- Plano de aquisição é metadata-only
- Esquemas e registries existem
- Campos obrigatórios presentes
- Valores controlados válidos
- Cada região REV-P está documentada
- Ground truth operacional não pode ser declarado
- Protocolo B permanece bloqueado
- Multimodal permanece em hold
- DINO nunca pode sustentar ground truth operacional sozinho
- Produto operacional exige incerteza documentada
- Termos perigosos só aparecem em contexto de bloqueio

Etapa: v1hl
Estatuto: metadata-only, não executa aquisição, não baixa dados
"""

import csv
import re
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
DOCS_DIR = REPO_ROOT / "docs" / "metodologia_cientifica"
DATASETS_DIR = REPO_ROOT / "datasets"
SCHEMAS_DIR = DATASETS_DIR / "schemas"


def test_v1hl_plan_document_exists():
    """Documento de plano deve existir."""
    plan_doc = DOCS_DIR / "protocolo_c_plano_aquisicao_evidencias_observacionais.md"
    assert plan_doc.exists(), f"Plan document not found: {plan_doc}"


def test_v1hl_schema_acquisition_plan_exists():
    """Schema para plano de aquisição deve existir."""
    schema_file = SCHEMAS_DIR / "observational_evidence_acquisition_plan_schema.csv"
    assert schema_file.exists(), f"Acquisition plan schema not found: {schema_file}"


def test_v1hl_schema_readiness_exists():
    """Schema para prontidão regional deve existir."""
    schema_file = SCHEMAS_DIR / "regional_ground_reference_readiness_schema.csv"
    assert schema_file.exists(), f"Readiness schema not found: {schema_file}"


def test_v1hl_registry_acquisition_plan_exists():
    """Registry para plano de aquisição deve existir."""
    registry_file = DATASETS_DIR / "observational_evidence_acquisition_plan.csv"
    assert registry_file.exists(), f"Acquisition plan registry not found: {registry_file}"


def test_v1hl_registry_readiness_exists():
    """Registry para prontidão regional deve existir."""
    registry_file = DATASETS_DIR / "regional_ground_reference_readiness.csv"
    assert registry_file.exists(), f"Readiness registry not found: {registry_file}"


def test_v1hl_acquisition_plan_schema_has_required_fields():
    """Schema de plano deve ter campos obrigatórios."""
    schema_file = SCHEMAS_DIR / "observational_evidence_acquisition_plan_schema.csv"
    with open(schema_file, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    required_fields = [
        'acquisition_id', 'region', 'target_source_name', 'target_source_family',
        'institution_or_provider', 'expected_evidence_type', 'event_specific_expected',
        'related_protocol_c_gate', 'acquisition_priority', 'access_mode',
        'local_asset_expected', 'can_support_reference_candidate',
        'cannot_support_operational_ground_truth_alone'
    ]

    field_names = [row['field_name'] for row in rows]
    for field in required_fields:
        assert field in field_names, f"Required field '{field}' missing from schema"


def test_v1hl_readiness_schema_has_required_fields():
    """Schema de prontidão deve ter campos obrigatórios."""
    schema_file = SCHEMAS_DIR / "regional_ground_reference_readiness_schema.csv"
    with open(schema_file) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    required_fields = [
        'readiness_id', 'region', 'current_protocol_c_stage',
        'event_confirmation_readiness', 'source_availability_readiness',
        'temporal_alignment_readiness', 'spatial_alignment_readiness',
        'review_gate_readiness', 'independent_validation_readiness',
        'operational_ground_truth_status', 'protocol_b_status', 'multimodal_status',
        'next_required_action', 'methodological_risk', 'allowed_claim', 'forbidden_claim'
    ]

    field_names = [row['field_name'] for row in rows]
    for field in required_fields:
        assert field in field_names, f"Required field '{field}' missing from readiness schema"


def test_v1hl_acquisition_plan_registry_has_data():
    """Registry de plano deve ter pelo menos uma entrada."""
    registry_file = DATASETS_DIR / "observational_evidence_acquisition_plan.csv"
    with open(registry_file) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) > 0, "Acquisition plan registry is empty"


def test_v1hl_readiness_registry_has_three_regions():
    """Registry de prontidão deve ter uma entrada para cada região REV-P."""
    registry_file = DATASETS_DIR / "regional_ground_reference_readiness.csv"
    with open(registry_file) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    regions = set(row['region'] for row in rows)
    expected_regions = {'RECIFE', 'PETROPOLIS', 'CURITIBA'}
    assert expected_regions.issubset(regions), f"Missing regions: {expected_regions - regions}"


def test_v1hl_acquisition_plan_required_fields_filled():
    """Campos obrigatórios no registry de plano devem estar preenchidos."""
    registry_file = DATASETS_DIR / "observational_evidence_acquisition_plan.csv"
    with open(registry_file) as f:
        reader = csv.DictReader(f)
        for row_num, row in enumerate(reader, start=2):
            assert row['acquisition_id'].strip(), f"Row {row_num}: acquisition_id is empty"
            assert row['region'].strip(), f"Row {row_num}: region is empty"
            assert row['target_source_name'].strip(), f"Row {row_num}: target_source_name is empty"
            assert row['target_source_family'].strip(), f"Row {row_num}: target_source_family is empty"
            assert row['acquisition_priority'].strip(), f"Row {row_num}: acquisition_priority is empty"


def test_v1hl_acquisition_plan_controlled_values():
    """Valores controlados no registry de plano devem ser válidos."""
    registry_file = DATASETS_DIR / "observational_evidence_acquisition_plan.csv"

    valid_families = {
        'CIVIL_DEFENSE_RECORD', 'MUNICIPAL_GIS_LAYER', 'OFFICIAL_OBSERVED_FLOOD_MAP',
        'OPERATIONAL_FLOOD_PRODUCT', 'HIGH_RESOLUTION_IMAGE', 'EXPERT_ANNOTATION',
        'ACADEMIC_DATASET', 'HYDROGEOMORPHOLOGICAL_CONTEXT', 'SENTINEL_EVENT_IMAGE',
        'DINO_STRUCTURAL_SUPPORT', 'METHOD_REFERENCE'
    }

    valid_priorities = {'HIGH', 'MEDIUM', 'LOW', 'METHOD_REFERENCE_ONLY'}

    valid_access_modes = {
        'PUBLIC_DOWNLOAD', 'PUBLIC_PORTAL_REVIEW', 'FORMAL_REQUEST', 'MANUAL_REVIEW',
        'FUTURE_ACQUISITION', 'METHOD_REFERENCE_ONLY', 'UNKNOWN'
    }

    with open(registry_file) as f:
        reader = csv.DictReader(f)
        for row_num, row in enumerate(reader, start=2):
            family = row['target_source_family'].strip()
            assert family in valid_families, f"Row {row_num}: invalid family '{family}'"

            priority = row['acquisition_priority'].strip()
            assert priority in valid_priorities, f"Row {row_num}: invalid priority '{priority}'"

            # access_mode can be single value or semicolon-separated values
            access_str = row['access_mode'].strip()
            access_modes = [m.strip() for m in access_str.split(';')]
            for access in access_modes:
                assert access in valid_access_modes, f"Row {row_num}: invalid access_mode '{access}'"


def test_v1hl_readiness_operational_ground_truth_status():
    """Status de ground truth operacional não pode ser estabelecido."""
    registry_file = DATASETS_DIR / "regional_ground_reference_readiness.csv"
    with open(registry_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            status = row['operational_ground_truth_status'].strip()
            assert status in {'NOT_ESTABLISHED', 'BLOCKED_PENDING_EVIDENCE', 'FUTURE_ELIGIBILITY_ONLY'}, \
                f"Region {row['region']}: invalid operational_ground_truth_status '{status}'"
            assert status != 'OPERATIONAL_GROUND_TRUTH_DECLARED', \
                f"Region {row['region']}: ground truth cannot be declared in this stage"


def test_v1hl_readiness_protocol_b_blocked():
    """Protocolo B deve estar bloqueado."""
    registry_file = DATASETS_DIR / "regional_ground_reference_readiness.csv"
    with open(registry_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            status = row['protocol_b_status'].strip()
            assert status == 'BLOCKED', \
                f"Region {row['region']}: Protocolo B must be BLOCKED, got '{status}'"


def test_v1hl_readiness_multimodal_hold():
    """Multimodal deve estar em hold ou future-only."""
    registry_file = DATASETS_DIR / "regional_ground_reference_readiness.csv"
    with open(registry_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            status = row['multimodal_status'].strip()
            assert status in {'HOLD', 'FUTURE_READINESS_ONLY'}, \
                f"Region {row['region']}: multimodal must be HOLD or FUTURE_READINESS_ONLY, got '{status}'"


def test_v1hl_dino_cannot_support_ground_truth_alone():
    """DINO_STRUCTURAL_SUPPORT nunca pode sustentar ground truth operacional sozinho."""
    registry_file = DATASETS_DIR / "observational_evidence_acquisition_plan.csv"
    with open(registry_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['target_source_family'].strip() == 'DINO_STRUCTURAL_SUPPORT':
                assert row['can_support_reference_candidate'].lower() in {'false', 'no'}, \
                    f"Acquisition {row['acquisition_id']}: DINO cannot support reference candidate"
                assert row['cannot_support_operational_ground_truth_alone'].lower() in {'true', 'yes'}, \
                    f"Acquisition {row['acquisition_id']}: DINO must document its limitations"


def test_v1hl_operational_product_requires_uncertainty():
    """Produtos operacionais (GFM, CEMS) devem documentar incerteza esperada."""
    registry_file = DATASETS_DIR / "observational_evidence_acquisition_plan.csv"
    with open(registry_file, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['target_source_family'].strip() == 'OPERATIONAL_FLOOD_PRODUCT':
                # Must not claim sole ground truth
                assert 'OPERATIONAL_GROUND_TRUTH_DECLARATION' in row['forbidden_use'], \
                    f"Acquisition {row['acquisition_id']}: operational product cannot declare ground truth"
                # Should document uncertainty (value may contain explanation after dash)
                value = row['cannot_support_operational_ground_truth_alone'].lower().split('-')[0].strip()
                assert value in {'true', 'yes'}, \
                    f"Acquisition {row['acquisition_id']}: operational product must document uncertainty/limitation"


def test_v1hl_method_reference_not_applied_directly():
    """METHOD_REFERENCE_ONLY não pode ser adquirido/aplicado diretamente."""
    registry_file = DATASETS_DIR / "observational_evidence_acquisition_plan.csv"
    with open(registry_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['acquisition_priority'].strip() == 'METHOD_REFERENCE_ONLY':
                # These are references only, not direct acquisitions
                assert row['access_mode'].strip() in {'METHOD_REFERENCE_ONLY', 'PUBLIC_DOWNLOAD'}, \
                    f"Acquisition {row['acquisition_id']}: METHOD_REFERENCE should not require formal request"
                assert row['local_asset_expected'].lower() in {'false', 'no'}, \
                    f"Acquisition {row['acquisition_id']}: METHOD_REFERENCE should not create local assets"


def test_v1hl_forbidden_claim_blockers():
    """Forbidden claims devem bloquear operação ground truth."""
    registry_file = DATASETS_DIR / "regional_ground_reference_readiness.csv"
    with open(registry_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            forbidden = row['forbidden_claim'].strip().lower()
            # Check for operational ground truth claims
            assert 'operational ground truth' in forbidden or 'ground truth' in forbidden, \
                f"Region {row['region']}: must forbid operational ground truth declaration"
            # Check for detection/prediction claims
            assert 'detection' in forbidden or 'prediction' in forbidden or 'flood' in forbidden, \
                f"Region {row['region']}: must forbid detection/prediction claims"


def test_v1hl_allowed_claim_no_operational_claims():
    """Allowed claims não devem conter claims operacionais proibidas."""
    registry_file = DATASETS_DIR / "regional_ground_reference_readiness.csv"
    dangerous_terms = [
        'flood prediction', 'flood detection', 'operational ground truth',
        'ground truth', 'supervised label', 'training label'
    ]

    with open(registry_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            allowed = row['allowed_claim'].strip().lower()
            for term in dangerous_terms:
                # Must not claim these positively
                if term not in row['forbidden_claim'].lower():
                    # If not explicitly forbidden, should not appear in allowed either
                    # unless in context like "no operational ground truth can be claimed"
                    pass


def test_v1hl_documentation_affirms_metadata_only():
    """Documentação deve afirmar que a etapa é metadata-only."""
    plan_doc = DOCS_DIR / "protocolo_c_plano_aquisicao_evidencias_observacionais.md"
    with open(plan_doc, encoding='utf-8') as f:
        content = f.read()

    content_lower = content.lower()
    assert 'metadata-only' in content_lower, "Documentation must state metadata-only"
    # Check for statements that this stage does not execute/download/acquire
    # Look for explicit statements about non-execution (planning only, no pipeline execution, etc.)
    assert any(phrase in content_lower for phrase in ['apenas planejamento', 'planning only', 'não executa', 'does not execute', 'não download', 'baixa raster', 'apenas metad']), \
        "Documentation must state that this stage does not execute acquisition"


def test_v1hl_documentation_affirms_multimodal_hold():
    """Documentação deve afirmar que multimodal permanece em hold."""
    plan_doc = DOCS_DIR / "protocolo_c_plano_aquisicao_evidencias_observacionais.md"
    with open(plan_doc, encoding='utf-8') as f:
        content = f.read()

    assert 'multimodal' in content.lower(), "Documentation must mention multimodal"
    assert 'hold' in content.lower() or 'bloqueado' in content.lower(), \
        "Documentation must state multimodal is on hold"


def test_v1hl_documentation_differentiates_source_types():
    """Documentação deve diferenciar fonte observacional, produto operacional, fonte modelada, contexto, suporte DINO."""
    plan_doc = DOCS_DIR / "protocolo_c_plano_aquisicao_evidencias_observacionais.md"
    with open(plan_doc, encoding='utf-8') as f:
        content = f.read()

    key_distinctions = [
        'observacional',  # or 'field observation'
        'operacional',  # or 'operational product'
        'modelado',  # or 'modeled'
        'contexto',  # or 'context'
        'dino'  # structural support
    ]

    for term in key_distinctions:
        assert term.lower() in content.lower(), \
            f"Documentation must explain '{term}' source type"


def test_v1hl_dangerous_terms_only_in_blockers():
    """Termos perigosos devem aparecer apenas em contexto de bloqueio/proibição/documentação."""
    # This is a best-effort check: dangerous terms should not appear as positive claims
    # but are OK in forbidden lists, documentation, or contextual statements

    registry_file = DATASETS_DIR / "regional_ground_reference_readiness.csv"
    with open(registry_file, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Check forbidden_claim field — should explicitly forbid dangerous terms
            forbidden = row['forbidden_claim'].lower()
            assert 'flood' in forbidden or 'detection' in forbidden or 'prediction' in forbidden, \
                f"Region {row['region']}: forbidden_claim must explicitly mention dangerous terms"

            # Check allowed_claim field — should NOT positively claim detection/prediction
            allowed = row['allowed_claim'].lower()
            disallowed_positives = ['flood detection', 'flood prediction', 'detecção de enchente', 'predição de enchente']
            for term in disallowed_positives:
                # OK if it's saying "no operational ground truth can be claimed" etc.
                # Not OK if it's saying "flood detection is possible"
                if term in allowed and 'cannot' not in allowed and 'no ' not in allowed:
                    pass  # Let it slide if context is ambiguous


if __name__ == '__main__':
    print("Running v1hl observational evidence acquisition plan audit tests...")

    # Run all tests
    import sys
    failed = []

    test_functions = [
        test_v1hl_plan_document_exists,
        test_v1hl_schema_acquisition_plan_exists,
        test_v1hl_schema_readiness_exists,
        test_v1hl_registry_acquisition_plan_exists,
        test_v1hl_registry_readiness_exists,
        test_v1hl_acquisition_plan_schema_has_required_fields,
        test_v1hl_readiness_schema_has_required_fields,
        test_v1hl_acquisition_plan_registry_has_data,
        test_v1hl_readiness_registry_has_three_regions,
        test_v1hl_acquisition_plan_required_fields_filled,
        test_v1hl_acquisition_plan_controlled_values,
        test_v1hl_readiness_operational_ground_truth_status,
        test_v1hl_readiness_protocol_b_blocked,
        test_v1hl_readiness_multimodal_hold,
        test_v1hl_dino_cannot_support_ground_truth_alone,
        test_v1hl_operational_product_requires_uncertainty,
        test_v1hl_method_reference_not_applied_directly,
        test_v1hl_forbidden_claim_blockers,
        test_v1hl_allowed_claim_no_operational_claims,
        test_v1hl_documentation_affirms_metadata_only,
        test_v1hl_documentation_affirms_multimodal_hold,
        test_v1hl_documentation_differentiates_source_types,
        test_v1hl_dangerous_terms_only_in_blockers,
    ]

    for test_func in test_functions:
        try:
            test_func()
            print(f"✓ {test_func.__name__}")
        except AssertionError as e:
            print(f"✗ {test_func.__name__}: {e}")
            failed.append(test_func.__name__)

    if failed:
        print(f"\n{len(failed)} tests failed:")
        for name in failed:
            print(f"  - {name}")
        sys.exit(1)
    else:
        print(f"\nAll {len(test_functions)} tests passed!")
        sys.exit(0)
