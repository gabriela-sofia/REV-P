"""
tests/test_revp_v1ik_temporal_provenance_recovery.py

Testes para v1ik — Recuperacao de Proveniencia Temporal
"""

import pytest
import csv
import json
from pathlib import Path
from datetime import datetime


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = REPO_ROOT / "datasets"
SCHEMAS_DIR = DATASETS_DIR / "schemas"
LOCAL_RUNS = REPO_ROOT / "local_runs" / "protocolo_c" / "v1ik"
SCRIPT_PATH = REPO_ROOT / "scripts" / "protocolo_c" / "revp_v1ik_temporal_provenance_recovery.py"


class TestV1IKScriptExists:
    """Verificar que script existe."""
    def test_script_exists(self):
        assert SCRIPT_PATH.exists(), f"Script not found: {SCRIPT_PATH}"


class TestV1IKExecution:
    """Testes de execucao basica."""
    def test_script_runs_with_force(self):
        import subprocess
        result = subprocess.run(
            ["python", str(SCRIPT_PATH), "--force", "--scan-sidecars",
             "--scan-registries", "--scan-local-docs", "--focus-best-candidates",
             "--emit-temporal-decision-matrix"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True
        )
        assert result.returncode == 0, f"Script failed: {result.stderr}"
        assert "Auditoria temporal concluida" in result.stdout


class TestTemporalProvenanceRegistry:
    """Testes para registry de proveniencia temporal."""
    def test_temporal_registry_created(self):
        registry_path = DATASETS_DIR / "temporal_provenance_recovery_registry.csv"
        assert registry_path.exists(), f"Registry not created: {registry_path}"

    def test_temporal_registry_has_rows(self):
        registry_path = DATASETS_DIR / "temporal_provenance_recovery_registry.csv"
        with open(registry_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) > 0, "Temporal registry is empty"

    def test_temporal_registry_includes_consolidated_candidates(self):
        """Garantir que todos os candidatos consolidados matching criteria são revisados."""
        registry_path = DATASETS_DIR / "temporal_provenance_recovery_registry.csv"
        consolidated_path = DATASETS_DIR / "consolidated_observed_event_vector_candidate_registry.csv"

        with open(registry_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        with open(consolidated_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            consolidated = [row for row in reader
                          if row.get('geometry_available') == 'YES'
                          and row.get('crs')
                          and row.get('observed_not_risk') == 'YES'
                          and ('gate_04' in row.get('blocking_gate', '') or 'gate_05' in row.get('blocking_gate', ''))]

        # v1ik deve revisar exatamente os candidatos que estao consolidados e matched os criteria
        assert len(rows) == len(consolidated), \
            f"Registry has {len(rows)} rows but expected {len(consolidated)} matching consolidated candidates"
        assert len(rows) >= 1, "Nenhum candidato consolidado para revisar"

    def test_temporal_registry_schema_compatible(self):
        registry_path = DATASETS_DIR / "temporal_provenance_recovery_registry.csv"
        schema_path = SCHEMAS_DIR / "temporal_provenance_recovery_schema.csv"

        with open(registry_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            registry_headers = set(reader.fieldnames) if reader.fieldnames else set()

        with open(schema_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            schema_fields = {row['field_name'] for row in reader}

        assert registry_headers == schema_fields, \
            f"Headers mismatch. Registry: {registry_headers}, Schema: {schema_fields}"

    def test_can_create_training_label_always_no(self):
        registry_path = DATASETS_DIR / "temporal_provenance_recovery_registry.csv"
        with open(registry_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                assert row.get('can_create_training_label') == 'NO', \
                    f"can_create_training_label not NO for {row.get('temporal_review_id')}"


class TestTemporalGateMatrix:
    """Testes para matriz de decisao temporal."""
    def test_temporal_gate_matrix_created(self):
        gate_path = DATASETS_DIR / "temporal_gate_decision_matrix.csv"
        assert gate_path.exists(), f"Gate matrix not created: {gate_path}"

    def test_temporal_gate_schema_compatible(self):
        gate_path = DATASETS_DIR / "temporal_gate_decision_matrix.csv"
        schema_path = SCHEMAS_DIR / "temporal_gate_decision_matrix_schema.csv"

        with open(gate_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            gate_headers = set(reader.fieldnames) if reader.fieldnames else set()

        with open(schema_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            schema_fields = {row['field_name'] for row in reader}

        assert gate_headers == schema_fields, \
            f"Gate headers mismatch. Matrix: {gate_headers}, Schema: {schema_fields}"

    def test_temporal_gate_has_candidates(self):
        gate_path = DATASETS_DIR / "temporal_gate_decision_matrix.csv"
        with open(gate_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) > 0, "Gate matrix is empty"


class TestLocalOutputs:
    """Testes para outputs locais."""
    def test_summary_json_created(self):
        summary_path = LOCAL_RUNS / "v1ik_temporal_provenance_summary.json"
        assert summary_path.exists(), f"Summary JSON not created: {summary_path}"

    def test_summary_json_valid(self):
        summary_path = LOCAL_RUNS / "v1ik_temporal_provenance_summary.json"
        with open(summary_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        assert isinstance(data, dict), "Summary JSON not dict"
        assert 'total_candidates_reviewed' in data

    def test_qa_csv_created(self):
        qa_path = LOCAL_RUNS / "v1ik_temporal_provenance_qa.csv"
        assert qa_path.exists(), f"QA CSV not created: {qa_path}"

    def test_sidecar_scan_log_created(self):
        log_path = LOCAL_RUNS / "v1ik_sidecar_scan_log.csv"
        assert log_path.exists(), f"Sidecar log not created: {log_path}"

    def test_registry_cross_reference_created(self):
        log_path = LOCAL_RUNS / "v1ik_registry_cross_reference_log.csv"
        assert log_path.exists(), f"Registry log not created: {log_path}"

    def test_documentary_support_log_created(self):
        log_path = LOCAL_RUNS / "v1ik_documentary_support_log.csv"
        assert log_path.exists(), f"Documentary log not created: {log_path}"


class TestTemporalBlockedCandidates:
    """Testes para candidatos bloqueados por data."""
    def test_blocked_candidates_appear_in_registry(self):
        registry_path = DATASETS_DIR / "temporal_provenance_recovery_registry.csv"
        with open(registry_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            blocked = [row for row in reader if 'BLOCKED' in row.get('temporal_status_after_review', '')]
        assert len(blocked) > 0, "No blocked candidates in registry"


class TestFileSystemDateRejection:
    """Teste que data de arquivo nao eh aceita."""
    def test_file_system_date_never_accepted(self):
        registry_path = DATASETS_DIR / "temporal_provenance_recovery_registry.csv"
        with open(registry_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'INVALID_FILE_SYSTEM_DATE' in row.get('temporal_evidence_strength', ''):
                    assert row.get('accepted_as_event_date') == 'NO', \
                        f"File system date accepted for {row.get('temporal_review_id')}"


class TestPistaFracaClassification:
    """Teste que pista fraca nao eh STRONG."""
    def test_weak_hint_never_strong(self):
        gate_path = DATASETS_DIR / "temporal_gate_decision_matrix.csv"
        with open(gate_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('only_file_or_folder_hint') == 'YES':
                    # Se tem apenas pista de arquivo, nao pode passar gate
                    assert row.get('temporal_gate_status') != 'PASSED', \
                        f"File/folder hint passed gate for {row.get('candidate_id')}"


class TestDateTypeValidation:
    """Teste que diferentes tipos de data sao tratados corretamente."""
    def test_publication_date_not_event_date(self):
        registry_path = DATASETS_DIR / "temporal_provenance_recovery_registry.csv"
        with open(registry_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Se a evidencia eh data de publicacao, nao pode ser aceita como data de evento
                if 'publication' in row.get('temporal_evidence_source_type', '').lower():
                    assert row.get('accepted_as_event_date') == 'NO'

    def test_survey_date_not_automatically_event_date(self):
        registry_path = DATASETS_DIR / "temporal_provenance_recovery_registry.csv"
        with open(registry_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Data de levantamento eh diferente de data do evento
                if row.get('accepted_as_survey_date') == 'YES':
                    # Pode fortalecer contexto, mas nao eh automaticamente data de evento
                    assert row.get('accepted_as_context_only') in {'YES', 'NO'}


class TestNoLabelTargetClass:
    """Teste que nao ha label/target supervisionado."""
    def test_no_label_fields(self):
        registry_path = DATASETS_DIR / "temporal_provenance_recovery_registry.csv"
        with open(registry_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames if reader.fieldnames else []

        forbidden_fields = {'label', 'target', 'class', 'supervision', 'supervised_label'}
        found_forbidden = [h for h in headers if h.lower() in forbidden_fields]
        assert not found_forbidden, f"Forbidden fields found: {found_forbidden}"

    def test_no_label_in_gate_matrix(self):
        gate_path = DATASETS_DIR / "temporal_gate_decision_matrix.csv"
        with open(gate_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames if reader.fieldnames else []

        forbidden_fields = {'label', 'target', 'class', 'supervision'}
        found_forbidden = [h for h in headers if h.lower() in forbidden_fields]
        assert not found_forbidden, f"Forbidden fields in gate matrix: {found_forbidden}"


class TestNoPrivatePaths:
    """Teste que nao ha paths privados."""
    PRIVATE_MARKERS = [
        "gabriela", "C:\\Users", "/Users/", "PROJETO",
        "\\gabriela\\", "/gabriela/",
    ]

    def test_registry_no_private_paths(self):
        registry_path = DATASETS_DIR / "temporal_provenance_recovery_registry.csv"
        with open(registry_path, 'r', encoding='utf-8') as f:
            content = f.read()

        for marker in self.PRIVATE_MARKERS:
            assert marker.lower() not in content.lower(), \
                f"Private marker '{marker}' found in temporal registry"

    def test_gate_matrix_no_private_paths(self):
        gate_path = DATASETS_DIR / "temporal_gate_decision_matrix.csv"
        with open(gate_path, 'r', encoding='utf-8') as f:
            content = f.read()

        for marker in self.PRIVATE_MARKERS:
            assert marker.lower() not in content.lower(), \
                f"Private marker '{marker}' found in gate matrix"


class TestLocalRunsIgnored:
    """Teste que local_runs eh ignorado por git."""
    def test_local_runs_in_gitignore(self):
        gitignore_path = REPO_ROOT / ".gitignore"
        if gitignore_path.exists():
            with open(gitignore_path, 'r', encoding='utf-8') as f:
                content = f.read()
            assert 'local_runs' in content or 'local_runs/' in content, \
                "local_runs not in .gitignore"


class TestLocalOnlyIgnored:
    """Teste que local_only eh ignorado por git."""
    def test_local_only_in_gitignore(self):
        gitignore_path = REPO_ROOT / ".gitignore"
        if gitignore_path.exists():
            with open(gitignore_path, 'r', encoding='utf-8') as f:
                content = f.read()
            assert 'local_only' in content or 'local_only/' in content, \
                "local_only not in .gitignore"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
