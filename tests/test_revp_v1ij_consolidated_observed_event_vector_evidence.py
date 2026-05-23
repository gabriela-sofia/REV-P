"""
tests/test_revp_v1ij_consolidated_observed_event_vector_evidence.py

Testes para v1ij — Consolidacao de Candidatos Vetoriais Observados
"""

import pytest
import csv
import json
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = REPO_ROOT / "datasets"
SCHEMAS_DIR = DATASETS_DIR / "schemas"
LOCAL_RUNS = REPO_ROOT / "local_runs" / "protocolo_c" / "v1ij"
SCRIPT_PATH = REPO_ROOT / "scripts" / "protocolo_c" / "revp_v1ij_consolidated_observed_event_vector_evidence.py"


class TestV1IJScriptExists:
    """Verificar que script existe."""
    def test_script_exists(self):
        assert SCRIPT_PATH.exists(), f"Script not found: {SCRIPT_PATH}"


class TestV1IJExecution:
    """Testes de execucao basica."""
    def test_script_runs_with_force(self):
        import subprocess
        result = subprocess.run(
            ["python", str(SCRIPT_PATH), "--force", "--enrich-metadata",
             "--scan-local-sidecars", "--emit-patch-binding-preflight"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True
        )
        assert result.returncode == 0, f"Script failed: {result.stderr}"
        assert "Consolidacao concluida" in result.stdout


class TestConsolidatedRegistry:
    """Testes para registry consolidado."""
    def test_consolidated_registry_created(self):
        registry_path = DATASETS_DIR / "consolidated_observed_event_vector_candidate_registry.csv"
        assert registry_path.exists(), f"Registry not created: {registry_path}"

    def test_consolidated_registry_has_rows(self):
        registry_path = DATASETS_DIR / "consolidated_observed_event_vector_candidate_registry.csv"
        with open(registry_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) > 0, "Registry is empty"

    def test_consolidated_registry_schema_compatible(self):
        registry_path = DATASETS_DIR / "consolidated_observed_event_vector_candidate_registry.csv"
        schema_path = SCHEMAS_DIR / "consolidated_observed_event_vector_candidate_schema.csv"

        with open(registry_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            registry_headers = set(reader.fieldnames) if reader.fieldnames else set()

        with open(schema_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            schema_fields = {row['field_name'] for row in reader}

        assert registry_headers == schema_fields, \
            f"Headers mismatch. Registry: {registry_headers}, Schema: {schema_fields}"

    def test_can_create_training_label_always_false(self):
        registry_path = DATASETS_DIR / "consolidated_observed_event_vector_candidate_registry.csv"
        with open(registry_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                assert row.get('can_create_training_label') == 'NO', \
                    f"can_create_training_label not NO for {row.get('consolidated_candidate_id')}"


class TestGateMatrix:
    """Testes para matriz de gates."""
    def test_gate_matrix_created(self):
        gate_path = DATASETS_DIR / "consolidated_event_vector_gate_matrix.csv"
        assert gate_path.exists(), f"Gate matrix not created: {gate_path}"

    def test_gate_matrix_schema_compatible(self):
        gate_path = DATASETS_DIR / "consolidated_event_vector_gate_matrix.csv"
        schema_path = SCHEMAS_DIR / "consolidated_event_vector_gate_matrix_schema.csv"

        with open(gate_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            gate_headers = set(reader.fieldnames) if reader.fieldnames else set()

        with open(schema_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            schema_fields = {row['field_name'] for row in reader}

        assert gate_headers == schema_fields, \
            f"Gate headers mismatch. Matrix: {gate_headers}, Schema: {schema_fields}"

    def test_gate_matrix_has_candidates(self):
        gate_path = DATASETS_DIR / "consolidated_event_vector_gate_matrix.csv"
        with open(gate_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) > 0, "Gate matrix is empty"


class TestPatchBindingPreflight:
    """Testes para patch binding preflight."""
    def test_patch_binding_preflight_created(self):
        preflight_path = DATASETS_DIR / "patch_binding_preflight_candidate_registry.csv"
        assert preflight_path.exists(), f"Preflight registry not created: {preflight_path}"

    def test_patch_binding_preflight_schema_compatible(self):
        preflight_path = DATASETS_DIR / "patch_binding_preflight_candidate_registry.csv"
        schema_path = SCHEMAS_DIR / "patch_binding_preflight_candidate_schema.csv"

        with open(preflight_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            preflight_headers = set(reader.fieldnames) if reader.fieldnames else set()

        with open(schema_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            schema_fields = {row['field_name'] for row in reader}

        assert preflight_headers == schema_fields, \
            f"Preflight headers mismatch. Registry: {preflight_headers}, Schema: {schema_fields}"

    def test_patch_binding_label_creation_always_no(self):
        preflight_path = DATASETS_DIR / "patch_binding_preflight_candidate_registry.csv"
        with open(preflight_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                assert row.get('label_creation_allowed') == 'NO', \
                    f"label_creation_allowed not NO for {row.get('patch_binding_candidate_id')}"


class TestNoLabelTargetClassFields:
    """Verificar que nao ha campos label/target/class."""
    def test_consolidated_registry_no_label_fields(self):
        registry_path = DATASETS_DIR / "consolidated_observed_event_vector_candidate_registry.csv"
        with open(registry_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames if reader.fieldnames else []

        forbidden_fields = {'label', 'target', 'class', 'supervision', 'supervised_label'}
        found_forbidden = [h for h in headers if h.lower() in forbidden_fields]
        assert not found_forbidden, f"Forbidden fields found: {found_forbidden}"

    def test_gate_matrix_no_label_fields(self):
        gate_path = DATASETS_DIR / "consolidated_event_vector_gate_matrix.csv"
        with open(gate_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames if reader.fieldnames else []

        forbidden_fields = {'label', 'target', 'class', 'supervision'}
        found_forbidden = [h for h in headers if h.lower() in forbidden_fields]
        assert not found_forbidden, f"Forbidden fields in gate matrix: {found_forbidden}"

    def test_preflight_no_label_fields(self):
        preflight_path = DATASETS_DIR / "patch_binding_preflight_candidate_registry.csv"
        with open(preflight_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames if reader.fieldnames else []

        forbidden_fields = {'label', 'target', 'class', 'supervision'}
        found_forbidden = [h for h in headers if h.lower() in forbidden_fields]
        assert not found_forbidden, f"Forbidden fields in preflight: {found_forbidden}"


class TestBlockingBehavior:
    """Testes de comportamento de bloqueio."""
    def test_candidates_without_date_blocked(self):
        registry_path = DATASETS_DIR / "consolidated_observed_event_vector_candidate_registry.csv"
        with open(registry_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('has_event_date') == 'NO':
                    assert row.get('ground_truth_candidate_status') == 'BLOCKED', \
                        f"Candidate without event_date not blocked: {row.get('consolidated_candidate_id')}"

    def test_risk_susceptibility_blocked(self):
        registry_path = DATASETS_DIR / "consolidated_observed_event_vector_candidate_registry.csv"
        with open(registry_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('observed_not_risk') == 'NO':
                    assert row.get('ground_truth_candidate_status') == 'BLOCKED', \
                        f"Candidate with risco/susceptibilidade not blocked: {row.get('consolidated_candidate_id')}"

    def test_pdf_blocked_as_vector(self):
        registry_path = DATASETS_DIR / "consolidated_observed_event_vector_candidate_registry.csv"
        with open(registry_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('asset_format', '').upper() == 'PDF':
                    assert row.get('ground_truth_candidate_status') == 'BLOCKED', \
                        f"PDF candidate not blocked: {row.get('consolidated_candidate_id')}"


class TestPatchBindingPreflightGates:
    """Testes para gates minimos de patch binding preflight."""
    def test_only_candidates_passing_all_gates_advance(self):
        registry_path = DATASETS_DIR / "consolidated_observed_event_vector_candidate_registry.csv"
        gate_path = DATASETS_DIR / "consolidated_event_vector_gate_matrix.csv"
        preflight_path = DATASETS_DIR / "patch_binding_preflight_candidate_registry.csv"

        # Load candidates
        candidates_by_id = {}
        with open(registry_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                candidates_by_id[row.get('consolidated_candidate_id')] = row

        # Load gates
        gates_by_id = {}
        with open(gate_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                gates_by_id[row.get('candidate_id')] = row

        # Check preflight
        with open(preflight_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            preflight_rows = list(reader)

        # For each preflight candidate, verify it passed all minimal gates
        for prow in preflight_rows:
            cand_id = prow.get('consolidated_candidate_id')
            if cand_id == 'N/A':
                continue  # status report row

            gate = gates_by_id.get(cand_id)
            assert gate is not None, f"Gate not found for {cand_id}"

            # Minimal gates
            minimal_gates = [
                gate.get('gate_vector_or_georeferenced_table'),
                gate.get('gate_crs_or_coordinate_reference'),
                gate.get('gate_event_date_available'),
                gate.get('gate_event_date_compatible'),
                gate.get('gate_phenomenon_available'),
                gate.get('gate_observed_not_risk'),
                gate.get('gate_spatial_unit_usable'),
            ]
            assert all(g == 'PASS' for g in minimal_gates), \
                f"Candidate {cand_id} in preflight but gates not all PASS"


class TestNoCandidatePassed:
    """Teste especifico para resultado nenhum candidato passou."""
    def test_no_candidate_passed_status_report(self):
        preflight_path = DATASETS_DIR / "patch_binding_preflight_candidate_registry.csv"
        with open(preflight_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # Should have exactly 1 row with status report
        assert len(rows) == 1, f"Expected 1 status report row, got {len(rows)}"

        row = rows[0]
        assert row.get('patch_binding_candidate_id') == 'STATUS_REPORT', \
            "Expected STATUS_REPORT in preflight"
        assert 'NO_CANDIDATE_PASSED' in row.get('asset_name', ''), \
            "Expected NO_CANDIDATE_PASSED message"


class TestNoPrivatePaths:
    """Verificar que nao ha paths privados em arquivos publicos."""
    PRIVATE_MARKERS = [
        "gabriela", "C:\\Users", "/Users/", "PROJETO",
        "\\gabriela\\", "/gabriela/",
    ]

    def test_consolidated_registry_no_private_paths(self):
        registry_path = DATASETS_DIR / "consolidated_observed_event_vector_candidate_registry.csv"
        with open(registry_path, 'r', encoding='utf-8') as f:
            content = f.read()

        for marker in self.PRIVATE_MARKERS:
            assert marker.lower() not in content.lower(), \
                f"Private marker '{marker}' found in consolidated registry"

    def test_gate_matrix_no_private_paths(self):
        gate_path = DATASETS_DIR / "consolidated_event_vector_gate_matrix.csv"
        with open(gate_path, 'r', encoding='utf-8') as f:
            content = f.read()

        for marker in self.PRIVATE_MARKERS:
            assert marker.lower() not in content.lower(), \
                f"Private marker '{marker}' found in gate matrix"

    def test_preflight_no_private_paths(self):
        preflight_path = DATASETS_DIR / "patch_binding_preflight_candidate_registry.csv"
        with open(preflight_path, 'r', encoding='utf-8') as f:
            content = f.read()

        for marker in self.PRIVATE_MARKERS:
            assert marker.lower() not in content.lower(), \
                f"Private marker '{marker}' found in preflight"


class TestLocalRunsNotVersionioned:
    """Verificar que local_runs nao esta versionado."""
    def test_local_runs_in_gitignore(self):
        gitignore_path = REPO_ROOT / ".gitignore"
        if gitignore_path.exists():
            with open(gitignore_path, 'r', encoding='utf-8') as f:
                content = f.read()
            assert 'local_runs' in content or 'local_runs/' in content, \
                "local_runs not in .gitignore"


class TestLocalOnlyNotVersionioned:
    """Verificar que local_only nao esta versionado."""
    def test_local_only_in_gitignore(self):
        gitignore_path = REPO_ROOT / ".gitignore"
        if gitignore_path.exists():
            with open(gitignore_path, 'r', encoding='utf-8') as f:
                content = f.read()
            assert 'local_only' in content or 'local_only/' in content, \
                "local_only not in .gitignore"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
